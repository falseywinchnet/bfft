"""The gauged positive-cone Bruun FFT: transport that conserves mass exactly.

CONTEXT. A positive-cone FFT lifts signed values x into two nonnegative
sheets (x+, x-) with x = x+ - x-. Every signed weight becomes routed
nonnegative transport: positive weights route straight, negative weights
route with a sheet SWAP. Subtraction disappears from the primitive set; in
its place stand splitter, coupler, route, sheet swap, absorber. Such a
machine is a candidate PHYSICAL Fourier device: the algebra is realizable as
two-rail circuits, power flows, or mass transport, with cancellation
happening only at explicit, local annihilator sites.

The reference implementation (bruun512_positive_forward_inverse.py) held the
sheets bounded with a crude global scalar, stage_scale = 0.5, compensated at
the end. That scalar is an anti-pathological hack, not an invariant: the
binomial wires of the Bruun circuit double their mass per stage while the
rotation wires grow only by c + s in (1, sqrt(2)], so one global constant
over-damps every rotation rail and the compensation factor re-inflates the
error floor.

THE THEOREM DELIVERED HERE (the mass-potential gauge).

  Let the forward circuit be a feed-forward composition of local cells M.
  Assign to every wire w a positive potential g_w by the backward recursion

      g_w = sum over cell outputs i fed by w of  g_i * |M_iw| ,

  with boundary g = 1 on the output (residue) wires -- one backward sweep of
  the all-ones covector through the ABSOLUTE-VALUE circuit. Re-express every
  cell in gauged units u_w = g_w * x_w, i.e. replace each weight M_iw by
  M~_iw = g_i * M_iw / g_w.  Then:

  (1) EXACT MASS PRESERVATION: every lifted cell is exactly column-
      stochastic (sum_i |M~_iw| = 1 for every input wire w). Total sheet
      mass is invariant under every transport cell. It is decremented only
      by explicit annihilation, giving the exact ledger
          mass_in = mass_out + sum over stages of annihilated mass.
  (2) EXACT PROJECTION: with input presented in gauged units (g_in * x,
      lifted) the machine's projected output equals the FFT exactly -- the
      gauges telescope, the output boundary is g = 1, no compensation
      factor exists.
  (3) UNIQUENESS: g is the unique positive gauge (up to the boundary
      normalization) achieving (1); it is the Perron left-vector of the
      absolute circuit, i.e. the physically calibrated "mass amplification
      to output" of each wire.

  Proof sketch: for the lift L(M) = [[M+, M-], [M-, M+]], column sums of
  |L(M~)| equal column sums of |M~|, which the recursion makes exactly 1;
  pi(p, n) = p - n intertwines L(M~) and M~, and annihilation subtracts
  equal amounts from both sheets, so it commutes with pi; composition
  telescopes g away except at the boundaries. Certified numerically below
  to machine precision, cell by cell, stage by stage.

Everything is parametric in N = 2^L (default 512 to match the reference).
Run this file for the full certificate suite and the stage-mass traces.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- tables --
def graydecode_int(g: int) -> int:
    x = int(g)
    while g:
        g >>= 1
        x ^= g
    return x


def bitrev_int(x: int, bits: int) -> int:
    y = 0
    for _ in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def bruun_idx_int(m: int, l: int) -> int:
    if m <= 0:
        return 0
    t = int(m).bit_length() - 1
    r = int(m) ^ (1 << t)
    return (2 * graydecode_int(bitrev_int(r, t)) + 1) << ((l - 2) - t)


def make_tables(n: int):
    l = n.bit_length() - 1
    idx = np.zeros(n // 2, dtype=np.int64)
    for m in range(1, n // 2):
        idx[m] = bruun_idx_int(m, l)
    c = np.zeros(n // 2, dtype=np.float64)
    c[1] = np.sqrt(0.5)
    for m in range(1, n // 4):
        cm = c[m]
        sm = np.sqrt(0.5) if m == 1 else c[m ^ 1]
        ce = np.sqrt(0.5 * (1.0 + cm))
        c[2 * m] = ce
        if 2 * m + 1 < n // 2:
            c[2 * m + 1] = sm / (2.0 * ce)
    return c, idx


def s_of(c: np.ndarray, m: int) -> float:
    return float(np.sqrt(0.5)) if m == 1 else float(c[m ^ 1])


# ------------------------------------------------------- circuit inventory --
# A cell is (out_slices, in_slices, edges); an edge is
# (dst_local, src_local, weight, flip). All slices in one cell have equal
# length (lane-aligned wires). This single description drives the signed
# reference, the backward potential sweep, and the gauged cone transport,
# so the three can never drift apart.

def forward_cells(n: int, c: np.ndarray):
    """Stages of the forward Bruun circuit as local cells."""
    stages = []
    half = n // 2
    # S0 fold: u = lo + hi, v = lo - hi
    stages.append([(
        [slice(0, half), slice(half, n)],          # out: sum, diff
        [slice(0, half), slice(half, n)],          # in: lo, hi
        [(0, 0, 1.0, False), (0, 1, 1.0, False),
         (1, 0, 1.0, False), (1, 1, 1.0, True)],
    )])
    lstages = n.bit_length() - 3                   # 512 -> stages 1..7
    for stage in range(1, lstages + 1):
        cells = []
        prefix = n >> stage
        h = prefix >> 1
        cells.append((
            [slice(0, h), slice(h, prefix)],
            [slice(0, h), slice(h, prefix)],
            [(0, 0, 1.0, False), (0, 1, 1.0, False),
             (1, 0, 1.0, False), (1, 1, 1.0, True)],
        ))
        q = (n // 4) >> stage
        for m in range(1, 2 ** stage):
            p = 4 * q * m
            cm = float(c[m])
            sm = s_of(c, m)
            sl = [slice(p, p + q), slice(p + q, p + 2 * q),
                  slice(p + 2 * q, p + 3 * q), slice(p + 3 * q, p + 4 * q)]
            # signed: R = c*B0 - s*B1 ; I = s*B0 + c*B1
            #   Y0 = A0 + R ; Y1 = A1 + I ; Y2 = A0 - R ; Y3 = I - A1
            # in locals: A0=0, B0=1, A1=2, B1=3 ; out Y0..Y3 = 0..3
            cells.append((sl, sl, [
                (0, 0, 1.0, False), (0, 1, cm, False), (0, 3, sm, True),
                (1, 2, 1.0, False), (1, 1, sm, False), (1, 3, cm, False),
                (2, 0, 1.0, False), (2, 1, cm, True), (2, 3, sm, False),
                (3, 1, sm, False), (3, 3, cm, False), (3, 2, 1.0, True),
            ]))
        stages.append(cells)
    return stages


def inverse_cells(n: int, c: np.ndarray):
    """Stages of the inverse circuit (cone geometry in reverse), composed to
    single 4x4 / 2x2 cells:
      A0 = (Y0+Y2)/2 ; A1 = (Y1-Y3)/2
      B0 = (c/2)Y0 + (s/2)Y1 - (c/2)Y2 + (s/2)Y3
      B1 = -(s/2)Y0 + (c/2)Y1 + (s/2)Y2 + (c/2)Y3
    """
    stages = []
    lstages = n.bit_length() - 3
    for stage in range(lstages, 0, -1):
        cells = []
        q = (n // 4) >> stage
        for m in range(1, 2 ** stage):
            p = 4 * q * m
            cm = float(c[m])
            sm = s_of(c, m)
            sl = [slice(p, p + q), slice(p + q, p + 2 * q),
                  slice(p + 2 * q, p + 3 * q), slice(p + 3 * q, p + 4 * q)]
            cells.append((sl, sl, [
                (0, 0, 0.5, False), (0, 2, 0.5, False),
                (1, 0, 0.5 * cm, False), (1, 1, 0.5 * sm, False),
                (1, 2, 0.5 * cm, True), (1, 3, 0.5 * sm, False),
                (2, 1, 0.5, False), (2, 3, 0.5, True),
                (3, 0, 0.5 * sm, True), (3, 1, 0.5 * cm, False),
                (3, 2, 0.5 * sm, False), (3, 3, 0.5 * cm, False),
            ]))
        prefix = n >> stage
        h = prefix >> 1
        cells.append((
            [slice(0, h), slice(h, prefix)],
            [slice(0, h), slice(h, prefix)],
            [(0, 0, 0.5, False), (0, 1, 0.5, False),
             (1, 0, 0.5, False), (1, 1, 0.5, True)],
        ))
        stages.append(cells)
    half = n // 2
    stages.append([(
        [slice(0, half), slice(half, n)],
        [slice(0, half), slice(half, n)],
        [(0, 0, 0.5, False), (0, 1, 0.5, False),
         (1, 0, 0.5, False), (1, 1, 0.5, True)],
    )])
    return stages


# ------------------------------------------------ the mass-potential gauge --
def backward_potential(n: int, stages) -> list[np.ndarray]:
    """g[t] = per-wire potential at the state boundary BEFORE stage t.
    Boundary condition g = 1 on the circuit's outputs; one backward sweep of
    the all-ones covector through the absolute-value circuit:
        g_in[src] = sum over edges of |w| * g_out[dst].
    Returns [g_before_stage_0, ..., g_before_stage_T, g_output]."""
    pots = [np.ones(n) for _ in range(len(stages) + 1)]
    for t in range(len(stages) - 1, -1, -1):
        g_out = pots[t + 1]
        g_in = g_out.copy()
        for out_sl, in_sl, edges in stages[t]:
            ins = {j for _, j, _, _ in edges}
            for j in ins:
                g_in[in_sl[j]] = 0.0
            for i, j, w, _flip in edges:
                g_in[in_sl[j]] += abs(w) * g_out[out_sl[i]]
        pots[t] = g_in
    return pots


# ------------------------------------------------------- cone transport --
def annihilate(wp: np.ndarray, wn: np.ndarray) -> float:
    """Explicit local annihilation; returns the sheet mass removed."""
    common = np.minimum(wp, wn)
    wp -= common
    wn -= common
    return float(2.0 * np.sum(common))


def apply_stage_cone(wp, wn, cells, g_in, g_out):
    """One stage of gauged nonnegative transport. Every effective weight
    g_out[dst] * |w| / g_in[src] is nonnegative; signs are carried purely by
    the flip (sheet swap) bit."""
    new_p = wp.copy()
    new_n = wn.copy()
    for out_sl, in_sl, edges in cells:
        outs = {i for i, _, _, _ in edges}
        for i in outs:
            new_p[out_sl[i]] = 0.0
            new_n[out_sl[i]] = 0.0
        for i, j, w, flip in edges:
            eff = g_out[out_sl[i]] * abs(w) / g_in[in_sl[j]]
            sp = wp[in_sl[j]]
            sn = wn[in_sl[j]]
            if flip != (w < 0):
                new_p[out_sl[i]] += eff * sn
                new_n[out_sl[i]] += eff * sp
            else:
                new_p[out_sl[i]] += eff * sp
                new_n[out_sl[i]] += eff * sn
    wp[:] = new_p
    wn[:] = new_n


def run_cone(x_or_work, stages, pots, annihilate_stage=True):
    """Gauged cone transport of a signed vector through `stages`.
    Input is lifted then expressed in gauged units (g_in * value); output is
    in true units because the output boundary gauge is 1.
    Returns (projected_output, trace)."""
    v = np.asarray(x_or_work, dtype=np.float64)
    wp = np.where(v >= 0, v, 0.0) * pots[0]
    wn = np.where(v < 0, -v, 0.0) * pots[0]
    trace = []
    mass_in = float(np.sum(wp) + np.sum(wn))
    burned_total = 0.0
    for t, cells in enumerate(stages):
        pre = float(np.sum(wp) + np.sum(wn))
        apply_stage_cone(wp, wn, cells, pots[t], pots[t + 1])
        post = float(np.sum(wp) + np.sum(wn))
        burned = annihilate(wp, wn) if annihilate_stage else 0.0
        burned_total += burned
        proj = wp - wn
        trace.append({
            "stage": t,
            "mass_before": pre,
            "mass_after_transport": post,
            "transport_defect": abs(post - pre) / max(pre, 1e-300),
            "annihilated": burned,
            "mass_after": post - burned,
            "projected_l1": float(np.sum(np.abs(proj))),
            "max_wire_mass": float(np.max(wp + wn)),
        })
    out = wp - wn
    ledger_defect = abs(mass_in - (float(np.sum(wp) + np.sum(wn)) + burned_total))
    return out, trace, mass_in, burned_total, ledger_defect


def run_signed(v, stages):
    """Signed reference through the SAME cell inventory (projection oracle)."""
    w = np.asarray(v, dtype=np.float64).copy()
    for cells in stages:
        nw = w.copy()
        for out_sl, in_sl, edges in cells:
            outs = {i for i, _, _, _ in edges}
            for i in outs:
                nw[out_sl[i]] = 0.0
            for i, j, wt, flip in edges:
                nw[out_sl[i]] += (-wt if flip else wt) * w[in_sl[j]]
        w = nw
    return w


# ------------------------------------------------------------ pack/unpack --
def pack_residue_to_rfft(work: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = work.shape[0]
    x = np.empty(n // 2 + 1, dtype=np.complex128)
    x[0] = work[0] + work[1]
    x[n // 2] = work[0] - work[1]
    for m in range(1, n // 2):
        x[idx[m]] = work[2 * m] - 1j * work[2 * m + 1]
    return x


def unpack_rfft_to_residue(x: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    work = np.empty(n, dtype=np.float64)
    work[0] = 0.5 * (x[0].real + x[n // 2].real)
    work[1] = 0.5 * (x[0].real - x[n // 2].real)
    for m in range(1, n // 2):
        work[2 * m] = x[idx[m]].real
        work[2 * m + 1] = -x[idx[m]].imag
    return work


# ----------------------------------------------------------- the machine --
class GaugedConeFFT:
    def __init__(self, n: int = 512):
        assert n >= 16 and (n & (n - 1)) == 0
        self.n = n
        self.C, self.IDX = make_tables(n)
        self.fwd = forward_cells(n, self.C)
        self.inv = inverse_cells(n, self.C)
        self.gf = backward_potential(n, self.fwd)
        self.gi = backward_potential(n, self.inv)

    def rfft(self, x, annihilate_stage=True):
        work, trace, m_in, burned, defect = run_cone(
            x, self.fwd, self.gf, annihilate_stage)
        return pack_residue_to_rfft(work, self.IDX), trace, m_in, burned, defect

    def irfft(self, X, annihilate_stage=True):
        work = unpack_rfft_to_residue(np.asarray(X), self.IDX, self.n)
        out, trace, m_in, burned, defect = run_cone(
            work, self.inv, self.gi, annihilate_stage)
        return out, trace, m_in, burned, defect


# ----------------------------------------------------- crude-scale baseline --
def run_cone_stage_scale(x, stages, scale=0.5):
    """The reference's crude stabilizer, on the same cell inventory, for the
    dynamic-range comparison. Compensated at the end."""
    v = np.asarray(x, dtype=np.float64)
    wp = np.where(v >= 0, v, 0.0)
    wn = np.where(v < 0, -v, 0.0)
    ones = np.ones(len(v))
    max_wire = []
    for cells in stages:
        apply_stage_cone(wp, wn, cells, ones, ones)   # ungauged
        annihilate(wp, wn)
        wp *= scale
        wn *= scale
        max_wire.append(float(np.max(wp + wn)))
    return (wp - wn) * scale ** (-len(stages)), max_wire


# ------------------------------------------------------------ certificates --
def cell_stochasticity(n, stages, pots):
    """Max deviation of any gauged column sum from 1 over every cell."""
    worst = 0.0
    for t, cells in enumerate(stages):
        g_in, g_out = pots[t], pots[t + 1]
        for out_sl, in_sl, edges in cells:
            ins = {j for _, j, _, _ in edges}
            for j in ins:
                colsum = np.zeros(in_sl[j].stop - in_sl[j].start)
                for i, jj, w, _f in edges:
                    if jj == j:
                        colsum += abs(w) * g_out[out_sl[i]] / g_in[in_sl[j]]
                worst = max(worst, float(np.max(np.abs(colsum - 1.0))))
    return worst


def main():
    n = 512
    M = GaugedConeFFT(n)

    print(f"== gauged positive-cone Bruun FFT, n={n} ==")
    print(f"forward potential range: g in [{M.gf[0].min():.4g}, {M.gf[0].max():.4g}]"
          f" at input; ratio {M.gf[0].max()/M.gf[0].min():.4g}")
    print(f"cell stochasticity defect (forward): {cell_stochasticity(n, M.fwd, M.gf):.2e}")
    print(f"cell stochasticity defect (inverse): {cell_stochasticity(n, M.inv, M.gi):.2e}")

    rng = np.random.default_rng(42)
    cases = {
        "random_positive": rng.random(n),
        "normal": np.random.default_rng(43).normal(size=n),
        "ramp": np.arange(n, dtype=np.float64),
        "alternating": np.where(np.arange(n) % 2 == 0, 1.0, -1.0),
        "impulse_0": np.eye(n)[0],
        "ones": np.ones(n),
    }
    print("\n== certificates ==")
    print(f"{'case':>16} {'fwd_vs_numpy':>12} {'roundtrip':>12} {'transport_defect':>16} "
          f"{'ledger_defect':>13} {'burned/mass_in':>14}")
    for name, x in cases.items():
        X, tr, m_in, burned, ldef = M.rfft(x)
        err_f = float(np.max(np.abs(X - np.fft.rfft(x))))
        xr, tri, mi_in, bi, ldi = M.irfft(X)
        err_rt = float(np.max(np.abs(xr - x)))
        tdef = max(row["transport_defect"] for row in tr)
        print(f"{name:>16} {err_f:>12.2e} {err_rt:>12.2e} {tdef:>16.2e} "
              f"{ldef:>13.2e} {burned/max(m_in,1e-300):>14.3f}")

    print("\n== stage trace (normal input): the dissipation ledger ==")
    x = cases["normal"]
    _, tr, m_in, burned, _ = M.rfft(x)
    print(f"   mass_in = {m_in:.4f}; total annihilated = {burned:.4f} "
          f"({100*burned/m_in:.1f}% of input mass becomes heat)")
    print(f"   {'stage':>5} {'mass_before':>12} {'annihilated':>12} {'mass_after':>12} "
          f"{'projected_l1':>13} {'max_wire':>10}")
    for row in tr:
        print(f"   {row['stage']:>5} {row['mass_before']:>12.3f} {row['annihilated']:>12.3f} "
              f"{row['mass_after']:>12.3f} {row['projected_l1']:>13.3f} {row['max_wire_mass']:>10.4f}")

    print("\n== gauge vs crude stage_scale: max wire mass by stage (normal input) ==")
    _, tr_g, _, _, _ = M.rfft(x)
    _, mw_s = run_cone_stage_scale(x, M.fwd, 0.5)
    print(f"   {'stage':>5} {'gauged':>10} {'stage_scale':>12}")
    for t in range(len(M.fwd)):
        print(f"   {t:>5} {tr_g[t]['max_wire_mass']:>10.4f} {mw_s[t]:>12.4f}")

    print("\n== no-annihilation variant (pure transport, cancellation deferred) ==")
    X2, tr2, m2, b2, _ = M.rfft(x, annihilate_stage=False)
    err2 = float(np.max(np.abs(X2 - np.fft.rfft(x))))
    print(f"   fwd_vs_numpy = {err2:.2e}; final sheet mass / mass_in = "
          f"{(tr2[-1]['mass_after'])/m2:.6f} (exactly 1: transport alone conserves)")


if __name__ == "__main__":
    main()
