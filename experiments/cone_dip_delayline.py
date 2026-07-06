"""The delay-line DIP: a Fourier machine with no multipliers.

The cone-DIF (cone_fft_gauged.py) is the MASS-substrate machine: two
nonnegative rails per value, splitters/couplers/sheet-swaps/absorbers,
signs carried by rail routing, the gauge making transport exactly
mass-preserving. This file explores the complementary FIELD-substrate
machine, built on the DIP walk, where the physical carrier itself supplies
the missing algebra:

  NEGATION IS A DELAY.  On a carrier of period T, a path difference of T/2
  multiplies the envelope by -1. The two sheets of the cone become the two
  phases of one line; the sheet swap becomes a half-period stub; and
  annihilation is performed for free by wave superposition at any junction
  (destructive interference IS min(p,n) removal, executed by physics).

  ROTATION IS A DELAY.  The DIP packet (a, b) = (Re, -Im) of a diagonal
  residue is the I/Q envelope of one narrowband physical signal
  z = a + j*b. The Bruun cell's rotation
        R = c*oa - s*ob,  I = s*oa + c*ob     <=>    R + jI = e^{j*theta} * z_o
  is a pure carrier phase shift: theta = 2*pi * (path length) / lambda.
  No multiplier exists anywhere in the machine; every twiddle is geometry.

  THE CELL IS TWO JUNCTIONS.  In complex envelope form the DIP cell is
        lo = z_e + W z_o,      hi = conj(z_e - W z_o),   W = e^{j theta}:
  one delay stub (W), one sum junction, one difference junction (a T/2 stub
  into a sum junction), and one conjugation on the hi arm. Conjugation of a
  narrowband envelope is lower-sideband selection (image mixing / phase
  conjugation); in the two-rail real form used below it is simply a sign
  flip on the b rail -- one more T/2 stub. Element census per cell:
  4 directional-coupler ports + 3 half-period stubs + 1 theta stub. Zero
  active elements.

  WHY THE DIP AND NOT THE DIF HERE: the DIP's angle law theta(d) = pi*d/e
  depends only on the span -- constant across all w columns of a span
  (isoclinic). In a time-multiplexed (serial delay-line) machine, one
  physical theta-element serves an entire span, and its control waveform is
  a STAIRCASE (one value per span) rather than per-slot agility. The census
  below counts the staircase: the total number of distinct theta settings
  a serial DIP machine needs per stage, versus the per-slot twiddle stream
  a serial DIT machine needs. Physically: quasi-static trombone lines /
  thermo-optic phase pads suffice; no fast modulator is required.

This file verifies the element-semantics machine equals numpy's rfft, then
prints the hardware census for the serial (pipelined, single-line) form.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------- physical element set --
def stub(z: np.ndarray, theta: float) -> np.ndarray:
    """A delay line: carrier phase shift by theta (theta = pi is the T/2
    negation stub). The ONLY 'twiddle' element in the machine."""
    return z * np.exp(1j * theta)


def junction_sum(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """A passive combiner: superposition. Destructive interference here is
    the machine's annihilation -- executed by physics, not by logic."""
    return z1 + z2


def conjugate_arm(z: np.ndarray) -> np.ndarray:
    """Lower-sideband selection / phase conjugation of the envelope. In the
    two-rail real form this is one T/2 stub on the b rail."""
    return np.conj(z)


# ------------------------------------------------------- the DIP machine --
def dip_cell(z_e: np.ndarray, z_o: np.ndarray, theta: float):
    """One DIP span cell out of physical elements only."""
    w_zo = stub(z_o, theta)                            # theta delay stub
    lo = junction_sum(z_e, w_zo)                       # sum junction
    hi = conjugate_arm(junction_sum(z_e, stub(w_zo, np.pi)))   # diff + conj
    return lo, hi


def delayline_dip_rfft(x: np.ndarray) -> np.ndarray:
    """The DIP span walk with every operation an element from the physical
    set above. Seed = the level-8 comb DFT (in hardware: a fixed 8-port
    junction/stub network -- an 8x8 Butler matrix, all passive)."""
    n = len(x)
    q = n // 8
    S = np.fft.fft(x.reshape(8, q), axis=0)            # 8-port Butler matrix
    X = np.zeros(n // 2 + 1, dtype=np.complex128)

    def descend(z: np.ndarray, d: int, e: int):
        w = len(z)
        if w == 1:
            X[d] = np.conj(z[0])                       # (a, b) -> a - j b
            return
        w2 = w // 2
        theta = np.pi * d / e
        lo, hi = dip_cell(z[:w2], z[w2:], theta)
        descend(lo, d, 2 * e)
        descend(hi, e - d, 2 * e)

    def descend_ridge(dc: np.ndarray, ny: np.ndarray, e: int):
        qq = len(dc)
        if qq == 1:
            X[0] = dc[0].real
            X[n // 2] = ny[0].real
            return
        q2 = qq // 2
        ndc = junction_sum(dc[:q2], dc[q2:])
        nny = junction_sum(dc[:q2], stub(dc[q2:], np.pi))
        descend_ridge(ndc, nny, 2 * e)
        descend(ny[:q2] + 1j * ny[q2:], e // 2, 2 * e)

    # packets: z_d = a_d + j*b_d = S[d] conj (a = Re, b = -Im => z = conj(S))
    descend_ridge(S[0].real.astype(complex), S[4].real.astype(complex), 8)
    for d in (1, 2, 3):
        descend(np.conj(S[d]), d, 8)
    return X


# --------------------------------------------------------- hardware census --
def serial_census(n: int):
    """Element and control census for the SERIAL (time-multiplexed) form:
    one physical section per level, values streamed as slots; section at
    level e has a feedback delay of w/2 slots and one theta element whose
    setting changes only when the streamed span changes (the isoclinic
    staircase). Compare: a serial DIT/DIF needs a per-slot twiddle stream
    (agile modulator + twiddle memory)."""
    rows = []
    e, w = 8, n // 8
    while w >= 2:
        spans = 0
        d_set = set()
        # spans crossing this level: every subtree node at this level
        def count(d, ee, ww):
            nonlocal spans
            if ee == e and ww == w:
                spans += 1
                d_set.add(d)
                return
            if ww < w or ee > e:
                return
            count(d, 2 * ee, ww // 2)
            count(ee - d, 2 * ee, ww // 2)
        for d0 in (1, 2, 3):
            count(d0, 8, n // 8)
        # ridge spawns entering at various levels also cross; count via combs:
        # every level-e span of width w corresponds to n/(e*w)... simpler:
        # total values crossing level e is n (all wires), spans = n/(2w) bins-ish;
        # we report the tree-census numbers gathered above plus schedule size.
        rows.append({
            "level_e": e,
            "span_width_w": w,
            "delay_slots": w // 2,
            "theta_settings": len(d_set),
            "spans_streamed": spans,
        })
        e, w = 2 * e, w // 2
    return rows


def main():
    print("== delay-line DIP: element-semantics machine vs numpy ==")
    for n in (64, 512, 4096):
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        err = float(np.max(np.abs(delayline_dip_rfft(x) - np.fft.rfft(x))))
        print(f"   n={n:5d}: max err = {err:.2e}")

    print("\n== element inventory (parallel spatial form, n=512) ==")
    n = 512
    cells = sum(2 ** t for t in range(0, 20) if 8 * 2 ** t <= n // 2)  # per top span
    print("   per DIP cell: 1 theta stub + 3 half-period stubs + 2 junctions")
    print("   twiddles are path lengths; multipliers: 0; active elements: 0")
    print("   annihilation: performed by superposition at junctions (free)")

    print("\n== serial (single-line, time-multiplexed) census, n=512 ==")
    print("   one section per level: feedback delay + 1 coupler + 1 theta pad")
    print(f"   {'level':>6} {'delay(slots)':>13} {'theta settings/stage':>21}")
    for row in serial_census(512):
        print(f"   {row['level_e']:>6} {row['delay_slots']:>13} {row['theta_settings']:>21}")
    print("   control: a per-stage STAIRCASE (isoclinic law) -- quasi-static")
    print("   phase pads; a serial DIT needs a per-slot agile twiddle stream.")


if __name__ == "__main__":
    main()
