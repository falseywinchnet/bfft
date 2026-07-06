"""Numba DIP (diagonal-in-packets) FFT with full stage capture.

Deliverable: a Python/Numba rendition of the BFFT DIP walk that returns EVERY
level's intermediate state as well as the final spectrum, plus the partial
operators (finish / backward / partial-forward) needed to treat any level as a
first-class representation.

THE STAGE STATES ARE ZAK TRANSFORMS (the load-bearing fact)
-----------------------------------------------------------
The complex diagonal walk (phase_fft.py :: phase_fft_ordered) carries state
B_t of shape (2^t, N/2^t), stored diagonal-major.  Unrolling the cell
recurrence gives the closed form

    B_t[delta, j] = sum_{r=0}^{2^t-1} x[j + r*q] * exp(-2i*pi*delta*r/2^t),
    q = N/2^t

i.e. the level-t state is exactly the FINITE ZAK TRANSFORM of the frame on
the dyadic lattice (2^t, N/2^t).  Equivalently, in spectrum terms,

    B_t[delta, j] = (2^t/N) * sum_{k == delta (mod 2^t)} X[k] * exp(2i*pi*k*j/N)

so a row delta is the signal restricted to the frequency comb
{k : k == delta mod 2^t} — the Bruun {jE +/- d} comb signature — critically
sampled in time.  The walk interpolates the whole lattice family from the
time axis (t=0, lattice 1 x N) to the frequency axis (t=m, lattice N x 1):
these are the "fractional phase Fourier states".  Coordinates at level t:

    row delta  = fine frequency (k mod 2^t : the LOW bits of the bin index)
    column j   = fine time      (n mod N/2^t : the LOW bits of the sample index)
    coarse frequency (which alias) = oscillation of the row along j
    coarse time  (which translate) = phase pattern down the column across delta

Two facts that matter for fusion (dip_zak_fusion.py):
  * two tones Delta-k bins apart land in DIFFERENT rows as soon as
    2^t does not divide Delta-k — close-tone separation is achieved in the
    first few levels and costs no time resolution;
  * every stage map U_t satisfies U_t U_t^H = 2 I (a scaled unitary), so the
    partial forward P_t obeys P_t P_t^H = 2^t I: descent, adjoints and
    momentum transfer between levels with fixed, well-conditioned geometry.

Both walks are here:
  * dip_fft_stages     — complex diagonal walk (angle = one coordinate, the
                         output row), final = FFT in natural order;
  * dip_rfft_stages    — the real Bruun residue-pair walk (pair_reduce_cs
                         cells, angle law theta = pi*d/e, walking DC/Nyquist
                         ridge), the shipped kernel's arithmetic; final row
                         extracts to rfft.

Every level t of either walk packs to exactly N scalars, so the full stage
stack is an (m+1, N) array: row t = level-t state, row m = the final product.

Self-test: python experiments/dip_numba.py
"""
import numpy as np
from numba import njit, prange

# ------------------------------------------------------------------ helpers --
def _log2(N):
    m = int(N).bit_length() - 1
    if (1 << m) != N:
        raise ValueError("N must be a power of two")
    return m

# =========================================================================== #
#  Complex diagonal walk                                                      #
#  state layout: level t row = vec of B_t, B_t[delta, j] = S[t, delta*q + j]  #
# =========================================================================== #

@njit(cache=True)
def _stage_fwd(src, dst, t, N):
    """One DIP level: state t (in src, length N) -> state t+1 (into dst)."""
    e = 1 << t
    q = N >> t
    q2 = q >> 1
    for db in range(e):
        base = db * q
        # low output row: delta = db ; high output row: delta = db + e
        thl = -np.pi * db / e        # = -2*pi*db / (2e)
        wl = complex(np.cos(thl), np.sin(thl))
        lo = db * q2
        hi = (db + e) * q2
        for j in range(q2):
            ev = src[base + j]
            od = wl * src[base + q2 + j]
            dst[lo + j] = ev + od
            dst[hi + j] = ev - od      # W[db+e] = -W[db]
    return dst


@njit(cache=True)
def _stage_bwd(src, dst, t, N):
    """Exact inverse of _stage_fwd: state t+1 (in src) -> state t (into dst)."""
    e = 1 << t
    q = N >> t
    q2 = q >> 1
    for db in range(e):
        base = db * q
        thl = -np.pi * db / e
        wl_conj = complex(np.cos(thl), -np.sin(thl))
        lo = db * q2
        hi = (db + e) * q2
        for j in range(q2):
            s = src[lo + j]
            d = src[hi + j]
            dst[base + j] = 0.5 * (s + d)
            dst[base + q2 + j] = 0.5 * wl_conj * (s - d)
    return dst


@njit(cache=True)
def dip_fft_stages(x):
    """Full complex DIP with stage capture.

    Returns S of shape (m+1, N) complex128:
      S[0] = x, S[t] = vec of the level-t Zak state B_t (row-major, rows of
      length N>>t), S[m] = fft(x) in natural bin order.
    """
    N = x.size
    m = 0
    while (1 << m) < N:
        m += 1
    S = np.empty((m + 1, N), np.complex128)
    S[0, :] = x.astype(np.complex128)
    for t in range(m):
        _stage_fwd(S[t], S[t + 1], t, N)
    return S


@njit(cache=True)
def dip_fft_finish(z, t, N):
    """Finish the DIP from a level-t state z (length N) -> final spectrum.

    Pure feed-forward: no descent, no data-dependent branching. This is the
    'synthetic intermediate -> final result' map.
    """
    m = 0
    while (1 << m) < N:
        m += 1
    a = z.copy()
    b = np.empty(N, np.complex128)
    for lv in range(t, m):
        _stage_fwd(a, b, lv, N)
        a, b = b, a
    return a


@njit(cache=True)
def dip_fft_backward(z, t, N):
    """Invert the DIP from a level-t state z back to the time signal (level 0)."""
    a = z.copy()
    b = np.empty(N, np.complex128)
    for lv in range(t - 1, -1, -1):
        _stage_bwd(a, b, lv, N)
        a, b = b, a
    return a


@njit(cache=True)
def dip_partial_forward(x, t, N):
    """Levels 0..t-1 only: time signal -> level-t state P_t x.

    P_t P_t^H = 2^t I, so the adjoint of dip_fft_backward(., t) is
    dip_partial_forward(., t)/2^t — the gradient plumbing for descent
    performed AT level t.
    """
    a = x.astype(np.complex128)
    b = np.empty(N, np.complex128)
    for lv in range(t):
        _stage_fwd(a, b, lv, N)
        a, b = b, a
    return a


def dip_state_2d(S, t, N):
    """View the level-t row of a stage stack as the (2^t, N>>t) Zak matrix."""
    return S[t].reshape(1 << t, N >> t)


# =========================================================================== #
#  Real Bruun residue-pair walk (the shipped kernel's arithmetic)             #
#                                                                             #
#  Level-t state (q = N>>t, e = 1<<t) packs to N floats:                      #
#    [ dc(q) | ny(q) | a_1(q) | b_1(q) | ... | a_{e/2-1}(q) | b_{e/2-1}(q) ]  #
#  slot of a_d = 2d*q, slot of b_d = (2d+1)*q.  (a,b) = (Re, -Im) of the      #
#  residue pair on diagonal d; the cell is pair_reduce_cs with the angle law  #
#  theta = pi*d/e; dc/ny is the walking DC/Nyquist ridge.  Level 0 = x.       #
# =========================================================================== #

@njit(cache=True)
def _rstage_fwd(src, dst, t, N):
    e = 1 << t
    q = N >> t
    q2 = q >> 1
    # walking DC/Nyquist ridge: dc -> (dc', ny')
    for j in range(q2):
        ev = src[j]
        od = src[q2 + j]
        dst[j] = ev + od           # new dc
        dst[q2 + j] = ev - od      # new ny
    if e >= 2:
        # quarter packet: old ny splits into packet d = e/2 of the new level
        d = e >> 1
        for j in range(q2):
            dst[(2 * d) * q2 + j] = src[q + j]            # a_{e/2} = ny[E]
            dst[(2 * d + 1) * q2 + j] = src[q + q2 + j]   # b_{e/2} = ny[O]
    # interior packets: pair_reduce_cs, angle theta = pi*d/e
    for d in range(1, e >> 1):
        th = np.pi * d / e
        c = np.cos(th)
        s = np.sin(th)
        sa = (2 * d) * q
        sb = (2 * d + 1) * q
        la = (2 * d) * q2                 # low child  -> diagonal d
        lb = (2 * d + 1) * q2
        ha = (2 * (e - d)) * q2           # high child -> mirror diagonal e-d
        hb = (2 * (e - d) + 1) * q2
        for j in range(q2):
            ea = src[sa + j]
            eb = src[sb + j]
            oa = src[sa + q2 + j]
            ob = src[sb + q2 + j]
            r = c * oa - s * ob
            i = s * oa + c * ob
            dst[la + j] = ea + r
            dst[lb + j] = eb + i
            dst[ha + j] = ea - r
            dst[hb + j] = i - eb
    return dst


@njit(cache=True)
def dip_rfft_stages(x):
    """Real Bruun-pair DIP with stage capture.

    Returns R of shape (m+1, N) float64; R[t] is the packed level-t state
    (see layout above), R[m] extracts to rfft(x) via dip_rfft_extract.
    """
    N = x.size
    m = 0
    while (1 << m) < N:
        m += 1
    R = np.empty((m + 1, N), np.float64)
    R[0, :] = x.astype(np.float64)
    for t in range(m):
        _rstage_fwd(R[t], R[t + 1], t, N)
    return R


def dip_rfft_extract(row_m, N):
    """Final packed row -> complex half spectrum (== np.fft.rfft)."""
    X = np.zeros(N // 2 + 1, np.complex128)
    X[0] = row_m[0]
    X[N // 2] = row_m[1]
    d = np.arange(1, N // 2)
    X[1:N // 2] = row_m[2 * d] - 1j * row_m[2 * d + 1]
    return X


def dip_rfft_packets(R, t, N):
    """Unpack the level-t real state into (dc, ny, a, b) views.

    a, b have shape (e/2-1, q) for diagonals d = 1..e/2-1 (empty at t<2).
    """
    q = N >> t
    e = 1 << t
    row = R[t]
    dc = row[:q]
    ny = row[q:2 * q] if e >= 2 else None
    nd = max(e // 2 - 1, 0)
    a = np.empty((nd, q))
    b = np.empty((nd, q))
    for d in range(1, e // 2):
        a[d - 1] = row[2 * d * q:(2 * d + 1) * q]
        b[d - 1] = row[(2 * d + 1) * q:(2 * d + 2) * q]
    return dc, ny, a, b


# ------------------------------------------------------------------ selftest --
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    print("== complex DIP: exactness, stage invertibility, Zak identity ==")
    for N in (16, 256, 4096, 65536):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        S = dip_fft_stages(x)
        m = _log2(N)
        err_f = np.max(np.abs(S[m] - np.fft.fft(x)))
        # backward from every level reproduces x
        err_b = max(np.max(np.abs(dip_fft_backward(S[t].copy(), t, N) - x))
                    for t in range(m + 1))
        # finish from every level reproduces the spectrum
        err_c = max(np.max(np.abs(dip_fft_finish(S[t].copy(), t, N) - S[m]))
                    for t in range(m + 1))
        # Zak identity at a mid level: B_t[delta, j] = e-point DFT of x[j::q]
        t = m // 2
        e, q = 1 << t, N >> t
        B = S[t].reshape(e, q)
        comb = x.reshape(e, q)                       # comb[r, j] = x[j + r*q]
        Zc = np.fft.fft(comb, axis=0)                # over r, for every j
        err_z = np.max(np.abs(B - Zc))
        # partial forward scaling: P_t P_t^H = 2^t I
        y = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        lhs = np.vdot(dip_partial_forward(x, t, N), dip_partial_forward(y, t, N))
        err_u = abs(lhs - (1 << t) * np.vdot(x, y)) / abs((1 << t) * np.vdot(x, y))
        print(f"  N={N:6d}: |fft err|={err_f:.2e}  bwd={err_b:.2e}  "
              f"finish={err_c:.2e}  zak={err_z:.2e}  unitary(2^t)={err_u:.2e}")

    print("\n== real Bruun-pair DIP: exactness vs rfft ==")
    for N in (16, 256, 4096, 65536):
        xr = rng.standard_normal(N)
        R = dip_rfft_stages(xr)
        m = _log2(N)
        err = np.max(np.abs(dip_rfft_extract(R[m], N) - np.fft.rfft(xr)))
        print(f"  N={N:6d}: |rfft err|={err:.2e}")

    print("\n== timing (jitted, N=65536, 50 reps) ==")
    import time
    x = rng.standard_normal(65536) + 1j * rng.standard_normal(65536)
    xr = rng.standard_normal(65536)
    dip_fft_stages(x); dip_rfft_stages(xr)      # warm
    t0 = time.perf_counter()
    for _ in range(50):
        dip_fft_stages(x)
    t1 = time.perf_counter()
    for _ in range(50):
        dip_rfft_stages(xr)
    t2 = time.perf_counter()
    print(f"  complex stages: {1e3*(t1-t0)/50:.2f} ms   "
          f"real stages: {1e3*(t2-t1)/50:.2f} ms")
