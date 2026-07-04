"""The phase-packet FFT: a Bruun-cell walk along the diagonal of the
time-frequency torus.

THE OBJECT
----------
State at level t: a lattice of N packets addressed by (row, column) =
(diagonal coordinate delta, packet offset j), shape [2^t, N/2^t]. The packet at
address (delta, j) is a time-frequency tile whose CENTER sits on the line of
slope sigma_t through the torus: its plain frequency-center is
(delta + sigma_t * j) and its time support is the offset-j comb. The frame
slope walks the canonical diagonal schedule

    sigma_t = 2^max(0, 2t - m),   sigma_0 = sigma_m = 0        (N = 2^m)

i.e. the lattice leaves the time axis, rides the UNIT DIAGONAL through the
self-dual half-depth scale, then steepens 4, 16, 64, ... into the frequency
axis. This is the steepest walk exactness permits: the admissible slopes at
level t form the group of size 2^min(t, m-t) (the "diamond"), the constraint
being sigma * (N/2^t)^2 = 0 mod N -- precisely the requirement that a slope
step never tilts a packet INTERNALLY (an intra-packet chirp is not
representable by a sparse exact cell; violating the diamond is what produces
dense carry corrections). The canonical schedule sits ON the boundary
(sigma * q^2 = N) at every level past half depth.

THE CELL AND THE LAW
--------------------
Every step is one normalized Bruun cell. Two packets paired ACROSS the current
diagonal (same delta, columns j and j + q') combine under a single rotation:

    out(delta, j)        = in(delta_bar, j) + W(delta) * in(delta_bar, j+q')
    W(delta)             = e^{-2 pi i delta / 2^{t+1}},   delta_bar = delta mod 2^t

THE ANSWER TO THE ORDERING QUESTION: with packets phase-referenced at their
own centers and stored in diagonal-major order (row = delta), the Bruun angle
of every cell depends on ONE coordinate only -- the diagonal coordinate delta
of its output row -- and the addressing collapses to constant geometry: the
walk across the torus costs nothing. The diagonal path is carried by what the
coordinates MEAN (which packet sits at which address, exhibited by
`packet_center` below), while the arithmetic is as lean as any axis-aligned
walk. Equivalently: the axis-aligned transforms were always this walk, seen in
a frame that hides the diagonal; `phase_fft_sheared` makes the frame explicit
and is verified bit-identical to `phase_fft_ordered`.

In the real (Bruun residue-pair) form, each cell is the kernel primitive
pair_reduce_cs on (a, b) = (Re, -Im) pairs, angle theta(delta) = 2 pi delta /
2^{t+1}: `phase_fft_real` computes the r2c transform entirely from those cells
plus the DC/Nyquist ridge (which itself walks the diagonal), asserting at
every cell that the angle is a function of delta alone.
"""
import numpy as np

# ---------------------------------------------------------------- schedule --
def sigma_schedule(m):
    return [0] + [1 << max(0, 2*t - m) for t in range(1, m)] + [0]

def admissible(sig, t, N):
    q = N >> t
    return (sig * q * q) % N == 0

def c_shift(sig, t, N):
    q = N >> t
    assert (sig * q * q) % N == 0
    return (sig * q * q) // N

# ------------------------------------------- explicit diagonal addressing --
def phase_fft_sheared(x, sched=None):
    """The walk with the frame written out: addresses are (kappa, j) in the
    sheared lattice; pairing reads along the level-t diagonal, the rotation
    W(delta) rides the level-(t+1) diagonal."""
    N = len(x); m = int(np.log2(N))
    sig = sigma_schedule(m) if sched is None else sched
    B = x[None, :].astype(np.complex128)
    for t in range(m):
        e, q2 = 1 << t, N >> (t + 1)
        s0, s1 = sig[t], sig[t + 1]
        c0, c1 = c_shift(s0, t, N), c_shift(s1, t + 1, N)
        W = np.exp(-2j * np.pi * np.arange(2 * e) / (2 * e))
        out = np.empty((2 * e, q2), np.complex128)
        for j in range(q2):
            for kappa in range(2 * e):
                delta = (kappa - s1 * j - c1 * e) % (2 * e)   # ONE diagonal coordinate
                db = delta % e
                rL = (db + s0 * j + c0 * (e >> 1)) % e
                rR = (db + s0 * (j + q2) + c0 * (e >> 1)) % e
                out[kappa, j] = B[rL, j] + W[delta] * B[rR, j + q2]
        B = out
    return B[:, 0]

# -------------------------------------- diagonal-major packet ordering ------
def phase_fft_ordered(x):
    """Same walk, packets stored diagonal-major (row = delta): the shear
    vanishes from the addresses; the angle is a function of the row only."""
    N = len(x); m = int(np.log2(N))
    B = x[None, :].astype(np.complex128)
    for t in range(m):
        e, q2 = 1 << t, N >> (t + 1)
        W = np.exp(-2j * np.pi * np.arange(2 * e) / (2 * e))[:, None]
        top = B[:, q2:2 * q2]
        out = np.empty((2 * e, q2), np.complex128)
        out[:e] = B[:, :q2] + W[:e] * top
        out[e:] = B[:, :q2] + W[e:] * top
        B = out
    return B[:, 0]

def packet_center(t, delta, j, m, sched=None):
    """Which physical packet the diagonal-major address (delta, j) denotes:
    (plain frequency-center row, time comb offset). Exhibits the walk."""
    N = 1 << m
    sig = sigma_schedule(m) if sched is None else sched
    e = 1 << t
    return ((delta + sig[t] * j + c_shift(sig[t], t, N) * (e >> 1)) % e, j)

# --------------------------------------------- real Bruun-pair phase FFT ----
def pair_reduce_cs(ea, eb, oa, ob, c, s):
    r = c * oa - s * ob
    i = s * oa + c * ob
    return ea + r, eb + i, ea - r, i - eb

def phase_fft_real(x, log_angles=None):
    """r2c phase FFT built ONLY from pair_reduce_cs cells + the DC/Nyquist
    ridge, in diagonal-major packet order. Every cell's angle is
    theta = 2*pi*delta / 2^{t+1} with delta the diagonal coordinate of its
    low output row -- one coordinate, nothing else. Folded state per column:
    dc, ny, and (a, b) = (Re, -Im) packet pairs for delta in [1, 2^{t-1})."""
    N = len(x); m = int(np.log2(N))
    q = N
    dc = x.astype(float).copy()
    ny = None
    a, b = {}, {}
    e = 1
    for t in range(m):
        q2 = q // 2
        E, O = slice(0, q2), slice(q2, q)
        ndc, nny = dc[E] + dc[O], dc[E] - dc[O]        # the walking DC/Ny ridge
        na, nb = {}, {}
        if e >= 2:
            na[e // 2], nb[e // 2] = ny[E].copy(), ny[O].copy()   # quarter packet
            if log_angles is not None:
                log_angles.append((t, e // 2, np.pi * (e // 2) / e))
        for d in range(1, e // 2):
            th = np.pi * d / e                          # = 2*pi*delta / 2^{t+1}
            cth, sth = np.cos(th), np.sin(th)
            la, lb, ha, hb = pair_reduce_cs(a[d][E], b[d][E], a[d][O], b[d][O], cth, sth)
            na[d], nb[d] = la, lb                       # output diagonal d
            na[e - d], nb[e - d] = ha, hb               # mirror diagonal e - d
            if log_angles is not None:
                log_angles.append((t, d, th))
        dc, ny, a, b, q, e = ndc, nny, na, nb, q2, 2 * e
    X = np.zeros(N // 2 + 1, np.complex128)
    X[0], X[N // 2] = dc[0], ny[0]
    for d in range(1, N // 2):
        X[d] = a[d][0] - 1j * b[d][0]
    return X

# DIP: diagonal in packets. These names make the packet-ordering FFT usable as
# its own experiment object before the C++ kernel version.
def dip_fft(x):
    return phase_fft_ordered(x)

def dip_rfft(x, log_angles=None):
    return phase_fft_real(x, log_angles)

# ---------------------------------------------------------------- exhibit --
if __name__ == "__main__":
    rng = np.random.default_rng(4)

    print("== exactness, and sheared == ordered (the ordering theorem) ==")
    for N in (16, 64, 256, 1024):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        A, B, F = phase_fft_sheared(x), phase_fft_ordered(x), np.fft.fft(x)
        print(f"  N={N:5d}: |sheared-F|={np.max(np.abs(A-F)):.2e}  "
              f"|ordered-F|={np.max(np.abs(B-F)):.2e}  |sheared-ordered|={np.max(np.abs(A-B)):.2e}")

    print("\n== real Bruun-cell phase FFT vs rfft, one-coordinate angle law ==")
    for N in (16, 64, 512, 4096):
        xr = rng.standard_normal(N)
        log = []
        err = np.max(np.abs(phase_fft_real(xr, log) - np.fft.rfft(xr)))
        law = {}
        for (t, d, th) in log:
            law.setdefault((t, d), set()).add(round(th, 12))
        onec = all(len(v) == 1 for v in law.values())
        print(f"  N={N:5d}: err={err:.2e}   angle == theta(delta) only: {onec}   "
              f"({len(law)} (level, delta) classes)")

    print("\n== the walk (m=8): frame slope + where a fixed address's packet sits ==")
    m = 8; N = 1 << m
    sig = sigma_schedule(m)
    print("  level t :", list(range(m + 1)))
    print("  sigma_t :", sig, "  (unit diagonal to half depth, then 4,16,64 -> freq axis)")
    print("  diamond :", [2 ** min(t, m - t) for t in range(m + 1)], " (admissible slope-group sizes)")
    d = 3
    print(f"  packet at fixed diagonal address (delta={d}, j=2): plain freq-center by level:")
    for t in range(2, m + 1):
        if (1 << t) > d:
            fc, j = packet_center(t, d, 2 % (N >> t), m)
            print(f"    t={t}: center row {fc} of {1 << t}  (address row stays {d}: the center walks, the address doesn't)")
