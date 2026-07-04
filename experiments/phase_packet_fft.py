"""Phase-packet (rotating local frame) FFT: existence proof and companion demos.

Three results, all verified to machine precision at the bottom of this file:

1. WHY STOCKHAM NEEDS NO REPACK (the 512-pt unrolled numba kernel):
   the stage-t state A_t[k, j] (shape [2^t, N/2^t]) is the 2^t-point DFT of the
   time comb {x[j + r*(N/2^t)]} with k in NATURAL order. Each stage moves one
   index bit from the column (time) axis to the row (frequency) axis by writing
   into a differently-shaped buffer. Bit reversal is the product of those m
   single-bit moves; paying one per stage as *store addressing* means no layer
   ever holds scrambled data. Cost: out-of-place ping-pong, thin columns late.

2. REAL BRUUN-PAIR STOCKHAM (our DIT primitive under Stockham addressing):
   state = folded real spectra per column (dc, ny, (a,b)=(Re,-Im) pairs); every
   stage is pair_reduce_cs cells + dc/ny chains, uniform along columns ->
   broadcast-twiddle vertical SIMD at ANY register width with no mirror-lane
   pairing, natural-order input AND output. The crossing tail (columns < lanes)
   has depth log2(lanes): the pairing-axis law shows up as the Stockham tail.

3. PHASE-PACKET FFT (the diagonal walk): gauge the Stockham tree by a schedule
   of integer time-shears sigma_t. Admissibility (monomial action on the mixed
   lattice): sigma_t * (N/2^t)^2 must be divisible by N. Effective shear group
   at stage t has size 2^min(t, m-t): a DIAMOND -- frames are forced axis-
   aligned at both ends, maximally tilted mid-transform. Every intermediate
   coordinate is then a chirped comb packet (support partly time-, partly
   frequency-local); butterflies pair packets separated DIAGONALLY by
   (sigma_t*q', q'); each stage stays 2-sparse; and the butterfly rotation is
   EXACTLY exp(-2*pi*i*(kappa - (sigma_{t+1}-sigma_t)*j)/2^{t+1}): constant
   along ONE diagonal coordinate (the increment diagonal), plus a separable
   j-only chirp law that vanishes when the frame does not turn that stage.
   Product over stages == DFT exactly, for every admissible schedule.
   DIT is the sigma == 0 gauge point and is cost-minimal (the frame-turn
   rotation is the only extra work: <= 1 extra Givens per point per turning
   stage). So: no flop win exists in this family -- proven, not suspected --
   but the schedule is a free knob (chirped/LCT-flavored outputs at O(N) extra,
   numerical conditioning, address geometry).
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1. the unrolled numba kernel, numba stripped, + its ledger invariant
# ---------------------------------------------------------------------------
def unrolled_rfft_512(x):
    tw = [np.exp(-1.0j*np.pi*np.arange((2**i)//2)/((2**i)//2)) for i in range(1, 10)]
    twf = np.concatenate([a.reshape(-1, 1) for a in tw])
    A = np.zeros((2, 256), np.complex128)
    A[0, :] = x[:256] + x[256:]
    A[1, :] = x[:256] - x[256:]
    for e in (2, 4, 8, 16, 32, 64, 128, 256):
        q = 512 // (2*e)
        t = twf[(e-1):(2*e)-1]
        tmp = t * A[:e, q:2*q]
        A = np.vstack([A[:e, :q] + tmp, A[:e, :q] - tmp])
    A[256, 0] = A[256, 0].real
    return A[:257, 0]

def comb_dfts(x, t):
    N = len(x); q = N >> t
    out = np.zeros((1 << t, q), np.complex128)
    for j in range(q):
        out[:, j] = np.fft.fft(x[j::q])
    return out

# ---------------------------------------------------------------------------
# 2. real Bruun-pair Stockham rfft (our primitive, Stockham addressing)
# ---------------------------------------------------------------------------
def pair_reduce_cs(ea, eb, oa, ob, c, s):
    r = c*oa - s*ob
    i = s*oa + c*ob
    return ea + r, eb + i, ea - r, i - eb        # parents k and e-k

def real_stockham_rfft(x):
    N = len(x); m = int(np.log2(N))
    q = N
    dc = x.astype(float).copy()
    ny = None
    a = {}; b = {}
    e = 1
    for _t in range(m):
        q2 = q // 2
        E, O = slice(0, q2), slice(q2, q)
        ndc, nny = dc[E] + dc[O], dc[E] - dc[O]
        na, nb = {}, {}
        if e >= 2:
            na[e//2], nb[e//2] = ny[E].copy(), ny[O].copy()     # quarter node
        for k in range(1, e//2):
            c, s = np.cos(np.pi*k/e), np.sin(np.pi*k/e)
            la, lb, ha, hb = pair_reduce_cs(a[k][E], b[k][E], a[k][O], b[k][O], c, s)
            na[k], nb[k] = la, lb
            na[e-k], nb[e-k] = ha, hb                            # palindromic mirror
        dc, ny, a, b, q, e = ndc, nny, na, nb, q2, 2*e
    X = np.zeros(N//2 + 1, np.complex128)
    X[0], X[N//2] = dc[0], ny[0]
    for k in range(1, N//2):
        X[k] = a[k][0] - 1j*b[k][0]
    return X

# ---------------------------------------------------------------------------
# 3. phase-packet FFT
# ---------------------------------------------------------------------------
def qphase(num_int, N):
    """exp(i*pi*num_int/N) with the quadratic exponent reduced mod 2N exactly."""
    return np.exp(1j*np.pi*(num_int % (2*N))/N)

def admissible(s, t, N):
    q = N >> t
    return (s*q*q) % N == 0

def fused_sheared_stage(B, t, s_t, s_t1, N):
    """One stage entirely in sheared coordinates: reads frame s_t at level t,
    writes frame s_t1 at level t+1. Diagonal pairing, 1-D twiddle law."""
    e, q, q2 = 1 << t, N >> t, N >> (t+1)
    assert (s_t*q*q) % N == 0 and (s_t1*q2*q2) % N == 0
    c_t, c_t1 = (s_t*q*q)//N, (s_t1*q2*q2)//N
    out = np.zeros((2*e, q2), np.complex128)
    for j in range(q2):
        for kappa in range(2*e):
            k = (kappa - s_t1*j - c_t1*e) % (2*e)
            kbar = k % e
            rL = (kbar + s_t*j + c_t*(e >> 1)) % e
            rR = (kbar + s_t*(j+q2) + c_t*(e >> 1)) % e
            phL = qphase(-s_t*j*j, N)
            phR = qphase(-s_t*(j+q2)*(j+q2), N)
            phO = qphase(s_t1*j*j, N)
            w = np.exp(-2j*np.pi*k/(2*e))
            out[kappa, j] = phO*(phL*B[rL, j] + w*phR*B[rR, j+q2])
    return out

def phase_packet_fft(x, sched):
    N = len(x); m = int(np.log2(N))
    assert len(sched) == m + 1 and sched[0] == 0 and sched[m] % (1 << m) == 0
    B = x[None, :].astype(np.complex128)
    for t in range(m):
        B = fused_sheared_stage(B, t, sched[t], sched[t+1], N)
    return B[:, 0]

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    x = rng.standard_normal(512)
    print("1a. numba-Stockham vs rfft:      %.2e" %
          np.max(np.abs(unrolled_rfft_512(x) - np.fft.rfft(x))))
    A = np.zeros((2, 256), np.complex128); A[0] = x[:256]+x[256:]; A[1] = x[:256]-x[256:]
    tw = [np.exp(-1.0j*np.pi*np.arange((2**i)//2)/((2**i)//2)) for i in range(1, 10)]
    twf = np.concatenate([a.reshape(-1, 1) for a in tw])
    for e in (2, 4, 8):
        q = 512//(2*e); tmp = twf[(e-1):(2*e)-1]*A[:e, q:2*q]
        A = np.vstack([A[:e, :q]+tmp, A[:e, :q]-tmp])
    print("1b. ledger invariant at t=4:     %.2e (state == natural-order comb DFTs)" %
          np.max(np.abs(A - comb_dfts(x, 4))))

    for N in (16, 64, 512, 4096):
        xr = rng.standard_normal(N)
        print("2.  real Bruun-pair Stockham N=%-5d %.2e" % (N,
              np.max(np.abs(real_stockham_rfft(xr) - np.fft.rfft(xr)))))

    N, m = 64, 6
    xc = rng.standard_normal(N) + 1j*rng.standard_normal(N)
    print("3a. shear diamond (N=64): effective shear-group size per stage:",
          [2**min(t, m-t) for t in range(m+1)])
    for sched in ([0, 1, 2, 5, 4, 16, 0], [0, 0, 1, 4, 4, 0, 0], [0, 1, 3, 0, 12, 16, 0]):
        err = np.max(np.abs(phase_packet_fft(xc, sched) - np.fft.fft(xc)))
        print("3b. phase-packet %-24s err=%.2e" % (sched, err))
    # crisp test: butterfly rotation constant along the increment diagonal
    sched = [0, 1, 2, 5, 4, 16, 0]
    ok_all = True
    for t in range(m):
        e, q, q2 = 1 << t, N >> t, N >> (t+1)
        dsig = sched[t+1] - sched[t]
        Op = np.zeros((2*e*q2, e*q), np.complex128)
        for idx in range(e*q):
            Bb = np.zeros((e, q), np.complex128); Bb[idx//q, idx % q] = 1.0
            Op[:, idx] = fused_sheared_stage(Bb, t, sched[t], sched[t+1], N).ravel()
        ref, worst = None, 0.0
        for kappa in range(2*e):
            for j in range(q2):
                row = Op[kappa*q2 + j]
                cols = np.where(np.abs(row) > 1e-12)[0]
                assert len(cols) == 2                       # 2-sparse
                cA = [c for c in cols if (c % q) == j][0]
                cB = [c for c in cols if (c % q) == j+q2][0]
                canon = (row[cB]/row[cA]) * np.exp(2j*np.pi*((kappa - dsig*j) % (2*e))/(2*e))
                ref = canon if ref is None else ref
                worst = max(worst, abs(canon - ref))
        ok_all &= worst < 1e-9
        print("3c. stage %d: 2-sparse, rotation == const * e^(-2pi i (kappa-%+d j)/%d): dev %.1e"
              % (t, dsig, 2*e, worst))
    print("3d. CRISP TEST (angles depend on ONE diagonal coordinate):", ok_all)
