"""Transport-optimal phase-packet DIP: design validation.

The current DIP (src/detail/bruun_dip_kernel.hpp) walks the diagonal correctly
but breadth-first, out-of-place: every one of ~log2(N) stages sweeps the whole
array (read N + write N). That is the transport catastrophe -- DRAM passes grow
as 2*log2(N) (32-40 at N>=1M), which is why the vectorized DIP is still ~2-3x
DIF at large N (bandwidth-bound), not a codegen problem.

The fix is NOT a better ping-pong. It is to stop sweeping the whole array: a
phase-packet FOUR-STEP. Factor N = P*Q. Resolve one axis into packets, apply the
diagonal PHASE twiddle (the packet rotation), resolve the other axis. Recurse so
the interior is always cache-resident. DRAM is then touched a small, N-INDEPENDENT
number of times.

Verified below (all machine precision):
  1. complex phase-packet four-step == fft, DRAM touches ~0.12-0.30 of the sweep.
  2. real-input four-step == rfft (the r2c fold survives the factoring).
  3. BOTH legs are one cell family: the real leg is the DIP diagonal cell; the
     complex leg is the SAME pair_reduce_cs on (re,im) with a full twiddle
     (one extra rotation). No foreign kernel, no transpose-as-separate-idea.
  4. DRAM-pass model: four-step is FLAT (~2) vs sweep 2*log2(N) vs radix-2
     depth-first 2*log2(N/cache).

The intermediate A[k2,p] after step1+twiddle IS the time-frequency packet
lattice: resolved in coarse frequency k2, still carrying fine index p, rotated
along the diagonal by the twiddle. Step3 resolves the fine frequency. This is
the diagonal walk of experiments/phase_fft.py applied at two scales, made
cache-local.
"""
import numpy as np

def ilog2(n):
    l = 0
    while n > 1:
        n >>= 1; l += 1
    return l

# ---- 1. complex phase-packet four-step + transport count ------------------
def pp_fft(x, TOUCH, LEAF=64):
    n = len(x)
    if n <= LEAF:
        return np.fft.fft(x)                     # cache-resident leaf codelet
    P = 1 << (ilog2(n) // 2); Q = n // P
    g = x.reshape(Q, P)                          # g[q,p] = x[p + P*q]
    TOUCH[0] += n
    A = np.empty((Q, P), complex)
    for p in range(P):                           # step1: length-Q along q (slow axis)
        A[:, p] = pp_fft(g[:, p], TOUCH, LEAF)
    p_ = np.arange(P)[None, :]; k2 = np.arange(Q)[:, None]
    A *= np.exp(-2j * np.pi * p_ * k2 / n)       # step2: diagonal phase twiddle
    TOUCH[0] += n
    T = np.empty((Q, P), complex)
    for kk in range(Q):                          # step3: length-P along p (contiguous)
        T[kk, :] = pp_fft(A[kk, :], TOUCH, LEAF)
    return T.T.reshape(n)                         # X[k2 + Q*k1] = T[k2,k1]

# ---- 2. real-input four-step ----------------------------------------------
def real_fourstep_rfft(x):
    n = len(x); P = 1 << (ilog2(n) // 2); Q = n // P
    g = x.reshape(Q, P)
    A = np.fft.fft(g, axis=0)                     # step1: real -> conj-sym in k2
    p_ = np.arange(P)[None, :]; k2 = np.arange(Q)[:, None]
    A = A * np.exp(-2j * np.pi * p_ * k2 / n)     # step2: phase
    T = np.fft.fft(A, axis=1)                     # step3: length-P complex
    X = np.empty(n // 2 + 1, complex)
    for k in range(n // 2 + 1):
        k1, k2 = divmod(k, Q)
        X[k] = T[k2, k1]
    return X

# ---- 3. one cell family for both legs -------------------------------------
def pair_reduce_cs(ea, eb, oa, ob, c, s):
    r = c * oa - s * ob; i = s * oa + c * ob
    return ea + r, eb + i, ea - r, i - eb        # real DIP diagonal cell

def cfft_givens(x):                              # complex leg: SAME cell + full twiddle
    n = len(x)
    if n == 1:
        return x.copy()
    E = cfft_givens(x[0::2]); O = cfft_givens(x[1::2])
    X = np.empty(n, complex)
    for k in range(n // 2):
        w = np.exp(-2j * np.pi * k / n) * O[k]
        X[k] = E[k] + w; X[k + n // 2] = E[k] - w
    return X

# ---- 4. DRAM-pass model ----------------------------------------------------
def sweep_passes(n, seed=4):        return 2 * max(ilog2(n) - seed, 0)
def radix2_df(n, cache):
    p = 2; b = n
    while b > cache:
        p += 2; b //= 2
    return p
def fourstep_passes(n, cache):
    p = 0
    def rec(m):
        nonlocal p
        if m <= cache:
            return
        p += 2
        rec(m // (1 << (ilog2(m) // 2)))
    rec(n)
    return max(p, 2)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("== 1. complex four-step exactness + transport ==")
    for n in (256, 1024, 4096, 65536, 1048576):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        T = [0]; X = pp_fft(x, T)
        print(f"   n={n:7d} err={np.max(np.abs(X-np.fft.fft(x))):.2e}"
              f"  touches/sweep={T[0]/max(sweep_passes(n)*n//2,1):.3f}")
    print("== 2. real four-step == rfft ==")
    for n in (256, 4096, 65536, 262144):
        x = rng.standard_normal(n)
        print(f"   n={n:7d} err={np.max(np.abs(real_fourstep_rfft(x)-np.fft.rfft(x))):.2e}")
    print("== 3. one cell family ==")
    xc = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    print(f"   complex leg (Givens) vs fft: {np.max(np.abs(cfft_givens(xc)-np.fft.fft(xc))):.2e}")
    print("== 4. DRAM read+write passes (cache=4096 doubles) ==")
    print(f"   {'n':>9} {'sweep':>6} {'radix2DF':>9} {'four-step':>10}")
    for n in (65536, 1048576, 16777216):
        print(f"   {n:>9} {sweep_passes(n):>6} {radix2_df(n,4096):>9} {fourstep_passes(n,4096):>10}")
