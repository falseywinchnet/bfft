"""Generalized Bruun, NORMALIZED + fully recursive (arbitrary N), FFT-grade.

Strategy: carry NATURAL residues through odd-prime splits (shallow, bounded
Chebyshev reductions); drop into the normalized cascade (stage-0 B-map +
_nb_subtree, or rfft_normbruun for minus-nodes) the moment a node is pure pow2.
"""
import numpy as np
try:
    from .scratch_normbruun import rfft_normbruun
except ImportError:  # pragma: no cover - supports direct script execution.
    from scratch_normbruun import rfft_normbruun

def _is_pow2(n): return n >= 1 and (n & (n-1)) == 0
def _odd_prime(n):
    while n % 2 == 0: n //= 2
    if n == 1: return None
    d = 3
    while d*d <= n:
        if n % d == 0: return d
        d += 2
    return n

def _cheb(phi, n):
    """w^i = a_i + b_i w  (mod w^2 - 2cos(phi) w + 1), i=0..n-1."""
    c = np.cos(phi)
    a = [1.0, 0.0]; b = [0.0, 1.0]
    for i in range(2, n):
        a.append(-b[i-1]); b.append(a[i-1] + 2*c*b[i-1])
    return a, b

def _nb_subtree(r, theta, N, out):
    d = len(r)
    if d == 2:
        k = int(round(theta*N/(2*np.pi))) % N
        out[k] = r[0] - 1j*r[1]; return
    q = d//4
    A0=r[0:q]; B0=r[q:2*q]; A1=r[2*q:3*q]; B1=r[3*q:4*q]
    c=np.cos(theta/2); s=np.sin(theta/2)
    R=c*B0-s*B1; I=s*B0+c*B1
    _nb_subtree(np.concatenate([A0+R, A1+I]), theta/2, N, out)
    _nb_subtree(np.concatenate([A0-R, -A1+I]), np.pi-theta/2, N, out)

def rfft_gen(x):
    x = np.asarray(x, dtype=np.float64); N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)

    def place(k, val):
        k %= N; X[k] = val
        if k != 0 and (N - k) != k: X[N-k] = np.conj(val)

    def reduce_bruun(r, twoM, theta):
        M = twoM // 2
        if M == 1:
            # leaf reached via an odd split: residue is NATURAL (lo + hi*z),
            # so the bin value is lo + hi*e^{-i theta} (not the cascade readout).
            place(int(round(theta*N/(2*np.pi))), r[0] + r[1]*np.exp(-1j*theta)); return
        if _is_pow2(M):
            lo = r[:M]; hi = r[M:]
            seed = np.concatenate([lo + np.cos(theta)*hi, np.sin(theta)*hi])
            out = {}; _nb_subtree(seed, theta, N, out)
            for k, val in out.items(): place(k, val)
            return
        p = _odd_prime(M); Mp = M // p
        R = [r[i*Mp:(i+1)*Mp] for i in range(2*p)]
        for t in range(p):
            phi = (theta + 2*np.pi*t) / p
            a, b = _cheb(phi, 2*p)
            lo = sum(R[i]*a[i] for i in range(2*p))
            hi = sum(R[i]*b[i] for i in range(2*p))
            reduce_bruun(np.concatenate([lo, hi]), 2*Mp, phi)

    def reduce_minus(r, D, sigma):
        if _is_pow2(D):
            Xloc = rfft_normbruun(r) if D > 1 else np.array([r[0]+0j])
            for k in range(len(Xloc)): place(sigma*k, Xloc[k])
            return
        p = _odd_prime(D); M = D // p
        R = [r[i*M:(i+1)*M] for i in range(p)]
        reduce_minus(sum(R), M, sigma*p)
        for j in range(1, (p-1)//2 + 1):
            theta = 2*np.pi*j/p
            a, b = _cheb(theta, p)
            lo = sum(R[i]*a[i] for i in range(p))
            hi = sum(R[i]*b[i] for i in range(p))
            reduce_bruun(np.concatenate([lo, hi]), 2*M, theta)

    reduce_minus(x, N, 1)
    return X[:N//2+1]

if __name__ == "__main__":
    import sympy
    sizes = [256, 6, 96, 45, 75, 1920, 3000, 240, 360, 720, 1080,
             15, 225, 405, 35, 49, 63, 1024, 600, 9, 27, 81]
    print(f"{'N':>6} {'factors':>18} {'err':>10}")
    worst = 0.0
    for N in sorted(set(sizes)):
        x = np.random.default_rng(N).standard_normal(N)
        e = np.abs(rfft_gen(x) - np.fft.rfft(x)).max() / max(np.abs(np.fft.rfft(x)).max(), 1)
        worst = max(worst, e)
        print(f"{N:>6} {str(sympy.factorint(N)):>18} {e:>10.2e}  {'OK' if e<1e-12 else 'FAIL'}")
    print(f"WORST {worst:.2e}")
