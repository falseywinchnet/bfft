"""Generalized Bruun, normalized + recursive + EXACT-PHASE twiddles.

Angles tracked as exact Fractions of a full turn (f in [0,1), angle = 2*pi*f).
Every cos/sin is computed fresh from the integer-reduced phase (no recurrence
accumulation); leaf bins are exact integers round(N*f).
"""
import numpy as np
import mpmath as mp
from fractions import Fraction as Fr
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

TWIDDLE_DPS = 160
_TWIDDLE_CACHE = {}

def _twiddle_key(kind, f):
    phase = f % 1
    return kind, phase.numerator, phase.denominator

def _phase_to_mpf(f):
    phase = f % 1
    return mp.mpf(phase.numerator) / mp.mpf(phase.denominator)

def _rounded_trig(kind, f):
    """Return a high-precision-generated binary64 twiddle component.

    This models the C++ design target: twiddle tables are generated offline or at
    plan construction with enough precision that each stored double is the
    correctly rounded value of the exact rational turn.
    """
    key = _twiddle_key(kind, f)
    if key in _TWIDDLE_CACHE:
        return _TWIDDLE_CACHE[key]

    with mp.workdps(TWIDDLE_DPS):
        angle = 2 * mp.pi * _phase_to_mpf(f)
        if kind == "cos":
            value = float(mp.cos(angle))
        else:
            value = float(mp.sin(angle))

    _TWIDDLE_CACHE[key] = value
    return value

def _cos(f):  # f: Fraction turns
    return _rounded_trig("cos", f)
def _sin(f):
    return _rounded_trig("sin", f)

def _pairwise_sum(terms):
    """Pairwise sum array terms to reduce source-order accumulation."""
    items = list(terms)
    if not items:
        return 0.0
    while len(items) > 1:
        next_items = []
        limit = len(items) - 1
        for i in range(0, limit, 2):
            next_items.append(items[i] + items[i + 1])
        if len(items) % 2 == 1:
            next_items.append(items[-1])
        items = next_items
    return items[0]

def _cheb_exact(g, n):
    """Chebyshev reduction coeffs a_i=-U_{i-2}(c), b_i=U_{i-1}(c), c=cos(2pi g).

    Two stable regimes:
      * long reductions (large n, e.g. prime p): closed form b_i=sin(i*phi)/sin(phi)
        with fresh integer-reduced phase -> no recurrence accumulation. Safe because
        the angles g=j/p are moderate (not pathologically near 0 or 1/2).
      * short reductions (small n, Bruun small-prime splits, possibly phi near 0/pi):
        the recurrence (polynomial in c) is stable where the closed form is 0/0.
    """
    if n > 16:
        sg = _sin(g)
        a = [-_sin(g*(i-1))/sg for i in range(n)]
        b = [_sin(g*i)/sg for i in range(n)]
        return a, b
    c = _cos(g)
    a = [1.0, 0.0]; b = [0.0, 1.0]
    for i in range(2, n):
        a.append(-b[i-1]); b.append(a[i-1] + 2*c*b[i-1])
    return a[:n], b[:n]

def _nb_subtree(r, f, N, out):
    """f: Fraction turns, node angle = 2*pi*f."""
    d = len(r)
    if d == 2:
        k = int(round(N*float(f % 1))) % N
        out[k] = r[0] - 1j*r[1]; return
    q = d//4
    A0=r[0:q]; B0=r[q:2*q]; A1=r[2*q:3*q]; B1=r[3*q:4*q]
    c=_cos(f/2); s=_sin(f/2)                 # cos/sin(theta/2), theta=2pi f
    R=c*B0-s*B1; I=s*B0+c*B1
    _nb_subtree(np.concatenate([A0+R, A1+I]), f/2, N, out)
    _nb_subtree(np.concatenate([A0-R, -A1+I]), Fr(1,2)-f/2, N, out)

def rfft_gen_exact(x):
    x = np.asarray(x, dtype=np.float64); N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)
    def place(k, val):
        k %= N; X[k] = val
        if k != 0 and (N-k) != k: X[N-k] = np.conj(val)

    def reduce_bruun_cascade(lo, hi, M, f):
        """Reduce a Bruun node already in the condition-1 cascade frame.

        The pair ``lo - 1j*hi`` is the node value in the normalized cascade
        frame.  Odd child splits can stay in that frame by evaluating the
        complex block coefficients with unit-modulus twiddles.  This avoids the
        Chebyshev inverse (1/sin(phi)) coefficients that made composite odd
        Bruun nodes lose FFT-grade accuracy for sizes like 2187 and 6075.
        """
        if M == 1:
            k = int(round(N*float(f % 1))) % N
            place(k, lo[0] - 1j*hi[0]); return
        if _is_pow2(M):
            seed = np.concatenate([lo, hi])
            out = {}; _nb_subtree(seed, f, N, out)
            for k, val in out.items(): place(k, val)
            return
        p = _odd_prime(M); Mp = M // p
        A = [lo[i*Mp:(i+1)*Mp] for i in range(p)]
        B = [hi[i*Mp:(i+1)*Mp] for i in range(p)]
        for t in range(p):
            phi = (f + t) / p             # (theta + 2pi t)/p in turns
            child_lo = _pairwise_sum(A[i]*_cos(i*phi) - B[i]*_sin(i*phi) for i in range(p))
            child_hi = _pairwise_sum(B[i]*_cos(i*phi) + A[i]*_sin(i*phi) for i in range(p))
            reduce_bruun_cascade(child_lo, child_hi, Mp, phi)

    def reduce_bruun(r, twoM, f):       # factor angle = 2*pi*f
        M = twoM // 2
        lo = r[:M]; hi = r[M:]
        cf = _cos(f); sf = _sin(f)
        reduce_bruun_cascade(lo + cf*hi, sf*hi, M, f)

    def reduce_minus(r, D, sigma):
        if _is_pow2(D):
            Xloc = rfft_normbruun(r) if D > 1 else np.array([r[0]+0j])
            for k in range(len(Xloc)): place(sigma*k, Xloc[k])
            return
        p = _odd_prime(D); M = D // p
        R = [r[i*M:(i+1)*M] for i in range(p)]
        reduce_minus(_pairwise_sum(R), M, sigma*p)
        for j in range(1, (p-1)//2 + 1):
            g = Fr(j, p)
            # CONDITION-1 cascade seed: B*(natural) = (sum_i R_i cos(i*phi),
            # sum_i R_i sin(i*phi)), bounded coeffs (no 1/sin(phi) growth). Exact phase.
            seed_lo = _pairwise_sum(R[i]*_cos(Fr(i*j, p)) for i in range(p))
            seed_hi = _pairwise_sum(R[i]*_sin(Fr(i*j, p)) for i in range(p))
            if M == 1:                                   # prime leaf: X = seed_lo - i seed_hi
                place(int(round(N*float(g % 1))) % N, seed_lo[0] - 1j*seed_hi[0])
            else:                                        # composite/pow2 M: stay in cascade frame
                reduce_bruun_cascade(seed_lo, seed_hi, M, g)

    reduce_minus(x, N, 1)
    return X[:N//2+1]

if __name__ == "__main__":
    import sympy
    try:
        from . import scratch_genbruun_rec as OLD
    except ImportError:  # pragma: no cover - supports direct script execution.
        import scratch_genbruun_rec as OLD
    sizes=[9,15,27,45,75,225,405,1024,1920,3000, 127,257,509,521,1021,511,2187,6075,10125]
    print(f"{'N':>6} {'factors':>16} {'old':>10} {'exact':>10}")
    for N in sorted(set(sizes)):
        x=np.random.default_rng(N).standard_normal(N)
        ref=np.fft.rfft(x); sc=max(np.abs(ref).max(),1)
        eo=np.abs(OLD.rfft_gen(x)-ref).max()/sc
        ee=np.abs(rfft_gen_exact(x)-ref).max()/sc
        print(f"{N:>6} {str(sympy.factorint(N)):>16} {eo:>10.2e} {ee:>10.2e}")
