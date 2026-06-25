"""Generalized Bruun: arbitrary-N real FFT by real factorization of z^N - 1.

Correctness-first reference (polynomial remainder; not yet the fast butterfly
form). The point is to show the *object*: z^N-1 factors over the reals into Bruun
pieces for ANY N, the 2-adic part reuses the angle-halving recursion, and odd
prime factors peel off via the real split

    z^(pM) - 1 = (z^M - 1) * prod_{j=1..(p-1)/2} (z^(2M) - 2cos(2*pi*j/p) z^M + 1)

No Bluestein, no Rader convolution, no chirp. Pure residue-domain factorization.
"""
import numpy as np

def _smallest_prime_factor(n):
    d = 2
    while d * d <= n:
        if n % d == 0:
            return d
        d += 1 if d == 2 else 2
    return n

def _polymod(p, d):
    return np.polydiv(p, d)[1]

def gen_bruun_rfft(x):
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    p = x[::-1].astype(np.complex128).copy()   # p(z)=sum x[n] z^n, descending
    X = np.zeros(N, dtype=np.complex128)
    WN = np.exp(-2j*np.pi/np.arange(1,2)[0])    # placeholder

    def emit_quad(rem, theta):
        rem = np.atleast_1d(rem)
        b = rem[-2] if rem.shape[0] >= 2 else 0.0
        a = rem[-1]
        k = int(round(theta * N / (2*np.pi)))
        X[k] = a + b*np.exp(-1j*theta)
        X[(N-k) % N] = np.conj(X[k]) if k != 0 else X[k]

    def div_minus(M):                # z^M - 1
        d = np.zeros(M+1); d[0]=1; d[M]=-1; return d
    def div_plus(M):                 # z^M + 1
        d = np.zeros(M+1); d[0]=1; d[M]=1;  return d
    def div_quad(M, theta):          # z^(2M) - 2cos(theta) z^M + 1
        d = np.zeros(2*M+1); d[0]=1; d[M]=-2*np.cos(theta); d[2*M]=1; return d

    def rec_minus(pr, M):            # pr reduced mod (z^M - 1)
        if M == 1:
            X[0] = _polymod(pr, np.array([1.0,-1.0]))[-1]; return
        pf = _smallest_prime_factor(M)
        if pf == 2:
            h = M//2
            rec_minus(_polymod(pr, div_minus(h)), h)
            rec_plus(_polymod(pr, div_plus(h)), h)
        else:
            Mp = M//pf
            rec_minus(_polymod(pr, div_minus(Mp)), Mp)
            for j in range(1, pf//2+1):
                theta = 2*np.pi*j/pf            # angle of the z^Mp variable
                rec_quad(_polymod(pr, div_quad(Mp, theta)), Mp, theta)

    def rec_plus(pr, M):             # pr reduced mod (z^M + 1); roots e^{i pi (2t+1)/M}
        if M == 1:
            X[N//2] = _polymod(pr, np.array([1.0,1.0]))[-1]; return
        # z^M + 1 == z^(2*(M/2)) - 2cos(pi/2) z^(M/2) + 1 when M even -> Bruun quad angle pi/2
        if M % 2 == 0:
            rec_quad(pr, M//2, np.pi/2); return
        # M odd: (z+1) * prod quads with angles (2t+1)pi/M
        rec_plus(_polymod(pr, np.array([1.0,1.0])), 1)   # z+1 -> Nyquist? handled below
        # general odd z^M+1 split via prime of M
        # roots: e^{i pi (2t+1)/M}, t=0..M-1 -> as conj pairs
        for t in range((M)//2):
            theta = np.pi*(2*t+1)/M
            emit_quad(_polymod(pr, div_quad_root(theta)), theta)

    def div_quad_root(theta):        # z^2 - 2cos(theta) z + 1
        return np.array([1.0, -2*np.cos(theta), 1.0])

    def rec_quad(pr, M, theta):      # pr mod (z^(2M) - 2cos(theta) z^M + 1)
        if M == 1:
            emit_quad(pr, theta); return
        pf = _smallest_prime_factor(M)
        if pf == 2:
            c0, c1 = theta/2, np.pi - theta/2
            rec_quad(_polymod(pr, div_quad(M//2, c0)), M//2, c0)
            rec_quad(_polymod(pr, div_quad(M//2, c1)), M//2, c1)
        else:
            # z^(2M)-2cos(th)z^M+1, M=pf*Mp: roots z^M = e^{±i th}; z = e^{i(±th+2pi a)/M}
            Mp = M//pf
            for t in range(pf):
                phi = (theta + 2*np.pi*t)/pf
                rec_quad(_polymod(pr, div_quad(Mp, phi)), Mp, phi)

    rec_minus(p, N)
    return X[:N//2+1]

if __name__ == "__main__":
    print(f"{'N':>6} {'factors':>14} {'max_rel_err vs numpy':>22}")
    import sympy
    for N in [3,5,6,7,9,10,12,15,24,45,48,96,120,127,511,512]:
        x = np.random.default_rng(N).standard_normal(N)
        try:
            got = gen_bruun_rfft(x); ref = np.fft.rfft(x)
            err = np.abs(got-ref).max()/max(np.abs(ref).max(),1)
            fac = str(sympy.factorint(N))
            flag = "OK" if err<1e-9 else "FAIL"
            print(f"{N:>6} {fac:>14} {err:>22.2e}  {flag}")
        except Exception as e:
            print(f"{N:>6}  ERROR {type(e).__name__}: {e}")
