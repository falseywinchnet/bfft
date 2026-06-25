"""Scratch: a clean, slow, obviously-correct reference for BFFT's core operation.

BFFT is a *normalized-basis Bruun real FFT*. The whole transform is one idea:
reduce the input polynomial

    p(z) = sum_n  x[n] * z^n        (real coefficients, degree N-1)

modulo the real factorization of  z^N - 1.  The DFT value X[k] is just p(z)
evaluated at the N-th root of unity  z = W^k = exp(-2*pi*i*k/N), so reducing p
modulo the (real) factor whose roots are those unit-circle points and then
evaluating the small remainder there gives the bin -- with all arithmetic real
until the very last step.

Real factorization of z^N - 1 (N a power of two):

    z^N - 1 = (z^(N/2) - 1) * (z^(N/2) + 1)

and every subsequent factor has the Bruun quadratic-in-z^(m/2) shape

    P(m, theta)(z) = z^m - 2*cos(theta)*z^(m/2) + 1            (degree m)

which splits exactly into two half-degree factors of the same shape:

    P(m, theta) = P(m/2, theta/2) * P(m/2, pi - theta/2)

(Proof: with y = z^(m/4), (y^2 - 2c y + 1)(y^2 + 2c y + 1) = y^4 - 2cos(theta) y^2 + 1
when 2 - 4c^2 = -2cos(theta), i.e. c = cos(theta/2).)

Leaves:
  * (z - 1)          -> DC,       X[0]   = p(1)
  * (z + 1)          -> Nyquist,  X[N/2] = p(-1)
  * z^2 - 2cos(t) z + 1  -> the bin k with theta = 2*pi*k/N; the remainder is
                            r(z) = a + b z, and X[k] = a + b*exp(-i*theta).

This file does the reductions with numpy.polydiv (descending-degree coeffs):
maximally readable, not fast. It is the spec the C kernel implements with
in-register butterflies instead of polynomial long division.
"""

import numpy as np


def _polymod(p, d):
    """Remainder of polynomial p modulo d (both highest-degree-first)."""
    _, r = np.polydiv(p, d)
    return r


def _quad_divisor(m, theta):
    """z^m - 2cos(theta) z^(m/2) + 1, highest-degree-first, length m+1."""
    d = np.zeros(m + 1)
    d[0] = 1.0            # z^m
    d[m // 2] = -2.0 * np.cos(theta)   # z^(m/2)
    d[m] = 1.0            # z^0
    return d


def bruun_rfft(x):
    """Reference real FFT via the Bruun real factorization. Returns N//2+1 bins."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    assert N >= 2 and (N & (N - 1)) == 0, "power of two only"

    # p(z) = sum x[n] z^n. polydiv wants highest degree first.
    p = x[::-1].copy()

    X = np.zeros(N // 2 + 1, dtype=np.complex128)

    def recurse_quad(pr, m, theta):
        # pr is p reduced modulo P(m, theta); now split or evaluate.
        if m == 2:
            r = _polymod(pr, _quad_divisor(2, theta))      # a + b z (deg <= 1)
            r = np.atleast_1d(r)
            b = r[-2] if r.shape[0] >= 2 else 0.0           # coeff of z^1
            a = r[-1]                                       # coeff of z^0
            k = int(round(theta * N / (2.0 * np.pi)))
            X[k] = a + b * np.exp(-1j * theta)
            return
        c0, c1 = theta / 2.0, np.pi - theta / 2.0
        recurse_quad(_polymod(pr, _quad_divisor(m // 2, c0)), m // 2, c0)
        recurse_quad(_polymod(pr, _quad_divisor(m // 2, c1)), m // 2, c1)

    def recurse_minus(pr, M):
        # pr is p reduced modulo (z^M - 1).
        if M == 1:                                          # divisor z - 1 -> DC
            X[0] = _polymod(pr, np.array([1.0, -1.0]))[-1]
            return
        half = M // 2
        p_minus = _polymod(pr, _minus_divisor(half))        # z^half - 1
        p_plus = _polymod(pr, _plus_divisor(half))          # z^half + 1
        recurse_minus(p_minus, half)
        if half == 1:                                       # divisor z + 1 -> Nyquist
            X[N // 2] = _polymod(p_plus, np.array([1.0, 1.0]))[-1]
        else:
            recurse_quad(p_plus, half, np.pi / 2.0)         # z^half + 1 == P(half, pi/2)

    def _minus_divisor(M):
        d = np.zeros(M + 1)
        d[0] = 1.0
        d[M] = -1.0
        return d

    def _plus_divisor(M):
        d = np.zeros(M + 1)
        d[0] = 1.0
        d[M] = 1.0
        return d

    recurse_minus(p, N)
    return X


def _max_rel_err(a, b):
    scale = np.maximum(np.abs(b).max(), 1.0)
    return np.abs(a - b).max() / scale


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'N':>8}  {'max_rel_err vs numpy.rfft':>26}")
    for L in range(1, 14):           # N = 2 .. 8192
        N = 1 << L
        x = rng.standard_normal(N)
        got = bruun_rfft(x)
        ref = np.fft.rfft(x)
        err = _max_rel_err(got, ref)
        flag = "OK" if err < 1e-9 else "FAIL"
        print(f"{N:>8}  {err:>26.3e}  {flag}")
