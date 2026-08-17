#!/usr/bin/env python3
"""Subquadratic exact-real engine for the genuine phase-interval DIP2.

The double-precision monomial tree in ``dip2_phase_axis.py`` exposes the walk
but fails numerically once its consecutive-arc product polynomials become ill
conditioned.  This module keeps the same genuinely different tree and cures
its quadratic arithmetic count with three real-algebra ingredients:

* divide-and-conquer Chebyshev-to-power conversion;
* Karatsuba polynomial products;
* Newton-reciprocal fast polynomial remainders.

All arithmetic is real ``decimal.Decimal``.  No FFT, complex transform, DCT,
or conjugate-pair split-radix call is hidden inside the multiplier.  For
Karatsuba cost M(n)=Theta(n^log2(3)), the complete balanced remainder walk is
Theta(M(n)): the work over successively smaller levels is a convergent
geometric series.  This cures N^2, although it does not yet reach FFT-class
N log N and Decimal is an oracle rather than a production representation.
"""

from decimal import Decimal, localcontext

import numpy as np


ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
HALF = Decimal("0.5")


def _trim(a):
    while len(a) > 1 and not a[-1]:
        a.pop()
    return a


def _add(a, b):
    out = [ZERO] * max(len(a), len(b))
    for j, v in enumerate(a):
        out[j] += v
    for j, v in enumerate(b):
        out[j] += v
    return _trim(out)


def _sub(a, b):
    out = [ZERO] * max(len(a), len(b))
    for j, v in enumerate(a):
        out[j] += v
    for j, v in enumerate(b):
        out[j] -= v
    return _trim(out)


def _school_mul(a, b):
    if not a or not b:
        return [ZERO]
    out = [ZERO] * (len(a) + len(b) - 1)
    for j, u in enumerate(a):
        for k, v in enumerate(b):
            out[j + k] += u * v
    return _trim(out)


def _karatsuba(a, b, cutoff=12):
    """Ascending-coefficient real polynomial product."""
    a = _trim(list(a))
    b = _trim(list(b))
    if min(len(a), len(b)) <= cutoff:
        return _school_mul(a, b)
    width = max(len(a), len(b))
    mid = width // 2
    a0, a1 = a[:mid], a[mid:]
    b0, b1 = b[:mid], b[mid:]
    if not a1 or not b1:
        return _school_mul(a, b)
    z0 = _karatsuba(a0, b0, cutoff)
    z2 = _karatsuba(a1, b1, cutoff)
    z1 = _sub(_sub(_karatsuba(_add(a0, a1), _add(b0, b1), cutoff), z0), z2)
    out = [ZERO] * (len(a) + len(b) - 1)
    for j, v in enumerate(z0):
        out[j] += v
    for j, v in enumerate(z1):
        out[mid + j] += v
    for j, v in enumerate(z2):
        out[2 * mid + j] += v
    return _trim(out)


def _series_inverse(h, count):
    """Newton inverse of h modulo x**count; requires h[0] != 0."""
    if count <= 0:
        return []
    inv = [ONE / h[0]]
    while len(inv) < count:
        width = min(2 * len(inv), count)
        prod = _karatsuba(h[:width], inv)[:width]
        error = [ZERO] * width
        error[0] = TWO - prod[0]
        for j in range(1, len(prod)):
            error[j] = -prod[j]
        inv = _karatsuba(inv, error)[:width]
    return inv[:count]


def _fast_remainder(f, g):
    """Remainder f mod g using reversed Newton division and Karatsuba."""
    f = _trim(list(f))
    g = _trim(list(g))
    if len(f) < len(g):
        return f
    qcount = len(f) - len(g) + 1
    rg = list(reversed(g))
    rf = list(reversed(f))[:qcount]
    qrev = _karatsuba(rf, _series_inverse(rg, qcount))[:qcount]
    q = list(reversed(qrev))
    prod = _karatsuba(q, g)
    rem = [ZERO] * (len(g) - 1)
    for j in range(len(rem)):
        rem[j] = f[j] - (prod[j] if j < len(prod) else ZERO)
    return _trim(rem)


class _ChebyshevConverter:
    """Fast conversion of sum c[n] T_n(t/2) to powers of t."""

    def __init__(self):
        self.basis = {0: [ONE], 1: [ZERO, HALF]}

    def t_basis(self, n):
        if n in self.basis:
            return self.basis[n]
        if n & (n - 1):
            raise ValueError("basis cache expects a power-of-two index")
        half = self.t_basis(n // 2)
        out = [TWO * v for v in _karatsuba(half, half)]
        out[0] -= ONE                       # T_2m = 2*T_m^2 - 1
        self.basis[n] = _trim(out)
        return self.basis[n]

    def power2_block(self, c):
        """Convert coefficients 0..L-1, for power-of-two L."""
        length = len(c)
        if length == 1:
            return [c[0]]
        if length <= 8:
            out = [ZERO] * length
            t0 = [ONE]
            for j, v in enumerate(t0):
                out[j] += c[0] * v
            t1 = [ZERO, HALF]
            for degree in range(1, length):
                if degree > 1:
                    shifted = [ZERO] + t1
                    if len(shifted) < len(t0):
                        shifted += [ZERO] * (len(t0) - len(shifted))
                    for j, v in enumerate(t0):
                        shifted[j] -= v
                    t0, t1 = t1, _trim(shifted)
                basis = t1
                for j, v in enumerate(basis):
                    out[j] += c[degree] * v
            return _trim(out)

        half = length // 2
        low = list(c[:half])
        high = c[half:]
        q = low
        h = [ZERO] * half
        h[0] = high[0]
        # T_(m+j) = 2*T_m*T_j - T_(m-j), j>0.
        for j in range(1, half):
            q[half - j] -= high[j]
            h[j] = TWO * high[j]
        qp = self.power2_block(q)
        hp = self.power2_block(h)
        return _add(qp, _karatsuba(self.t_basis(half), hp))

    def convert_with_endpoint(self, c):
        """Convert degree-M coefficients where M is a power of two."""
        m = len(c) - 1
        low = self.power2_block(c[:m])
        endpoint = [c[m] * v for v in self.t_basis(m)]
        return _add(low, endpoint)


def _u_to_t_coefficients(b):
    """O(N) conversion from U_n to T_n coefficients."""
    out = [ZERO] * len(b)
    parity_sum = [ZERO, ZERO]
    for j in range(len(b) - 1, -1, -1):
        parity_sum[j & 1] += b[j]
        out[j] = TWO * parity_sum[j & 1]
    if out:
        out[0] *= HALF
    return out


def _atan_inverse(q, eps):
    x = ONE / Decimal(q)
    x2 = x * x
    term = x
    total = term
    j = 1
    sign = -ONE
    while True:
        term *= x2
        add = sign * term / Decimal(2 * j + 1)
        total += add
        if abs(add) < eps:
            return total
        sign = -sign
        j += 1


def _pi(eps):
    return Decimal(16) * _atan_inverse(5, eps) - Decimal(4) * _atan_inverse(239, eps)


def _cos(x, eps):
    term = ONE
    total = ONE
    x2 = x * x
    j = 1
    while True:
        term *= -x2 / Decimal((2 * j - 1) * (2 * j))
        total += term
        if abs(term) < eps:
            return total
        j += 1


class _Node:
    __slots__ = ("lo", "hi", "modulus", "left", "right")

    def __init__(self, nodes, lo, hi):
        self.lo, self.hi = lo, hi
        if hi - lo == 1:
            self.left = self.right = None
            self.modulus = [-nodes[lo], ONE]
        else:
            mid = (lo + hi) // 2
            self.left = _Node(nodes, lo, mid)
            self.right = _Node(nodes, mid, hi)
            self.modulus = _karatsuba(self.left.modulus, self.right.modulus)


def _tree_eval(poly, node, out):
    rem = _fast_remainder(poly, node.modulus)
    if node.left is None:
        out[node.lo] = rem[0]
    else:
        _tree_eval(rem, node.left, out)
        _tree_eval(rem, node.right, out)


def dip2_subquadratic_rfft(x, precision=None):
    """Execute genuine contiguous-phase DIP2 in subquadratic real algebra."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n < 4 or (n & (n - 1)):
        raise ValueError("length must be a power of two >= 4")
    # Global monomial interval products lose roughly O(N) bits.  Keep a wide
    # guard region; callers can override this research-oracle policy.
    digits = int(precision or max(70, n))
    with localcontext() as ctx:
        ctx.prec = digits
        eps = Decimal(10) ** Decimal(-(digits - 12))
        pi = _pi(eps)
        xd = [Decimal(repr(float(v))) for v in x]
        m = n // 2

        at = [ZERO] * (m + 1)
        at[0], at[m] = xd[0], xd[m]
        bu = [ZERO] * (m - 1)
        for j in range(1, m):
            at[j] = xd[j] + xd[n - j]
            bu[j - 1] = HALF * (xd[j] - xd[n - j])

        converter = _ChebyshevConverter()
        a = converter.convert_with_endpoint(at)
        bt = _u_to_t_coefficients(bu)
        padded_bt = bt + [ZERO] * (m - len(bt))
        b = converter.power2_block(padded_bt)

        theta = [TWO * pi * Decimal(k) / Decimal(n) for k in range(m + 1)]
        nodes = [TWO * _cos(th, eps) for th in theta]
        root = _Node(nodes, 0, m + 1)
        av = [ZERO] * (m + 1)
        bv = [ZERO] * (m + 1)
        _tree_eval(a, root, av)
        _tree_eval(b, root, bv)

        # sin(theta)=cos(pi/2-theta), still entirely real Decimal arithmetic.
        out = np.empty(m + 1, dtype=np.complex128)
        pio2 = pi * HALF
        for k in range(m + 1):
            s = _cos(pio2 - theta[k], eps)
            out[k] = complex(float(av[k]), float(-TWO * s * bv[k]))
        return out


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(85)
    for n in (8, 16, 32, 64):
        x = rng.standard_normal(n)
        start = time.perf_counter()
        got = dip2_subquadratic_rfft(x)
        elapsed = time.perf_counter() - start
        err = np.max(np.abs(got - np.fft.rfft(x)))
        print(f"N={n:4d} err={err:.3e} elapsed={elapsed:.3f}s")
