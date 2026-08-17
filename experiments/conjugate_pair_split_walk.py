#!/usr/bin/env python3
"""Separate research object: the conjugate-pair split-radix FFT walk.

This is intentionally *not* DIP2 and is intentionally complex.  It uses the
4n+1 / 4n-1 conjugate-pair split

    X[k] = E[k] + W_N^k O_plus[k] + W_N^-k O_minus[k],

so the two odd branches carry complementary phase rotations.  The 4n-1 child
is a cyclically rotated reverse residue class; recursively, that changes the
opening distribution away from ordinary bit reversal while retaining natural
frequency order at the output.

The algebra belongs to the split-radix/conjugate-pair family.  The purpose of
this file is to preserve the walk, measure its exact resorting signature, and
avoid making a novelty claim merely because its permutation looks unusual.
"""

import numpy as np


def _power_of_two(n):
    return n >= 1 and (n & (n - 1)) == 0


def conjugate_pair_fft(x):
    """Exact recursive conjugate-pair split-radix DFT reference."""
    x = np.asarray(x, dtype=np.complex128)
    n = int(x.size)
    if not _power_of_two(n):
        raise ValueError("length must be a power of two")
    if n == 1:
        return x.copy()
    if n == 2:
        return np.array([x[0] + x[1], x[0] - x[1]])

    even = conjugate_pair_fft(x[::2])
    plus = conjugate_pair_fft(x[1::4])
    # 4*j-1 modulo N: N-1, 3, 7, ..., N-5.
    minus_input = np.r_[x[-1:], x[3:-1:4]]
    minus = conjugate_pair_fft(minus_input)

    k = np.arange(n)
    w = np.exp(-2.0j * np.pi * k / n)
    q = n // 4
    return (even[k % (n // 2)] + w * plus[k % q] +
            np.conj(w) * minus[k % q])


def _folded_residue_bins(a, b, k, n):
    """Read periodic bins from a stored real Bruun ``(Re,-Im)`` spectrum."""
    j = np.asarray(k, dtype=np.int64) % n
    reflected = j > n // 2
    q = np.where(reflected, n - j, j)
    return a[q], np.where(reflected, -b[q], b[q])


def conjugate_pair_rfft_residues(x):
    """Strict-real fold of the conjugate-pair walk.

    This belongs beside ``conjugate_pair_fft`` and is not DIP2.  It proves
    that the conjugate phase branches can be coupled using only real Bruun
    ``(a,b)=(Re,-Im)`` packets.  The recurrence is O(N log N).
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if not _power_of_two(n) or n < 4:
        raise ValueError("length must be a power of two >= 4")

    def rec(v):
        size = int(v.size)
        if size == 1:
            return v.copy(), np.zeros(1, dtype=np.float64)
        if size == 2:
            return (np.array([v[0] + v[1], v[0] - v[1]]),
                    np.zeros(2, dtype=np.float64))

        ea, eb = rec(v[::2])
        pa, pb = rec(v[1::4])
        ma, mb = rec(np.r_[v[-1:], v[3:-1:4]])

        k = np.arange(size // 2 + 1, dtype=np.int64)
        era, erb = _folded_residue_bins(ea, eb, k, size // 2)
        pra, prb = _folded_residue_bins(pa, pb, k, size // 4)
        mra, mrb = _folded_residue_bins(ma, mb, k, size // 4)
        theta = 2.0 * np.pi * k / size
        c = np.cos(theta)
        s = np.sin(theta)

        ta = c * (pra + mra) + s * (mrb - prb)
        tb = c * (prb + mrb) + s * (pra - mra)
        return era + ta, erb + tb

    return rec(x)


def conjugate_pair_rfft(x):
    """Public complex packaging for the strict-real conjugate-pair fold."""
    a, b = conjugate_pair_rfft_residues(x)
    return a - 1.0j * b


def real_fold_ledger(n):
    """Static generic arithmetic ledger for the strict-real fold."""
    if not _power_of_two(n) or n < 4:
        raise ValueError("length must be a power of two >= 4")

    def rec(size):
        if size <= 2:
            return 0, 0, 0
        m0, a0, p0 = rec(size // 2)
        m1, a1, p1 = rec(size // 4)
        points = size // 2 + 1
        return (m0 + 2 * m1 + 4 * points,
                a0 + 2 * a1 + 8 * points,
                p0 + 2 * p1 + points)

    muls, adds, pairs = rec(n)
    return {
        "real_multiplications": muls,
        "real_additions": adds,
        "paired_phase_outputs": pairs,
        "muls_per_nlog2n": muls / (n * np.log2(n)),
        "adds_per_nlog2n": adds / (n * np.log2(n)),
    }


def conjugate_pair_leaf_order(n):
    """Opening-distribution order induced by a leaves-up implementation."""
    if not _power_of_two(n):
        raise ValueError("length must be a power of two")

    def rec(indices):
        size = len(indices)
        if size <= 2:
            return list(map(int, indices))
        even = indices[::2]
        plus = indices[1::4]
        minus = np.r_[indices[-1:], indices[3:-1:4]]
        return rec(even) + rec(plus) + rec(minus)

    return rec(np.arange(n))


def bit_reversal(n):
    bits = n.bit_length() - 1
    return [int(f"{j:0{bits}b}"[::-1], 2) for j in range(n)]


def permutation_ledger(order, scalar_bytes=16, line_bytes=64):
    """Transport metrics for gather reads in ``order``.

    ``scalar_bytes=16`` models one complex128 sample.  Line transitions are
    counted without pretending to be a cache simulator.
    """
    p = np.asarray(order, dtype=np.int64)
    gaps = np.abs(np.diff(p))
    lines = (p * scalar_bytes) // line_bytes
    return {
        "travel": int(np.sum(gaps)),
        "mean_gap": float(np.mean(gaps)) if gaps.size else 0.0,
        "line_transitions": int(np.count_nonzero(np.diff(lines))) if gaps.size else 0,
        "descending_edges": int(np.count_nonzero(np.diff(p) < 0)),
    }


def comparison_ledger(n):
    cp = conjugate_pair_leaf_order(n)
    br = bit_reversal(n)
    a = permutation_ledger(cp)
    b = permutation_ledger(br)
    return {
        "cp": a,
        "bit_reversal": b,
        "same_permutation": cp == br,
        "travel_ratio": a["travel"] / b["travel"],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(83)
    print("Conjugate-pair split-radix walk (separate from DIP2)")
    for n in (8, 16, 64, 256, 1024):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        err = np.max(np.abs(conjugate_pair_fft(x) - np.fft.fft(x)))
        led = comparison_ledger(n)
        print(f"  N={n:4d} error={err:.3e} "
              f"cp/bitrev travel={led['travel_ratio']:.3f} "
              f"same={led['same_permutation']}")
    print("Strict-real Bruun fold of the same known walk")
    for n in (8, 16, 64, 256, 1024, 4096):
        x = rng.standard_normal(n)
        err = np.max(np.abs(conjugate_pair_rfft(x) - np.fft.rfft(x)))
        led = real_fold_ledger(n)
        print(f"  N={n:4d} error={err:.3e}; "
              f"mul/(NlogN)={led['muls_per_nlog2n']:.3f}")
