#!/usr/bin/env python3
"""Arbitrary dyadic-offset Fourier channels as one twisted phase-packet cell.

F_N^alpha[p] = sum_n x[n] exp(-2 pi i (p+alpha)n/N).
Under even/odd splitting alpha is inherited unchanged by both children and only
adds to the parent twiddle angle. No input pre-twist is required.
"""

import numpy as np


def twisted_fft(x, alpha):
    x = np.asarray(x, np.complex128)
    N = x.size
    if N == 1:
        return x.copy()
    E = twisted_fft(x[0::2], alpha)
    O = twisted_fft(x[1::2], alpha)
    p = np.arange(N // 2)
    w = np.exp(-2j * np.pi * (p + alpha) / N)
    Y = np.empty(N, np.complex128)
    Y[:N // 2] = E + w * O
    Y[N // 2:] = E - w * O
    return Y


def direct_fractional(x, alpha):
    x = np.asarray(x, np.complex128)
    N = x.size
    return np.exp(-2j * np.pi * np.outer(np.arange(N) + alpha,
                                         np.arange(N)) / N) @ x


def twisted_fft_bins(x, alpha, bins):
    """Pruned twisted butterfly: return only requested output bins + cell count."""
    x = np.asarray(x, np.complex128)
    bins = np.unique(np.asarray(bins, np.int64))
    N = x.size
    if bins.size == 0:
        return bins, np.empty(0, complex), 0
    if N == 1:
        return bins, np.array([x[0]], complex), 0
    half = N // 2
    child_bins = np.unique(bins % half)
    cb, E, ce = twisted_fft_bins(x[0::2], alpha, child_bins)
    _, O, co = twisted_fft_bins(x[1::2], alpha, child_bins)
    lookup = {int(k): j for j, k in enumerate(cb)}
    out = np.empty(bins.size, complex)
    for j, p in enumerate(bins):
        r = int(p % half)
        w = np.exp(-2j * np.pi * (r + alpha) / N)
        v = E[lookup[r]] + w * O[lookup[r]]
        if p >= half:
            v = E[lookup[r]] - w * O[lookup[r]]
        out[j] = v
    return bins, out, ce + co + child_bins.size


def pruned_cell_count(N, bins):
    bins = np.unique(np.asarray(bins, np.int64))
    if bins.size == 0 or N == 1:
        return 0
    child = np.unique(bins % (N // 2))
    return 2 * pruned_cell_count(N // 2, child) + child.size


def block_global_bins(x, start, L, delta):
    """One (block,delta/e) channel supplies k=p*e+delta, p=0..L-1."""
    N = x.size
    e = N // L
    alpha = delta / e
    local = twisted_fft(x[start:start + L], alpha)
    p = np.arange(L)
    k = p * e + delta
    # Translate local block coordinates back to global sample positions.
    local *= np.exp(-2j * np.pi * k * start / N)
    return k, local


def self_test():
    rng = np.random.default_rng(93)
    worst = 0.0
    for N in (16, 64, 256):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        for den in (1, 2, 4, 8, 16):
            for num in range(den):
                alpha = num / den
                err = np.max(np.abs(twisted_fft(x, alpha) -
                                    direct_fractional(x, alpha)))
                worst = max(worst, float(err))
    print(f"twisted fractional cell worst direct error: {worst:.2e}")
    assert worst < 1e-10

    N = 256
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    t = np.arange(N)
    worst_block = 0.0
    for L in (8, 16, 32, 64, 128):
        e = N // L
        start = 2 * L if 3 * L <= N else 0
        for delta in range(e):
            k, got = block_global_bins(x, start, L, delta)
            ref = np.array([
                np.sum(x[start:start + L] *
                       np.exp(-2j * np.pi * kk * t[start:start + L] / N))
                for kk in k])
            worst_block = max(worst_block, float(np.max(np.abs(got - ref))))
    print(f"(block,delta/e) global-bin channel error: {worst_block:.2e}")
    assert worst_block < 1e-10

    for N in (32, 128, 256):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        full = twisted_fft(x, 3 / 8)
        for count in (1, 3, N // 4, N):
            bins = np.sort(rng.choice(N, min(count, N), replace=False))
            got_bins, got, cells = twisted_fft_bins(x, 3 / 8, bins)
            assert np.array_equal(got_bins, bins)
            assert np.max(np.abs(got - full[bins])) < 1e-10
            assert cells == pruned_cell_count(N, bins)
    print("pruned twisted outputs: exact; cell-count recurrence verified")


if __name__ == "__main__":
    self_test()
