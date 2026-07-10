#!/usr/bin/env python3
"""Regression checks for the Intrinsic Phase-Disk DIP research reference."""

import os
import sys
from pathlib import Path

import numpy as np

# Avoid cross-module Numba cache names in this small algebra regression; the
# production DIP speed path has its own tests.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.intrinsic_dip_boundary import (  # noqa: E402
    boundary_coords,
    promote_boundary,
    truncated_zak_staircase,
)
from experiments.intrinsic_dip_branchbound import ipd_transform  # noqa: E402
from experiments.dip_numba import dip_fft_finish, dip_partial_forward  # noqa: E402
from experiments.spectral_gate_selector import (  # noqa: E402
    BlockSpectra,
    select_support_hybrid,
)
from experiments.grid_disk_pyramid import (  # noqa: E402
    GridDiskStore,
    select_support_grid,
    self_test as grid_disk_self_test,
)


def check_spectral_hybrid():
    """Certified Dirichlet envelope + two-regime hybrid walk stay exact."""
    rng = np.random.default_rng(74)
    N = 128
    t = np.arange(N)
    worst = -np.inf
    for _ in range(10):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        sp = BlockSpectra(x, lazy=True)
        for L in (16, 32, 64):
            lo = int(rng.integers(0, N // L)) * L
            for k in rng.uniform(0, N, 4):
                v = abs(np.sum(x[lo:lo + L] *
                               np.exp(-2j * np.pi * k * t[lo:lo + L] / N)))
                q = sp.envelope(lo, L, k)
                assert q >= v - 1e-9
                worst = max(worst, v - q)
    # adversarial multi-bursts: the hybrid asserts certified global support
    # for every bin internally
    for _ in range(10):
        x = 0.2 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        for _ in range(3):
            a = int(rng.integers(0, N - 8))
            b = int(rng.integers(a + 4, N + 1))
            kk = float(rng.uniform(-N / 2, N / 2))
            x[a:b] += rng.uniform(.5, 2) * np.exp(
                2j * np.pi * kk * t[a:b] / N + 1j * rng.uniform(0, 6.28))
        sp = BlockSpectra(x, lazy=True)
        for k in range(0, N, 5):
            select_support_hybrid(x, 2 * np.pi * k / N, sp, Lf=32)
    print(f"PASS spectral hybrid N={N}: envelope certified "
          f"(max |V|-Q={worst:.2e}), adversarial bins exact")


def check_grid_disk_walk():
    """Grid-shared fine disks: containment property + exact walk results."""
    grid_disk_self_test()
    rng = np.random.default_rng(75)
    N = 128
    t = np.arange(N)
    for _ in range(8):
        x = 0.2 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        for _ in range(3):
            a = int(rng.integers(0, N - 8))
            b = int(rng.integers(a + 4, N + 1))
            kk = float(rng.uniform(-N / 2, N / 2))
            x[a:b] += rng.uniform(.5, 2) * np.exp(
                2j * np.pi * kk * t[a:b] / N + 1j * rng.uniform(0, 6.28))
        sp = BlockSpectra(x, lazy=True)
        store = GridDiskStore(x, G=4, Lf=32)
        for k in range(0, N, 5):
            # asserts the certified global optimum internally
            select_support_grid(x, 2 * np.pi * k / N, sp, store)
    print(f"PASS grid disk walk N={N}: shared fine disks stay exact "
          f"({store.builds} shared builds last trial)")


def main():
    rng = np.random.default_rng(73)
    for N in (32, 64, 128):
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        C, tau, nodes, leaves = ipd_transform(x)
        t = np.arange(N)
        U = x[None, :] * np.exp(-2j * np.pi * np.outer(np.arange(N), t) / N)
        P = np.cumsum(U, axis=1)
        S = np.abs(P) ** 2 / (t + 1)
        truth = np.argmax(S, axis=1) + 1
        assert np.array_equal(tau, truth)
        assert np.max(np.abs(C - P[np.arange(N), tau - 1])) < 1e-9
        assert np.median(leaves) <= 3

        # Selected supports are ordinary paused DIP states with a closed (a,b)
        # boundary type; there is no special correlation egress.
        for k in range(0, N, max(1, N // 8)):
            tk = int(tau[k])
            masked = np.zeros(N, complex)
            masked[:tk] = x[:tk]
            for level in (0, int(np.log2(N)) // 2, int(np.log2(N))):
                B = truncated_zak_staircase(x, level, tk)
                ref = dip_partial_forward(masked, level, N)
                assert np.max(np.abs(B.ravel() - ref)) < 1e-9
                out = dip_fft_finish(B.ravel(), level, N)
                assert abs(out[k] - C[k]) < 1e-9
                if level < int(np.log2(N)):
                    a, b = boundary_coords(N, level, tk)
                    assert promote_boundary(N, level, a, b) == \
                           boundary_coords(N, level + 1, tk)
        print(f"PASS IPD N={N}: exact global support, median "
              f"nodes/leaves={np.median(nodes):.0f}/{np.median(leaves):.0f}")

    check_spectral_hybrid()
    check_grid_disk_walk()


if __name__ == "__main__":
    main()
