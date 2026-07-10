#!/usr/bin/env python3
"""The intrinsic leading-edge boundary as a two-height Zak/DIP type.

At DIP level t, q=N/2^t and n=j+r*q.  Writing tau=a*q+b makes the natural
prefix mask exactly `a` complete Zak rows plus row `a` over columns j<b.
The pair (a,b) is therefore a closed support type. Advancing one DIP level
promotes one binary bit from b into a.

This script verifies:
  1. the staircase formula equals the level-t DIP state of x[:tau];
  2. finishing that paused state equals FFT(x[:tau]);
  3. boundary promotion is bit-exact at every level;
  4. phase-disk selection can egress through these paused states without an
     endpoint recurrence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dip_numba import dip_fft_finish, dip_partial_forward
from experiments.intrinsic_dip_branchbound import select_support_disk, make_cases


def boundary_coords(N, t, tau):
    q = N >> t
    return divmod(int(tau), q)


def promote_boundary(N, t, a, b):
    q2 = N >> (t + 1)
    return 2 * a + (b // q2), b % q2


def truncated_zak_staircase(x, t, tau):
    x = np.asarray(x, np.complex128)
    N = x.size
    e, q = 1 << t, N >> t
    a, b = boundary_coords(N, t, tau)
    B = np.zeros((e, q), np.complex128)
    delta = np.arange(e)
    # Complete time-comb rows r<a contribute to every Zak column.
    for r in range(min(a, e)):
        ph = np.exp(-2j * np.pi * delta * r / e)[:, None]
        B += ph * x[r * q:(r + 1) * q][None, :]
    # One partial row carries the residual boundary b.
    if a < e and b > 0:
        ph = np.exp(-2j * np.pi * delta * a / e)[:, None]
        B[:, :b] += ph * x[a * q:a * q + b][None, :]
    return B


def verify_boundaries(N=256):
    rng = np.random.default_rng(71)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    taus = sorted(set([1, 2, 3, N // 3, N // 2 - 1, N // 2,
                       N // 2 + 1, N - 1, N] +
                      rng.integers(1, N + 1, 12).tolist()))
    worst_state = 0.0
    worst_finish = 0.0
    for tau in taus:
        masked = np.zeros(N, np.complex128)
        masked[:tau] = x[:tau]
        for t in range(int(np.log2(N)) + 1):
            B = truncated_zak_staircase(x, t, tau)
            ref = dip_partial_forward(masked, t, N).reshape(1 << t, N >> t)
            worst_state = max(worst_state, float(np.max(np.abs(B - ref))))
            out = dip_fft_finish(B.ravel(), t, N)
            worst_finish = max(worst_finish,
                               float(np.max(np.abs(out - np.fft.fft(masked)))))
            if t < int(np.log2(N)):
                a, b = boundary_coords(N, t, tau)
                assert promote_boundary(N, t, a, b) == \
                       boundary_coords(N, t + 1, tau)
    print(f"boundary identity N={N}: state err={worst_state:.2e}, "
          f"finish err={worst_finish:.2e}")
    assert worst_state < 1e-10 and worst_finish < 1e-10


def verify_selected_egress(N=256, pause_t=4):
    rng = np.random.default_rng(72)
    for name, x in make_cases(N, rng).items():
        mean = np.mean(np.abs(x) ** 2)
        selected = []
        err = 0.0
        for k in range(N):
            tau, z, score, audit = select_support_disk(x, 2 * np.pi * k / N)
            if score <= (np.log(N) + 2) * mean:
                continue
            B = truncated_zak_staircase(x, pause_t, tau)
            X = dip_fft_finish(B.ravel(), pause_t, N)
            err = max(err, abs(X[k] - z))
            selected.append((k, tau, audit.nodes, audit.leaves))
        unique = len({v[1] for v in selected})
        med_nodes = np.median([v[2] for v in selected]) if selected else 0
        med_leaves = np.median([v[3] for v in selected]) if selected else 0
        print(f"{name:8s}: active={len(selected):3d}, unique tau={unique:3d}, "
              f"egress err={err:.2e}, disk nodes/leaves med="
              f"{med_nodes:.0f}/{med_leaves:.0f}")
        assert err < 1e-9


if __name__ == "__main__":
    for N in (64, 256):
        verify_boundaries(N)
    verify_selected_egress()
