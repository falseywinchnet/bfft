#!/usr/bin/env python3
"""Deterministic adversarial comparison: heuristic FCT vs exact IPD-DIP."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft
from experiments.intrinsic_dip_branchbound import ipd_transform


def run(trials=200, N=128):
    rng = np.random.default_rng(882)
    plan = bfft.FctPlan(N, act=0.0)
    worst = (1.0, None)
    for trial in range(trials):
        x = 0.2 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        t = np.arange(N)
        for _ in range(3):
            a = int(rng.integers(0, N - 8))
            b = int(rng.integers(a + 4, N + 1))
            k = float(rng.uniform(-N / 2, N / 2))
            x[a:b] += rng.uniform(0.5, 2.0) * np.exp(
                2j * np.pi * k * t[a:b] / N + 1j * rng.uniform(0, 2 * np.pi))
        C, tau = plan.fct_complex(x)
        P = np.cumsum(
            x[None, :] * np.exp(-2j * np.pi * np.outer(np.arange(N), t) / N),
            axis=1)
        S = np.abs(P) ** 2 / (t + 1)
        ratio = np.abs(C) ** 2 / tau / (S.max(axis=1) + 1e-30)
        ratio[0] = 1.0  # FCT intentionally leaves DC at full support
        kbad = int(np.argmin(ratio))
        if ratio[kbad] < worst[0]:
            worst = (float(ratio[kbad]),
                     (trial, kbad, int(tau[kbad]),
                      int(np.argmax(S[kbad]) + 1), x.copy()))

    ratio, (trial, kbad, tfct, ttrue, x) = worst
    Cipd, tipd, _, _ = ipd_transform(x)
    assert int(tipd[kbad]) == ttrue
    ref = np.sum(x[:ttrue] * np.exp(-2j * np.pi * kbad * np.arange(ttrue) / N))
    assert abs(Cipd[kbad] - ref) < 1e-9
    print(f"worst deterministic case: trial={trial} bin={kbad}")
    print(f"  heuristic FCT tau={tfct}, exact IPD tau={ttrue}")
    print(f"  FCT score / global score = {ratio:.6f}")


if __name__ == "__main__":
    run()
