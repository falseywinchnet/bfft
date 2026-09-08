"""Probe-bank robustness audit for transported zonotopic CONV--ITD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.conv_itd_multicomponent import make_signal
from experiments.conv_itd_vs_canonical import representation_metrics
from experiments.conv_itd_zonotopic_enrichment import enriched_decompose, plain_decompose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20260830,20260831,20260832,20260833")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--out", type=Path, default=Path("/tmp/conv_itd_zonotopic_seed_audit.json"))
    args = parser.parse_args()

    t, signal, truth = make_signal(args.n)
    plain_rotation, plain_baseline, _ = plain_decompose(signal, args.levels)
    plain, _ = representation_metrics(plain_rotation, plain_baseline, truth)
    records = []
    for seed in (int(value) for value in args.seeds.split(",")):
        start = time.perf_counter()
        rotation, baseline, levels = enriched_decompose(
            signal,
            alpha=args.alpha,
            probe_count=args.probes,
            max_levels=args.levels,
            transported=True,
            normalize_rotation=False,
            seed=seed,
        )
        metric, _ = representation_metrics(rotation, baseline, truth)
        records.append({
            "seed": seed,
            "mean_best_abs_correlation": metric["mean_best_abs_correlation"],
            "mean_correlation_concentration": metric["mean_correlation_concentration"],
            "trend_mse": metric["trend_mse"],
            "burst_energy_localization": metric["burst_energy_localization"],
            "rotation_count": len(rotation),
            "closure_linf": float(np.max(np.abs(signal-(baseline+np.sum(rotation, axis=0))))),
            "elapsed_seconds": time.perf_counter()-start,
            "noise_rotation_rms": [row["noise_rotation_rms"] for row in levels],
        })

    result = {
        "plain": {
            "mean_best_abs_correlation": plain["mean_best_abs_correlation"],
            "mean_correlation_concentration": plain["mean_correlation_concentration"],
            "trend_mse": plain["trend_mse"],
            "burst_energy_localization": plain["burst_energy_localization"],
            "rotation_count": len(plain_rotation),
        },
        "transported": records,
        "summary": {},
    }
    for key in (
        "mean_best_abs_correlation",
        "mean_correlation_concentration",
        "trend_mse",
        "burst_energy_localization",
        "rotation_count",
    ):
        value = np.asarray([row[key] for row in records], dtype=np.float64)
        result["summary"][key] = {
            "minimum": float(np.min(value)),
            "mean": float(np.mean(value)),
            "maximum": float(np.max(value)),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
