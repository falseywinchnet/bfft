"""Matched canon-versus-rebuilt FMMT support-donation battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dcnt import target_excluded_conv_family
from .fmmt_certified import denoise_fmmt
from .fmmt_rebuilt import (
    coherent_support_donation,
    transport_supported_donation,
)
from .run_2d_denoiser_battery import CONDITIONS, metrics, sources
from .sample_series import corrupt


def _means(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    selected = [row[method] for row in rows]
    keys = ("mse", "ssim", "edge_retention", "variance_ratio")
    return {key: float(np.mean([value[key] for value in selected])) for key in keys}


METHODS = (
    "canon_fmmt",
    "raw_support_donated_fmmt",
    "transport_support_1",
    "transport_support_2",
    "transport_support_3",
)


def _method_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {method: _means(rows, method) for method in METHODS}


def run(size: int = 64, seeds: int = 1) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cases = (("clean", "none", 0.0, 0.0),) + CONDITIONS
    for source_name, truth in sources(size).items():
        for condition_name, law, amount, density in cases:
            condition_seeds = (0,) if condition_name == "clean" else range(seeds)
            for seed in condition_seeds:
                observation = corrupt(
                    truth, law, amount=amount, density=density, seed=seed)
                canon, _diagnostic = denoise_fmmt(observation)
                family = target_excluded_conv_family(observation)
                canon_family = target_excluded_conv_family(canon)
                raw_donated, donation = coherent_support_donation(
                    observation, canon, family, canon_family)
                rebuilt, observer = transport_supported_donation(
                    observation, canon, raw_donated)
                rebuilt_2, observer_2 = transport_supported_donation(
                    observation, canon, rebuilt)
                rebuilt_3, observer_3 = transport_supported_donation(
                    observation, canon, rebuilt_2)
                rows.append({
                    "source": source_name,
                    "condition": condition_name,
                    "seed": int(seed),
                    "observation": metrics(observation, truth),
                    "canon_fmmt": metrics(canon, truth),
                    "raw_support_donated_fmmt": metrics(raw_donated, truth),
                    "transport_support_1": metrics(rebuilt, truth),
                    "transport_support_2": metrics(rebuilt_2, truth),
                    "transport_support_3": metrics(rebuilt_3, truth),
                    "donation": donation,
                    "donation_observer": observer,
                    "observer_action_trace": [
                        observer["transport_supported_action"],
                        observer_2["transport_supported_action"],
                        observer_3["transport_supported_action"],
                    ],
                })
    noisy = [row for row in rows if row["condition"] != "clean"]
    clean = [row for row in rows if row["condition"] == "clean"]
    per_condition = {}
    for condition_name, *_rest in cases:
        selected = [row for row in rows if row["condition"] == condition_name]
        per_condition[condition_name] = _method_means(selected)
    return {
        "size": int(size),
        "seeds": int(seeds),
        "cases": len(rows),
        "summary": {
            "all": {
                **_method_means(rows),
            },
            "noisy": _method_means(noisy),
            "clean": _method_means(clean),
            "per_condition": per_condition,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
