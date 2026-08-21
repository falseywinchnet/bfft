"""Measure how V3's local support volume reacts before predictive transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bfft.effects import srgb_to_lab
from port_needed.density_population import curvature_limited_geometry
from port_needed.frozen_meyer_geometry import build_frozen_geometry

from .probes import hair_edge_scene
from .sample_series import corrupt


def summarize(field: np.ndarray, texture_weight: float) -> dict[str, float]:
    rgb = np.repeat(np.asarray(field, dtype=np.float64)[..., None], 3, axis=2)
    geometry = build_frozen_geometry(
        rgb,
        target_lab=srgb_to_lab(rgb),
        tgfd_sweeps=1,
        meyer_operator="jump_measure",
        flow_sweeps=1,
        texture_support_weight=texture_weight,
        glass_support_weight=0.0,
        null_evidence_strength=1.0 if texture_weight == 0.0 else 0.5,
        threads=4,
    )
    curved = curvature_limited_geometry(geometry)
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    trace = qxx + qyy
    coherence = np.hypot(qxx - qyy, 2.0 * qxy) / np.maximum(trace, 1e-30)
    return {
        "straight_implied_cells": float(geometry["implied_cells"]),
        "curvature_implied_cells": float(curved["implied_cells"]),
        "median_precision_trace": float(np.median(trace)),
        "median_coherence": float(np.median(coherence)),
        "p90_coherence": float(np.quantile(coherence, 0.90)),
    }


def run(size: int) -> dict:
    clean, _ = hair_edge_scene(size=size, seed=719)
    observations = {
        "clean": clean,
        "mixed replacement + uniform": corrupt(
            clean, "mixed replacement + uniform",
            amount=0.15, density=0.25, seed=7),
        "uniform additive": corrupt(
            clean, "uniform additive", amount=0.15, density=0.25, seed=7),
    }
    modes = {
        "half-cartoon structural measure": 0.0,
        "full texture-bearing measure": 0.65,
    }
    result = {
        "purpose": (
            "diagnose local V3 support before the fused predictive transport; "
            "these measurements are not denoiser parameters"
        ),
        "size": size,
        "modes": {},
    }
    for mode, texture_weight in modes.items():
        records = {
            name: summarize(field, texture_weight)
            for name, field in observations.items()
        }
        clean_cells = records["clean"]["straight_implied_cells"]
        for record in records.values():
            record["population_ratio_to_clean"] = (
                record["straight_implied_cells"] / clean_cells)
        result["modes"][mode] = records
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
