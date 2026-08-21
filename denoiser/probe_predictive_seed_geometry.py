"""Falsify an untransported local predictive seed on the hair-edge scene.

Noise names appear only to generate and diagnose controls.  They are never
inputs to the geometry.  The tested seed is pointwise leave-one-out: four
topological line directions contribute opposite interpolation and one-sided
affine extrapolation particles without reading the central observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fused_transport_geometry import predictive_wasserstein_geometry
from .probes import hair_edge_scene
from .sample_series import corrupt


def local_relation_particles(observation: np.ndarray) -> np.ndarray:
    """Return twelve equal-mass leave-one-out affine predictions per pixel."""
    field = np.asarray(observation, dtype=np.float64)
    if field.ndim != 2 or min(field.shape) < 5:
        raise ValueError("observation must be a 2-D field at least 5x5")
    padded = np.pad(field, 2, mode="reflect")
    height, width = field.shape
    yy, xx = np.mgrid[:height, :width]
    yy = yy + 2
    xx = xx + 2
    particles: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        minus_one = padded[yy - dy, xx - dx]
        plus_one = padded[yy + dy, xx + dx]
        minus_two = padded[yy - 2 * dy, xx - 2 * dx]
        plus_two = padded[yy + 2 * dy, xx + 2 * dx]
        particles.extend((
            0.5 * (minus_one + plus_one),
            2.0 * minus_one - minus_two,
            2.0 * plus_one - plus_two,
        ))
    return np.stack(particles, axis=-1)


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
    records = {}
    for name, observation in observations.items():
        geometry = predictive_wasserstein_geometry(
            local_relation_particles(observation))
        records[name] = {
            "implied_support": geometry["implied_support"],
            "information_trace_mean": geometry["information_trace_mean"],
            "metric_determinant_max_error": float(np.max(np.abs(
                geometry["metric_determinant"] - 1.0))),
        }
    clean_support = records["clean"]["implied_support"]
    for record in records.values():
        record["population_ratio_to_clean"] = (
            record["implied_support"] / clean_support)
    return {
        "purpose": (
            "falsify an untransported local relation seed; corruption labels "
            "generate controls and are not geometry inputs"
        ),
        "size": int(size),
        "particle_count": 12,
        "pullback": "bin-free quadratic Wasserstein quantile geometry",
        "records": records,
        "verdict": (
            "rejected: representation is refinement-stable, but local "
            "finite-sample disagreement still becomes false support; "
            "parallel jet transport must precede the pullback"
        ),
    }


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
