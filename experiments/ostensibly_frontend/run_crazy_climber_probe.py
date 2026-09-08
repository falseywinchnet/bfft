#!/usr/bin/env python3
"""Build a continuous ridge surface and stochastic occupation field."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .crazy_climber_geometry import (
    CrazyClimberConfig,
    HessianRidgeConfig,
    crazy_climber_occupation,
    multiscale_hessian_ridge_surface,
)
from .depthmap_geometry import DepthmapGeometryConfig, multiscale_morphological_residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step3", type=Path)
    parser.add_argument("--frames", type=int, default=94)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("frames must be positive")

    with np.load(args.step3) as saved:
        trace = np.asarray(saved["trace_field"][:, : args.frames], dtype=np.float64)
    morphology_config = DepthmapGeometryConfig()
    ridge_config = HessianRidgeConfig()
    climber_config = CrazyClimberConfig()
    residual, residual_scales = multiscale_morphological_residual(
        trace, morphology_config
    )
    ridge = multiscale_hessian_ridge_surface(residual, ridge_config)
    occupation = crazy_climber_occupation(ridge.saliency, climber_config)

    args.out.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out / "crazy_climber_probe.npz"
    np.savez_compressed(
        arrays_path,
        trace=trace,
        morphological_residual=residual,
        morphological_scale_stack=residual_scales,
        ridge_saliency=ridge.saliency,
        ridge_tangent=ridge.tangent,
        ridge_scale=ridge.scale,
        occupation_unweighted=occupation.unweighted,
        occupation_weighted=occupation.weighted,
    )
    metadata = {
        "source": str(args.step3),
        "shape": list(trace.shape),
        "morphology": asdict(morphology_config),
        "ridge": asdict(ridge_config),
        "climbers": asdict(climber_config),
        "occupation_visits": occupation.visits,
        "accepted_vertical_fraction": occupation.accepted_vertical_fraction,
    }
    metadata_path = args.out / "crazy_climber_probe.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"arrays": str(arrays_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
