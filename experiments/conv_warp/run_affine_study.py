"""First direct-warp census against procedural affine ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import affine_about_center
from .operators import direct_warp, positive_warped_basin_average, source_validity_mask
from .synthetic import canonical_fourier_fields


def _metrics(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = np.asarray(estimate, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    selected = residual[mask]
    return {
        "mse": float(np.mean(selected * selected)),
        "linf": float(np.max(np.abs(selected))),
        "range_excursion": max(
            float(np.max(np.min(truth) - estimate, initial=0.0)),
            float(np.max(estimate - np.max(truth), initial=0.0)),
        ),
    }


def run(source_side: int = 25, target_side: int = 41) -> dict[str, object]:
    transforms = {
        "rotate_23": affine_about_center(angle_degrees=23.0, scale_x=0.72, scale_y=0.72),
        "skew": affine_about_center(scale_x=0.72, scale_y=0.72, shear_x=0.38),
        "anisotropic": affine_about_center(angle_degrees=-17.0, scale_x=0.55, scale_y=0.86),
    }
    methods = ("conv", "lanczos3", "bilinear")
    records: list[dict[str, object]] = []
    for field_name, field in canonical_fourier_fields().items():
        source = field.sample_nodes((source_side, source_side)).astype(np.float32)
        for transform_name, transform in transforms.items():
            point_truth = field.warped_nodes(transform, (target_side, target_side))
            basin_truth = field.affine_basin_average(transform, (target_side, target_side))
            mask = source_validity_mask(
                transform, (target_side, target_side), source.shape[:2]
            )
            record: dict[str, object] = {
                "field": field_name,
                "transform": transform_name,
                "valid_points": int(np.sum(mask)),
                "point": {},
                "basin": {},
            }
            for method in methods:
                point = direct_warp(source, transform, (target_side, target_side), method=method)
                # Basin evaluation is expensive for CONV's reference point
                # oracle, so the first census uses order two.  The positive
                # weight theorem is independent of this accuracy choice.
                basin = positive_warped_basin_average(
                    source, transform, (target_side, target_side),
                    method=method, quadrature_order=2,
                )
                record["point"][method] = _metrics(point, point_truth, mask)
                record["basin"][method] = _metrics(basin, basin_truth, mask)
            records.append(record)
    return {
        "source_shape": [source_side, source_side],
        "target_shape": [target_side, target_side],
        "coordinate_convention": "one direct target-to-source normalized map",
        "point_truth": "closed-form procedural field evaluation",
        "basin_truth": "closed-form affine Fourier basin integral",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-side", type=int, default=25)
    parser.add_argument("--target-side", type=int, default=41)
    parser.add_argument("--out", type=Path, default=Path("/tmp/conv_warp_affine.json"))
    args = parser.parse_args()
    result = run(args.source_side, args.target_side)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
