"""Audit the V3 eikonal-owner mechanism as an image-transport scaffold.

This is not an interpolation candidate.  It isolates the geometric mechanism
that a later CONV transport must use: tensor-implied population, anisotropic
first arrival, front collision, and curvature-limited support.  Synthetic
truth is used only to score whether the resulting owners follow known long
features; it is never supplied to the construction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfft.effects import srgb_to_lab
from port_needed.continuous_eikonal_transport import (
    continuous_first_partition_prepared,
    prepare_continuous_metric,
)
from port_needed.density_population import (
    curvature_limited_geometry,
    emit_density_population,
)
from port_needed.frozen_meyer_geometry import build_frozen_geometry
from port_needed.wide_stencil_transport import _metric_fields


Array = np.ndarray


def _ridge(side: int, angle_degrees: float, width: float = 1.15) -> Array:
    y, x = np.indices((side, side), dtype=np.float64)
    center = 0.5 * (side - 1)
    angle = math.radians(angle_degrees)
    normal = np.array((-math.sin(angle), math.cos(angle)))
    distance = normal[0] * (x - center) + normal[1] * (y - center)
    return 0.12 + 0.76 * np.exp(-0.5 * (distance / width) ** 2)


def _ring(side: int, radius: float, width: float = 1.15) -> Array:
    y, x = np.indices((side, side), dtype=np.float64)
    center = 0.5 * (side - 1)
    distance = np.hypot(x - center, y - center) - radius
    return 0.12 + 0.76 * np.exp(-0.5 * (distance / width) ** 2)


def _crossing(side: int, width: float = 1.15) -> Array:
    first = _ridge(side, 30.0, width)
    second = _ridge(side, 150.0, width)
    return np.maximum(first, second)


def _v3_front(value: Array) -> tuple[dict, Array, Array, dict]:
    rgb = np.repeat(np.asarray(value)[..., None], 3, axis=2)
    geometry = build_frozen_geometry(
        rgb,
        target_lab=srgb_to_lab(rgb),
        tgfd_sweeps=1,
        meyer_operator="jump_measure",
        flow_sweeps=1,
        texture_support_weight=0.65,
        glass_support_weight=0.0,
        null_evidence_strength=0.5,
        threads=4,
    )
    geometry = curvature_limited_geometry(geometry)
    centers, population = emit_density_population(
        geometry, safety_cells=value.size
    )
    metric = prepare_continuous_metric(*_metric_fields(
        geometry, 1.5, 24.0
    ))
    forest = continuous_first_partition_prepared(
        centers, metric, compact=True, source_gradients=False
    )
    return geometry, centers, forest["labels"], population


def _owner_boundaries(labels: Array) -> Array:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def _straight_statistics(
    labels: Array,
    angle_degrees: float,
    band_half_width: float = 3.0,
) -> dict[str, float | int]:
    side = labels.shape[0]
    y, x = np.indices(labels.shape, dtype=np.float64)
    center = 0.5 * (side - 1)
    angle = math.radians(angle_degrees)
    tangent = np.array((math.cos(angle), math.sin(angle)))
    normal = np.array((-math.sin(angle), math.cos(angle)))
    distance = normal[0] * (x - center) + normal[1] * (y - center)
    ridge_owners = np.unique(labels[np.abs(distance) <= band_half_width])

    aspects: list[float] = []
    alignments: list[float] = []
    sizes: list[int] = []
    for owner in ridge_owners:
        yy, xx = np.nonzero(labels == owner)
        if len(xx) < 5:
            continue
        centered = np.column_stack((xx - xx.mean(), yy - yy.mean()))
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        aspects.append(float(math.sqrt(
            (eigenvalues[1] + 1e-12) / (eigenvalues[0] + 1e-12)
        )))
        alignments.append(float(abs(eigenvectors[:, 1] @ tangent)))
        sizes.append(int(len(xx)))
    return {
        "ridge_owner_count": int(len(ridge_owners)),
        "scored_owner_count": int(len(aspects)),
        "median_aspect": float(np.median(aspects)),
        "median_tangent_alignment": float(np.median(alignments)),
        "minimum_tangent_alignment": float(np.min(alignments)),
        "median_pixels": float(np.median(sizes)),
        "maximum_pixels": int(np.max(sizes)),
    }


def _geometry_fields(geometry: dict) -> tuple[Array, Array, Array]:
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    trace = qxx + qyy
    coherence = np.hypot(qxx - qyy, 2.0 * qxy) / np.maximum(trace, 1e-30)
    density = (
        np.asarray(geometry["measure"], dtype=np.float64)
        * float(geometry["implied_cells"])
    )
    curvature_factor = np.asarray(
        geometry["curvature_population_factor"], dtype=np.float64
    )
    return coherence, density, curvature_factor


def run(output: Path, side: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, Array, float | None]] = [
        ("straight_15", _ridge(side, 15.0), 15.0),
        ("straight_22_5", _ridge(side, 22.5), 22.5),
        ("straight_45", _ridge(side, 45.0), 45.0),
        ("straight_75", _ridge(side, 75.0), 75.0),
        ("curved_ring", _ring(side, 0.28 * side), None),
        ("crossing", _crossing(side), None),
    ]
    records: dict[str, dict] = {}
    figure, axes = plt.subplots(
        len(cases), 5, figsize=(13.5, 2.5 * len(cases)), squeeze=False
    )
    for row, (name, value, angle) in enumerate(cases):
        geometry, centers, labels, population = _v3_front(value)
        coherence, density, curvature_factor = _geometry_fields(geometry)
        record = {
            "site_count": int(len(centers)),
            "implied_cells": float(geometry["implied_cells"]),
            "maximum_owner_pixels": int(np.max(np.bincount(labels.ravel()))),
            "median_coherence": float(np.median(coherence)),
            "maximum_density": float(np.max(density)),
            "maximum_curvature_population_factor": float(
                np.max(curvature_factor)
            ),
            "population_quantization_error": float(
                population["quantization_error"]
            ),
        }
        if angle is not None:
            record.update(_straight_statistics(labels, angle))
        records[name] = record

        panels = (
            (value, "source", "gray"),
            (coherence, "tensor coherence", "magma"),
            (density, "population density", "viridis"),
            (curvature_factor, "curvature factor", "plasma"),
            (labels, "eikonal owners", "nipy_spectral"),
        )
        for column, (panel, title, cmap) in enumerate(panels):
            axes[row, column].imshow(panel, cmap=cmap, interpolation="nearest")
            if title == "eikonal owners":
                boundary = _owner_boundaries(labels)
                axes[row, column].contour(
                    boundary.astype(float), levels=(0.5,), colors="white",
                    linewidths=0.35,
                )
                axes[row, column].scatter(
                    centers[:, 0] * side - 0.5,
                    centers[:, 1] * side - 0.5,
                    s=5,
                    c="black",
                )
            axes[row, column].set_title(f"{name}: {title}", fontsize=8)
            axes[row, column].set_axis_off()
    figure.suptitle(
        "V3 tensor population and eikonal first-arrival ownership",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output / "owner_audit.png", dpi=180)
    plt.close(figure)

    result = {
        "purpose": (
            "geometry audit only; no interpolation result and no optimizer"
        ),
        "side": int(side),
        "construction": {
            "geometry": "existing frozen V3 Meyer/BFFT tensor",
            "population": "sqrt(det(Q))/pi with curvature limiter",
            "partition": "continuous anisotropic first arrival",
            "truth_usage": "scoring only",
        },
        "records": records,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/support_geometry/eikonal_owner_audit"),
    )
    parser.add_argument("--side", type=int, default=65)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.side), indent=2))


if __name__ == "__main__":
    main()
