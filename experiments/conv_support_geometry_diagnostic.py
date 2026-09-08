"""Orientation-resolved diagnosis of CONV's two-dimensional support geometry.

Every truth raster is an analytic cell-average discretization of a continuum
indicator.  The diagnostic separates the unconstrained transported moment
proposal from the admitted face-current moment.  It therefore identifies
whether an orientation-dependent loss enters before or during admission.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.conv_conservative_multiresolution import (
    BlockMoments,
    admit_moments_2d,
    block_moments_2d,
    conv_moment_proposal_2d,
    conservative_restrict_2d,
    synthesize_block_moments_2d,
)


Array = np.ndarray


def _clip_halfplane(
    polygon: list[Array], normal: Array, threshold: float
) -> list[Array]:
    result: list[Array] = []
    for start, stop in zip(polygon, polygon[1:] + polygon[:1]):
        first = float(normal @ start - threshold)
        second = float(normal @ stop - threshold)
        inside_first = first >= 0.0
        inside_second = second >= 0.0
        if inside_first:
            result.append(start)
        if inside_first != inside_second:
            result.append(start + (stop - start) * (first / (first - second)))
    return result


def _polygon_area(polygon: list[Array]) -> float:
    if len(polygon) < 3:
        return 0.0
    point = np.stack(polygon)
    return 0.5 * abs(float(
        point[:, 0] @ np.roll(point[:, 1], -1)
        - point[:, 1] @ np.roll(point[:, 0], -1)
    ))


def half_plane_cell_averages(side: int, angle: float, phase: float) -> Array:
    """Exact square-cell coverage of an oriented continuum half-plane."""

    theta = math.radians(angle)
    normal = np.array((-math.sin(theta), math.cos(theta)), dtype=np.float64)
    center = np.array((side / 2.0, side / 2.0), dtype=np.float64)
    threshold = float(normal @ center + phase)
    output = np.empty((side, side), dtype=np.float64)
    for row in range(side):
        for column in range(side):
            polygon = _clip_halfplane([
                np.array((column, row), dtype=np.float64),
                np.array((column + 1.0, row), dtype=np.float64),
                np.array((column + 1.0, row + 1.0), dtype=np.float64),
                np.array((column, row + 1.0), dtype=np.float64),
            ], normal, threshold)
            output[row, column] = _polygon_area(polygon)
    return output


def rotated_square_cell_averages(
    side: int, angle: float, phase_x: float, phase_y: float
) -> Array:
    """Exact square-cell coverage of a rotated continuum square."""

    theta = math.radians(angle)
    first = np.array((math.cos(theta), math.sin(theta)), dtype=np.float64)
    second = np.array((-math.sin(theta), math.cos(theta)), dtype=np.float64)
    center = np.array(
        (side / 2.0 + phase_x, side / 2.0 + phase_y), dtype=np.float64
    )
    half_extent = side * 0.21875
    constraints = (
        (first, float(first @ center - half_extent)),
        (-first, float((-first) @ center - half_extent)),
        (second, float(second @ center - half_extent)),
        (-second, float((-second) @ center - half_extent)),
    )
    output = np.empty((side, side), dtype=np.float64)
    for row in range(side):
        for column in range(side):
            polygon = [
                np.array((column, row), dtype=np.float64),
                np.array((column + 1.0, row), dtype=np.float64),
                np.array((column + 1.0, row + 1.0), dtype=np.float64),
                np.array((column, row + 1.0), dtype=np.float64),
            ]
            for normal, threshold in constraints:
                polygon = _clip_halfplane(polygon, normal, threshold)
                if not polygon:
                    break
            output[row, column] = _polygon_area(polygon)
    return output


def _crop(value: Array, border: int = 4) -> Array:
    return np.asarray(value)[border:-border, border:-border]


def _isotropic_tv(value: Array) -> float:
    field = np.asarray(value, dtype=np.float64)
    dx = field[:-1, 1:] - field[:-1, :-1]
    dy = field[1:, :-1] - field[:-1, :-1]
    return float(np.sum(np.hypot(dx, dy)))


def _anisotropic_tv(value: Array) -> float:
    field = np.asarray(value, dtype=np.float64)
    return float(np.sum(np.abs(np.diff(field, axis=0)))
                 + np.sum(np.abs(np.diff(field, axis=1))))


def _moment_error(estimate: BlockMoments, truth: BlockMoments, mask: Array) -> dict[str, float]:
    residual_x = estimate.horizontal[mask] - truth.horizontal[mask]
    residual_y = estimate.vertical[mask] - truth.vertical[mask]
    residual_xy = estimate.mixed[mask] - truth.mixed[mask]
    vector_error = residual_x * residual_x + residual_y * residual_y
    exact_norm = np.hypot(truth.horizontal[mask], truth.vertical[mask])
    estimate_norm = np.hypot(estimate.horizontal[mask], estimate.vertical[mask])
    valid = (exact_norm > 1.0e-14) & (estimate_norm > 1.0e-14)
    if np.any(valid):
        cosine = (
            estimate.horizontal[mask][valid] * truth.horizontal[mask][valid]
            + estimate.vertical[mask][valid] * truth.vertical[mask][valid]
        ) / (exact_norm[valid] * estimate_norm[valid])
        mean_cosine = float(np.mean(cosine))
    else:
        mean_cosine = 1.0
    return {
        "vector_moment_rmse": float(np.sqrt(np.mean(vector_error))),
        "mixed_moment_rmse": float(np.sqrt(np.mean(residual_xy * residual_xy))),
        "mean_direction_cosine": mean_cosine,
    }


def diagnose_case(truth: Array) -> dict[str, object]:
    coarse = conservative_restrict_2d(truth)
    exact = block_moments_2d(truth)
    proposal = conv_moment_proposal_2d(coarse)
    admitted = admit_moments_2d(coarse, proposal)
    raw = synthesize_block_moments_2d(proposal)
    constrained = synthesize_block_moments_2d(admitted)
    activity = np.hypot(exact.horizontal, exact.vertical) + np.abs(exact.mixed)
    mask = activity > 1.0e-14
    if not np.any(mask):
        mask = np.ones_like(activity, dtype=bool)

    proposal_norm = np.sqrt(
        proposal.horizontal * proposal.horizontal
        + proposal.vertical * proposal.vertical
        + proposal.mixed * proposal.mixed
    )
    admitted_norm = np.sqrt(
        admitted.horizontal * admitted.horizontal
        + admitted.vertical * admitted.vertical
        + admitted.mixed * admitted.mixed
    )
    active_proposal = proposal_norm[mask]
    retained = np.divide(
        admitted_norm[mask], active_proposal,
        out=np.ones_like(active_proposal), where=active_proposal > 1.0e-14,
    )

    def output_metrics(value: Array) -> dict[str, float]:
        estimate = _crop(value)
        reference = _crop(truth)
        residual = estimate - reference
        return {
            "mse": float(np.mean(residual * residual)),
            "linf": float(np.max(np.abs(residual), initial=0.0)),
            "threshold_disagreement": float(np.mean(
                (estimate >= 0.5) != (reference >= 0.5)
            )),
            "minimum": float(np.min(value)),
            "maximum": float(np.max(value)),
            "anisotropic_tv_ratio": _anisotropic_tv(estimate)
                / max(_anisotropic_tv(reference), 1.0e-300),
            "isotropic_tv_ratio": _isotropic_tv(estimate)
                / max(_isotropic_tv(reference), 1.0e-300),
            "mean_error": float(np.mean(value) - np.mean(truth)),
        }

    return {
        "proposal_moments": _moment_error(proposal, exact, mask),
        "admitted_moments": _moment_error(admitted, exact, mask),
        "admission": {
            "mean_retained_moment_norm": float(np.mean(retained)),
            "minimum_retained_moment_norm": float(np.min(retained, initial=1.0)),
            "fraction_contracted": float(np.mean(retained < 1.0 - 1.0e-12)),
        },
        "raw_synthesis": output_metrics(raw),
        "admitted_synthesis": output_metrics(constrained),
    }


def run(side: int, angle_step: float) -> dict[str, object]:
    half_planes: list[dict[str, object]] = []
    squares: list[dict[str, object]] = []
    angles = np.arange(0.0, 90.0 + angle_step / 2.0, angle_step)
    for angle in angles:
        for phase in (-0.375, -0.125, 0.125, 0.375):
            half_planes.append({
                "angle_degrees": float(angle),
                "phase": phase,
                "diagnosis": diagnose_case(
                    half_plane_cell_averages(side, float(angle), phase)
                ),
            })
        folded = float(min(angle, 90.0 - angle))
        if folded <= 45.0 and angle <= 45.0:
            for phase_x, phase_y in ((-0.25, 0.125), (0.25, -0.125)):
                squares.append({
                    "angle_degrees": float(angle),
                    "phase": [phase_x, phase_y],
                    "diagnosis": diagnose_case(rotated_square_cell_averages(
                        side, float(angle), phase_x, phase_y
                    )),
                })

    def angle_summary(records: list[dict[str, object]]) -> list[dict[str, float]]:
        result = []
        for angle in sorted({float(row["angle_degrees"]) for row in records}):
            group = [row["diagnosis"] for row in records
                     if float(row["angle_degrees"]) == angle]
            result.append({
                "angle_degrees": angle,
                "proposal_vector_moment_rmse": float(np.mean([
                    row["proposal_moments"]["vector_moment_rmse"] for row in group
                ])),
                "admitted_vector_moment_rmse": float(np.mean([
                    row["admitted_moments"]["vector_moment_rmse"] for row in group
                ])),
                "mean_retained_moment_norm": float(np.mean([
                    row["admission"]["mean_retained_moment_norm"] for row in group
                ])),
                "raw_mse": float(np.mean([
                    row["raw_synthesis"]["mse"] for row in group
                ])),
                "admitted_mse": float(np.mean([
                    row["admitted_synthesis"]["mse"] for row in group
                ])),
                "raw_range_excess": float(max(
                    max(0.0, row["raw_synthesis"]["maximum"] - 1.0,
                        -row["raw_synthesis"]["minimum"])
                    for row in group
                )),
                "admitted_range_excess": float(max(
                    max(0.0, row["admitted_synthesis"]["maximum"] - 1.0,
                        -row["admitted_synthesis"]["minimum"])
                    for row in group
                )),
            })
        return result

    return {
        "design": {
            "fine_side": side,
            "coarse_side": side // 2,
            "truth": "analytic continuum-indicator cell averages",
            "comparison": "raw transported moments versus face-current admission",
            "angle_step_degrees": angle_step,
        },
        "half_plane_angle_summary": angle_summary(half_planes),
        "square_angle_summary": angle_summary(squares),
        "half_plane_records": half_planes,
        "square_records": squares,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("/tmp/conv_support_geometry_diagnostic.json"),
    )
    args = parser.parse_args()
    if args.side < 18 or args.side % 2:
        raise ValueError("side must be even and at least 18")
    result = run(args.side, args.angle_step)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "half_plane_angle_summary": result["half_plane_angle_summary"],
        "square_angle_summary": result["square_angle_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
