#!/usr/bin/env python3
"""Analytic-truth census for conservative four-child moment refinement.

Source and target entries are exact uniform-cell averages of declared
continuous fields.  The baseline transports the source cell averages as a
piecewise-constant density.  The two candidate arms apply one or two native
four-child moment-atlas levels before the same positive footprint integral.
No competing interpolator supplies a reference value.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from experiments.conv_high_frequency_transport_audit import (
    _cell_validity_mask,
    _exact_affine_cell_overlap,
    _uniform_cell_average,
    _warped_uniform_cell_average,
)
from experiments.conv_warp.geometry import ProjectiveMap, affine_about_center
from standalone_conv_resize_demo.conservative import (
    restrict_2x2,
    zero_detail_synthesis,
)


Array = np.ndarray


@lru_cache(maxsize=None)
def _overlap_matrix(source_count: int, target_count: int) -> Array:
    """Exact target-cell means of a piecewise-constant source line."""

    source_count, target_count = int(source_count), int(target_count)
    matrix = np.zeros((target_count, source_count), dtype=np.float64)
    for target in range(target_count):
        target_left = target / target_count
        target_right = (target + 1) / target_count
        first = max(0, int(math.floor(target_left * source_count)))
        last = min(
            source_count - 1,
            int(math.ceil(target_right * source_count) - 1),
        )
        for source in range(first, last + 1):
            left = max(target_left, source / source_count)
            right = min(target_right, (source + 1) / source_count)
            matrix[target, source] = target_count * max(0.0, right - left)
    return matrix


def _identity_cell_overlap(values: Array, target_shape: tuple[int, int]) -> Array:
    source = np.asarray(values, dtype=np.float64)
    if tuple(map(int, target_shape)) == source.shape[:2]:
        return source.copy()
    wy = _overlap_matrix(source.shape[0], int(target_shape[0]))
    wx = _overlap_matrix(source.shape[1], int(target_shape[1]))
    if source.ndim == 2:
        return wy @ source @ wx.T
    return np.einsum("ai,ijc,bj->abc", wy, source, wx, optimize=True)


def _polygon_halfplane_fraction(
    shape: tuple[int, int], normal: Array, threshold: float
) -> Array:
    """Exact cell coverage of ``normal dot x >= threshold`` on [0,1]^2."""

    height, width = map(int, shape)
    normal = np.asarray(normal, dtype=np.float64)
    output = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        y0, y1 = row / height, (row + 1) / height
        for column in range(width):
            x0, x1 = column / width, (column + 1) / width
            polygon = [
                np.array((x0, y0)), np.array((x1, y0)),
                np.array((x1, y1)), np.array((x0, y1)),
            ]
            clipped: list[Array] = []
            for start, stop in zip(polygon, polygon[1:] + polygon[:1]):
                first = float(normal @ start - threshold)
                second = float(normal @ stop - threshold)
                first_inside, second_inside = first >= 0.0, second >= 0.0
                if first_inside:
                    clipped.append(start)
                if first_inside != second_inside:
                    clipped.append(
                        start + (stop - start) * (first / (first - second))
                    )
            if len(clipped) < 3:
                output[row, column] = 0.0
                continue
            points = np.stack(clipped)
            area = 0.5 * abs(float(
                points[:, 0] @ np.roll(points[:, 1], -1)
                - points[:, 1] @ np.roll(points[:, 0], -1)
            ))
            output[row, column] = area * height * width
    return output


def _mapped_halfplane_average(
    shape: tuple[int, int], normal: Array, threshold: float,
    transform: ProjectiveMap,
) -> Array:
    linear, offset = transform.affine_parts()
    mapped_normal = linear.T @ np.asarray(normal, dtype=np.float64)
    mapped_threshold = float(threshold - np.asarray(normal) @ offset)
    return _polygon_halfplane_fraction(shape, mapped_normal, mapped_threshold)


def _range_excursion(value: Array, lower: float, upper: float) -> float:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    return max(lower - float(np.min(finite)), float(np.max(finite)) - upper, 0.0)


def _field_metrics(result: Array, truth: Array, source: Array, mask: Array) -> dict[str, float]:
    estimate = np.asarray(result, dtype=np.float64)[mask]
    reference = np.asarray(truth, dtype=np.float64)[mask]
    residual = estimate - reference
    centered = reference - float(np.mean(reference))
    denominator = float(centered @ centered)
    gain = (
        float((estimate - float(np.mean(estimate))) @ centered / denominator)
        if denominator else 1.0
    )
    return {
        "mse": float(np.mean(residual * residual)),
        "maximum_absolute_error": float(np.max(np.abs(residual), initial=0.0)),
        "signed_target_gain": gain,
        "source_range_excursion": _range_excursion(
            estimate, float(np.min(source)), float(np.max(source))
        ),
        "exact_target_range_excursion": _range_excursion(
            estimate, float(np.min(reference)), float(np.max(reference))
        ),
    }


def _sign_changes(values: Array, tolerance: float) -> int:
    sign = np.sign(np.where(np.abs(values) > tolerance, values, 0.0))
    sign = sign[sign != 0.0]
    return int(np.count_nonzero(sign[1:] != sign[:-1])) if sign.size else 0


def _face_sign_surplus(coarse: Array, fine: Array) -> dict[str, int]:
    """Executable factorwise certificate used by the 2-D theorem."""

    tolerance = 3.0e-6 * max(
        1.0, float(np.max(np.abs(coarse), initial=0.0))
    )
    horizontal = np.diff(fine, axis=1)
    horizontal_surplus = 0
    for row in range(coarse.shape[0]):
        coarse_changes = _sign_changes(np.diff(coarse[row]), tolerance)
        horizontal_surplus = max(
            horizontal_surplus,
            _sign_changes(horizontal[2 * row], tolerance) - coarse_changes,
            _sign_changes(horizontal[2 * row + 1], tolerance) - coarse_changes,
        )
    vertical = np.diff(fine, axis=0)
    vertical_surplus = 0
    for column in range(coarse.shape[1]):
        coarse_changes = _sign_changes(np.diff(coarse[:, column]), tolerance)
        vertical_surplus = max(
            vertical_surplus,
            _sign_changes(vertical[:, 2 * column], tolerance) - coarse_changes,
            _sign_changes(vertical[:, 2 * column + 1], tolerance) - coarse_changes,
        )
    return {
        "horizontal_sign_surplus": int(horizontal_surplus),
        "vertical_sign_surplus": int(vertical_surplus),
    }


def _transport(
    values: Array, transform: ProjectiveMap, target_shape: tuple[int, int],
    *, identity: bool,
) -> Array:
    if identity:
        return _identity_cell_overlap(values, target_shape)
    return _exact_affine_cell_overlap(values, transform, target_shape)


def _evaluate_case(
    source: Array,
    truth: Array,
    transform: ProjectiveMap,
    target_shape: tuple[int, int],
    *,
    identity: bool,
) -> dict[str, object]:
    source = np.asarray(source, dtype=np.float64)
    valid = np.ones(target_shape, dtype=bool) if identity else _cell_validity_mask(
        transform, target_shape
    )
    baseline = _transport(source, transform, target_shape, identity=identity)
    valid &= np.isfinite(baseline)
    methods: dict[str, dict[str, float]] = {
        "piecewise_constant_source_cells": _field_metrics(
            baseline, truth, source, valid
        )
    }
    refined = source.astype(np.float32)
    internal: list[dict[str, float | int]] = []
    for level in (1, 2):
        parent = refined
        refined = zero_detail_synthesis(parent)
        recovered = restrict_2x2(refined)
        certificate = _face_sign_surplus(parent, refined)
        internal.append({
            "level": level,
            "scale": 2**level,
            "source_range_excursion": _range_excursion(
                refined, float(np.min(source)), float(np.max(source))
            ),
            "parent_mean_residual": float(np.max(
                np.abs(recovered.astype(np.float64) - parent.astype(np.float64)),
                initial=0.0,
            )),
            **certificate,
        })
        transported = _transport(
            refined, transform, target_shape, identity=identity
        )
        methods[f"native_moment_atlas_{2**level}x"] = _field_metrics(
            transported, truth, source, valid & np.isfinite(transported)
        )
    baseline_mse = methods["piecewise_constant_source_cells"]["mse"]
    atlas_mse = methods["native_moment_atlas_4x"]["mse"]
    return {
        "valid_target_cells": int(np.count_nonzero(valid)),
        "internal_atlas": internal,
        "methods": methods,
        "baseline_to_atlas_4x_mse_ratio": (
            baseline_mse / atlas_mse if atlas_mse else float("inf")
        ),
        "baseline_minus_atlas_4x_mse": baseline_mse - atlas_mse,
    }


def _wave_case(
    source_shape: tuple[int, int], target_shape: tuple[int, int],
    radius_fraction: float, angle_degrees: float, phase: float,
    transform: ProjectiveMap,
) -> tuple[Array, Array, dict[str, object]]:
    limiting_side = min(min(source_shape), min(target_shape))
    radius = radius_fraction * limiting_side / 2.0
    angle = math.radians(angle_degrees)
    wave = (radius * math.cos(angle), radius * math.sin(angle))
    source = _uniform_cell_average(source_shape, wave, phase)
    truth = _warped_uniform_cell_average(transform, target_shape, wave, phase)
    return source, truth, {
        "radius_fraction_of_limiting_nyquist": radius_fraction,
        "limiting_nyquist_side": limiting_side,
        "angle_degrees": angle_degrees,
        "phase_radians": phase,
        "wave": list(wave),
    }


def _crossed_wave_case(
    source_shape: tuple[int, int], target_shape: tuple[int, int],
    first_fraction: float, second_fraction: float,
    angle_degrees: float, separation_degrees: float, phase_index: int,
) -> tuple[Array, Array, dict[str, object]]:
    phases = (
        (0.0, math.pi / 2.0),
        (math.pi / 2.0, math.pi),
        (math.pi, 3.0 * math.pi / 2.0),
        (3.0 * math.pi / 2.0, 0.0),
    )
    phase_a, phase_b = phases[phase_index]

    def wave(fraction: float, angle_value: float) -> tuple[float, float]:
        radius = fraction * min(min(source_shape), min(target_shape)) / 2.0
        theta = math.radians(angle_value)
        return radius * math.cos(theta), radius * math.sin(theta)

    first_wave = wave(first_fraction, angle_degrees)
    second_wave = wave(second_fraction, angle_degrees + separation_degrees)
    source = (
        0.6 * _uniform_cell_average(source_shape, first_wave, phase_a)
        + 0.4 * _uniform_cell_average(source_shape, second_wave, phase_b)
    )
    identity = ProjectiveMap(np.eye(3))
    truth = (
        0.6 * _warped_uniform_cell_average(
            identity, target_shape, first_wave, phase_a
        )
        + 0.4 * _warped_uniform_cell_average(
            identity, target_shape, second_wave, phase_b
        )
    )
    return source, truth, {
        "first_fraction_of_limiting_nyquist": first_fraction,
        "second_fraction_of_limiting_nyquist": second_fraction,
        "angle_degrees": angle_degrees,
        "separation_degrees": separation_degrees,
        "phase_index": phase_index,
    }


def _interface_case(
    source_shape: tuple[int, int], target_shape: tuple[int, int],
    angle_degrees: float, phase_pixels: float, transform: ProjectiveMap,
) -> tuple[Array, Array, dict[str, object]]:
    theta = math.radians(angle_degrees)
    normal = np.array((-math.sin(theta), math.cos(theta)))
    threshold = float(normal @ np.array((0.5, 0.5))) + phase_pixels / min(source_shape)
    source = _polygon_halfplane_fraction(source_shape, normal, threshold)
    truth = _mapped_halfplane_average(target_shape, normal, threshold, transform)
    return source, truth, {
        "angle_degrees": angle_degrees,
        "phase_source_pixels": phase_pixels,
    }


def _strip_case(
    source_shape: tuple[int, int], target_shape: tuple[int, int],
    angle_degrees: float, width_target_pixels: float,
    phase_target_pixels: float, transform: ProjectiveMap,
) -> tuple[Array, Array, dict[str, object]]:
    theta = math.radians(angle_degrees)
    normal = np.array((-math.sin(theta), math.cos(theta)))
    center = (
        float(normal @ np.array((0.5, 0.5)))
        + phase_target_pixels / min(target_shape)
    )
    half_width = 0.5 * width_target_pixels / min(target_shape)
    lower, upper = center - half_width, center + half_width
    source = (
        _polygon_halfplane_fraction(source_shape, normal, lower)
        - _polygon_halfplane_fraction(source_shape, normal, upper)
    )
    truth = (
        _mapped_halfplane_average(target_shape, normal, lower, transform)
        - _mapped_halfplane_average(target_shape, normal, upper, transform)
    )
    return source, truth, {
        "angle_degrees": angle_degrees,
        "width_target_pixels": width_target_pixels,
        "phase_target_pixels": phase_target_pixels,
    }


def _geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    positive = array[np.isfinite(array) & (array > 0.0)]
    return float(np.exp(np.mean(np.log(positive)))) if positive.size else float("nan")


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    numerical_mse_floor = 1.0e-14
    groups: dict[str, list[dict[str, object]]] = {"all": records}
    for record in records:
        groups.setdefault(str(record["family"]), []).append(record)
        groups.setdefault(f"transform:{record['transform']}", []).append(record)
        source_side = int(record["source_shape"][0])
        target_side = int(record["target_shape"][0])
        scale_regime = (
            "reduction" if target_side < source_side else
            "identity" if target_side == source_side else "enlargement"
        )
        groups.setdefault(f"scale:{scale_regime}", []).append(record)
        if record["family"] == "plane_wave":
            fraction = float(record["parameters"]["radius_fraction_of_limiting_nyquist"])
            band = "below_half" if fraction <= 0.5 else (
                "resolvable_high" if fraction <= 1.0 else "above_limiting_nyquist"
            )
            groups.setdefault(f"wave_band:{band}", []).append(record)

    summary: dict[str, object] = {}
    for name, items in groups.items():
        ratios = np.array([
            float(item["result"]["baseline_to_atlas_4x_mse_ratio"])
            for item in items
        ])
        baseline_mse = np.array([
            float(item["result"]["methods"]["piecewise_constant_source_cells"]["mse"])
            for item in items
        ])
        atlas_mse = np.array([
            float(item["result"]["methods"]["native_moment_atlas_4x"]["mse"])
            for item in items
        ])
        mse_advantage = baseline_mse - atlas_mse
        numerical_tie = np.maximum(baseline_mse, atlas_mse) <= numerical_mse_floor
        win = (~numerical_tie) & (mse_advantage > numerical_mse_floor)
        loss = (~numerical_tie) & (mse_advantage < -numerical_mse_floor)
        close = ~(win | loss)
        nondegenerate_baseline = baseline_mse > numerical_mse_floor
        final_source_excursion = np.array([
            float(item["result"]["methods"]["native_moment_atlas_4x"][
                "source_range_excursion"
            ]) for item in items
        ])
        final_truth_excursion = np.array([
            float(item["result"]["methods"]["native_moment_atlas_4x"][
                "exact_target_range_excursion"
            ]) for item in items
        ])
        internal_excursion = np.array([
            float(item["result"]["internal_atlas"][-1]["source_range_excursion"])
            for item in items
        ])
        horizontal_surplus = max(
            int(level["horizontal_sign_surplus"])
            for item in items for level in item["result"]["internal_atlas"]
        )
        vertical_surplus = max(
            int(level["vertical_sign_surplus"])
            for item in items for level in item["result"]["internal_atlas"]
        )
        worst = int(np.argmin(mse_advantage))
        best = int(np.argmax(mse_advantage))
        informative_ratios = ratios[nondegenerate_baseline]
        summary[name] = {
            "case_count": len(items),
            "numerical_mse_floor": numerical_mse_floor,
            "atlas_4x_material_win_count": int(np.count_nonzero(win)),
            "atlas_4x_material_loss_count": int(np.count_nonzero(loss)),
            "atlas_4x_numerical_or_close_tie_count": int(np.count_nonzero(close)),
            "atlas_4x_material_win_fraction": float(np.mean(win)),
            "atlas_4x_material_loss_fraction": float(np.mean(loss)),
            "exact_baseline_case_count": int(np.count_nonzero(
                baseline_mse <= numerical_mse_floor
            )),
            "exact_baseline_atlas_material_loss_count": int(np.count_nonzero(
                (baseline_mse <= numerical_mse_floor) & loss
            )),
            "nondegenerate_baseline_case_count": int(np.count_nonzero(
                nondegenerate_baseline
            )),
            "nondegenerate_geometric_mean_baseline_to_atlas_mse_ratio": (
                _geometric_mean(informative_ratios)
            ),
            "median_baseline_to_atlas_mse_ratio": float(np.median(ratios)),
            "minimum_baseline_to_atlas_mse_ratio": float(np.min(ratios)),
            "maximum_baseline_to_atlas_mse_ratio": float(np.max(ratios)),
            "largest_absolute_mse_reduction": float(np.max(mse_advantage)),
            "largest_absolute_mse_increase": float(max(np.max(-mse_advantage), 0.0)),
            "maximum_internal_source_range_excursion": float(np.max(internal_excursion)),
            "maximum_final_source_range_excursion": float(np.max(final_source_excursion)),
            "maximum_final_exact_target_range_excursion": float(np.max(final_truth_excursion)),
            "final_source_range_excursion_nonzero_count": int(np.count_nonzero(
                final_source_excursion > 2.0e-7
            )),
            "final_exact_target_range_excursion_nonzero_count": int(np.count_nonzero(
                final_truth_excursion > 2.0e-7
            )),
            "maximum_horizontal_sign_surplus": int(horizontal_surplus),
            "maximum_vertical_sign_surplus": int(vertical_surplus),
            "worst_case_id": items[worst]["id"],
            "best_case_id": items[best]["id"],
        }
    return summary


def run_census(*, quick: bool = False, include_affine: bool = True) -> dict[str, object]:
    identity = ProjectiveMap(np.eye(3))
    identity_sizes = ((17, 9), (25, 13)) if quick else (
        (17, 9), (25, 9), (25, 13), (33, 17), (49, 25),
        (25, 19), (25, 25), (25, 37), (25, 49), (25, 97),
    )
    radii = (0.35, 0.8, 1.05) if quick else (
        0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.05
    )
    angles = (0.0, 45.0, 90.0, 135.0) if quick else tuple(
        float(value) for value in range(0, 180, 15)
    )
    phases = (0.0, math.pi) if quick else tuple(
        2.0 * math.pi * value / 8.0 for value in range(8)
    )
    interface_phases = (-0.35, 0.35) if quick else (-0.45, -0.15, 0.15, 0.45)
    records: list[dict[str, object]] = []
    started = time.perf_counter()

    def append(
        family: str, source_shape: tuple[int, int], target_shape: tuple[int, int],
        transform_name: str, transform: ProjectiveMap, source: Array, truth: Array,
        parameters: dict[str, object],
    ) -> None:
        identifier = (
            f"{family}:{transform_name}:{source_shape[0]}x{target_shape[0]}:"
            f"{len(records):05d}"
        )
        records.append({
            "id": identifier,
            "family": family,
            "transform": transform_name,
            "source_shape": list(source_shape),
            "target_shape": list(target_shape),
            "parameters": parameters,
            "result": _evaluate_case(
                source, truth, transform, target_shape,
                identity=transform_name == "identity",
            ),
        })
        if len(records) % (20 if quick else 100) == 0:
            print(
                f"census {len(records)} cases, {time.perf_counter()-started:.1f} s",
                flush=True,
            )

    for source_side, target_side in identity_sizes:
        source_shape, target_shape = (source_side, source_side), (target_side, target_side)
        for radius in radii:
            for angle in angles:
                for phase in phases:
                    source, truth, parameters = _wave_case(
                        source_shape, target_shape, radius, angle, phase, identity
                    )
                    append(
                        "plane_wave", source_shape, target_shape, "identity", identity,
                        source, truth, parameters,
                    )
        for angle in angles:
            for phase_pixels in interface_phases:
                source, truth, parameters = _interface_case(
                    source_shape, target_shape, angle, phase_pixels, identity
                )
                append(
                    "half_plane", source_shape, target_shape, "identity", identity,
                    source, truth, parameters,
                )
        strip_widths = (1.0, 2.0) if quick else (0.75, 1.0, 2.0, 4.0)
        for angle in angles:
            for width in strip_widths:
                for phase_pixels in interface_phases:
                    source, truth, parameters = _strip_case(
                        source_shape, target_shape, angle, width, phase_pixels, identity
                    )
                    append(
                        "thin_strip", source_shape, target_shape, "identity", identity,
                        source, truth, parameters,
                    )

    crossed_angles = (0.0, 45.0) if quick else (0.0, 30.0, 60.0, 90.0)
    crossed_pairs = ((0.5, 0.8),) if quick else ((0.5, 0.8), (0.8, 0.95))
    crossed_separations = (60.0,) if quick else (30.0, 60.0, 90.0)
    crossed_phases = range(2) if quick else range(4)
    source_shape, target_shape = (25, 25), (13, 13)
    for first, second in crossed_pairs:
        for angle in crossed_angles:
            for separation in crossed_separations:
                for phase_index in crossed_phases:
                    source, truth, parameters = _crossed_wave_case(
                        source_shape, target_shape, first, second,
                        angle, separation, phase_index,
                    )
                    append(
                        "crossed_wave", source_shape, target_shape, "identity", identity,
                        source, truth, parameters,
                    )

    if include_affine:
        transforms = {
            "affine_17": affine_about_center(
                angle_degrees=17.0, scale_x=0.86, scale_y=0.82, shear_x=0.06
            ),
        } if quick else {
            "affine_17": affine_about_center(
                angle_degrees=17.0, scale_x=0.86, scale_y=0.82, shear_x=0.06
            ),
            "affine_m31": affine_about_center(
                angle_degrees=-31.0, scale_x=0.80, scale_y=0.88, shear_x=-0.05
            ),
            "affine_anisotropic": affine_about_center(
                angle_degrees=24.0, scale_x=0.62, scale_y=0.92, shear_x=0.10
            ),
        }
        affine_radii = (0.8,) if quick else (0.5, 0.8, 0.95, 1.05)
        affine_angles = (0.0, 90.0) if quick else tuple(
            float(value) for value in range(0, 180, 30)
        )
        affine_phases = (0.0, math.pi) if quick else tuple(
            2.0 * math.pi * value / 4.0 for value in range(4)
        )
        for transform_name, transform in transforms.items():
            for radius in affine_radii:
                for angle in affine_angles:
                    for phase in affine_phases:
                        source, truth, parameters = _wave_case(
                            source_shape, target_shape, radius, angle, phase, transform
                        )
                        append(
                            "plane_wave", source_shape, target_shape,
                            transform_name, transform, source, truth, parameters,
                        )
            for angle in affine_angles:
                for phase_pixels in interface_phases[:2]:
                    source, truth, parameters = _interface_case(
                        source_shape, target_shape, angle, phase_pixels, transform
                    )
                    append(
                        "half_plane", source_shape, target_shape,
                        transform_name, transform, source, truth, parameters,
                    )
            affine_widths = (1.0,) if quick else (0.75, 1.0, 2.0)
            for angle in affine_angles:
                for width in affine_widths:
                    for phase_pixels in interface_phases[:2]:
                        source, truth, parameters = _strip_case(
                            source_shape, target_shape, angle, width,
                            phase_pixels, transform,
                        )
                        append(
                            "thin_strip", source_shape, target_shape,
                            transform_name, transform, source, truth, parameters,
                        )

    return {
        "design": {
            "truth": "exact continuous synthetic target-cell averages",
            "baseline": "exact positive overlap of source piecewise-constant cells",
            "candidate": "one or two native admitted four-child atlas levels followed by the same overlap",
            "quick": bool(quick),
            "include_affine": bool(include_affine),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "record_count": len(records),
        "summary": _summarize(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path("/tmp/conv_four_child_atlas_census.json"),
    )
    args = parser.parse_args()
    result = run_census(
        quick=bool(args.quick), include_affine=not args.identity_only
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
