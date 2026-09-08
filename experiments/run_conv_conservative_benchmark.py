"""Focused synthetic evaluation of CONV multiresolution and matched resize.

The benchmark starts from fine cell rasters, applies each method's own
analysis, discards its detail, and applies the matched synthesis.  Retained-
detail CONV round trips are audited separately.  The technical plate uses the
current basin/CONV* raster cycle rather than the conservative ablation.  No
natural-image corpus or learned choice enters the design.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.conv_conservative_multiresolution import (
    block_moments_2d,
    conservative_analysis_1d,
    conservative_analysis_2d,
    conservative_restrict_1d,
    conservative_restrict_2d,
    conservative_synthesis_1d,
    conservative_synthesis_2d,
    face_sign_certificate_2d,
    sign_changes,
)
from experiments.conv_distilled_core import distilled_conv_resize
from standalone_conv_resize_demo.backend import (
    lanczos3_resize,
    linear_resize,
)


Array = np.ndarray


def _mse(value: Array, truth: Array, crop: int = 0) -> float:
    estimate = np.asarray(value, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    if crop:
        slices = (slice(crop, -crop),) * min(reference.ndim, 2)
        estimate = estimate[slices]
        reference = reference[slices]
    residual = estimate - reference
    return float(np.mean(residual * residual))


def _linf(value: Array, truth: Array, crop: int = 0) -> float:
    estimate = np.asarray(value, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    if crop:
        slices = (slice(crop, -crop),) * min(reference.ndim, 2)
        estimate = estimate[slices]
        reference = reference[slices]
    return float(np.max(np.abs(estimate - reference), initial=0.0))


def _pixel_center_lanczos_matrix(
    source_count: int, target_count: int, radius: int = 3
) -> Array:
    """Return a normalized pixel-center Lanczos matrix with widened analysis."""

    scale = target_count / source_count
    contraction = min(1.0, scale)
    source_coordinate = (
        (np.arange(target_count, dtype=np.float64) + 0.5) / scale - 0.5
    )
    source = np.arange(source_count, dtype=np.float64)
    matrix = np.zeros((target_count, source_count), dtype=np.float64)
    for row, coordinate in enumerate(source_coordinate):
        distance = (coordinate - source) * contraction
        admitted = np.abs(distance) < radius
        weight = np.zeros(source_count, dtype=np.float64)
        weight[admitted] = (
            np.sinc(distance[admitted])
            * np.sinc(distance[admitted] / radius)
        )
        total = float(np.sum(weight))
        if total == 0.0:
            weight[int(np.clip(round(coordinate), 0, source_count - 1))] = 1.0
            total = 1.0
        matrix[row] = weight / total
    return matrix


def _lanczos_resample(value: Array, shape: tuple[int, ...]) -> Array:
    source = np.asarray(value, dtype=np.float64)
    if source.ndim == 1:
        return _pixel_center_lanczos_matrix(source.size, shape[0]) @ source
    if source.ndim != 2:
        raise ValueError("benchmark Lanczos supports scalar 1-D or 2-D arrays")
    wy = _pixel_center_lanczos_matrix(source.shape[0], shape[0])
    wx = _pixel_center_lanczos_matrix(source.shape[1], shape[1])
    return wy @ source @ wx.T


def _linear_synthesis_1d(coarse: Array) -> Array:
    c = np.asarray(coarse, dtype=np.float64)
    coordinate = np.arange(2 * c.size, dtype=np.float64) / 2.0 - 0.25
    return np.interp(coordinate, np.arange(c.size), c)


def _bilinear_synthesis_2d(coarse: Array) -> Array:
    c = np.asarray(coarse, dtype=np.float64)
    x = np.arange(2 * c.shape[1], dtype=np.float64) / 2.0 - 0.25
    y = np.arange(2 * c.shape[0], dtype=np.float64) / 2.0 - 0.25
    rows = np.stack([
        np.interp(x, np.arange(c.shape[1]), row) for row in c
    ])
    return np.stack([
        np.interp(y, np.arange(c.shape[0]), rows[:, column])
        for column in range(rows.shape[1])
    ], axis=1)


def _half_plane_cell_average(side: int, angle: float, offset: float) -> Array:
    """Return exact polygonal coverage of a half-plane in every square cell."""

    theta = math.radians(angle)
    normal = np.array((-math.sin(theta), math.cos(theta)))
    center = np.array((side / 2.0, side / 2.0))
    threshold = float(normal @ center + offset)

    def clip(polygon: list[Array]) -> list[Array]:
        result: list[Array] = []
        for start, stop in zip(polygon, polygon[1:] + polygon[:1]):
            fs = float(normal @ start - threshold)
            ft = float(normal @ stop - threshold)
            inside_s, inside_t = fs >= 0.0, ft >= 0.0
            if inside_s:
                result.append(start)
            if inside_s != inside_t:
                result.append(start + (stop - start) * (fs / (fs - ft)))
        return result

    output = np.empty((side, side), dtype=np.float64)
    for row in range(side):
        for column in range(side):
            polygon = clip([
                np.array((column, row), dtype=np.float64),
                np.array((column + 1.0, row), dtype=np.float64),
                np.array((column + 1.0, row + 1.0), dtype=np.float64),
                np.array((column, row + 1.0), dtype=np.float64),
            ])
            if len(polygon) < 3:
                output[row, column] = 0.0
                continue
            points = np.stack(polygon)
            output[row, column] = 0.5 * abs(float(
                points[:, 0] @ np.roll(points[:, 1], -1)
                - points[:, 1] @ np.roll(points[:, 0], -1)
            ))
    return output


def _one_dimensional_cases(side: int) -> list[tuple[str, Array]]:
    index = np.arange(side, dtype=np.float64) + 0.5
    cases: list[tuple[str, Array]] = []
    cases.append(("nyquist", (-1.0) ** np.arange(side)))
    for phase in np.arange(8) * math.pi / 4.0:
        cases.append((
            "near_nyquist",
            0.5 + 0.45 * np.sin(0.92 * math.pi * index + phase),
        ))
        coordinate = index / side
        cases.append((
            "chirp",
            0.5 + 0.45 * np.sin(
                2.0 * math.pi * (1.0 * coordinate + 0.22 * side * coordinate**2)
                + phase
            ),
        ))
    return cases


def _two_dimensional_cases(side: int) -> list[tuple[str, Array]]:
    cases: list[tuple[str, Array]] = []
    y, x = np.indices((side, side), dtype=np.float64)
    cases.append(("checkerboard", (-1.0) ** (x + y)))
    cases.append(("vertical_nyquist", (-1.0) ** x))
    for angle in (17.5, 32.5, 47.5, 62.5):
        for phase in (-0.35, 0.35):
            cases.append((
                "diagonal_edge",
                _half_plane_cell_average(side, angle, phase),
            ))
    coordinate = (x + 0.5) / side
    for phase in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
        cases.append((
            "chirp",
            0.5 + 0.45 * np.sin(
                2.0 * math.pi * (coordinate + 0.22 * side * coordinate**2)
                + phase
            ),
        ))
    return cases


def _metrics_1d(fine: Array) -> dict[str, dict[str, float | int]]:
    coarse = conservative_restrict_1d(fine)
    conv = conservative_synthesis_1d(coarse)
    lanczos_coarse = _lanczos_resample(fine, (fine.size // 2,))
    lanczos = _lanczos_resample(lanczos_coarse, (fine.size,))
    linear = _linear_synthesis_1d(coarse)
    methods = {
        "CONV-C": (conv, coarse),
        "Lanczos-3": (lanczos, lanczos_coarse),
        "box-linear": (linear, coarse),
    }
    return {
        name: {
            "mse": _mse(value, fine, crop=4),
            "linf": _linf(value, fine, crop=4),
            "sign_surplus": sign_changes(np.diff(value))
            - sign_changes(np.diff(restricted)),
            "coarse_rms": float(np.sqrt(np.mean(restricted * restricted))),
            "coarse_maximum_absolute": float(
                np.max(np.abs(restricted), initial=0.0)
            ),
        }
        for name, (value, restricted) in methods.items()
    }


def _metrics_2d(fine: Array) -> dict[str, dict[str, float | int]]:
    coarse = conservative_restrict_2d(fine)
    conv = conservative_synthesis_2d(coarse)
    lanczos_coarse = _lanczos_resample(
        fine, (fine.shape[0] // 2, fine.shape[1] // 2)
    )
    lanczos = _lanczos_resample(lanczos_coarse, fine.shape)
    linear = _bilinear_synthesis_2d(coarse)
    certificate = face_sign_certificate_2d(coarse, block_moments_2d(conv))
    methods = {
        "CONV-C": (conv, coarse),
        "Lanczos-3": (lanczos, lanczos_coarse),
        "box-bilinear": (linear, coarse),
    }
    result = {
        name: {
            "mse": _mse(value, fine, crop=4),
            "linf": _linf(value, fine, crop=4),
            "minimum": float(np.min(value)),
            "maximum": float(np.max(value)),
            "coarse_rms": float(np.sqrt(np.mean(restricted * restricted))),
            "coarse_maximum_absolute": float(
                np.max(np.abs(restricted), initial=0.0)
            ),
        }
        for name, (value, restricted) in methods.items()
    }
    result["CONV-C"].update(certificate)
    return result


def _geometric_mean(value: list[float]) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.exp(np.mean(np.log(np.maximum(array, 1.0e-300)))))


def run(sizes: tuple[int, ...]) -> dict[str, object]:
    records_1d: list[dict[str, object]] = []
    records_2d: list[dict[str, object]] = []
    exact_round_trip = 0.0
    rng = np.random.default_rng(20260826)
    for side in sizes:
        for name, fine in _one_dimensional_cases(side):
            records_1d.append({
                "fine_side": side, "case": name, "metrics": _metrics_1d(fine)
            })
        for name, fine in _two_dimensional_cases(side):
            records_2d.append({
                "fine_side": side, "case": name, "metrics": _metrics_2d(fine)
            })
        sample_1d = rng.normal(size=(side, 3))
        state_1d = conservative_analysis_1d(sample_1d)
        exact_round_trip = max(exact_round_trip, _linf(
            conservative_synthesis_1d(state_1d.coarse, state_1d.detail),
            sample_1d,
        ))
        sample_2d = rng.normal(size=(side, side, 3))
        state_2d = conservative_analysis_2d(sample_2d)
        exact_round_trip = max(exact_round_trip, _linf(
            conservative_synthesis_2d(
                state_2d.coarse,
                state_2d.horizontal_detail,
                state_2d.vertical_detail,
                state_2d.mixed_detail,
            ),
            sample_2d,
        ))

    summary: dict[str, object] = {}
    for dimension, records in (("one_dimensional", records_1d), ("two_dimensional", records_2d)):
        dimension_summary: dict[str, object] = {}
        for side in sizes:
            group = [row for row in records if row["fine_side"] == side]
            methods = group[0]["metrics"].keys()
            dimension_summary[str(side)] = {
                method: {
                    "geometric_mean_mse": _geometric_mean([
                        float(row["metrics"][method]["mse"]) for row in group
                    ]),
                    "maximum_linf": max(
                        float(row["metrics"][method]["linf"]) for row in group
                    ),
                }
                for method in methods
            }
        summary[dimension] = dimension_summary
    summary["maximum_retained_detail_round_trip_error"] = exact_round_trip
    summary["maximum_conv_2d_horizontal_sign_surplus"] = max(
        int(row["metrics"]["CONV-C"]["horizontal_sign_surplus"])
        for row in records_2d
    )
    summary["maximum_conv_2d_vertical_sign_surplus"] = max(
        int(row["metrics"]["CONV-C"]["vertical_sign_surplus"])
        for row in records_2d
    )
    summary["nyquist_alias_leakage"] = {
        "one_dimensional": {
            method: max(
                float(row["metrics"][method]["coarse_maximum_absolute"])
                for row in records_1d if row["case"] == "nyquist"
            )
            for method in records_1d[0]["metrics"]
        },
        "two_dimensional": {
            method: max(
                float(row["metrics"][method]["coarse_maximum_absolute"])
                for row in records_2d
                if row["case"] in {"checkerboard", "vertical_nyquist"}
            )
            for method in records_2d[0]["metrics"]
        },
    }
    return {
        "design": {
            "fine_sizes": list(sizes),
            "restriction": "each method's own matched analysis",
            "synthesis": "each method's own matched synthesis",
            "score_crop": 4,
            "synthetics": [
                "Nyquist stripes/checkerboard", "near-Nyquist carriers",
                "polygon-exact diagonal cell coverage", "quadratic chirps",
            ],
        },
        "summary": summary,
        "one_dimensional_records": records_1d,
        "two_dimensional_records": records_2d,
    }


def _plot_summary(result: dict[str, object], output: Path) -> None:
    colors = {"CONV-C": "#0072B2", "Lanczos-3": "#D55E00", "box-bilinear": "#009E73"}
    figure, axes = plt.subplots(1, 2, figsize=(7.08, 2.45))
    table = result["summary"]["two_dimensional"]
    sizes = np.array(sorted(map(int, table)), dtype=float)
    for method in ("CONV-C", "Lanczos-3", "box-bilinear"):
        values = [table[str(int(side))][method]["geometric_mean_mse"] for side in sizes]
        axes[0].semilogy(sizes, values, marker="o", linewidth=1.4,
                         label=method, color=colors[method])
    axes[0].set_title("Matched 2-D discarded-detail cycle")
    axes[0].set_xlabel("fine cells per axis")
    axes[0].set_ylabel("geometric-mean MSE")
    axes[0].set_xticks(sizes, [str(int(value)) for value in sizes])
    axes[0].grid(True, which="both", linewidth=0.35, alpha=0.45)
    axes[0].legend(frameon=False, fontsize=7)

    records = result["two_dimensional_records"]
    for method in ("CONV-C", "Lanczos-3", "box-bilinear"):
        values = []
        for side in sizes:
            group = [
                row["metrics"][method]["mse"] for row in records
                if row["fine_side"] == int(side) and row["case"] == "diagonal_edge"
            ]
            values.append(float(np.mean(group)))
        axes[1].semilogy(sizes, values, marker="o", linewidth=1.4,
                         label=method, color=colors[method])
    axes[1].set_title("Polygon-exact diagonal interfaces")
    axes[1].set_xlabel("fine cells per axis")
    axes[1].set_ylabel("mean MSE")
    axes[1].set_xticks(sizes, [str(int(value)) for value in sizes])
    axes[1].grid(True, which="both", linewidth=0.35, alpha=0.45)
    figure.tight_layout(pad=0.55, w_pad=1.0)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_current_matched_plate(side: int, output: Path) -> dict[str, object]:
    """Render the selected fields through each current matched resize cycle."""

    cases = _two_dimensional_cases(side)
    edge = [value for name, value in cases if name == "diagonal_edge"][5]
    chirp = [value for name, value in cases if name == "chirp"][1]
    scenes = (("Diagonal interface", edge), ("Quadratic chirp", chirp))
    operations = {
        "CONV*": distilled_conv_resize,
        "Lanczos-3": lanczos3_resize,
        "bilinear": linear_resize,
    }
    rows = ("Truth", *operations)
    figure, axes = plt.subplots(2, len(rows), figsize=(7.08, 3.15))
    records: dict[str, object] = {}
    for scene_row, (title, fine) in enumerate(scenes):
        images: dict[str, Array] = {"Truth": fine}
        scene_records: dict[str, object] = {}
        for label, operation in operations.items():
            coarse = operation(fine, (side // 2, side // 2))
            cycle = operation(coarse, fine.shape)
            images[label] = cycle
            scene_records[label] = {
                "mse": _mse(cycle, fine, crop=4),
                "linf": _linf(cycle, fine, crop=4),
                "excursion": max(
                    float(np.max(np.min(fine) - cycle, initial=0.0)),
                    float(np.max(cycle - np.max(fine), initial=0.0)),
                ),
                "minimum": float(np.min(cycle)),
                "maximum": float(np.max(cycle)),
            }
        records[title] = scene_records
        for method_column, label in enumerate(rows):
            image = images[label]
            axis = axes[scene_row, method_column]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0,
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if scene_row == 0:
                axis.set_title(label, fontsize=8)
            if method_column == 0:
                axis.set_ylabel(title, fontsize=8)
    figure.tight_layout(pad=0.35, h_pad=0.2, w_pad=0.2)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return {
        "fine_shape": [side, side],
        "coarse_shape": [side // 2, side // 2],
        "cycle": "each method's declared reduction followed by the same method's synthesis",
        "score_crop": 4,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="32,64")
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/pdf/conv_conservative_evidence/results.json"),
    )
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if any(side % 2 or side < 10 for side in sizes):
        raise ValueError("every fine side must be even and at least ten")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(sizes)
    result["current_matched_technical"] = _plot_current_matched_plate(
        max(sizes), args.out.parent / "current_matched_plate.pdf"
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    _plot_summary(result, args.out.parent / "conservative_cycle.pdf")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
