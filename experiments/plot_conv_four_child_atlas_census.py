#!/usr/bin/env python3
"""Render scale-resolved diagnostics for the four-child atlas census."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.conv_four_child_atlas_census import (
    _identity_cell_overlap,
    _strip_case,
    _wave_case,
)
from experiments.conv_warp.geometry import ProjectiveMap
from standalone_conv_resize_demo.conservative import zero_detail_synthesis


METHODS = {
    "2x atlas": "native_moment_atlas_2x",
    "4x atlas": "native_moment_atlas_4x",
}


def _ratio(record: dict[str, object]) -> float:
    return float(record["target_shape"][0]) / float(record["source_shape"][0])


def _scale_summary(records: list[dict[str, object]], method: str) -> dict[float, dict[str, float]]:
    result: dict[float, dict[str, float]] = {}
    for scale in sorted({_ratio(record) for record in records}):
        group = [record for record in records if _ratio(record) == scale]
        ratios, increases, excursions = [], [], []
        for record in group:
            methods = record["result"]["methods"]
            baseline = float(methods["piecewise_constant_source_cells"]["mse"])
            candidate = float(methods[method]["mse"])
            if baseline > 1.0e-14 and candidate > 0.0:
                ratios.append(baseline / candidate)
            increases.append(candidate - baseline)
            excursions.append(float(methods[method]["source_range_excursion"]))
        result[scale] = {
            "geometric_mean_ratio": (
                math.exp(float(np.mean(np.log(ratios)))) if ratios else float("nan")
            ),
            "loss_fraction": float(np.mean(np.asarray(increases) > 1.0e-14)),
            "maximum_source_excursion": float(np.max(excursions)),
        }
    return result


def render_scale_figure(records: list[dict[str, object]], path: Path) -> None:
    plane_waves = [record for record in records if record["family"] == "plane_wave"]
    summaries = {
        label: _scale_summary(plane_waves, method) for label, method in METHODS.items()
    }
    scales = sorted(next(iter(summaries.values())))
    colors = {"2x atlas": "#2171b5", "4x atlas": "#cb181d"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)

    for label, summary in summaries.items():
        axes[0, 0].plot(
            scales,
            [summary[scale]["geometric_mean_ratio"] for scale in scales],
            marker="o", linewidth=1.8, color=colors[label], label=label,
        )
        axes[0, 1].plot(
            scales,
            [summary[scale]["loss_fraction"] for scale in scales],
            marker="o", linewidth=1.8, color=colors[label], label=label,
        )
        axes[1, 0].plot(
            scales,
            [max(summary[scale]["maximum_source_excursion"], 1.0e-9) for scale in scales],
            marker="o", linewidth=1.8, color=colors[label], label=label,
        )

    axes[0, 0].axhline(1.0, color="0.35", linewidth=0.9, linestyle="--")
    axes[0, 0].axvline(1.0, color="0.35", linewidth=0.9, linestyle=":")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Plane-wave MSE advantage")
    axes[0, 0].set_ylabel("geometric mean baseline / atlas MSE")

    axes[0, 1].axvline(1.0, color="0.35", linewidth=0.9, linestyle=":")
    axes[0, 1].set_title("Plane-wave material-loss fraction")
    axes[0, 1].set_ylabel("fraction of cases")
    axes[0, 1].set_ylim(-0.01, 1.01)

    axes[1, 0].axvline(1.0, color="0.35", linewidth=0.9, linestyle=":")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Maximum final source-range excursion")
    axes[1, 0].set_ylabel("absolute excursion")

    regime_colors = {"reduction": "#238b45", "identity": "#756bb1", "enlargement": "#d95f0e"}
    for regime in ("reduction", "identity", "enlargement"):
        points = []
        for record in records:
            scale = _ratio(record)
            actual = "reduction" if scale < 1.0 else "identity" if scale == 1.0 else "enlargement"
            if actual != regime:
                continue
            internal = float(record["result"]["internal_atlas"][-1]["source_range_excursion"])
            final = float(record["result"]["methods"]["native_moment_atlas_4x"]["source_range_excursion"])
            points.append((max(internal, 1.0e-9), max(final, 1.0e-9)))
        array = np.asarray(points)
        axes[1, 1].scatter(
            array[:, 0], array[:, 1], s=5, alpha=0.18,
            color=regime_colors[regime], edgecolors="none", label=regime,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Footprint suppression of internal excursion")
    axes[1, 1].set_xlabel("internal atlas excursion")
    axes[1, 1].set_ylabel("final source-range excursion")
    axes[1, 1].legend(frameon=False, markerscale=2.0)

    for axis in axes.flat:
        axis.set_xlabel("target/source side ratio") if axis is not axes[1, 1] else None
        axis.grid(True, alpha=0.18)
    axes[0, 0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _reconstruct(record: dict[str, object]) -> tuple[np.ndarray, ...]:
    identity = ProjectiveMap(np.eye(3))
    source_shape = tuple(record["source_shape"])
    target_shape = tuple(record["target_shape"])
    parameters = record["parameters"]
    if record["family"] == "plane_wave":
        source, truth, _ = _wave_case(
            source_shape, target_shape,
            float(parameters["radius_fraction_of_limiting_nyquist"]),
            float(parameters["angle_degrees"]),
            float(parameters["phase_radians"]), identity,
        )
    elif record["family"] == "thin_strip":
        source, truth, _ = _strip_case(
            source_shape, target_shape,
            float(parameters["angle_degrees"]),
            float(parameters["width_target_pixels"]),
            float(parameters["phase_target_pixels"]), identity,
        )
    else:
        raise ValueError(f"unsupported diagnostic family {record['family']}")
    baseline = _identity_cell_overlap(source, target_shape)
    atlas2_state = zero_detail_synthesis(source.astype(np.float32))
    atlas4_state = zero_detail_synthesis(atlas2_state)
    atlas2 = _identity_cell_overlap(atlas2_state, target_shape)
    atlas4 = _identity_cell_overlap(atlas4_state, target_shape)
    return source, truth, baseline, atlas2, atlas4


def render_case_figure(records: list[dict[str, object]], identifiers: list[str], path: Path) -> None:
    selected = [next(record for record in records if record["id"] == identifier) for identifier in identifiers]
    fig, axes = plt.subplots(len(selected), 6, figsize=(13.2, 2.7 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, record in enumerate(selected):
        source, truth, baseline, atlas2, atlas4 = _reconstruct(record)
        arrays = (source, truth, baseline, atlas2, atlas4, atlas4 - truth)
        titles = ("source means", "exact target means", "positive overlap", "2x atlas", "4x atlas", "4x error")
        lower = min(float(np.min(array)) for array in arrays[:-1])
        upper = max(float(np.max(array)) for array in arrays[:-1])
        error_limit = max(float(np.max(np.abs(arrays[-1]))), 1.0e-12)
        for column, (array, title) in enumerate(zip(arrays, titles)):
            if column == 5:
                axes[row, column].imshow(array, cmap="coolwarm", vmin=-error_limit, vmax=error_limit, interpolation="nearest")
            else:
                axes[row, column].imshow(array, cmap="gray", vmin=lower, vmax=upper, interpolation="nearest")
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(record["id"], fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("census", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.census.read_text(encoding="utf-8"))["records"]
    render_scale_figure(records, args.out_dir / "conv_four_child_atlas_scale_census.png")
    render_case_figure(
        records,
        [
            "thin_strip:identity:17x9:00820",
            "plane_wave:identity:25x37:06628",
            "plane_wave:identity:25x97:08789",
        ],
        args.out_dir / "conv_four_child_atlas_worst_cases.png",
    )


if __name__ == "__main__":
    main()
