"""Analytic synthetic bank for owner versus source-conditioned frames."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import conv_resize, lanczos3_resize  # noqa: E402
from experiments.conv_fermi_owner_demo import (  # noqa: E402
    direct_distance_eikonal_conv_resize,
    distance_eikonal_conv_resize,
    eikonal_basin_conv_resize,
)


Field = Callable[[np.ndarray, np.ndarray], np.ndarray]
GRID_PAIRS = (
    ((17, 19), (65, 73)),
    ((25, 27), (97, 105)),
    ((33, 35), (129, 137)),
)
METHODS = (
    "owner_frames",
    "direct_frames_inverse_2",
    "owner_frames_inverse_2",
    "cartesian_conv",
    "lanczos3_sinc",
)


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, shape[0], dtype=np.float64)
    x = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)
    return np.meshgrid(x, y)


def _oriented(theta: float) -> tuple[Field, Field]:
    cosine, sine = math.cos(theta), math.sin(theta)
    return (
        lambda x, y: cosine * x + sine * y,
        lambda x, y: -sine * x + cosine * y,
    )


def _cases() -> list[tuple[str, str, Field]]:
    cases: list[tuple[str, str, Field]] = []
    angles = (0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0)
    phases = (0.0, 0.37)
    for angle in angles:
        q, _ = _oriented(angle)
        degrees = int(round(math.degrees(angle)))
        for phase in phases:
            cases.append((
                "plane_wave",
                f"plane_{degrees:02d}_p{phase:.2f}",
                lambda x, y, q=q, phase=phase: (
                    0.5 + 0.36 * np.sin(2.0 * np.pi * (2.1 * q(x, y) + phase))
                ),
            ))
    for angle in angles:
        q, t = _oriented(angle)
        degrees = int(round(math.degrees(angle)))
        cases.append((
            "filament",
            f"filament_{degrees:02d}",
            lambda x, y, q=q, t=t: (
                0.12 + 0.76 * np.exp(-0.5 * ((q(x, y) - 0.08) / 0.075) ** 2)
                * (0.82 + 0.18 * np.cos(np.pi * t(x, y)))
            ),
        ))
        cases.append((
            "interface",
            f"interface_{degrees:02d}",
            lambda x, y, q=q: 0.5 + 0.4 * np.tanh((q(x, y) - 0.035) / 0.055),
        ))
    for phase in (0.0, 0.21, 0.43):
        cases.append((
            "quadratic_chirp",
            f"quadratic_chirp_p{phase:.2f}",
            lambda x, y, phase=phase: (
                0.5 + 0.34 * np.sin(
                    2.0 * np.pi * (
                        0.55 * (0.8 * x + 0.6 * y)
                        + 1.25 * (0.8 * x + 0.6 * y) ** 2
                        + phase
                    )
                )
            ),
        ))
    for phase in (0.0, 0.31):
        cases.append((
            "radial_chirp",
            f"radial_chirp_p{phase:.2f}",
            lambda x, y, phase=phase: (
                0.5 + 0.34 * np.sin(
                    2.0 * np.pi * (0.45 * np.hypot(x + 0.08, y - 0.05)
                                   + 1.5 * (x * x + y * y) + phase)
                )
            ),
        ))
    for radius in (0.38, 0.62):
        cases.append((
            "curved_interface",
            f"circle_r{radius:.2f}",
            lambda x, y, radius=radius: (
                0.5 + 0.4 * np.tanh(
                    (radius - np.hypot(x + 0.08, y - 0.06)) / 0.05
                )
            ),
        ))
    for angle in (math.pi / 12.0, math.pi / 4.0):
        q, t = _oriented(angle)
        degrees = int(round(math.degrees(angle)))
        cases.append((
            "crossing_filaments",
            f"crossing_{degrees:02d}",
            lambda x, y, q=q, t=t: np.clip(
                0.08
                + 0.58 * np.exp(-0.5 * (q(x, y) / 0.065) ** 2)
                + 0.48 * np.exp(-0.5 * ((t(x, y) - 0.13) / 0.085) ** 2),
                0.0,
                1.0,
            ),
        ))
    cases.extend((
        (
            "smooth_warp",
            "bilinear_phase",
            lambda x, y: 0.5 + 0.34 * np.sin(
                2.0 * np.pi * (1.15 * x + 0.7 * x * y + 0.45 * y * y)
            ),
        ),
        (
            "smooth_blob",
            "anisotropic_gaussian",
            lambda x, y: 0.1 + 0.8 * np.exp(
                -0.5 * (((x + 0.13 + 0.35 * y) / 0.18) ** 2 + ((y - 0.09) / 0.43) ** 2)
            ),
        ),
        (
            "corner",
            "rounded_corner",
            lambda x, y: 0.5 + 0.4 * np.tanh(
                (np.minimum(x + 0.16, y - 0.04)) / 0.055
            ),
        ),
        (
            "affine",
            "affine_plane",
            lambda x, y: 0.5 + 0.18 * x - 0.13 * y,
        ),
    ))
    return cases


def _evaluate_methods(
    coarse: np.ndarray, fine_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    return {
        "owner_frames": eikonal_basin_conv_resize(coarse, fine_shape),
        "direct_frames_inverse_2": direct_distance_eikonal_conv_resize(
            coarse, fine_shape, response="inverse_power_2"
        ),
        "owner_frames_inverse_2": distance_eikonal_conv_resize(
            coarse, fine_shape, response="inverse_power_2"
        ),
        "cartesian_conv": conv_resize(coarse, fine_shape),
        "lanczos3_sinc": lanczos3_resize(coarse, fine_shape),
    }


def _plot_ratios(records: list[dict[str, object]], output: Path) -> None:
    records = [record for record in records if record["coarse_shape"] == [25, 27]]
    names = [str(record["name"]) for record in records]
    x = np.arange(len(records))
    figure, axis = plt.subplots(figsize=(13.5, 5.5), constrained_layout=True)
    for method in METHODS[:-1]:
        ratio = [
            float(record[method]["mse"]) / float(record["lanczos3_sinc"]["mse"])
            for record in records
        ]
        axis.plot(x, ratio, marker=".", linewidth=1.0, label=method)
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylabel("MSE / Lanczos-3 MSE")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_direct_owner(records: list[dict[str, object]], output: Path) -> None:
    families = sorted(set(str(record["family"]) for record in records))
    figure, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    for family_index, family in enumerate(families):
        rows = [record for record in records if record["family"] == family]
        ratios = np.array([
            float(record["direct_frames_inverse_2"]["mse"])
            / float(record["owner_frames"]["mse"])
            for record in rows
        ])
        offsets = np.linspace(-0.22, 0.22, len(ratios)) if len(ratios) > 1 else np.zeros(1)
        axis.scatter(
            family_index + offsets, ratios, s=19, alpha=0.72, color="#2878b5"
        )
        axis.plot(
            [family_index - 0.28, family_index + 0.28],
            [np.median(ratios), np.median(ratios)],
            color="#d1495b", linewidth=2.0,
        )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylabel("direct-frame MSE / owner-frame MSE")
    axis.set_xticks(np.arange(len(families)))
    axis.set_xticklabels(families, rotation=35, ha="right")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_selected(
    selected: list[tuple[str, np.ndarray, dict[str, np.ndarray]]], output: Path
) -> None:
    columns = 1 + len(METHODS)
    figure, axes = plt.subplots(
        len(selected), columns,
        figsize=(2.15 * columns, 2.15 * len(selected)),
        constrained_layout=True,
    )
    for row, (name, truth, results) in enumerate(selected):
        images = [("truth", truth)] + [(method, results[method]) for method in METHODS]
        for column, (label, image) in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(label.replace("_", "\n"), fontsize=8)
            if column == 0:
                axis.set_ylabel(name.replace("_", "\n"), fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    selected = []
    selected_names = {
        "plane_45_p0.37", "filament_30", "interface_45",
        "quadratic_chirp_p0.21", "circle_r0.62", "crossing_45",
    }
    started = time.perf_counter()
    for coarse_shape, fine_shape in GRID_PAIRS:
        coarse_x, coarse_y = _grid(coarse_shape)
        fine_x, fine_y = _grid(fine_shape)
        for family, name, function in _cases():
            coarse = np.clip(function(coarse_x, coarse_y), 0.0, 1.0).astype(np.float32)
            truth = np.clip(function(fine_x, fine_y), 0.0, 1.0).astype(np.float64)
            results = _evaluate_methods(coarse, fine_shape)
            record: dict[str, object] = {
                "family": family,
                "name": name,
                "coarse_shape": list(coarse_shape),
                "fine_shape": list(fine_shape),
            }
            for method, result in results.items():
                residual = np.asarray(result, dtype=np.float64) - truth
                source_min = float(np.min(coarse))
                source_max = float(np.max(coarse))
                record[method] = {
                    "mse": float(np.mean(residual * residual)),
                    "max_abs": float(np.max(np.abs(residual))),
                    "range_excess": float(max(
                        source_min - float(np.min(result)),
                        float(np.max(result)) - source_max,
                        0.0,
                    )),
                }
            records.append(record)
            if coarse_shape == (25, 27) and name in selected_names:
                selected.append((name, truth, results))
    elapsed = time.perf_counter() - started

    summary = {}
    for method in METHODS:
        values = np.array([float(record[method]["mse"]) for record in records])
        summary[method] = {
            "mean_mse": float(np.mean(values)),
            "median_mse": float(np.median(values)),
            "geometric_mean_mse": float(np.exp(np.mean(np.log(values)))),
            "wins": int(sum(
                float(record[method]["mse"])
                == min(float(record[candidate]["mse"]) for candidate in METHODS)
                for record in records
            )),
            "mean_max_abs": float(np.mean([
                float(record[method]["max_abs"]) for record in records
            ])),
            "maximum_range_excess": float(max(
                float(record[method]["range_excess"]) for record in records
            )),
        }
    payload = {
        "grid_pairs": GRID_PAIRS,
        "case_count": len(records),
        "elapsed_seconds": elapsed,
        "truth": "direct binary64 evaluation of each listed analytic field",
        "summary": summary,
        "records": records,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "coarse_shape", "fine_shape", "family", "name", "method",
            "mse", "max_abs", "range_excess",
        ))
        for record in records:
            for method in METHODS:
                metrics = record[method]
                writer.writerow((
                    "x".join(map(str, record["coarse_shape"])),
                    "x".join(map(str, record["fine_shape"])),
                    record["family"], record["name"], method,
                    metrics["mse"], metrics["max_abs"], metrics["range_excess"],
                ))
    _plot_ratios(records, output / "mse_ratio_to_lanczos.png")
    _plot_direct_owner(records, output / "direct_vs_owner.png")
    _plot_selected(selected, output / "selected_fields.png")


if __name__ == "__main__":
    run(ROOT / "output/support_geometry/eikonal_frame_bank")
