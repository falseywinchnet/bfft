"""Use Lanczos-sinc only as a destination oracle for CONV scale transport.

The diagnostic separates analysis from synthesis, then decomposes the direct
scale-8 Lanczos analysis matrix in an orthonormal Haar basis.  It does not use
optimization or insert Lanczos values into CONV.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from skimage import data

from standalone_conv_resize_demo.backend import (
    conv_resize,
    lanczos3_resize,
    polyphase_fir_resize,
)
from standalone_conv_resize_demo.conservative import restrict_to, zero_detail_expand
from standalone_conv_resize_demo.gui_common import mse, range_excess, unit_image


Array = np.ndarray


def lanczos_analysis_matrix(
    source_count: int, target_count: int, radius: int = 3
) -> Array:
    """Reproduce the demo's endpoint-aligned normalized analysis matrix."""

    rate = (target_count - 1) / (source_count - 1)
    cutoff = min(1.0, rate)
    support = radius / cutoff
    matrix = np.zeros((target_count, source_count), dtype=np.float64)
    for row in range(target_count):
        coordinate = row * (source_count - 1) / (target_count - 1)
        begin = math.ceil(coordinate - support)
        end = math.floor(coordinate + support)
        taps = np.arange(begin, end + 1)
        distance = coordinate - taps
        weight = (
            cutoff * np.sinc(cutoff * distance)
            * np.sinc(distance / support)
        )
        weight /= np.sum(weight)
        for tap, value in zip(taps, weight):
            matrix[row, int(np.clip(tap, 0, source_count - 1))] += value
    return matrix


def haar_operator_bands(matrix: Array, levels: int) -> tuple[Array, tuple[Array, ...]]:
    """Return scaling and detail columns of an operator in a Haar basis."""

    current = np.asarray(matrix, dtype=np.float64)
    details: list[Array] = []
    for _ in range(levels):
        even = current[:, 0::2]
        odd = current[:, 1::2]
        details.append((odd - even) / np.sqrt(2.0))
        current = (even + odd) / np.sqrt(2.0)
    return current, tuple(details)


def haar_signal_bands(value: Array, levels: int, axis: int = 1) -> tuple[Array, tuple[Array, ...]]:
    """Analyze an array into orthonormal Haar scale/detail coordinates."""

    current = np.moveaxis(np.asarray(value, dtype=np.float64), axis, 0)
    details: list[Array] = []
    for _ in range(levels):
        even = current[0::2]
        odd = current[1::2]
        details.append((odd - even) / np.sqrt(2.0))
        current = (even + odd) / np.sqrt(2.0)
    return np.moveaxis(current, 0, axis), tuple(
        np.moveaxis(detail, 0, axis) for detail in details
    )


def synthesize_haar_band(
    scale: Array, details: tuple[Array, ...], selected: int, axis: int = 1
) -> Array:
    """Reconstruct exactly one detail band, or the terminal scaling band."""

    scaling_selected = selected == len(details)
    current = np.moveaxis(
        scale if scaling_selected else np.zeros_like(scale), axis, 0
    )
    moved_details = [np.moveaxis(item, axis, 0) for item in details]
    for level in range(len(details) - 1, -1, -1):
        detail = (
            moved_details[level]
            if selected == level else np.zeros_like(moved_details[level])
        )
        fine = np.empty((2 * current.shape[0],) + current.shape[1:])
        fine[0::2] = (current - detail) / np.sqrt(2.0)
        fine[1::2] = (current + detail) / np.sqrt(2.0)
        current = fine
    return np.moveaxis(current, 0, axis)


def cross_cycle(image: Array, target: tuple[int, int]) -> dict[str, object]:
    """Cross every admitted analysis state with every reconstruction family."""

    source = np.asarray(image, dtype=np.float32)
    cell_average, levels = restrict_to(source, target)
    analyses = {
        "Lanczos-3": lanczos3_resize(source, target),
        "Lanczos-8": polyphase_fir_resize(source, target, radius=8),
        "point-CONV": conv_resize(source, target),
        "cell-average": cell_average,
    }
    syntheses = {
        "Lanczos-3": lambda value: lanczos3_resize(value, source.shape[:2]),
        "Lanczos-8": lambda value: polyphase_fir_resize(
            value, source.shape[:2], radius=8
        ),
        "point-CONV": lambda value: conv_resize(value, source.shape[:2]),
        "moment-CONV": lambda value: zero_detail_expand(value, levels),
    }
    records: dict[str, object] = {}
    for analysis_name, coarse in analyses.items():
        records[analysis_name] = {
            "coarse_minimum": float(np.min(coarse)),
            "coarse_maximum": float(np.max(coarse)),
            "coarse_mean": float(np.mean(coarse)),
            "syntheses": {},
        }
        for synthesis_name, operation in syntheses.items():
            returned = operation(coarse)
            records[analysis_name]["syntheses"][synthesis_name] = {
                "mse": mse(returned, source),
                "range_excess": range_excess(returned, source),
            }
    return records


def direct_scale_detail_audit(image: Array) -> dict[str, object]:
    """Measure which Haar coordinates direct scale-8 Lanczos carries forward."""

    source = np.asarray(image, dtype=np.float64)
    matrix = lanczos_analysis_matrix(source.shape[1], source.shape[1] // 8)
    scaling_operator, detail_operators = haar_operator_bands(matrix, 3)
    total_energy = float(np.sum(matrix * matrix))
    operator_bands = {
        f"detail_level_{level}": {
            "squared_norm": float(np.sum(band * band)),
            "fraction_of_operator_squared_norm": float(
                np.sum(band * band) / total_energy
            ),
        }
        for level, band in enumerate(detail_operators, 1)
    }
    operator_bands["scale_level_3"] = {
        "squared_norm": float(np.sum(scaling_operator * scaling_operator)),
        "fraction_of_operator_squared_norm": float(
            np.sum(scaling_operator * scaling_operator) / total_energy
        ),
    }

    scale, details = haar_signal_bands(source, 3, axis=1)
    components = []
    for selected in range(4):
        band = synthesize_haar_band(scale, details, selected, axis=1)
        components.append(np.einsum("mn,hnc->hmc", matrix, band, optimize=True))
    output = sum(components)
    detail_total = sum(components[:3])
    signal_bands = {
        name: {
            "output_rms": float(np.sqrt(np.mean(component * component))),
            "output_squared_rms_fraction": float(
                np.mean(component * component) / np.mean(output * output)
            ),
        }
        for name, component in zip(
            ("detail_level_1", "detail_level_2", "detail_level_3", "scale_level_3"),
            components,
        )
    }
    return {
        "operator_total_squared_norm": total_energy,
        "operator_bands": operator_bands,
        "astronaut_horizontal_signal_bands": signal_bands,
        "astronaut_horizontal_detail_total_rms": float(
            np.sqrt(np.mean(detail_total * detail_total))
        ),
        "astronaut_horizontal_scale_only_mse_to_full_analysis": float(
            np.mean(detail_total * detail_total)
        ),
        "recomposition_maximum_absolute_error": float(np.max(np.abs(
            output - np.einsum("mn,hnc->hmc", matrix, source, optimize=True)
        ))),
    }


def run() -> dict[str, object]:
    astronaut = unit_image(data.astronaut())
    return {
        "design": {
            "source": "skimage.data.astronaut",
            "cycle": [512, 64, 512],
            "purpose": "Lanczos destination audit; no solver or fitted parameter",
        },
        "cross_cycle": cross_cycle(astronaut, (64, 64)),
        "direct_scale_detail_audit": direct_scale_detail_audit(astronaut),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/support_geometry/lanczos_basin_diagnostic.json"),
    )
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
