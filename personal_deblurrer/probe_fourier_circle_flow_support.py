#!/usr/bin/env python3
"""Probe ring-dependent phase transport for non-fold flow ambiguity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .circles import phase_circle_translation
from .dense_estimation import _luminance
from .run_visibility_benchmark import _layered_pair


def _peak(correlation: np.ndarray) -> tuple[np.ndarray, float, float]:
    height, width = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    value = float(correlation[peak_y, peak_x])
    excluded = correlation.copy()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            excluded[(peak_y + dy) % height, (peak_x + dx) % width] = -np.inf
    competitor = float(np.max(excluded))
    offset_x = peak_x if peak_x <= width // 2 else peak_x - width
    offset_y = peak_y if peak_y <= height // 2 else peak_y - height
    return np.asarray((offset_x, offset_y), dtype=np.float64), value, competitor


def fourier_circle_support(
    first: np.ndarray,
    second: np.ndarray,
    *,
    ring_count: int = 7,
) -> dict[str, object]:
    _, record = phase_circle_translation(
        _luminance(first), _luminance(second), ring_count=ring_count)
    return record


def run(size: int, output: Path) -> dict[str, object]:
    rows = []
    for source_index, (source, background) in enumerate(sources(size).items()):
        for case, displacement in (
            ("moderate_disocclusion", 2.0),
            ("folded_disocclusion", 3.0),
        ):
            _, observations = _layered_pair(
                background,
                displacement,
                noise_sigma=0.002,
                seed=21000 + 41 * source_index,
            )
            rows.append({
                "source": source,
                "case": case,
                **fourier_circle_support(observations[0], observations[1]),
            })
    result = {
        "experiment": "fourier_circle_cross_predictive_flow_support_v1",
        "size": int(size),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument(
        "--out", type=Path,
        default=Path("personal_deblurrer/fourier_circle_flow_support.json"),
    )
    args = parser.parse_args()
    result = run(args.size, args.out)
    for row in result["rows"]:
        print(
            f"{row['source']:22s} {row['case']:23s} "
            f"disp {row['translation_dispersion_pixels']:.3f}  "
            f"path {row['circle_path_length_pixels']:.2f}  "
            f"ambiguity {row['weighted_peak_ambiguity']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
