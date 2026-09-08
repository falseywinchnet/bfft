#!/usr/bin/env python3
"""Stratified 2x/4x/8x depth probe for the conservative CONV atlas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.conv_four_child_atlas_census import (
    _field_metrics,
    _identity_cell_overlap,
    _range_excursion,
    _wave_case,
)
from experiments.conv_warp.geometry import ProjectiveMap
from standalone_conv_resize_demo.conservative import zero_detail_synthesis


def run(census: dict[str, object]) -> dict[str, object]:
    identity = ProjectiveMap(np.eye(3))
    selected: list[dict[str, object]] = []
    for record in census["records"]:
        parameters = record["parameters"]
        if (
            record["family"] != "plane_wave"
            or record["transform"] != "identity"
            or int(record["source_shape"][0]) != 25
            or int(record["target_shape"][0]) not in (9, 13, 37, 49, 97)
            or int(round(float(parameters["angle_degrees"]))) % 45 != 0
            or int(round(float(parameters["phase_radians"]) / (math.pi / 4.0))) % 4 != 0
        ):
            continue
        source_shape = tuple(record["source_shape"])
        target_shape = tuple(record["target_shape"])
        source, truth, _ = _wave_case(
            source_shape,
            target_shape,
            float(parameters["radius_fraction_of_limiting_nyquist"]),
            float(parameters["angle_degrees"]),
            float(parameters["phase_radians"]),
            identity,
        )
        atlas = source.astype(np.float32)
        for _ in range(3):
            atlas = zero_detail_synthesis(atlas)
        estimate = _identity_cell_overlap(atlas, target_shape)
        metrics = _field_metrics(
            estimate, truth, source, np.ones(target_shape, dtype=bool)
        )
        metrics["internal_source_range_excursion"] = _range_excursion(
            atlas, float(np.min(source)), float(np.max(source))
        )
        selected.append({
            "id": record["id"],
            "source_shape": record["source_shape"],
            "target_shape": record["target_shape"],
            "parameters": parameters,
            "atlas_2x_mse": record["result"]["methods"][
                "native_moment_atlas_2x"
            ]["mse"],
            "atlas_4x_mse": record["result"]["methods"][
                "native_moment_atlas_4x"
            ]["mse"],
            "atlas_8x": metrics,
        })

    summary: dict[str, object] = {}
    for target in (9, 13, 37, 49, 97):
        group = [item for item in selected if int(item["target_shape"][0]) == target]
        mse = np.array([
            [
                float(item["atlas_2x_mse"]),
                float(item["atlas_4x_mse"]),
                float(item["atlas_8x"]["mse"]),
            ]
            for item in group
        ])
        summary[f"25_to_{target}"] = {
            "case_count": len(group),
            "best_depth_counts_2x_4x_8x": np.bincount(
                np.argmin(mse, axis=1), minlength=3
            ).tolist(),
            "geometric_mean_4x_to_8x_mse_ratio": float(
                np.exp(np.mean(np.log(mse[:, 1] / mse[:, 2])))
            ),
            "mean_mse_2x_4x_8x": np.mean(mse, axis=0).tolist(),
            "maximum_8x_internal_source_range_excursion": max(
                float(item["atlas_8x"]["internal_source_range_excursion"])
                for item in group
            ),
            "maximum_8x_final_source_range_excursion": max(
                float(item["atlas_8x"]["source_range_excursion"])
                for item in group
            ),
        }
    return {
        "design": {
            "source_side": 25,
            "target_sides": [9, 13, 37, 49, 97],
            "radii": "all seven census radii",
            "angles_degrees": [0, 45, 90, 135],
            "phases_radians": [0.0, math.pi],
            "cases_per_scale": 56,
        },
        "record_count": len(selected),
        "summary": summary,
        "records": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("census", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.census.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
