#!/usr/bin/env python3
"""Calibrate topology/physical phone fusion on cross-speaker ARCTIC holds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import (
    calibration_objective,
    rank_summary,
    target_rank,
)
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .physical_interval_geometry import (
    PhysicalIntervalAtlas,
    compile_physical_interval_atlas,
)
from .run_occupation_context_battery import _cache_name, _window_list


DEFAULT_UTTERANCES = (
    "arctic_a0007",
    "arctic_a0014",
    "arctic_a0033",
    "arctic_a0041",
    "arctic_a0052",
    "arctic_a0066",
    "arctic_a0078",
    "arctic_a0088",
    "arctic_a0095",
    "arctic_a0102",
    "arctic_a0108",
    "arctic_a0112",
    "arctic_a0117",
    "arctic_a0120",
)


def _payload() -> dict[str, object]:
    return {
        "rows": 128,
        "noise_db": 8.0,
        "morphology": asdict(DepthmapGeometryConfig()),
        "ridge": asdict(HessianRidgeConfig()),
        "climbers": asdict(CrazyClimberConfig()),
        "point_cloud": asdict(PointCloudConfig(point_count=32768)),
    }


def _cloud_path(cache: Path, window: object, payload: dict[str, object]) -> Path:
    return cache / _cache_name(
        window,
        {
            **payload,
            "key": window.key,
            "representation": "raw_occupation_point_cloud_v1",
        },
    )


def _windows(root: Path, speaker: str, utterances: tuple[str, ...]):
    return tuple(
        (utterance, ordinal, window)
        for utterance in utterances
        for ordinal, window in enumerate(_window_list(root, speaker, utterance))
    )


def _physical_atlas(
    root: Path,
    cache: Path,
    speaker: str,
    utterances: tuple[str, ...],
    payload: dict[str, object],
) -> PhysicalIntervalAtlas:
    occurrences = []
    for utterance, ordinal, window in _windows(root, speaker, utterances):
        cloud = np.asarray(np.load(_cloud_path(cache, window, payload)), dtype=np.float64)
        occurrences.append(
            (window.label, f"{speaker}:{utterance}:{ordinal}", cloud)
        )
    return compile_physical_interval_atlas(occurrences)


def _save_physical_atlas(path: Path, atlas: PhysicalIntervalAtlas) -> None:
    metadata = {
        "maximum_shift": atlas.maximum_shift,
        "trajectory_weight": atlas.trajectory_weight,
        "mass_weight": atlas.mass_weight,
        "row_scale": atlas.row_scale,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        labels=atlas.labels,
        witnesses=atlas.witnesses,
        surfaces=atlas.surfaces,
        masses=atlas.masses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_topology_atlas", type=Path)
    parser.add_argument("slt_topology_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--weights", default="0,0.25,0.5,1,2,4")
    parser.add_argument(
        "--policies", default="arithmetic,geometric,minimum,maximum"
    )
    parser.add_argument("--physical-interval-weight", type=float, default=1.0)
    parser.add_argument("--physical-trajectory-weight", type=float, default=0.5)
    parser.add_argument("--physical-mass-weight", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    weights = tuple(float(value) for value in args.weights.split(",") if value)
    policies = tuple(value for value in args.policies.split(",") if value)
    if not utterances or not weights or any(weight < 0.0 for weight in weights):
        raise ValueError("calibration inventory is empty or invalid")
    physical_component_weights = (
        args.physical_interval_weight,
        args.physical_trajectory_weight,
        args.physical_mass_weight,
    )
    if (
        any(not np.isfinite(value) or value < 0.0 for value in physical_component_weights)
        or sum(physical_component_weights) <= 0.0
    ):
        raise ValueError("physical component weights must be finite and nonzero")
    configurations = tuple(
        (policy, weight)
        for policy in policies
        for weight in (
            (1.0,) if policy in ("minimum", "maximum") else weights
        )
    )
    def configuration_key(policy: str, weight: float) -> str:
        return f"{policy}:{weight:g}"
    payload = _payload()
    topology = {
        "bdl": load_conditional_ridge_atlas(args.bdl_topology_atlas),
        "slt": load_conditional_ridge_atlas(args.slt_topology_atlas),
    }
    physical = {
        speaker: _physical_atlas(
            args.arctic_root, args.raw_cache, speaker, utterances, payload
        )
        for speaker in ("bdl", "slt")
    }
    for speaker in ("bdl", "slt"):
        if not np.array_equal(topology[speaker].witnesses, physical[speaker].witnesses):
            raise ValueError(f"{speaker} physical/topology witnesses are misaligned")

    rows: list[dict[str, object]] = []
    for reference_speaker, query_speaker in (("bdl", "slt"), ("slt", "bdl")):
        query_windows = _windows(args.arctic_root, query_speaker, utterances)
        for completed, (utterance, ordinal, window) in enumerate(
            query_windows, start=1
        ):
            cloud = np.asarray(
                np.load(_cloud_path(args.raw_cache, window, payload)),
                dtype=np.float64,
            )
            topology_ranking = topology[reference_speaker].rank(cloud)
            physical_ranking = physical[reference_speaker].rank_with_weights(
                cloud,
                interval_weight=args.physical_interval_weight,
                trajectory_weight=args.physical_trajectory_weight,
                mass_weight=args.physical_mass_weight,
            )
            topology_target_item = next(
                item
                for item in topology_ranking
                if item["phone"] == window.label
            )
            physical_target_item = next(
                item
                for item in physical_ranking
                if item["phone"] == window.label
            )
            fused = {
                configuration_key(policy, weight): target_rank(
                    topology_ranking,
                    physical_ranking,
                    window.label,
                    weight,
                    policy=policy,
                )
                for policy, weight in configurations
            }
            rows.append(
                {
                    "reference_speaker": reference_speaker,
                    "query_speaker": query_speaker,
                    "utterance": utterance,
                    "ordinal": ordinal,
                    "phone": window.label,
                    "topology_rank": int(topology_target_item["rank"]),
                    "physical_rank": int(physical_target_item["rank"]),
                    "topology_target_distance": float(
                        topology_target_item["distance"]
                    ),
                    "physical_target_distance": float(
                        physical_target_item["distance"]
                    ),
                    "topology_best_distance": float(
                        topology_ranking[0]["distance"]
                    ),
                    "physical_best_distance": float(
                        physical_ranking[0]["distance"]
                    ),
                    "topology_best_margin": float(
                        topology_ranking[1]["distance"]
                        - topology_ranking[0]["distance"]
                    ),
                    "physical_best_margin": float(
                        physical_ranking[1]["distance"]
                        - physical_ranking[0]["distance"]
                    ),
                    "fused_ranks": fused,
                }
            )
            if completed % 64 == 0 or completed == len(query_windows):
                print(
                    json.dumps(
                        {
                            "direction": f"{reference_speaker}_to_{query_speaker}",
                            "completed": completed,
                            "total": len(query_windows),
                        }
                    ),
                    flush=True,
                )

    summaries = {
        configuration_key(policy, weight): rank_summary(
            int(row["fused_ranks"][configuration_key(policy, weight)])
            for row in rows
        )
        for policy, weight in configurations
    }
    selected_policy, selected_weight = min(
        configurations,
        key=lambda item: (
            calibration_objective(
                summaries[configuration_key(item[0], item[1])]
            ),
            item[1],
            item[0],
        ),
    )
    holdout_rows = []
    holdout_ranks = []
    for held_out in utterances:
        training = [row for row in rows if row["utterance"] != held_out]
        validation = [row for row in rows if row["utterance"] == held_out]
        training_summaries = {
            configuration_key(policy, weight): rank_summary(
                int(row["fused_ranks"][configuration_key(policy, weight)])
                for row in training
            )
            for policy, weight in configurations
        }
        fold_policy, fold_weight = min(
            configurations,
            key=lambda item: (
                calibration_objective(
                    training_summaries[configuration_key(item[0], item[1])]
                ),
                item[1],
                item[0],
            ),
        )
        fold_ranks = [
            int(
                row["fused_ranks"][
                    configuration_key(fold_policy, fold_weight)
                ]
            )
            for row in validation
        ]
        holdout_ranks.extend(fold_ranks)
        holdout_rows.append(
            {
                "utterance": held_out,
                "selected_policy": fold_policy,
                "selected_weight": fold_weight,
                "validation": rank_summary(fold_ranks),
            }
        )

    result = {
        "method": "cross_speaker_rank_percentile_topology_physical_fusion",
        "selection_status": "weights selected without radio queries",
        "physical_component_weights": {
            "interval": args.physical_interval_weight,
            "trajectory": args.physical_trajectory_weight,
            "mass": args.physical_mass_weight,
        },
        "utterances": list(utterances),
        "weights": list(weights),
        "policies": list(policies),
        "configurations": [
            {"policy": policy, "weight": weight}
            for policy, weight in configurations
        ],
        "summaries": summaries,
        "selected_policy": selected_policy,
        "selected_weight": selected_weight,
        "leave_one_utterance_out": {
            "summary": rank_summary(holdout_ranks),
            "folds": holdout_rows,
        },
        "topology_only": rank_summary(int(row["topology_rank"]) for row in rows),
        "physical_only": rank_summary(int(row["physical_rank"]) for row in rows),
        "query_count": len(rows),
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _save_physical_atlas(
        args.out.with_name(args.out.stem + "_bdl_physical_atlas.npz"),
        physical["bdl"],
    )
    _save_physical_atlas(
        args.out.with_name(args.out.stem + "_slt_physical_atlas.npz"),
        physical["slt"],
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "selected_weight": selected_weight,
                "selected_policy": selected_policy,
                "topology_only": result["topology_only"],
                "physical_only": result["physical_only"],
                "selected": summaries[
                    configuration_key(selected_policy, selected_weight)
                ],
                "leave_one_utterance_out": result["leave_one_utterance_out"]["summary"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
