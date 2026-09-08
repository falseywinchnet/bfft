#!/usr/bin/env python3
"""Route reference occurrences by duration before unchanged geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import (
    calibration_objective,
    fuse_rank_channels,
    rank_summary,
)
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .occurrence_statistics import rank_routed_occurrence_minimum
from .physical_interval_geometry import load_physical_interval_atlas
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)


def configuration_key(maximum_occurrences: int | None) -> str:
    return "all" if maximum_occurrences is None else f"top:{maximum_occurrences}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_topology_atlas", type=Path)
    parser.add_argument("slt_topology_atlas", type=Path)
    parser.add_argument("bdl_physical_atlas", type=Path)
    parser.add_argument("slt_physical_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--occurrence-counts", default="1,2,4,8,all")
    parser.add_argument("--baseline-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    configurations = tuple(
        None if value == "all" else int(value)
        for value in (item for item in args.occurrence_counts.split(",") if item)
    )
    if (
        not utterances
        or not configurations
        or len(set(configurations)) != len(configurations)
        or any(value is not None and value < 1 for value in configurations)
    ):
        raise ValueError("duration-routed occurrence grid is invalid")
    payload = _payload()
    topology = {
        "bdl": load_conditional_ridge_atlas(args.bdl_topology_atlas),
        "slt": load_conditional_ridge_atlas(args.slt_topology_atlas),
    }
    physical = {
        "bdl": load_physical_interval_atlas(args.bdl_physical_atlas),
        "slt": load_physical_interval_atlas(args.slt_physical_atlas),
    }
    reference_log_durations = {}
    for speaker in ("bdl", "slt"):
        windows = _windows(args.arctic_root, speaker, utterances)
        expected_witnesses = np.asarray(
            [f"{speaker}:{utterance}:{ordinal}" for utterance, ordinal, _ in windows]
        )
        if (
            not np.array_equal(topology[speaker].witnesses.astype(str), expected_witnesses)
            or not np.array_equal(physical[speaker].witnesses.astype(str), expected_witnesses)
        ):
            raise ValueError(f"{speaker} duration route is not aligned to occurrences")
        reference_log_durations[speaker] = np.log(
            np.asarray([window.duration for _, _, window in windows], dtype=np.float64)
        )

    rows = []
    for reference, query in (("bdl", "slt"), ("slt", "bdl")):
        windows = _windows(args.arctic_root, query, utterances)
        for completed, (utterance, ordinal, window) in enumerate(windows, start=1):
            cloud = np.asarray(
                np.load(_cloud_path(args.raw_cache, window, payload)),
                dtype=np.float64,
            )
            topology_distances = topology[reference].occurrence_distances(cloud)
            mass_distances = physical[reference].occurrence_distances_with_weights(
                cloud,
                interval_weight=0.0,
                trajectory_weight=0.0,
                mass_weight=1.0,
            )
            query_duration = float(np.log(window.duration))
            ranks = {}
            winners = {}
            for maximum in configurations:
                topology_ranking = rank_routed_occurrence_minimum(
                    topology[reference].labels,
                    topology[reference].witnesses,
                    topology_distances,
                    reference_log_durations[reference],
                    query_duration,
                    maximum,
                )
                mass_ranking = rank_routed_occurrence_minimum(
                    physical[reference].labels,
                    physical[reference].witnesses,
                    mass_distances,
                    reference_log_durations[reference],
                    query_duration,
                    maximum,
                )
                fused = fuse_rank_channels(
                    topology_ranking,
                    mass_ranking,
                    physical_weight=1.0,
                    policy="geometric",
                )
                key = configuration_key(maximum)
                ranks[key] = next(
                    int(item["rank"])
                    for item in fused
                    if item["phone"] == window.label
                )
                winners[key] = str(fused[0]["phone"])
            rows.append({
                "reference_speaker": reference,
                "query_speaker": query,
                "utterance": utterance,
                "ordinal": ordinal,
                "phone": window.label,
                "duration_seconds": float(window.duration),
                "fused_ranks": ranks,
                "fused_winners": winners,
            })
            if completed % 64 == 0 or completed == len(windows):
                print(json.dumps({
                    "direction": f"{reference}_to_{query}",
                    "completed": completed,
                    "total": len(windows),
                }), flush=True)

    summaries = {
        configuration_key(configuration): rank_summary(
            row["fused_ranks"][configuration_key(configuration)] for row in rows
        )
        for configuration in configurations
    }
    selected = min(
        configurations,
        key=lambda value: (
            calibration_objective(summaries[configuration_key(value)]),
            float("inf") if value is None else value,
        ),
    )
    folds = []
    crossfit_ranks = []
    for held_out in utterances:
        training = [row for row in rows if row["utterance"] != held_out]
        validation = [row for row in rows if row["utterance"] == held_out]
        training_summaries = {
            configuration_key(configuration): rank_summary(
                row["fused_ranks"][configuration_key(configuration)]
                for row in training
            )
            for configuration in configurations
        }
        fold_selection = min(
            configurations,
            key=lambda value: (
                calibration_objective(
                    training_summaries[configuration_key(value)]
                ),
                float("inf") if value is None else value,
            ),
        )
        held_ranks = [
            row["fused_ranks"][configuration_key(fold_selection)]
            for row in validation
        ]
        crossfit_ranks.extend(held_ranks)
        folds.append({
            "held_out_utterance": held_out,
            "selected_maximum_occurrences": fold_selection,
            "validation": rank_summary(held_ranks),
        })

    if args.baseline_audit is not None:
        audit = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
        expected = {
            (
                str(row["reference_speaker"]),
                str(row["query_speaker"]),
                str(row["utterance"]),
                int(row["ordinal"]),
            ): int(row["fused_ranks"]["geometric:1"])
            for row in audit["rows"]
        }
        for row in rows:
            key = (
                row["reference_speaker"],
                row["query_speaker"],
                row["utterance"],
                row["ordinal"],
            )
            if row["fused_ranks"]["all"] != expected[key]:
                raise ValueError(f"unrouted minimum identity failed for {key}")

    selection_counts = {
        configuration_key(configuration): sum(
            fold["selected_maximum_occurrences"] == configuration
            for fold in folds
        )
        for configuration in configurations
    }
    result = {
        "method": "duration_routed_occurrence_minimum_topology_by_mass",
        "selection_status": "duration routes witnesses; it never scores identity",
        "utterances": list(utterances),
        "configurations": [configuration_key(value) for value in configurations],
        "summaries": summaries,
        "selected_maximum_occurrences": selected,
        "selected_summary": summaries[configuration_key(selected)],
        "minimum_control": summaries["all"],
        "leave_one_utterance_out": {
            "selection_counts": selection_counts,
            "summary": rank_summary(crossfit_ranks),
            "folds": folds,
        },
        "query_count": len(rows),
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "selected_maximum_occurrences": selected,
        "selected": result["selected_summary"],
        "minimum_control": result["minimum_control"],
        "leave_one_utterance_out": {
            "selection_counts": {
                key: value for key, value in selection_counts.items() if value
            },
            "summary": result["leave_one_utterance_out"]["summary"],
        },
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
