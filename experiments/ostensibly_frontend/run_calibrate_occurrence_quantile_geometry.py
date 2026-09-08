#!/usr/bin/env python3
"""Calibrate label-balanced occurrence statistics for phone geometry."""

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
from .occurrence_statistics import rank_occurrence_quantiles
from .physical_interval_geometry import load_physical_interval_atlas
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)


def configuration_key(topology_quantile: float, mass_quantile: float) -> str:
    return f"topology:{topology_quantile:g}|mass:{mass_quantile:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_topology_atlas", type=Path)
    parser.add_argument("slt_topology_atlas", type=Path)
    parser.add_argument("bdl_physical_atlas", type=Path)
    parser.add_argument("slt_physical_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--quantiles", default="0,0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--baseline-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    quantiles = tuple(float(value) for value in args.quantiles.split(",") if value)
    if (
        not utterances
        or not quantiles
        or len(set(quantiles)) != len(quantiles)
        or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in quantiles)
    ):
        raise ValueError("occurrence quantile calibration grid is invalid")
    configurations = tuple((top, mass) for top in quantiles for mass in quantiles)
    payload = _payload()
    topology = {
        "bdl": load_conditional_ridge_atlas(args.bdl_topology_atlas),
        "slt": load_conditional_ridge_atlas(args.slt_topology_atlas),
    }
    physical = {
        "bdl": load_physical_interval_atlas(args.bdl_physical_atlas),
        "slt": load_physical_interval_atlas(args.slt_physical_atlas),
    }
    for speaker in ("bdl", "slt"):
        if not np.array_equal(topology[speaker].witnesses, physical[speaker].witnesses):
            raise ValueError(f"{speaker} topology/mass occurrences are misaligned")

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
            topology_rankings = {
                quantile: rank_occurrence_quantiles(
                    topology[reference].labels,
                    topology[reference].witnesses,
                    topology_distances,
                    quantile,
                )
                for quantile in quantiles
            }
            mass_rankings = {
                quantile: rank_occurrence_quantiles(
                    physical[reference].labels,
                    physical[reference].witnesses,
                    mass_distances,
                    quantile,
                )
                for quantile in quantiles
            }
            fused_ranks = {}
            fused_winners = {}
            for top_quantile, mass_quantile in configurations:
                key = configuration_key(top_quantile, mass_quantile)
                ranking = fuse_rank_channels(
                    topology_rankings[top_quantile],
                    mass_rankings[mass_quantile],
                    physical_weight=1.0,
                    policy="geometric",
                )
                fused_ranks[key] = next(
                    int(item["rank"])
                    for item in ranking
                    if item["phone"] == window.label
                )
                fused_winners[key] = str(ranking[0]["phone"])
            rows.append({
                "reference_speaker": reference,
                "query_speaker": query,
                "utterance": utterance,
                "ordinal": ordinal,
                "phone": window.label,
                "fused_ranks": fused_ranks,
                "fused_winners": fused_winners,
            })
            if completed % 64 == 0 or completed == len(windows):
                print(json.dumps({
                    "direction": f"{reference}_to_{query}",
                    "completed": completed,
                    "total": len(windows),
                }), flush=True)

    summaries = {
        configuration_key(*configuration): rank_summary(
            int(row["fused_ranks"][configuration_key(*configuration)])
            for row in rows
        )
        for configuration in configurations
    }
    selected = min(
        configurations,
        key=lambda item: (
            calibration_objective(summaries[configuration_key(*item)]),
            item,
        ),
    )
    folds = []
    crossfit_ranks = []
    for held_out in utterances:
        training = [row for row in rows if row["utterance"] != held_out]
        validation = [row for row in rows if row["utterance"] == held_out]
        training_summaries = {
            configuration_key(*configuration): rank_summary(
                int(row["fused_ranks"][configuration_key(*configuration)])
                for row in training
            )
            for configuration in configurations
        }
        fold_selection = min(
            configurations,
            key=lambda item: (
                calibration_objective(training_summaries[configuration_key(*item)]),
                item,
            ),
        )
        held_ranks = [
            int(row["fused_ranks"][configuration_key(*fold_selection)])
            for row in validation
        ]
        crossfit_ranks.extend(held_ranks)
        folds.append({
            "held_out_utterance": held_out,
            "selected_topology_quantile": fold_selection[0],
            "selected_mass_quantile": fold_selection[1],
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
        baseline_key = configuration_key(0.0, 0.0)
        for row in rows:
            key = (
                str(row["reference_speaker"]),
                str(row["query_speaker"]),
                str(row["utterance"]),
                int(row["ordinal"]),
            )
            if int(row["fused_ranks"][baseline_key]) != expected[key]:
                raise ValueError(f"minimum-statistic identity failed for {key}")

    selection_counts = {
        configuration_key(*configuration): sum(
            fold["selected_topology_quantile"] == configuration[0]
            and fold["selected_mass_quantile"] == configuration[1]
            for fold in folds
        )
        for configuration in configurations
    }
    result = {
        "method": "label_balanced_occurrence_quantile_topology_by_mass",
        "selection_status": "quantiles selected without radio queries",
        "utterances": list(utterances),
        "quantiles": list(quantiles),
        "fusion": {"policy": "geometric", "physical_weight": 1.0},
        "summaries": summaries,
        "selected_topology_quantile": selected[0],
        "selected_mass_quantile": selected[1],
        "selected_summary": summaries[configuration_key(*selected)],
        "minimum_control": summaries[configuration_key(0.0, 0.0)],
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
        "selected": {
            "topology_quantile": selected[0],
            "mass_quantile": selected[1],
            "summary": result["selected_summary"],
        },
        "minimum_control": result["minimum_control"],
        "leave_one_utterance_out": {
            "selection_counts": {
                key: count for key, count in selection_counts.items() if count
            },
            "summary": result["leave_one_utterance_out"]["summary"],
        },
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
