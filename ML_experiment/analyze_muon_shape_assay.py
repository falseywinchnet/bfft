#!/usr/bin/env python3
"""Summarize paired Muon, transported-Muon, Lepton, and Anchor runs."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ARMS = (
    "muon", "muon_transport", "lepton", "lepton_transport", "anchor",
)
METRICS = ("learning_auc", "validation_score", "score", "tail_score", "seconds")


def finite_mean(rows, key):
    values = []
    for row in rows:
        value = row.get(key, row.get("score") if key == "tail_score" else None)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return statistics.fmean(values) if values else float("nan")


def first_at(history, target):
    return next(
        (
            int(point["step"])
            for point in history
            if math.isfinite(float(point["score"]))
            and float(point["score"]) >= target - 1e-12
        ),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    runs = payload["runs"]
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["task"], row["variant"], row["optimizer"]].append(row)

    overall = []
    for arm in ARMS:
        rows = [row for row in runs if row["optimizer"] == arm]
        overall.append({
            "optimizer": arm,
            "runs": len(rows),
            **{metric: finite_mean(rows, metric) for metric in METRICS},
            "failures": sum(row.get("status") != "complete" for row in rows),
        })

    pairs = []
    seed_deltas = defaultdict(list)
    target_speedups = defaultdict(list)
    tasks = list(dict.fromkeys(row["task"] for row in runs))
    variants = list(dict.fromkeys(row["variant"] for row in runs))
    for task in tasks:
        for variant in variants:
            arm_rows = {
                arm: sorted(
                    grouped[task, variant, arm], key=lambda row: row["seed"]
                )
                for arm in ARMS
            }
            if not all(arm_rows.values()):
                continue
            pair = {"task": task, "variant": variant}
            for arm, rows in arm_rows.items():
                pair[arm] = {
                    metric: finite_mean(rows, metric) for metric in METRICS
                }
            for candidate in ("muon_transport", "lepton", "lepton_transport"):
                pair[f"{candidate}_auc_delta"] = (
                    pair[candidate]["learning_auc"] - pair["muon"]["learning_auc"]
                )
                pair[f"{candidate}_validation_delta"] = (
                    pair[candidate]["validation_score"]
                    - pair["muon"]["validation_score"]
                )
            pair["lepton_transport_over_lepton_auc_delta"] = (
                pair["lepton_transport"]["learning_auc"]
                - pair["lepton"]["learning_auc"]
            )
            pair["lepton_transport_over_lepton_validation_delta"] = (
                pair["lepton_transport"]["validation_score"]
                - pair["lepton"]["validation_score"]
            )

            by_seed = {
                arm: {int(row["seed"]): row for row in rows}
                for arm, rows in arm_rows.items()
            }
            for seed, muon_row in by_seed["muon"].items():
                target = float(muon_row["validation_score"])
                muon_step = first_at(muon_row["history"], target)
                for candidate in ("muon_transport", "lepton", "lepton_transport"):
                    candidate_row = by_seed[candidate][seed]
                    seed_deltas[f"{candidate}_auc"].append(
                        float(candidate_row["learning_auc"])
                        - float(muon_row["learning_auc"])
                    )
                    seed_deltas[f"{candidate}_validation"].append(
                        float(candidate_row["validation_score"]) - target
                    )
                    candidate_step = first_at(candidate_row["history"], target)
                    if muon_step is not None and candidate_step is not None:
                        target_speedups[candidate].append(muon_step / candidate_step)
                lepton_row = by_seed["lepton"][seed]
                transported_lepton_row = by_seed["lepton_transport"][seed]
                seed_deltas["lepton_transport_over_lepton_auc"].append(
                    float(transported_lepton_row["learning_auc"])
                    - float(lepton_row["learning_auc"])
                )
                seed_deltas["lepton_transport_over_lepton_validation"].append(
                    float(transported_lepton_row["validation_score"])
                    - float(lepton_row["validation_score"])
                )
                lepton_target = float(lepton_row["validation_score"])
                lepton_step = first_at(lepton_row["history"], lepton_target)
                transported_step = first_at(
                    transported_lepton_row["history"], lepton_target
                )
                if lepton_step is not None and transported_step is not None:
                    target_speedups["lepton_transport_over_lepton"].append(
                        lepton_step / transported_step
                    )
            pairs.append(pair)

    comparisons = {}
    for candidate in ("muon_transport", "lepton", "lepton_transport"):
        auc = seed_deltas[f"{candidate}_auc"]
        validation = seed_deltas[f"{candidate}_validation"]
        speeds = target_speedups[candidate]
        comparisons[candidate] = {
            "paired_runs": len(auc),
            "auc_wins": sum(value > 0 for value in auc),
            "auc_mean_delta": statistics.fmean(auc),
            "validation_wins": sum(value > 0 for value in validation),
            "validation_mean_delta": statistics.fmean(validation),
            "muon_final_target_reached": len(speeds),
            "median_speedup_to_muon_final": (
                statistics.median(speeds) if speeds else None
            ),
        }
    direct_auc = seed_deltas["lepton_transport_over_lepton_auc"]
    direct_validation = seed_deltas[
        "lepton_transport_over_lepton_validation"
    ]
    direct_speeds = target_speedups["lepton_transport_over_lepton"]
    comparisons["lepton_transport_over_lepton"] = {
        "paired_runs": len(direct_auc),
        "auc_wins": sum(value > 0 for value in direct_auc),
        "auc_mean_delta": statistics.fmean(direct_auc),
        "validation_wins": sum(value > 0 for value in direct_validation),
        "validation_mean_delta": statistics.fmean(direct_validation),
        "lepton_final_target_reached": len(direct_speeds),
        "median_speedup_to_lepton_final": (
            statistics.median(direct_speeds) if direct_speeds else None
        ),
    }

    result = {
        "source": str(args.results),
        "configuration": payload["configuration"],
        "arms": list(ARMS),
        "overall": overall,
        "comparisons": comparisons,
        "pairs": pairs,
    }
    out = args.out or args.results.with_name("muon_shape_summary.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "overall": overall,
        "comparisons": comparisons,
        "pair_deltas": [
            {
                key: value
                for key, value in pair.items()
                if key in {
                    "task", "variant", "muon_transport_auc_delta",
                    "lepton_auc_delta", "lepton_transport_auc_delta",
                    "muon_transport_validation_delta", "lepton_validation_delta",
                    "lepton_transport_validation_delta",
                    "lepton_transport_over_lepton_auc_delta",
                    "lepton_transport_over_lepton_validation_delta",
                }
            }
            for pair in pairs
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
