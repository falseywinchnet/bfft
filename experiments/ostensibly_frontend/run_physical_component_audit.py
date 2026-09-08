#!/usr/bin/env python3
"""Select physical phone evidence components on cross-speaker holds only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import calibration_objective, rank_summary
from .physical_interval_geometry import load_physical_interval_atlas
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)


CONFIGURATIONS = {
    "interval": (1.0, 0.0, 0.0),
    "trajectory": (0.0, 1.0, 0.0),
    "mass": (0.0, 0.0, 1.0),
    "interval_trajectory": (1.0, 0.5, 0.0),
    "interval_mass": (1.0, 0.0, 0.25),
    "trajectory_mass": (0.0, 1.0, 0.5),
    "all": (1.0, 0.5, 0.25),
}


def _target(ranking: list[dict[str, object]], label: str) -> dict[str, object]:
    return next(row for row in ranking if row["phone"] == label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_physical_atlas", type=Path)
    parser.add_argument("slt_physical_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if not utterances:
        raise ValueError("component audit requires utterances")
    payload = _payload()
    atlases = {
        "bdl": load_physical_interval_atlas(args.bdl_physical_atlas),
        "slt": load_physical_interval_atlas(args.slt_physical_atlas),
    }
    rows: list[dict[str, object]] = []
    for reference_speaker, query_speaker in (("bdl", "slt"), ("slt", "bdl")):
        windows = _windows(args.arctic_root, query_speaker, utterances)
        for completed, (utterance, ordinal, window) in enumerate(windows, start=1):
            cloud = np.asarray(
                np.load(_cloud_path(args.raw_cache, window, payload)),
                dtype=np.float64,
            )
            evidence: dict[str, dict[str, object]] = {}
            for name, (interval, trajectory, mass) in CONFIGURATIONS.items():
                ranking = atlases[reference_speaker].rank_with_weights(
                    cloud,
                    interval_weight=interval,
                    trajectory_weight=trajectory,
                    mass_weight=mass,
                )
                target = _target(ranking, window.label)
                evidence[name] = {
                    "rank": int(target["rank"]),
                    "target_distance": float(target["distance"]),
                    "best_distance": float(ranking[0]["distance"]),
                    "best_phone": str(ranking[0]["phone"]),
                }
            rows.append(
                {
                    "reference_speaker": reference_speaker,
                    "query_speaker": query_speaker,
                    "utterance": utterance,
                    "ordinal": ordinal,
                    "phone": window.label,
                    "evidence": evidence,
                }
            )
            if completed % 64 == 0 or completed == len(windows):
                print(
                    json.dumps(
                        {
                            "direction": f"{reference_speaker}_to_{query_speaker}",
                            "completed": completed,
                            "total": len(windows),
                        }
                    ),
                    flush=True,
                )

    summaries = {
        name: rank_summary(int(row["evidence"][name]["rank"]) for row in rows)
        for name in CONFIGURATIONS
    }
    selected = min(
        CONFIGURATIONS,
        key=lambda name: (calibration_objective(summaries[name]), name),
    )
    folds = []
    held_out_ranks = []
    for utterance in utterances:
        training = [row for row in rows if row["utterance"] != utterance]
        validation = [row for row in rows if row["utterance"] == utterance]
        training_summaries = {
            name: rank_summary(
                int(row["evidence"][name]["rank"]) for row in training
            )
            for name in CONFIGURATIONS
        }
        fold_selected = min(
            CONFIGURATIONS,
            key=lambda name: (
                calibration_objective(training_summaries[name]),
                name,
            ),
        )
        ranks = [
            int(row["evidence"][fold_selected]["rank"]) for row in validation
        ]
        held_out_ranks.extend(ranks)
        folds.append(
            {
                "utterance": utterance,
                "selected": fold_selected,
                "validation": rank_summary(ranks),
            }
        )

    per_phone = {}
    for phone in sorted({str(row["phone"]) for row in rows}):
        phone_rows = [row for row in rows if row["phone"] == phone]
        per_phone[phone] = {
            name: rank_summary(
                int(row["evidence"][name]["rank"]) for row in phone_rows
            )
            for name in CONFIGURATIONS
        }
    result = {
        "method": "cross_speaker_physical_evidence_component_ablation",
        "selection_status": "selected without radio queries",
        "configurations": {
            name: {
                "interval_weight": values[0],
                "trajectory_weight": values[1],
                "mass_weight": values[2],
            }
            for name, values in CONFIGURATIONS.items()
        },
        "summaries": summaries,
        "selected": selected,
        "selected_weights": {
            "interval": CONFIGURATIONS[selected][0],
            "trajectory": CONFIGURATIONS[selected][1],
            "mass": CONFIGURATIONS[selected][2],
        },
        "leave_one_utterance_out": {
            "summary": rank_summary(held_out_ranks),
            "folds": folds,
        },
        "per_phone": per_phone,
        "query_count": len(rows),
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "selected": selected,
                "selected_summary": summaries[selected],
                "leave_one_utterance_out": result["leave_one_utterance_out"][
                    "summary"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
