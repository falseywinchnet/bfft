#!/usr/bin/env python3
"""Measure joint duration and Cleanup/SHARK proposal coverage cross-speaker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phone_duration_gate import load_phone_duration_gate
from .phone_rank_coverage import wilson_interval
from .run_calibrate_geometry_fusion import DEFAULT_UTTERANCES, _windows


def _pairs(raw: str) -> tuple[tuple[int, int], ...]:
    try:
        values = tuple(
            tuple(int(part) for part in item.split(":"))
            for item in raw.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("proposal depths must be D:S pairs") from error
    if not values or any(len(value) != 2 or min(value) < 1 for value in values):
        raise argparse.ArgumentTypeError("proposal depths must be positive D:S pairs")
    return tuple((value[0], value[1]) for value in values)


def _coverage(successes: int, count: int) -> dict[str, float | int]:
    lower, upper = wilson_interval(successes, count)
    return {
        "successes": successes,
        "count": count,
        "estimate": successes / count,
        "lower_95": lower,
        "upper_95": upper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("state_profile_audit", type=Path)
    parser.add_argument("bdl_duration_gate", type=Path)
    parser.add_argument("slt_duration_gate", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--depths", type=_pairs, default=((24, 23), (24, 28)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if not utterances:
        raise ValueError("proposal calibration requires utterances")

    state_audit = json.loads(args.state_profile_audit.read_text(encoding="utf-8"))
    gates = {
        "bdl": load_phone_duration_gate(args.bdl_duration_gate),
        "slt": load_phone_duration_gate(args.slt_duration_gate),
    }
    windows = {
        f"{speaker}:{utterance}:{ordinal}": window
        for speaker in ("bdl", "slt")
        for utterance, ordinal, window in _windows(
            args.arctic_root, speaker, utterances
        )
    }
    rows = []
    for state_row in state_audit["rows"]:
        witness = str(state_row["witness"])
        window = windows.get(witness)
        if window is None or window.label != str(state_row["phone"]):
            raise ValueError("state profile and duration windows do not align")
        reference = str(state_row["reference_speaker"])
        duration_order = gates[reference].rank_labels(window.duration)
        rows.append(
            {
                "reference_speaker": reference,
                "query_speaker": str(state_row["query_speaker"]),
                "witness": witness,
                "phone": window.label,
                "duration_rank": duration_order.index(window.label) + 1,
                "state_rank": int(state_row["target_rank"]),
            }
        )

    experiments = []
    count = len(rows)
    for duration_depth, state_depth in args.depths:
        duration_successes = sum(
            int(row["duration_rank"]) <= duration_depth for row in rows
        )
        state_successes = sum(int(row["state_rank"]) <= state_depth for row in rows)
        joint_successes = sum(
            int(row["duration_rank"]) <= duration_depth
            and int(row["state_rank"]) <= state_depth
            for row in rows
        )
        union_successes = sum(
            int(row["duration_rank"]) <= duration_depth
            or int(row["state_rank"]) <= state_depth
            for row in rows
        )
        duration_coverage = _coverage(duration_successes, count)
        state_coverage = _coverage(state_successes, count)
        joint_coverage = _coverage(joint_successes, count)
        union_coverage = _coverage(union_successes, count)
        experiments.append(
            {
                "duration_depth": duration_depth,
                "state_depth": state_depth,
                "duration_coverage": duration_coverage,
                "state_coverage": state_coverage,
                "joint_coverage": joint_coverage,
                "union_coverage": union_coverage,
                "independence_product": (
                    float(duration_coverage["estimate"])
                    * float(state_coverage["estimate"])
                ),
            }
        )

    result = {
        "method": "cross_speaker_duration_cleanup_shark_proposal_intersection",
        "selection_status": "coverage audit only; no radio query or identity score",
        "state_profile_audit": str(args.state_profile_audit),
        "duration_gates": {
            "bdl": str(args.bdl_duration_gate),
            "slt": str(args.slt_duration_gate),
        },
        "utterances": list(utterances),
        "query_count": count,
        "experiments": experiments,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "rows": f"{count} measured rows"}, indent=2))


if __name__ == "__main__":
    main()
