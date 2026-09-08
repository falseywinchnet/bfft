#!/usr/bin/env python3
"""Cross-audit Cleanup/SHARK-supported occupation extraction floors."""

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
from .conditional_ridge_atlas import compile_conditional_ridge_atlas
from .physical_interval_geometry import compile_physical_interval_atlas
from .run_arctic_calibration import TARGET_SAMPLE_RATE
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)
from .temporal_evidence_reweighting import (
    cleanup_shark_extraction_support,
    temporal_reweight_cloud,
)


def _floors(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("support floors must be decimals") from error
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("support floors must lie in [0, 1]")
    return tuple(sorted(set(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cloud_cache", type=Path)
    parser.add_argument("state_cache", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--floors", type=_floors, default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if not utterances:
        raise ValueError("temporal reweighting calibration requires utterances")
    started = perf_counter()
    payload = _payload()

    records: dict[str, list[dict[str, object]]] = {"bdl": [], "slt": []}
    support_cache: dict[tuple[str, str], np.ndarray] = {}
    for speaker in ("bdl", "slt"):
        for utterance, ordinal, window in _windows(
            args.arctic_root, speaker, utterances
        ):
            support_key = (speaker, utterance)
            if support_key not in support_cache:
                cache_path = (
                    args.state_cache / f"{speaker}_{utterance}_cleanup_shark.npz"
                )
                with np.load(cache_path, allow_pickle=False) as document:
                    support_cache[support_key] = cleanup_shark_extraction_support(
                        document["cleanup_probability"],
                        document["fused_score"],
                        document["unvoiced_score"],
                    )
            full_support = support_cache[support_key]
            centers = (
                np.arange(full_support.size, dtype=np.float64)
                * 512
                / TARGET_SAMPLE_RATE
            )
            indices = np.flatnonzero(
                (centers >= window.seconds0) & (centers < window.seconds1)
            )
            if not indices.size:
                midpoint = 0.5 * (window.seconds0 + window.seconds1)
                indices = np.asarray((int(np.argmin(abs(centers - midpoint))),))
            records[speaker].append(
                {
                    "utterance": utterance,
                    "ordinal": ordinal,
                    "label": window.label,
                    "witness": f"{speaker}:{utterance}:{ordinal}",
                    "cloud_path": _cloud_path(args.raw_cloud_cache, window, payload),
                    "support": full_support[indices],
                }
            )

    atlases = {}
    for floor in args.floors:
        for speaker in ("bdl", "slt"):
            def occurrences():
                for record in records[speaker]:
                    cloud = np.asarray(np.load(record["cloud_path"]), dtype=np.float64)
                    yield (
                        str(record["label"]),
                        str(record["witness"]),
                        temporal_reweight_cloud(
                            cloud,
                            np.asarray(record["support"], dtype=np.float64),
                            support_floor=floor,
                        ),
                    )

            topology = compile_conditional_ridge_atlas(occurrences())
            physical = compile_physical_interval_atlas(occurrences())
            atlases[(floor, speaker)] = (topology, physical)
        print(json.dumps({"compiled_support_floor": floor}), flush=True)

    rows: dict[tuple[str, str], dict[str, object]] = {}
    for floor in args.floors:
        for reference, query in (("bdl", "slt"), ("slt", "bdl")):
            topology, physical = atlases[(floor, reference)]
            for completed, record in enumerate(records[query], start=1):
                cloud = np.asarray(np.load(record["cloud_path"]), dtype=np.float64)
                weighted = temporal_reweight_cloud(
                    cloud,
                    np.asarray(record["support"], dtype=np.float64),
                    support_floor=floor,
                )
                ranking = fuse_rank_channels(
                    topology.rank(weighted),
                    physical.rank_with_weights(
                        weighted,
                        interval_weight=0.0,
                        trajectory_weight=0.0,
                        mass_weight=1.0,
                    ),
                    physical_weight=1.0,
                    policy="geometric",
                )
                target = next(
                    row for row in ranking if row["phone"] == record["label"]
                )
                key = (reference, str(record["witness"]))
                row = rows.setdefault(
                    key,
                    {
                        "reference_speaker": reference,
                        "query_speaker": query,
                        "utterance": record["utterance"],
                        "ordinal": record["ordinal"],
                        "phone": record["label"],
                        "target_ranks": {},
                    },
                )
                row["target_ranks"][f"{floor:g}"] = int(target["rank"])
                if completed % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "support_floor": floor,
                                "reference": reference,
                                "completed_queries": completed,
                                "total_queries": len(records[query]),
                            }
                        ),
                        flush=True,
                    )

    output_rows = list(rows.values())
    summaries = {
        f"{floor:g}": rank_summary(
            int(row["target_ranks"][f"{floor:g}"]) for row in output_rows
        )
        for floor in args.floors
    }
    fold_rows = []
    crossfit_ranks = []
    for heldout in utterances:
        training = [row for row in output_rows if row["utterance"] != heldout]
        chosen = min(
            args.floors,
            key=lambda floor: (
                calibration_objective(
                    rank_summary(
                        int(row["target_ranks"][f"{floor:g}"])
                        for row in training
                    )
                ),
                -floor,
            ),
        )
        heldout_rows = [row for row in output_rows if row["utterance"] == heldout]
        heldout_ranks = [
            int(row["target_ranks"][f"{chosen:g}"]) for row in heldout_rows
        ]
        crossfit_ranks.extend(heldout_ranks)
        fold_rows.append(
            {
                "heldout_utterance": heldout,
                "selected_support_floor": chosen,
                "heldout_summary": rank_summary(heldout_ranks),
            }
        )

    result = {
        "method": "cross_speaker_cleanup_shark_temporal_occupation_reweighting",
        "selection_status": "generic extraction audit; no radio query used",
        "support": "max(cleanup probability, voiced sigmoid, unvoiced score)",
        "floors": list(args.floors),
        "utterances": list(utterances),
        "query_count": len(output_rows),
        "summaries": summaries,
        "leave_one_utterance_out": fold_rows,
        "crossfit_summary": rank_summary(crossfit_ranks),
        "rows": output_rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "summaries": summaries,
                "selected_floors": [
                    row["selected_support_floor"] for row in fold_rows
                ],
                "crossfit_summary": result["crossfit_summary"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
