#!/usr/bin/env python3
"""Audit label-balanced occurrence-distribution phone geometry."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr

from .calibrated_geometry_fusion import fuse_rank_channels, rank_summary
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .occurrence_statistics import (
    rank_occurrence_corrected_minimum,
    rank_occurrence_dominance,
)
from .physical_interval_geometry import load_physical_interval_atlas
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)


def _target_rank(ranking: list[dict[str, object]], target: str) -> int:
    return next(int(row["rank"]) for row in ranking if row["phone"] == target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_topology_atlas", type=Path)
    parser.add_argument("slt_topology_atlas", type=Path)
    parser.add_argument("bdl_physical_atlas", type=Path)
    parser.add_argument("slt_physical_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument(
        "--statistic",
        choices=("dominance", "corrected_minimum"),
        default="dominance",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if not utterances:
        raise ValueError("occurrence dominance audit needs utterances")
    payload = _payload()
    rank_occurrences = {
        "dominance": rank_occurrence_dominance,
        "corrected_minimum": rank_occurrence_corrected_minimum,
    }[args.statistic]
    topology = {
        "bdl": load_conditional_ridge_atlas(args.bdl_topology_atlas),
        "slt": load_conditional_ridge_atlas(args.slt_topology_atlas),
    }
    physical = {
        "bdl": load_physical_interval_atlas(args.bdl_physical_atlas),
        "slt": load_physical_interval_atlas(args.slt_physical_atlas),
    }
    rows = []
    for reference, query in (("bdl", "slt"), ("slt", "bdl")):
        if not np.array_equal(topology[reference].witnesses, physical[reference].witnesses):
            raise ValueError(f"{reference} occurrence atlases are misaligned")
        windows = _windows(args.arctic_root, query, utterances)
        for completed, (utterance, ordinal, window) in enumerate(windows, start=1):
            cloud = np.asarray(
                np.load(_cloud_path(args.raw_cache, window, payload)),
                dtype=np.float64,
            )
            topology_ranking = rank_occurrences(
                topology[reference].labels,
                topology[reference].witnesses,
                topology[reference].occurrence_distances(cloud),
            )
            mass_ranking = rank_occurrences(
                physical[reference].labels,
                physical[reference].witnesses,
                physical[reference].occurrence_distances_with_weights(
                    cloud,
                    interval_weight=0.0,
                    trajectory_weight=0.0,
                    mass_weight=1.0,
                ),
            )
            fused = fuse_rank_channels(
                topology_ranking,
                mass_ranking,
                physical_weight=1.0,
                policy="geometric",
            )
            rows.append({
                "reference_speaker": reference,
                "query_speaker": query,
                "utterance": utterance,
                "ordinal": ordinal,
                "phone": window.label,
                "topology_rank": _target_rank(topology_ranking, window.label),
                "mass_rank": _target_rank(mass_ranking, window.label),
                "fused_rank": _target_rank(fused, window.label),
                "topology_winner": str(topology_ranking[0]["phone"]),
                "mass_winner": str(mass_ranking[0]["phone"]),
                "fused_winner": str(fused[0]["phone"]),
            })
            if completed % 64 == 0 or completed == len(windows):
                print(json.dumps({
                    "direction": f"{reference}_to_{query}",
                    "completed": completed,
                    "total": len(windows),
                }), flush=True)

    count_bias = {}
    for reference in ("bdl", "slt"):
        selected = [row for row in rows if row["reference_speaker"] == reference]
        counts = Counter(map(str, topology[reference].labels))
        labels = sorted(counts)
        channel_rows = {}
        for channel in ("topology", "mass", "fused"):
            winners = Counter(
                row[f"{channel}_winner"]
                for row in selected
                if row[f"{channel}_winner"] != row["phone"]
            )
            correlation = spearmanr(
                [counts[label] for label in labels],
                [winners[label] for label in labels],
            )
            channel_rows[channel] = {
                "spearman_rho": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            }
        count_bias[reference] = channel_rows

    result = {
        "method": f"occurrence_{args.statistic}_topology_by_mass",
        "statistic": args.statistic,
        "selection_status": "parameter-free generic audit; no radio queries",
        "utterances": list(utterances),
        "topology": rank_summary(row["topology_rank"] for row in rows),
        "mass": rank_summary(row["mass_rank"] for row in rows),
        "fused": rank_summary(row["fused_rank"] for row in rows),
        "reference_count_false_winner_correlation": count_bias,
        "query_count": len(rows),
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "topology": result["topology"],
        "mass": result["mass"],
        "fused": result["fused"],
        "reference_count_false_winner_correlation": count_bias,
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
