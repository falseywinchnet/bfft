#!/usr/bin/env python3
"""Compile every frozen topology-by-mass label rank for ARCTIC queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrated_geometry_fusion import fuse_rank_channels
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .phone_rank_matrix import PhoneRankMatrix, complete_rank_vector, save_phone_rank_matrix
from .physical_interval_geometry import load_physical_interval_atlas
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
    _windows,
)
from .word_lattice import ARPABET_39


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("bdl_topology_atlas", type=Path)
    parser.add_argument("slt_topology_atlas", type=Path)
    parser.add_argument("bdl_physical_atlas", type=Path)
    parser.add_argument("slt_physical_atlas", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--topology-occurrence-quantile", type=float, default=0.0)
    parser.add_argument("--mass-occurrence-quantile", type=float, default=0.0)
    parser.add_argument("--target-rank-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if not utterances:
        raise ValueError("phone rank matrix requires utterances")
    labels = tuple(sorted(ARPABET_39))
    payload = _payload()
    topology = {
        "bdl": load_conditional_ridge_atlas(args.bdl_topology_atlas),
        "slt": load_conditional_ridge_atlas(args.slt_topology_atlas),
    }
    physical = {
        "bdl": load_physical_interval_atlas(args.bdl_physical_atlas),
        "slt": load_physical_interval_atlas(args.slt_physical_atlas),
    }
    references = []
    queries = []
    utterance_rows = []
    ordinals = []
    targets = []
    ranks = []
    for reference, query in (("bdl", "slt"), ("slt", "bdl")):
        windows = _windows(args.arctic_root, query, utterances)
        for completed, (utterance, ordinal, window) in enumerate(windows, start=1):
            cloud = np.asarray(
                np.load(_cloud_path(args.raw_cache, window, payload)),
                dtype=np.float64,
            )
            ranking = fuse_rank_channels(
                topology[reference].rank(
                    cloud,
                    occurrence_quantile=args.topology_occurrence_quantile,
                ),
                physical[reference].rank_with_weights(
                    cloud,
                    interval_weight=0.0,
                    trajectory_weight=0.0,
                    mass_weight=1.0,
                    occurrence_quantile=args.mass_occurrence_quantile,
                ),
                physical_weight=1.0,
                policy="geometric",
            )
            references.append(reference)
            queries.append(query)
            utterance_rows.append(utterance)
            ordinals.append(ordinal)
            targets.append(window.label)
            ranks.append(complete_rank_vector(ranking, labels))
            if completed % 64 == 0 or completed == len(windows):
                print(json.dumps({
                    "direction": f"{reference}_to_{query}",
                    "completed": completed,
                    "total": len(windows),
                }), flush=True)
    matrix = PhoneRankMatrix(
        labels=np.asarray(labels),
        reference_speakers=np.asarray(references),
        query_speakers=np.asarray(queries),
        utterances=np.asarray(utterance_rows),
        ordinals=np.asarray(ordinals, dtype=np.int64),
        target_phones=np.asarray(targets),
        ranks=np.stack(ranks),
        provenance={
            "method": "complete_equal_weight_geometric_topology_by_mass_ranks",
            "utterances": list(utterances),
            "occurrence_quantiles": {
                "topology": args.topology_occurrence_quantile,
                "mass": args.mass_occurrence_quantile,
            },
            "physical_components": {"interval": 0.0, "trajectory": 0.0, "mass": 1.0},
            "fusion": {"policy": "geometric", "physical_weight": 1.0},
        },
    )
    if args.target_rank_audit is not None:
        audit = json.loads(args.target_rank_audit.read_text(encoding="utf-8"))
        expected = {
            (
                str(row["reference_speaker"]),
                str(row["query_speaker"]),
                str(row["utterance"]),
                int(row["ordinal"]),
            ): int(row["fused_ranks"]["geometric:1"])
            for row in audit["rows"]
        }
        label_columns = {str(label): i for i, label in enumerate(matrix.labels)}
        for key, row in matrix.index().items():
            target = str(matrix.target_phones[row])
            actual = int(matrix.ranks[row, label_columns[target]])
            if expected.get(key) != actual:
                raise ValueError(f"target-rank identity failed for {key}")
    save_phone_rank_matrix(args.out, matrix)
    print(json.dumps({
        "output": str(args.out),
        "query_count": int(matrix.ranks.shape[0]),
        "label_count": int(matrix.ranks.shape[1]),
        "target_rank_audit": args.target_rank_audit is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
