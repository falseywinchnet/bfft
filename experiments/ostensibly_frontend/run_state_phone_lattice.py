#!/usr/bin/env python3
"""Build a coverage-bounded phone DAG from Cleanup/SHARK state crops."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import fuse_rank_channels
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .occurrence_statistics import union_occurrence_proposals
from .phone_rank_coverage import load_phone_rank_coverage
from .phone_duration_gate import load_phone_duration_gate
from .phone_state_profile import (
    load_phone_state_profile_atlas,
    phone_state_features,
)
from .physical_interval_geometry import (
    combine_physical_interval_atlases,
    load_physical_interval_atlas,
)
from .run_segmental_pronunciation_ablation import _trace_to_cloud
from .state_phone_lattice import bounded_lattice_spans, state_speech_crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("speech_state", type=Path)
    parser.add_argument("topology_atlas", type=Path)
    parser.add_argument("physical_atlas", type=Path, nargs="+")
    parser.add_argument("rank_coverage", type=Path)
    parser.add_argument(
        "--balanced-rank-coverage",
        type=Path,
        help="Coverage law for an optional balanced occurrence-tail proposal",
    )
    parser.add_argument(
        "--balanced-topology-occurrence-quantile",
        type=float,
        help="Frozen topology quantile for a separate proposal channel",
    )
    parser.add_argument(
        "--duration-gate",
        type=Path,
        help="Optional robust duration inventory; annotates candidates without reordering",
    )
    parser.add_argument(
        "--state-profile-atlas",
        type=Path,
        help="Optional Cleanup/SHARK manner atlas; annotates candidates without reordering",
    )
    parser.add_argument("--crop0", type=int, default=0)
    parser.add_argument("--crop1", type=int)
    parser.add_argument("--minimum-state-run-frames", type=int, default=3)
    parser.add_argument("--maximum-compartments-per-phone", type=int, default=3)
    parser.add_argument("--coverage-lower-target", type=float, default=0.95)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    if (
        args.crop0 < 0
        or args.minimum_state_run_frames < 1
        or args.maximum_compartments_per_phone < 1
        or not 0.0 < args.coverage_lower_target < 1.0
        or args.points < 1
        or min(args.sample_rate, args.hop_length) < 1
    ):
        raise ValueError("state phone lattice configuration is invalid")
    if (args.balanced_rank_coverage is None) != (
        args.balanced_topology_occurrence_quantile is None
    ):
        raise ValueError("balanced proposal needs both quantile and coverage law")
    with np.load(args.recording, allow_pickle=False) as document:
        trace = np.asarray(document["trace_field"], dtype=np.float64)
    with np.load(args.speech_state, allow_pickle=False) as document:
        state = np.asarray(document["state"], dtype=np.int8)
        speech_mask = np.asarray(document["speech_mask"], dtype=bool)
        state_evidence = {
            key: np.asarray(document[key])
            for key in (
                "pitch_periodicity",
                "spectral_prominence_db",
                "cleanup_probability",
                "fused_score",
                "unvoiced_score",
                "energy_snr_db",
                "state",
            )
        }
    if trace.ndim != 2 or trace.shape[1] != state.size:
        raise ValueError("recording and speech state have incompatible frames")
    crops = state_speech_crops(
        state,
        speech_mask,
        minimum_run_frames=args.minimum_state_run_frames,
    )
    crop1 = len(crops) if args.crop1 is None else args.crop1
    if not args.crop0 < crop1 <= len(crops):
        raise ValueError("requested crop range is outside measured speech")
    selected_crops = crops[args.crop0:crop1]
    topology = load_conditional_ridge_atlas(args.topology_atlas)
    physical = combine_physical_interval_atlases(
        load_physical_interval_atlas(path) for path in args.physical_atlas
    )
    coverage = load_phone_rank_coverage(args.rank_coverage)
    balanced_coverage = (
        load_phone_rank_coverage(args.balanced_rank_coverage)
        if args.balanced_rank_coverage is not None
        else None
    )
    duration_gate = (
        load_phone_duration_gate(args.duration_gate)
        if args.duration_gate is not None
        else None
    )
    state_profile_atlas = (
        load_phone_state_profile_atlas(args.state_profile_atlas)
        if args.state_profile_atlas is not None
        else None
    )
    labels = tuple(sorted(set(topology.labels.astype(str))))
    if coverage.label_count != len(labels) or set(physical.labels.astype(str)) != set(labels):
        raise ValueError("lattice atlases and coverage law have different inventories")
    if state_profile_atlas is not None and set(state_profile_atlas.labels) != set(labels):
        raise ValueError("state-profile and phone atlases have different inventories")
    retained_phone_count = coverage.minimum_rank_for_lower_coverage(
        args.coverage_lower_target
    )
    balanced_retained_phone_count = (
        balanced_coverage.minimum_rank_for_lower_coverage(
            args.coverage_lower_target
        )
        if balanced_coverage is not None
        else None
    )
    if balanced_coverage is not None and balanced_coverage.label_count != len(labels):
        raise ValueError("balanced coverage law has a different phone inventory")
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climber = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=args.points)
    output_crops = []
    completed = 0
    total = sum(
        len(
            bounded_lattice_spans(
                crop.landmarks,
                maximum_compartments=args.maximum_compartments_per_phone,
            )
        )
        for crop in selected_crops
    )
    for crop in selected_crops:
        edges = []
        spans = bounded_lattice_spans(
            crop.landmarks,
            maximum_compartments=args.maximum_compartments_per_phone,
        )
        for node0, node1, frame0, frame1 in spans:
            cloud = _trace_to_cloud(
                trace[:128, frame0:frame1],
                morphology,
                ridge,
                climber,
                cloud_config,
            )
            topology_ranking = topology.rank(cloud)
            mass_ranking = physical.rank_with_weights(
                cloud,
                interval_weight=0.0,
                trajectory_weight=0.0,
                mass_weight=1.0,
            )
            ranking = fuse_rank_channels(
                topology_ranking,
                mass_ranking,
                physical_weight=1.0,
                policy="geometric",
            )
            topology_by_phone = {
                str(row["phone"]): row for row in topology_ranking
            }
            mass_by_phone = {str(row["phone"]): row for row in mass_ranking}
            for row in ranking:
                phone = str(row["phone"])
                row["topology_rank"] = int(topology_by_phone[phone]["rank"])
                row["mass_rank"] = int(mass_by_phone[phone]["rank"])
            annotated = coverage.annotate(ranking)
            if balanced_coverage is not None:
                balanced_topology = topology.rank(
                    cloud,
                    occurrence_quantile=args.balanced_topology_occurrence_quantile,
                )
                balanced_ranking = fuse_rank_channels(
                    balanced_topology,
                    mass_ranking,
                    physical_weight=1.0,
                    policy="geometric",
                )
                annotated = union_occurrence_proposals(
                    annotated,
                    balanced_coverage.annotate(balanced_ranking),
                    minimum_depth=retained_phone_count,
                    balanced_depth=balanced_retained_phone_count,
                )
            else:
                annotated = annotated[:retained_phone_count]
            if duration_gate is not None:
                duration = (frame1 - frame0) * args.hop_length / args.sample_rate
                duration_ranks = {
                    phone: rank
                    for rank, phone in enumerate(
                        duration_gate.rank_labels(duration), start=1
                    )
                }
                for row in annotated:
                    row["duration_rank"] = duration_ranks[str(row["phone"])]
            if state_profile_atlas is not None:
                features = phone_state_features(
                    *(state_evidence[key][frame0:frame1] for key in (
                        "pitch_periodicity",
                        "spectral_prominence_db",
                        "cleanup_probability",
                        "fused_score",
                        "unvoiced_score",
                        "energy_snr_db",
                        "state",
                    ))
                )
                state_ranking = state_profile_atlas.rank(features)
                state_by_phone = {
                    str(row["phone"]): row for row in state_ranking
                }
                for row in annotated:
                    state_row = state_by_phone[str(row["phone"])]
                    row["state_profile_rank"] = int(state_row["rank"])
                    row["state_profile_distance"] = float(state_row["distance"])
            edges.append(
                {
                    "node0": node0,
                    "node1": node1,
                    "frame0": frame0,
                    "frame1": frame1,
                    "seconds0": frame0 * args.hop_length / args.sample_rate,
                    "seconds1": frame1 * args.hop_length / args.sample_rate,
                    "candidates": annotated,
                }
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "completed_edges": completed,
                        "total_edges": total,
                        "crop": crop.crop_index,
                        "span": [node0, node1],
                    }
                ),
                flush=True,
            )
        output_crops.append(
            {
                "crop_index": crop.crop_index,
                "frame0": crop.frame0,
                "frame1": crop.frame1,
                "seconds0": crop.frame0 * args.hop_length / args.sample_rate,
                "seconds1": crop.frame1 * args.hop_length / args.sample_rate,
                "landmarks": list(crop.landmarks),
                "state_audit": list(crop.state_audit),
                "edges": edges,
            }
        )
    result = {
        "method": "cleanup_shark_state_phone_dag_geometric_topology_mass",
        "selection_status": "label-free radio inference; geometry and coverage frozen cross-speaker",
        "recording": str(args.recording),
        "speech_state": str(args.speech_state),
        "topology_atlas": str(args.topology_atlas),
        "physical_atlases": [str(path) for path in args.physical_atlas],
        "rank_coverage": str(args.rank_coverage),
        "balanced_rank_coverage": (
            str(args.balanced_rank_coverage)
            if args.balanced_rank_coverage is not None
            else None
        ),
        "duration_gate": (
            str(args.duration_gate) if args.duration_gate is not None else None
        ),
        "state_profile_atlas": (
            str(args.state_profile_atlas)
            if args.state_profile_atlas is not None
            else None
        ),
        "crop_range": [args.crop0, crop1],
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "maximum_compartments_per_phone": args.maximum_compartments_per_phone,
        "coverage_lower_target": args.coverage_lower_target,
        "retained_phone_count": retained_phone_count,
        "retained_coverage": coverage.coverage(retained_phone_count),
        "balanced_topology_occurrence_quantile": (
            args.balanced_topology_occurrence_quantile
        ),
        "balanced_retained_phone_count": balanced_retained_phone_count,
        "balanced_retained_coverage": (
            balanced_coverage.coverage(balanced_retained_phone_count)
            if balanced_coverage is not None
            else None
        ),
        "geometry": {
            "topology_mass_fusion": "equal-weight geometric rank percentile",
            "physical_component": "normalized temporal occupation mass",
            "morphology": asdict(morphology),
            "ridge": asdict(ridge),
            "climbers": asdict(climber),
            "point_cloud": asdict(cloud_config),
        },
        "crop_count": len(output_crops),
        "edge_count": completed,
        "crops": output_crops,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "crop_count": result["crop_count"],
                "edge_count": result["edge_count"],
                "retained_phone_count": retained_phone_count,
                "retained_coverage": result["retained_coverage"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
