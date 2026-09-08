#!/usr/bin/env python3
"""Screen Cleanup frequency gains before unrolled occupation extraction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibrated_geometry_fusion import fuse_rank_channels, rank_summary
from .cleanup_shark_vad import CleanupSharkConfig, analyze_cleanup_shark
from .conditional_ridge_atlas import compile_conditional_ridge_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)
from .occupation_point_cloud import PointCloudConfig
from .physical_interval_geometry import compile_physical_interval_atlas
from .run_arctic_calibration import (
    DEFAULT_UTTERANCES as LEGACY_CACHE_UTTERANCES,
    HOP_LENGTH,
    TARGET_SAMPLE_RATE,
)
from .run_calibrate_geometry_fusion import (
    DEFAULT_UTTERANCES,
    _cloud_path,
    _payload,
)
from .run_occupation_cloud_battery import (
    _load_channel_audio,
    _trace_to_cloud,
    _window_trace,
)
from .run_occupation_context_battery import _window_list
from .spectral_gain_transport import transported_spectral_gain


LABEL_COMPLETE_SCREEN = (
    "arctic_a0007",
    "arctic_a0041",
    "arctic_a0095",
    "arctic_a0117",
    "arctic_a0120",
)


def _floors(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("gain floors must be decimals") from error
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("gain floors must lie in [0, 1]")
    return tuple(sorted(set(values)))


def _legacy_cache_seed_index(speaker: str, utterance: str) -> int:
    """Recover the seed schedule omitted from the v1 raw-cloud cache key."""

    if speaker == "bdl" and utterance in LEGACY_CACHE_UTTERANCES:
        return LEGACY_CACHE_UTTERANCES.index(utterance)
    return DEFAULT_UTTERANCES.index(utterance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("raw_cloud_cache", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--gain-floors", type=_floors, default=(0.0, 0.5, 1.0))
    parser.add_argument(
        "--gain-mode",
        choices=("global", "voiced-only"),
        default="global",
    )
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--voice-certifier-strength", type=float, default=0.0)
    parser.add_argument("--structure-certifier-strength", type=float, default=0.125)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if (
        not utterances
        or any(value not in DEFAULT_UTTERANCES for value in utterances)
        or not np.isfinite(args.noise_db)
        or not 0.0 <= args.voice_certifier_strength <= 0.25
        or not 0.0 <= args.structure_certifier_strength <= 0.25
    ):
        raise ValueError("spectral-gain screen configuration is invalid")

    started = perf_counter()
    morphology = DepthmapGeometryConfig()
    ridge = HessianRidgeConfig()
    climbers = CrazyClimberConfig()
    cloud_config = PointCloudConfig(point_count=32768)
    payload = _payload()
    cleanup_config = CleanupSharkConfig(
        cleanup_voice_certifier_strength=args.voice_certifier_strength,
        cleanup_structure_certifier_strength=args.structure_certifier_strength,
    )
    rows: dict[tuple[str, str], dict[str, object]] = {}
    summaries = {}
    gain_statistics = {}
    exact_baseline_clouds = 0
    query_count = 0
    if args.resume and args.out.exists():
        checkpoint = json.loads(args.out.read_text(encoding="utf-8"))
        summaries.update(checkpoint.get("summaries", {}))
        gain_statistics.update(checkpoint.get("gain_statistics", {}))
        for row in checkpoint.get("rows", []):
            witness = (
                f"{row['query_speaker']}:{row['utterance']}:{row['ordinal']}"
            )
            rows[(str(row["reference_speaker"]), witness)] = dict(row)

    for floor in args.gain_floors:
        if f"{floor:g}" in summaries:
            continue
        speaker_occurrences = {}
        floor_gain_values = []
        for speaker in ("bdl", "slt"):
            occurrences = []
            for completed, utterance in enumerate(utterances, start=1):
                utterance_index = _legacy_cache_seed_index(speaker, utterance)
                samples = _load_channel_audio(
                    args.arctic_root,
                    speaker,
                    utterance,
                    utterance_index,
                    args.noise_db,
                )
                analysis = analyze_cleanup_shark(
                    samples,
                    TARGET_SAMPLE_RATE,
                    config=cleanup_config,
                )
                effective_gain = transported_spectral_gain(
                    analysis.spectral_gain,
                    analysis.state,
                    gain_floor=floor,
                    mode=args.gain_mode,
                )
                floor_gain_values.append(effective_gain.reshape(-1))
                densified = fourier_unrolled_densified(
                    samples,
                    apertures=DEFAULT_APERTURES,
                    target_n_fft=2048,
                    target_crop_rows=256,
                    hop_length=HOP_LENGTH,
                    spectral_gain=effective_gain,
                )
                baseline = reassigned_texture_baseline(
                    densified.fused, densified.support_union
                ).merged
                for ordinal, window in enumerate(
                    _window_list(args.arctic_root, speaker, utterance)
                ):
                    trace, _ = _window_trace(
                        baseline, window.seconds0, window.seconds1, 128
                    )
                    cloud = _trace_to_cloud(
                        trace, morphology, ridge, climbers, cloud_config
                    )
                    witness = f"{speaker}:{utterance}:{ordinal}"
                    if floor == 1.0:
                        cached = np.asarray(
                            np.load(_cloud_path(args.raw_cloud_cache, window, payload))
                        )
                        if not np.array_equal(cloud.astype(np.float32), cached):
                            raise ValueError(
                                "all-ones gain failed cached cloud identity for "
                                f"{speaker}:{utterance}:{ordinal} under seed index "
                                f"{utterance_index}"
                            )
                        exact_baseline_clouds += 1
                    occurrences.append((window.label, witness, cloud, utterance, ordinal))
                print(
                    json.dumps(
                        {
                            "gain_floor": floor,
                            "speaker": speaker,
                            "completed_utterances": completed,
                            "total_utterances": len(utterances),
                            "phone_clouds": len(occurrences),
                        }
                    ),
                    flush=True,
                )
            speaker_occurrences[speaker] = occurrences

        atlases = {}
        for speaker in ("bdl", "slt"):
            compact = [
                (label, witness, cloud)
                for label, witness, cloud, _, _ in speaker_occurrences[speaker]
            ]
            atlases[speaker] = (
                compile_conditional_ridge_atlas(compact),
                compile_physical_interval_atlas(compact),
            )

        floor_ranks = []
        for reference, query in (("bdl", "slt"), ("slt", "bdl")):
            topology, physical = atlases[reference]
            for label, witness, cloud, utterance, ordinal in speaker_occurrences[query]:
                ranking = fuse_rank_channels(
                    topology.rank(cloud),
                    physical.rank_with_weights(
                        cloud,
                        interval_weight=0.0,
                        trajectory_weight=0.0,
                        mass_weight=1.0,
                    ),
                    physical_weight=1.0,
                    policy="geometric",
                )
                target = next(row for row in ranking if row["phone"] == label)
                rank = int(target["rank"])
                floor_ranks.append(rank)
                key = (reference, witness)
                row = rows.setdefault(
                    key,
                    {
                        "reference_speaker": reference,
                        "query_speaker": query,
                        "utterance": utterance,
                        "ordinal": ordinal,
                        "phone": label,
                        "target_ranks": {},
                    },
                )
                row["target_ranks"][f"{floor:g}"] = rank
        query_count = len(floor_ranks)
        summaries[f"{floor:g}"] = rank_summary(floor_ranks)
        gains = np.concatenate(floor_gain_values)
        gain_statistics[f"{floor:g}"] = {
            "minimum": float(np.min(gains)),
            "q25": float(np.quantile(gains, 0.25)),
            "median": float(np.median(gains)),
            "q75": float(np.quantile(gains, 0.75)),
            "maximum": float(np.max(gains)),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "method": "cleanup_frequency_gain_before_multiscale_unrolled_extraction",
                    "selection_status": "partial checkpoint; no radio query used",
                    "gain_mode": args.gain_mode,
                    "cleanup_shark_config": asdict(cleanup_config),
                    "utterances": list(utterances),
                    "completed_gain_floors": [
                        value
                        for value in args.gain_floors
                        if f"{value:g}" in summaries
                    ],
                    "summaries": summaries,
                    "gain_statistics": gain_statistics,
                    "rows": list(rows.values()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        del speaker_occurrences, atlases

    result = {
        "method": "cleanup_frequency_gain_before_multiscale_unrolled_extraction",
        "selection_status": "label-complete cross-speaker screen; no radio query used",
        "utterances": list(utterances),
        "label_complete_screen": tuple(utterances) == LABEL_COMPLETE_SCREEN,
        "gain_floors": list(args.gain_floors),
        "gain_mode": args.gain_mode,
        "cleanup_shark_config": asdict(cleanup_config),
        "channel_seed_schedule": {
            speaker: {
                utterance: _legacy_cache_seed_index(speaker, utterance)
                for utterance in utterances
            }
            for speaker in ("bdl", "slt")
        },
        "channel_seed_provenance": (
            "legacy v1 BDL raw-cloud cache reused five-sentence entries; SLT "
            "was built under the expanded schedule; seed was absent from key"
        ),
        "query_count": query_count,
        "summaries": summaries,
        "gain_statistics": gain_statistics,
        "exact_floor_one_cached_clouds": exact_baseline_clouds,
        "geometry": {
            "morphology": asdict(morphology),
            "ridge": asdict(ridge),
            "climbers": asdict(climbers),
            "point_cloud": asdict(cloud_config),
            "fusion": "equal-weight geometric topology by occupation mass",
        },
        "rows": list(rows.values()),
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "query_count": query_count,
                "summaries": summaries,
                "gain_statistics": gain_statistics,
                "exact_floor_one_cached_clouds": exact_baseline_clouds,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
