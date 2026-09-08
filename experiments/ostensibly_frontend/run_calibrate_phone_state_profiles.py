#!/usr/bin/env python3
"""Compile and cross-audit Cleanup/SHARK phone state-profile atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .calibrated_geometry_fusion import rank_summary
from .cleanup_shark_vad import CleanupSharkConfig, analyze_cleanup_shark
from .phone_rank_coverage import fit_phone_rank_coverage
from .phone_state_profile import (
    FEATURE_NAMES,
    compile_phone_state_profile_atlas,
    phone_state_features,
)
from .run_calibrate_geometry_fusion import DEFAULT_UTTERANCES, _windows
from .run_arctic_calibration import TARGET_SAMPLE_RATE
from .synthetic_vowels import apply_ssb_probe_channel
from .uncertainty_fusion import _as_float_audio


ARRAY_KEYS = (
    "pitch_periodicity",
    "spectral_prominence_db",
    "cleanup_probability",
    "fused_score",
    "unvoiced_score",
    "energy_snr_db",
    "state",
)


def _analysis_arrays(
    root: Path,
    speaker: str,
    utterance: str,
    utterance_index: int,
    noise_db: float,
    cache: Path | None,
) -> dict[str, np.ndarray]:
    cache_path = (
        cache / f"{speaker}_{utterance}_cleanup_shark.npz"
        if cache is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as document:
            return {key: np.asarray(document[key]) for key in ARRAY_KEYS}
    sample_rate, raw = wavfile.read(root / speaker / "wav" / f"{utterance}.wav")
    samples = _as_float_audio(raw)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        samples = resample_poly(samples, TARGET_SAMPLE_RATE, int(sample_rate))
    samples = apply_ssb_probe_channel(
        samples,
        sample_rate=TARGET_SAMPLE_RATE,
        noise_db=noise_db,
        seed=10_000 * (1 + utterance_index) + sum(map(ord, speaker)),
    )
    analysis = analyze_cleanup_shark(
        samples,
        TARGET_SAMPLE_RATE,
        config=CleanupSharkConfig(),
    )
    values = {key: np.asarray(getattr(analysis, key)) for key in ARRAY_KEYS}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **values)
    return values


def _speaker_occurrences(
    root: Path,
    speaker: str,
    utterances: tuple[str, ...],
    noise_db: float,
    cache: Path | None,
) -> tuple[tuple[str, str, np.ndarray], ...]:
    output = []
    config = CleanupSharkConfig()
    for utterance_index, utterance in enumerate(utterances):
        arrays = _analysis_arrays(
            root,
            speaker,
            utterance,
            utterance_index,
            noise_db,
            cache,
        )
        frame_count = arrays["state"].size
        centers_seconds = (
            np.arange(frame_count, dtype=np.float64)
            * config.hop_length
            / TARGET_SAMPLE_RATE
        )
        for _, ordinal, window in _windows(root, speaker, (utterance,)):
            indices = np.flatnonzero(
                (centers_seconds >= window.seconds0)
                & (centers_seconds < window.seconds1)
            )
            if not indices.size:
                midpoint = 0.5 * (window.seconds0 + window.seconds1)
                indices = np.asarray((int(np.argmin(abs(centers_seconds - midpoint))),))
            features = phone_state_features(
                *(arrays[key][indices] for key in ARRAY_KEYS)
            )
            output.append(
                (
                    window.label,
                    f"{speaker}:{utterance}:{ordinal}",
                    features,
                )
            )
        print(
            json.dumps(
                {
                    "speaker": speaker,
                    "completed_utterances": utterance_index + 1,
                    "total_utterances": len(utterances),
                    "phone_profiles": len(output),
                }
            ),
            flush=True,
        )
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--utterances", default=",".join(DEFAULT_UTTERANCES))
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    utterances = tuple(value for value in args.utterances.split(",") if value)
    if not utterances or not np.isfinite(args.noise_db):
        raise ValueError("state-profile calibration configuration is invalid")
    occurrences = {
        speaker: _speaker_occurrences(
            args.arctic_root,
            speaker,
            utterances,
            args.noise_db,
            args.cache,
        )
        for speaker in ("bdl", "slt")
    }
    atlases = {
        speaker: compile_phone_state_profile_atlas(
            (label, features) for label, _, features in values
        )
        for speaker, values in occurrences.items()
    }
    combined_atlas = compile_phone_state_profile_atlas(
        (label, features)
        for speaker in ("bdl", "slt")
        for label, _, features in occurrences[speaker]
    )
    if atlases["bdl"].labels != atlases["slt"].labels:
        raise ValueError("state-profile speaker atlases have different labels")
    rows = []
    for reference_speaker, query_speaker in (("bdl", "slt"), ("slt", "bdl")):
        for label, witness, features in occurrences[query_speaker]:
            ranking = atlases[reference_speaker].rank(features)
            target = next(row for row in ranking if row["phone"] == label)
            rows.append(
                {
                    "reference_speaker": reference_speaker,
                    "query_speaker": query_speaker,
                    "witness": witness,
                    "phone": label,
                    "target_rank": int(target["rank"]),
                    "target_distance": float(target["distance"]),
                    "best_phone": str(ranking[0]["phone"]),
                    "best_distance": float(ranking[0]["distance"]),
                }
            )
    ranks = [int(row["target_rank"]) for row in rows]
    coverage = fit_phone_rank_coverage(
        ranks,
        len(atlases["bdl"].labels),
        provenance={"channel": "cleanup_shark_phone_state_profile"},
    )
    bdl_path = args.out.with_name(args.out.stem + "_bdl_atlas.npz")
    slt_path = args.out.with_name(args.out.stem + "_slt_atlas.npz")
    coverage_path = args.out.with_name(args.out.stem + "_coverage.npz")
    combined_path = args.out.with_name(args.out.stem + "_combined_atlas.npz")
    atlases["bdl"].save(bdl_path)
    atlases["slt"].save(slt_path)
    coverage.save(coverage_path)
    combined_atlas.save(combined_path)
    result = {
        "method": "cross_speaker_cleanup_shark_phone_state_profiles",
        "selection_status": "proposal channel only; no radio queries used",
        "utterances": list(utterances),
        "noise_db": args.noise_db,
        "feature_names": list(FEATURE_NAMES),
        "labels": list(atlases["bdl"].labels),
        "query_count": len(rows),
        "summary": rank_summary(ranks),
        "coverage": [coverage.coverage(rank) for rank in range(1, coverage.label_count + 1)],
        "minimum_rank_for_conservative_coverage": {
            str(target): coverage.minimum_rank_for_lower_coverage(target)
            for target in (0.5, 0.8, 0.9, 0.95)
        },
        "atlases": {
            "bdl": str(bdl_path),
            "slt": str(slt_path),
            "combined": str(combined_path),
        },
        "coverage_atlas": str(coverage_path),
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "summary": result["summary"],
                "minimum_rank_for_conservative_coverage": result[
                    "minimum_rank_for_conservative_coverage"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
