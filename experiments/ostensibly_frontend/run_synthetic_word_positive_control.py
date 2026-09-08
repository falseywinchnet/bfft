#!/usr/bin/env python3
"""Rank a radio word patch against whole-word macOS speech realizations.

This is a positive-control inventory, not a proposed recognition dependency.
It asks whether complete word coarticulation closes the domain gap that remains
when generic words are assembled from independently recorded phone fragments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.io import wavfile

from .conditional_ridge_atlas import (
    conditional_ridge_distance,
    conditional_ridge_signature,
)
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .fourier_unrolled_densified import (
    DEFAULT_APERTURES,
    fourier_unrolled_densified,
    reassigned_texture_baseline,
)
from .generic_word_template_bank import (
    load_generic_word_template_bank,
    word_signature_distance,
)
from .occupation_point_cloud import PointCloudConfig
from .run_occupation_cloud_battery import _trace_to_cloud
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks
from .synthetic_vowels import apply_ssb_probe_channel


def _as_float(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        return values.astype(np.float64) / max(abs(info.min), info.max)
    return values.astype(np.float64)


def _synthesize(text: str, voice: str, rate: int, directory: Path) -> np.ndarray:
    stem = f"item-{abs(hash((text, voice, rate)))}"
    aiff = directory / f"{stem}.aiff"
    wav = directory / f"{stem}.wav"
    subprocess.run(
        ("say", "-v", voice, "-r", str(rate), "-o", str(aiff), text),
        check=True,
    )
    subprocess.run(
        (
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@48000",
            str(aiff),
            str(wav),
        ),
        check=True,
    )
    sample_rate, samples = wavfile.read(wav)
    if int(sample_rate) != 48_000:
        raise ValueError("synthetic positive control must be 48 kHz")
    return _as_float(samples)


def _synthetic_trace(
    text: str,
    voice: str,
    rate: int,
    noise_db: float,
    seed: int,
    directory: Path,
) -> np.ndarray:
    samples = apply_ssb_probe_channel(
        _synthesize(text, voice, rate, directory),
        noise_db=noise_db,
        seed=seed,
    )
    densified = fourier_unrolled_densified(
        samples,
        apertures=DEFAULT_APERTURES,
        target_n_fft=2048,
        hop_length=512,
        target_crop_rows=256,
    )
    baseline = reassigned_texture_baseline(
        densified.fused, densified.support_union
    ).merged
    return np.asarray(baseline[:128], dtype=np.float64)


def _representative(words: list[str]) -> str:
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return min(words, key=lambda word: (len(word), word))
    return max(words, key=lambda word: (zipf_frequency(word, "en"), -len(word), word))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("region_clouds", type=Path)
    parser.add_argument("generic_word_bank", type=Path)
    parser.add_argument("--phone0", type=int, required=True)
    parser.add_argument("--phone1", type=int, required=True)
    parser.add_argument("--target-phones", required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument(
        "--candidate-length",
        type=int,
        help="Restrict the positive-control inventory to one phone length",
    )
    parser.add_argument("--voice", default="Daniel")
    parser.add_argument("--rate", type=int, default=180)
    parser.add_argument("--noise-db", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    document = json.loads(args.lattice.read_text(encoding="utf-8"))
    edge = next(
        item
        for item in document["edges"]
        if (int(item["phone0"]), int(item["phone1"]))
        == (args.phone0, args.phone1)
    )
    classes = {
        tuple(item["phones"]): item
        for item in edge["ambiguity_classes"]
        if args.candidate_length is None
        or len(item["phones"]) == args.candidate_length
    }
    target = tuple(value for value in args.target_phones.split(",") if value)
    if args.candidate_length is not None and len(target) != args.candidate_length:
        raise ValueError("target disagrees with requested candidate length")
    with np.load(args.region_clouds, allow_pickle=False) as clouds_document:
        frame0 = np.asarray(clouds_document["frame0"])
        frame1 = np.asarray(clouds_document["frame1"])
    with np.load(args.recording, allow_pickle=False) as recording_document:
        trace = np.asarray(recording_document["trace_field"], dtype=np.float64)
    query_trace = trace[
        :128,
        int(frame0[args.phone0]) : int(frame1[args.phone1 - 1]),
    ]
    query_cloud = _trace_to_cloud(
        query_trace,
        DepthmapGeometryConfig(),
        HessianRidgeConfig(),
        CrazyClimberConfig(),
        PointCloudConfig(point_count=32768),
    )
    query = conditional_ridge_signature(
        lane_chunks(joint_gauge_chunks((query_cloud,), 8192)), 64, 24
    )
    bank = load_generic_word_template_bank(args.generic_word_bank)
    generic = []
    for phones in classes:
        signatures = bank.word_signatures(
            phones, 4, 64, 24, context_policy="unconditional"
        )
        generic.append(
            (
                word_signature_distance(query, signatures, 4, "mean"),
                phones,
            )
        )
    generic.sort()
    selected = [phones for _, phones in generic[: args.candidate_limit]]
    if target not in selected:
        selected.append(target)
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ostensibly-synthetic-word-") as temporary:
        directory = Path(temporary)
        for ordinal, phones in enumerate(selected):
            if phones == target:
                text = args.target_text
            else:
                text = _representative([str(word) for word in classes[phones]["words"]])
            trace_field = _synthetic_trace(
                text,
                args.voice,
                args.rate,
                args.noise_db,
                20260826 + ordinal,
                directory,
            )
            cloud = _trace_to_cloud(
                trace_field,
                DepthmapGeometryConfig(),
                HessianRidgeConfig(),
                CrazyClimberConfig(),
                PointCloudConfig(point_count=32768),
            )
            signature = conditional_ridge_signature(
                lane_chunks(joint_gauge_chunks((cloud,), 8192)), 64, 24
            )
            rows.append(
                {
                    "phones": list(phones),
                    "text": text,
                    "distance": conditional_ridge_distance(
                        query[0], query[1], signature[0], signature[1], 4
                    ),
                    "surface": signature[0].astype(np.float32),
                    "mass": signature[1].astype(np.float32),
                    "trace": trace_field.astype(np.float32),
                }
            )
            print(
                json.dumps(
                    {
                        "completed": ordinal + 1,
                        "total": len(selected),
                        "text": text,
                    }
                ),
                flush=True,
            )
    rows.sort(key=lambda row: (float(row["distance"]), row["phones"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    target_row = next(row for row in rows if tuple(row["phones"]) == target)
    surfaces = np.stack([row.pop("surface") for row in rows])
    masses = np.stack([row.pop("mass") for row in rows])
    traces = [row.pop("trace") for row in rows]
    trace_offsets = np.concatenate(
        ([0], np.cumsum([value.shape[1] for value in traces]))
    ).astype(np.int64)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        query_trace=np.asarray(query_trace, dtype=np.float32),
        query_surface=query[0].astype(np.float32),
        query_mass=query[1].astype(np.float32),
        surfaces=surfaces,
        masses=masses,
        trace_fields=np.concatenate(traces, axis=1),
        trace_offsets=trace_offsets,
    )
    output = {
        "method": "whole_word_macos_speech_positive_control",
        "voice": args.voice,
        "rate": args.rate,
        "noise_db": args.noise_db,
        "phone_span": [args.phone0, args.phone1],
        "frame_span": [int(frame0[args.phone0]), int(frame1[args.phone1 - 1])],
        "candidate_source": (
            f"top_{args.candidate_limit}_generic_whole_trace"
            + (
                ""
                if args.candidate_length is None
                else f"_length_{args.candidate_length}"
            )
        ),
        "target": target_row,
        "ranking": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
