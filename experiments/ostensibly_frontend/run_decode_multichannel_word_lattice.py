#!/usr/bin/env python3
"""Decode independent route and generic-geometry sentence hypotheses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from .sentence_decoder import decode_sentence_paths, path_display


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word_lattice", type=Path)
    parser.add_argument(
        "--channels",
        default=(
            "route,generic_word:unconditional,"
            "generic_word:maximum,generic_word:mixed"
        ),
    )
    parser.add_argument("--path-count", type=int, default=16)
    parser.add_argument("--word-penalty", type=float, default=0.12)
    parser.add_argument(
        "--generic-score-mode",
        choices=("per_edge_robust", "raw", "length_cdf"),
        default="length_cdf",
    )
    parser.add_argument("--unknown-phone-penalty", type=float, default=1.5)
    parser.add_argument("--comma-gap-seconds", type=float, default=0.20)
    parser.add_argument("--period-gap-seconds", type=float, default=0.45)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    channels = tuple(
        value.strip() for value in args.channels.split(",") if value.strip()
    )
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("sentence evidence channels are empty or repeated")
    document = json.loads(args.word_lattice.read_text())
    edges = document.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("word lattice has no lexical edges")
    phone_count = max(int(edge["phone1"]) for edge in edges)
    started = perf_counter()
    channel_paths = {}
    for channel in channels:
        paths = decode_sentence_paths(
            edges,
            phone_count,
            path_count=args.path_count,
            word_penalty=args.word_penalty,
            unknown_phone_penalty=args.unknown_phone_penalty,
            evidence_channel=channel,
            generic_score_mode=args.generic_score_mode,
        )
        channel_paths[channel] = [
            {
                "rank": rank,
                "cost": asdict(path.cost),
                "display": path_display(
                    path,
                    comma_gap_seconds=args.comma_gap_seconds,
                    period_gap_seconds=args.period_gap_seconds,
                ),
                "segments": [asdict(segment) for segment in path.segments],
            }
            for rank, path in enumerate(paths, start=1)
        ]
    result = {
        "method": "independent_evidence_channel_k_best_lexical_dags",
        "selection_status": (
            "channel hypotheses retained independently; no cross-channel fusion"
        ),
        "word_lattice": str(args.word_lattice),
        "parameters": {
            "channels": list(channels),
            "path_count": args.path_count,
            "word_penalty": args.word_penalty,
            "generic_score_mode": args.generic_score_mode,
            "unknown_phone_penalty": args.unknown_phone_penalty,
            "comma_gap_seconds": args.comma_gap_seconds,
            "period_gap_seconds": args.period_gap_seconds,
        },
        "phone_count": phone_count,
        "edge_count": len(edges),
        "elapsed_seconds": perf_counter() - started,
        "channels": channel_paths,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "elapsed_seconds": result["elapsed_seconds"],
                "best": {
                    channel: (paths[0]["display"] if paths else None)
                    for channel, paths in channel_paths.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
