#!/usr/bin/env python3
"""Decode acoustic-only k-best sentence paths from a saved word lattice."""

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
    parser.add_argument("--path-count", type=int, default=16)
    parser.add_argument("--word-penalty", type=float, default=0.12)
    parser.add_argument("--unknown-phone-penalty", type=float, default=1.5)
    parser.add_argument("--comma-gap-seconds", type=float, default=0.20)
    parser.add_argument("--period-gap-seconds", type=float, default=0.45)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.word_lattice.read_text())
    edges = document.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("word lattice has no lexical edges")
    phone_count = max(int(edge["phone1"]) for edge in edges)
    started = perf_counter()
    paths = decode_sentence_paths(
        edges,
        phone_count,
        path_count=args.path_count,
        word_penalty=args.word_penalty,
        unknown_phone_penalty=args.unknown_phone_penalty,
    )
    elapsed = perf_counter() - started
    serialized = []
    for rank, path in enumerate(paths, start=1):
        serialized.append(
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
        )
    result = {
        "method": "acoustic_only_k_best_lexical_dag",
        "selection_status": "homophone_ambiguity_preserved; no language model",
        "word_lattice": str(args.word_lattice),
        "parameters": {
            "path_count": args.path_count,
            "word_penalty": args.word_penalty,
            "unknown_phone_penalty": args.unknown_phone_penalty,
            "comma_gap_seconds": args.comma_gap_seconds,
            "period_gap_seconds": args.period_gap_seconds,
        },
        "phone_count": phone_count,
        "edge_count": len(edges),
        "elapsed_seconds": elapsed,
        "paths": serialized,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "phone_count": phone_count,
                "edge_count": len(edges),
                "path_count": len(paths),
                "elapsed_seconds": elapsed,
                "best": (
                    {
                        "cost": serialized[0]["cost"],
                        "display": serialized[0]["display"],
                        "segment_count": len(serialized[0]["segments"]),
                    }
                    if serialized
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
