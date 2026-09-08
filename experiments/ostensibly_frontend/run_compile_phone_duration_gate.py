#!/usr/bin/env python3
"""Compile a robust duration-proposal inventory from labeled ARCTIC phones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phone_duration_gate import compile_phone_duration_gate, save_phone_duration_gate
from .run_occupation_context_battery import _window_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arctic_root", type=Path)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--minimum-scale", type=float, default=0.12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    occurrences = [
        (window.label, window.duration)
        for utterance in (value for value in args.utterances.split(",") if value)
        for window in _window_list(args.arctic_root, args.speaker, utterance)
    ]
    gate = compile_phone_duration_gate(occurrences, args.minimum_scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_phone_duration_gate(args.out, gate)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "occurrences": len(occurrences),
                "labels": len(gate.label_statistics()),
                "size_bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
