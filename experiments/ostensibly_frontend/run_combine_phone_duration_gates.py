#!/usr/bin/env python3
"""Combine duration-proposal occurrence inventories without losing samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phone_duration_gate import (
    combine_phone_duration_gates,
    load_phone_duration_gate,
    save_phone_duration_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gates", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gate = combine_phone_duration_gates(
        load_phone_duration_gate(path) for path in args.gates
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_phone_duration_gate(args.out, gate)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "occurrences": int(gate.labels.size),
                "labels": len(gate.label_statistics()),
                "size_bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
