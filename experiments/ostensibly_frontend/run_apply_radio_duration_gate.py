#!/usr/bin/env python3
"""Apply duration admission to a radio ranking while preserving geometry order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phone_duration_gate import load_phone_duration_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_lattice", type=Path)
    parser.add_argument("duration_gate", type=Path)
    parser.add_argument("--ranking-field", default="top5")
    parser.add_argument("--label-count", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.phone_lattice.read_text())
    gate = load_phone_duration_gate(args.duration_gate)
    for row in document["phones"]:
        duration = float(row["seconds1"]) - float(row["seconds0"])
        row[args.ranking_field] = gate.filter_ranking(
            duration, row[args.ranking_field], args.label_count
        )
    document["method"] = "duration_admission_then_unchanged_geometric_ranking"
    document["duration_gate"] = {
        "inventory": str(args.duration_gate),
        "label_count": args.label_count,
        "ranking_field": args.ranking_field,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.out), "phone_count": len(document["phones"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
