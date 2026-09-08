#!/usr/bin/env python3
"""Combine raw occurrence banks while preserving every speaker witness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generic_word_template_bank import (
    combine_generic_word_template_banks,
    load_generic_word_template_bank,
    save_generic_word_template_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("banks", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bank = combine_generic_word_template_banks(
        load_generic_word_template_bank(path) for path in args.banks
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_generic_word_template_bank(args.out, bank)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "occurrences": int(bank.labels.size),
                "labels": len(set(bank.labels.tolist())),
                "points_per_occurrence": bank.points_per_occurrence,
                "size_bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
