#!/usr/bin/env python3
"""Concatenate compatible compiled support phone atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiled_support_phone_atlas import (
    combine_support_phone_atlases,
    load_support_phone_atlas,
    save_support_phone_atlas,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlases", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combined = combine_support_phone_atlases(
        load_support_phone_atlas(path) for path in args.atlases
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_support_phone_atlas(args.out, combined)
    print(json.dumps({"output": str(args.out), "occurrences": combined.supports.shape[0], "labels": len(set(combined.labels.tolist())), "size_bytes": args.out.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
