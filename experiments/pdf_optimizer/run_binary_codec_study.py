#!/usr/bin/env python3
"""Compare PDF-native lossless binary-image payload representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import zlib

import numpy as np
from PIL import Image


def _pages(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    return result


def _tag_sum(image: Image.Image, tag: int) -> int:
    value = image.tag_v2[tag]
    if isinstance(value, int):
        return value
    return sum(int(item) for item in value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-dir", type=Path, required=True)
    parser.add_argument("--generic-dir", type=Path, required=True)
    parser.add_argument("--pages", default="1-468")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"pages": []}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="binary-codec-") as directory:
        work = Path(directory)
        for page in _pages(args.pages):
            source = args.bitmap_dir / f"{page:04d}.pbm"
            with Image.open(source) as opened:
                image = opened.convert("1")
                pixels = np.asarray(image, dtype=bool)
                packed = np.packbits(np.logical_not(pixels), axis=1, bitorder="big").tobytes()
                tiff = work / f"{page:04d}.tif"
                image.save(tiff, compression="group4", rows_per_strip=image.height)
            with Image.open(tiff) as encoded:
                group4 = _tag_sum(encoded, 279)
            item = {
                "page": page,
                "jbig2_generic_bytes": (args.generic_dir / f"{page:04d}.jb2").stat().st_size,
                "ccitt_group4_payload_bytes": group4,
                "flate_packed_bitmap_bytes": len(zlib.compress(packed, 9)),
            }
            report["pages"].append(item)
        for field in ("jbig2_generic_bytes", "ccitt_group4_payload_bytes", "flate_packed_bitmap_bytes"):
            report[field.replace("_bytes", "_total_bytes")] = sum(item[field] for item in report["pages"])
    report["total_seconds"] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key.endswith("total_bytes") or key == "total_seconds"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
