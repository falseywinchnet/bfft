#!/usr/bin/env python3
"""Terminal-measure entropy-proposed generic-region partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from pdf_optimizer.entropy_regions import partition_pixel_boxes, propose_entropy_partitions
from pdf_optimizer.jbig2 import _patch_cropped_stream, append_cropped_region


def _pages(value: str):
    result = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    return result


def _encode(encoder: Path, bitmap: Path):
    return subprocess.run([str(encoder), "-p", str(bitmap)], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _compose(encoder: Path, ink, boxes, width, height, work: Path, label: str):
    stream = None
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        bitmap = work / f"{label}-{index}.pbm"
        Image.fromarray(np.where(ink[y0:y1, x0:x1], 0, 255).astype(np.uint8), "L").convert("1").save(bitmap)
        region = _encode(encoder, bitmap)
        if stream is None:
            stream = _patch_cropped_stream(region, page_width=width, page_height=height, x=x0, y=y0)
        else:
            stream = append_cropped_region(stream, region, x=x0, y=y0, composition_operator=4)
    return stream


def _exact(decoder: Path, stream: bytes, source: Path, work: Path, label: str):
    encoded, decoded = work / f"{label}.jb2", work / f"{label}.pbm"
    encoded.write_bytes(stream)
    subprocess.run([str(decoder), "-q", "-e", "-t", "pbm", "-o", str(decoded), str(encoded)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with Image.open(source) as left, Image.open(decoded) as right:
        return np.array_equal(np.asarray(left), np.asarray(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--pages", default="1-64")
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--first-split-frontier", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"pages": [], "cell_size": args.cell_size, "max_regions": args.max_regions}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="entropy-region-") as directory:
        work = Path(directory)
        for page in _pages(args.pages):
            source = args.bitmap_dir / f"{page:04d}.pbm"
            with Image.open(source) as image:
                ink = np.logical_not(np.asarray(image.convert("1"), dtype=bool))
                width, height = image.size
            partitions = propose_entropy_partitions(
                ink, cell_size=args.cell_size, max_regions=args.max_regions,
                first_split_frontier=args.first_split_frontier,
            )
            candidates = []
            for index, partition in enumerate(partitions):
                boxes = partition_pixel_boxes(partition, ink)
                stream = _compose(args.encoder, ink, boxes, width, height, work, f"p{page}-k{index}")
                exact = _exact(args.decoder, stream, source, work, f"verify-{page}-{index}")
                candidates.append({
                    "regions": len(boxes), "bytes": len(stream), "pixel_exact": exact,
                    "estimated_bits": partition.estimated_bits,
                    "last_estimated_gain_bits": partition.last_gain_bits,
                    "boxes": boxes,
                })
            best = min((item for item in candidates if item["pixel_exact"]), key=lambda item: item["bytes"])
            item = {"page": page, "candidates": candidates, "best": best,
                    "delta_from_single": best["bytes"] - candidates[0]["bytes"]}
            report["pages"].append(item)
            print(json.dumps({"page": page, "best": best["regions"], "delta": item["delta_from_single"]}), flush=True)
    report["single_total_bytes"] = sum(item["candidates"][0]["bytes"] for item in report["pages"])
    report["best_total_bytes"] = sum(item["best"]["bytes"] for item in report["pages"])
    report["delta_bytes"] = report["best_total_bytes"] - report["single_total_bytes"]
    report["total_seconds"] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("single_total_bytes", "best_total_bytes", "delta_bytes", "total_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
