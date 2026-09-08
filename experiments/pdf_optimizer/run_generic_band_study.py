#!/usr/bin/env python3
"""Benchmark one generic bitmap versus disjoint cropped generic bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from pdf_optimizer.jbig2 import _patch_cropped_stream, append_cropped_region


def _pages(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    return result


def _encode(encoder: Path, bitmap: Path) -> bytes:
    return subprocess.run(
        [str(encoder), "-p", str(bitmap)], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _regions(ink: np.ndarray, minimum_gap: int):
    occupied = ink.any(axis=1)
    rows = np.flatnonzero(occupied)
    if not len(rows):
        return []
    starts = [int(rows[0])]
    ends = []
    for previous, current in zip(rows[:-1], rows[1:]):
        if current - previous - 1 >= minimum_gap:
            ends.append(int(previous) + 1)
            starts.append(int(current))
    ends.append(int(rows[-1]) + 1)
    result = []
    for y0, y1 in zip(starts, ends):
        columns = np.flatnonzero(ink[y0:y1].any(axis=0))
        result.append((int(columns[0]), y0, int(columns[-1]) + 1, y1))
    return result


def _compose(encoder: Path, ink: np.ndarray, boxes, work: Path, label: str) -> bytes:
    height, width = ink.shape
    stream = None
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        bitmap = work / f"{label}-{index:03d}.pbm"
        crop = ink[y0:y1, x0:x1]
        Image.fromarray(np.where(crop, 0, 255).astype(np.uint8), "L").convert("1").save(bitmap)
        encoded = _encode(encoder, bitmap)
        if stream is None:
            stream = _patch_cropped_stream(
                encoded, page_width=width, page_height=height, x=x0, y=y0
            )
        else:
            # Bands are vertically disjoint, so REPLACE is exact and avoids
            # depending on the page's internal black/white polarity.
            stream = append_cropped_region(
                stream, encoded, x=x0, y=y0, composition_operator=4
            )
    if stream is None:
        raise ValueError("blank page")
    return stream


def _exact(decoder: Path, stream: bytes, source: Path, work: Path, label: str) -> bool:
    encoded = work / f"{label}.jb2"
    decoded = work / f"{label}.pbm"
    encoded.write_bytes(stream)
    subprocess.run(
        [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(decoded), str(encoded)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    with Image.open(source) as left, Image.open(decoded) as right:
        return np.array_equal(np.asarray(left), np.asarray(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--pages", default="1-64")
    parser.add_argument("--gaps", default="8,16,24,32,48,64")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = _pages(args.pages)
    gaps = [int(value) for value in args.gaps.split(",")]
    report = {"pages": [], "gaps": gaps}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="jbig2-band-") as directory:
        work = Path(directory)
        for page in pages:
            source = args.bitmap_dir / f"{page:04d}.pbm"
            with Image.open(source) as image:
                ink = np.logical_not(np.asarray(image.convert("1"), dtype=bool))
            rows, columns = np.nonzero(ink)
            if not len(rows):
                continue
            baseline_box = [(int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)]
            candidates = []
            for name, boxes in [("single", baseline_box)] + [
                (f"gap_{gap}", _regions(ink, gap)) for gap in gaps
            ]:
                stream = _compose(args.encoder, ink, boxes, work, f"p{page}-{name}")
                exact = _exact(args.decoder, stream, source, work, f"verify-{page}-{name}")
                candidates.append({"name": name, "regions": len(boxes), "bytes": len(stream), "pixel_exact": exact})
            best = min((item for item in candidates if item["pixel_exact"]), key=lambda item: item["bytes"])
            item = {"page": page, "candidates": candidates, "best": best,
                    "delta_from_single": best["bytes"] - candidates[0]["bytes"]}
            report["pages"].append(item)
            print(json.dumps({"page": page, "best": best, "delta": item["delta_from_single"]}), flush=True)
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
