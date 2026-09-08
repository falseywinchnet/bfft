#!/usr/bin/env python3
"""Measure aligned XOR refinements as a two-dimensional JBIG2 sheet."""

from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
from pathlib import Path
import subprocess
import tempfile
import zlib

import numpy as np
from PIL import Image


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 128:
            return value, offset
        shift += 7


def _refinements(path: Path) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    data = path.read_bytes()
    if not data.startswith(b"GCF1"):
        raise ValueError("expected a GCF1 fixed-refinement atlas")
    raw = zlib.decompress(data[4:])
    offset = 0
    page_count, offset = _varint(raw, offset)
    for _ in range(page_count):
        _height, offset = _varint(raw, offset)
        _width, offset = _varint(raw, offset)
    glyph_count, offset = _varint(raw, offset)
    shapes = []
    for _ in range(glyph_count):
        height, offset = _varint(raw, offset)
        width, offset = _varint(raw, offset)
        byte_count, offset = _varint(raw, offset)
        offset += byte_count
        shapes.append((height, width))
    occurrence_count, offset = _varint(raw, offset)
    records = []
    for _ in range(occurrence_count):
        values = []
        for _field in range(4):
            value, offset = _varint(raw, offset)
            values.append(value)
        page, y, x, glyph = values
        height, width = shapes[glyph]
        byte_count = height * ((width + 7) // 8)
        packed = np.frombuffer(raw[offset : offset + byte_count], dtype=np.uint8)
        offset += byte_count
        bitmap = np.unpackbits(packed, bitorder="big").reshape(height, -1)[:, :width]
        records.append(((page, y, x, glyph), bitmap.astype(bool)))
    if offset != len(raw):
        raise ValueError("trailing fixed-refinement bytes")
    records.sort(key=lambda item: item[0])
    return [item[0] for item in records], [item[1] for item in records]


def _sheet(bitmaps: list[np.ndarray], columns: int) -> np.ndarray:
    height, width = bitmaps[0].shape
    rows = math.ceil(len(bitmaps) / columns)
    sheet = np.zeros((rows * height, columns * width), dtype=bool)
    for index, bitmap in enumerate(bitmaps):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = bitmap
    return sheet


def _encode(encoder: Path, bitmap: Path) -> bytes:
    return subprocess.run(
        [str(encoder.resolve()), "-p", bitmap.name],
        cwd=bitmap.parent,
        check=True,
        capture_output=True,
        timeout=600,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcf", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", default="1,2,4,8,16,32,64,128")
    args = parser.parse_args()
    positions, bitmaps = _refinements(args.gcf)
    if not bitmaps:
        raise ValueError("atlas has no refinements")
    if len({bitmap.shape for bitmap in bitmaps}) != 1:
        raise ValueError("probe currently expects a single fixed glyph shape")

    packed = b"".join(
        np.packbits(bitmap, axis=1, bitorder="big").tobytes()
        for bitmap in bitmaps
    )
    report = {
        "occurrences": len(bitmaps),
        "shape": list(bitmaps[0].shape),
        "set_pixels": sum(int(np.count_nonzero(bitmap)) for bitmap in bitmaps),
        "packed_bytes": len(packed),
        "zlib_bytes": len(zlib.compress(packed, 9)),
        "bz2_bytes": len(bz2.compress(packed, 9)),
        "lzma_bytes": len(lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME)),
        "layouts": [],
        "position_range": {
            "pages": [min(item[0] for item in positions), max(item[0] for item in positions)],
            "x": [min(item[2] for item in positions), max(item[2] for item in positions)],
            "y": [min(item[1] for item in positions), max(item[1] for item in positions)],
        },
    }
    with tempfile.TemporaryDirectory(prefix="refinement-sheet-") as directory:
        work = Path(directory)
        for columns in sorted({
            min(len(bitmaps), int(value))
            for value in args.columns.split(",") if value.strip()
        }):
            sheet = _sheet(bitmaps, columns)
            target = work / f"sheet-{columns:04d}.pbm"
            Image.fromarray(np.where(sheet, 255, 0).astype(np.uint8), "L").convert("1").save(target)
            encoded = _encode(args.encoder, target)
            stream = work / f"sheet-{columns:04d}.jb2"
            decoded = work / f"sheet-{columns:04d}-decoded.pbm"
            stream.write_bytes(encoded)
            subprocess.run(
                [
                    str(args.decoder.resolve()), "-q", "-e", "-t", "pbm",
                    "-o", str(decoded), str(stream),
                ],
                check=True,
                capture_output=True,
                timeout=600,
            )
            with Image.open(decoded) as image:
                exact = np.array_equal(sheet, np.asarray(image.convert("L")) > 0)
            report["layouts"].append(
                {
                    "columns": columns,
                    "rows": math.ceil(len(bitmaps) / columns),
                    "width": int(sheet.shape[1]),
                    "height": int(sheet.shape[0]),
                    "jbig2_bytes": len(encoded),
                    "exact": bool(exact),
                }
            )
            print(json.dumps(report["layouts"][-1]), flush=True)
    report["best"] = min(
        (item for item in report["layouts"] if item["exact"]),
        key=lambda item: item["jbig2_bytes"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"best": report["best"], "bz2_bytes": report["bz2_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
