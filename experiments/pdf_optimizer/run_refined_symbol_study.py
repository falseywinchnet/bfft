#!/usr/bin/env python3
"""Benchmark shared refined symbols plus a verified cropped XOR residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from pdf_optimizer.jbig2 import append_cropped_xor_region


def _pages(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.extend(range(start, end + 1))
        elif part:
            result.append(int(part))
    return result


def _decode(decoder: Path, globals_path: Path, page_path: Path, output: Path) -> None:
    subprocess.run(
        [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(output),
         str(globals_path), str(page_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _generic_encode(encoder: Path, bitmap: Path) -> bytes:
    return subprocess.run(
        [str(encoder), "-p", str(bitmap)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-dir", type=Path, required=True)
    parser.add_argument("--generic-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--pages", default="1-64")
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pages = _pages(args.pages)
    started = time.monotonic()
    report = {"threshold": args.threshold, "pages": [], "page_numbers": pages}
    with tempfile.TemporaryDirectory(prefix="jbig2-refined-symbol-") as directory:
        work = Path(directory)
        base = work / "symbol"
        inputs = [args.bitmap_dir / f"{page:04d}.pbm" for page in pages]
        subprocess.run(
            [str(args.encoder), "-p", "-s", "-r", "-t", str(args.threshold),
             "-b", str(base), *(str(path) for path in inputs)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        globals_path = base.with_suffix(".sym")
        global_bytes = globals_path.stat().st_size
        report["global_bytes"] = global_bytes

        for index, page in enumerate(pages):
            source_path = args.bitmap_dir / f"{page:04d}.pbm"
            symbol_path = Path(f"{base}.{index:04d}")
            symbol_decoded = work / f"decoded-{index:04d}.pbm"
            _decode(args.decoder, globals_path, symbol_path, symbol_decoded)
            with Image.open(source_path) as source_image, Image.open(symbol_decoded) as decoded_image:
                source = np.asarray(source_image.convert("1"), dtype=bool)
                decoded = np.asarray(decoded_image.convert("1"), dtype=bool)
            difference = source != decoded
            residual_pixels = int(difference.sum())
            hybrid = symbol_path.read_bytes()
            residual_bytes = 0
            box = None
            if residual_pixels:
                ys, xs = np.nonzero(difference)
                # Leptonica's input path rejects extremely small bitmaps; a
                # four-pixel white guard also makes edge behavior explicit.
                box = (
                    max(0, int(xs.min()) - 4),
                    max(0, int(ys.min()) - 4),
                    min(difference.shape[1], int(xs.max()) + 5),
                    min(difference.shape[0], int(ys.max()) + 5),
                )
                crop = difference[box[1]:box[3], box[0]:box[2]]
                # PBM black pixels are the values that the XOR region flips.
                residual_bitmap = work / f"residual-{index:04d}.pbm"
                Image.fromarray(np.where(crop, 0, 255).astype(np.uint8), "L").convert("1").save(residual_bitmap)
                encoded_residual = _generic_encode(args.encoder, residual_bitmap)
                hybrid = append_cropped_xor_region(
                    hybrid, encoded_residual, x=box[0], y=box[1]
                )
                residual_bytes = len(hybrid) - symbol_path.stat().st_size

            hybrid_path = work / f"hybrid-{index:04d}.jb2"
            hybrid_path.write_bytes(hybrid)
            verified_path = work / f"verified-{index:04d}.pbm"
            _decode(args.decoder, globals_path, hybrid_path, verified_path)
            with Image.open(source_path) as source_image, Image.open(verified_path) as verified_image:
                exact = np.array_equal(np.asarray(source_image), np.asarray(verified_image))
            if not exact:
                raise RuntimeError(f"page {page} failed exact hybrid verification")
            generic_bytes = (args.generic_dir / f"{page:04d}.jb2").stat().st_size
            item = {
                "page": page,
                "generic_bytes": generic_bytes,
                "symbol_page_bytes": symbol_path.stat().st_size,
                "residual_pixels": residual_pixels,
                "residual_box": box,
                "residual_segment_bytes": residual_bytes,
                "hybrid_page_bytes": len(hybrid),
                "pixel_exact": exact,
            }
            report["pages"].append(item)
            print(json.dumps(item), flush=True)

        report["generic_total_bytes"] = sum(item["generic_bytes"] for item in report["pages"])
        report["hybrid_total_bytes"] = global_bytes + sum(item["hybrid_page_bytes"] for item in report["pages"])
        report["delta_bytes"] = report["hybrid_total_bytes"] - report["generic_total_bytes"]
        report["exact_pages"] = sum(item["pixel_exact"] for item in report["pages"])
        report["total_seconds"] = time.monotonic() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "generic_total_bytes", "hybrid_total_bytes", "delta_bytes", "exact_pages", "total_seconds"
    )}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
