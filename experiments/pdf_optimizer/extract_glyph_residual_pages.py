#!/usr/bin/env python3
"""Extract a page range of simple MRC scans in one PDF traversal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pikepdf

from pdf_optimizer.mrc import _decode_jpx, _decode_mask, _mrc_pair


def _pages(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    return sorted(dict.fromkeys(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--pages", required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    page_numbers = _pages(args.pages)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []

    with tempfile.TemporaryDirectory(prefix="glyph-residual-extract-") as directory:
        work = Path(directory)
        with pikepdf.open(args.source) as pdf:
            for page_number in page_numbers:
                if not 1 <= page_number <= len(pdf.pages):
                    raise SystemExit(f"page {page_number} is outside the document")
                pair = _mrc_pair(pdf.pages[page_number - 1])
                if pair is None:
                    raise SystemExit(f"page {page_number} is not a simple MRC page")
                background, foreground, mask_stream = pair
                mask = _decode_mask(
                    mask_stream,
                    page_number=page_number,
                    decoder=args.decoder,
                    mask_cache=None,
                    timeout=180.0,
                    work=work,
                )
                foreground_image = _decode_jpx(foreground)
                background_image = _decode_jpx(background)
                destination = args.out / f"page-{page_number:04d}"
                destination.mkdir(parents=True, exist_ok=True)
                mask.save(destination / "source-mask.pbm")
                foreground_image.save(destination / "source-foreground.png")
                background_image.save(destination / "source-background.png")
                metadata = {
                    "source": str(args.source.resolve()),
                    "page": page_number,
                    "page_count": len(pdf.pages),
                    "mask_size": list(mask.size),
                    "mask_fraction": float(np.mean(np.asarray(mask.convert("L")) > 0)),
                    "mask_stream_bytes": len(mask_stream.read_raw_bytes()),
                    "foreground_size": [foreground_image.width, foreground_image.height],
                    "foreground_stream_bytes": len(foreground.read_raw_bytes()),
                    "background_size": [background_image.width, background_image.height],
                    "background_stream_bytes": len(background.read_raw_bytes()),
                }
                (destination / "source.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                )
                records.append(metadata)
                print(json.dumps(metadata), flush=True)

    (args.out / "manifest.json").write_text(
        json.dumps({"pages": records}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
