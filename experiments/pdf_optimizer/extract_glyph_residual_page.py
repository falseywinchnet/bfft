#!/usr/bin/env python3
"""Extract one simple MRC page into portable experiment assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pikepdf

from pdf_optimizer.mrc import _decode_jpx, _decode_mask, _mrc_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.page < 1:
        raise SystemExit("--page is one-based")
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="glyph-residual-extract-") as directory:
        with pikepdf.open(args.source) as pdf:
            if args.page > len(pdf.pages):
                raise SystemExit("page is outside the document")
            pair = _mrc_pair(pdf.pages[args.page - 1])
            if pair is None:
                raise SystemExit("page is not a simple background/foreground/JBIG2 MRC page")
            background, foreground, mask_stream = pair
            mask = _decode_mask(
                mask_stream,
                page_number=args.page,
                decoder=args.decoder,
                mask_cache=None,
                timeout=180.0,
                work=Path(directory),
            )
            foreground_image = _decode_jpx(foreground)
            background_image = _decode_jpx(background)
            mask.save(args.out / "source-mask.pbm")
            foreground_image.save(args.out / "source-foreground.png")
            background_image.save(args.out / "source-background.png")
            mask_stream.read_raw_bytes()
            metadata = {
                "source": str(args.source.resolve()),
                "page": args.page,
                "page_count": len(pdf.pages),
                "mask_size": list(mask.size),
                "mask_fraction": float(np.mean(np.asarray(mask.convert("L")) > 0)),
                "mask_stream_bytes": len(mask_stream.read_raw_bytes()),
                "foreground_size": [foreground_image.width, foreground_image.height],
                "foreground_stream_bytes": len(foreground.read_raw_bytes()),
                "background_size": [background_image.width, background_image.height],
                "background_stream_bytes": len(background.read_raw_bytes()),
            }
    (args.out / "source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
