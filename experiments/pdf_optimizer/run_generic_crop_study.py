#!/usr/bin/env python3
"""Apply the verified generic-region crop optimizer to a bitmap stream set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from pdf_optimizer.jbig2 import optimize_generic_stream


def _pages(value: str) -> list[int]:
    pages = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            pages.extend(range(first, last + 1))
        elif part:
            pages.append(int(part))
    return pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--bitmap-pattern", default="remainder-{page:04d}.pbm")
    parser.add_argument("--stream-pattern", default="remainder-{page:04d}.jb2")
    parser.add_argument("--pages", required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template-1", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for page in _pages(args.pages):
        bitmap = args.dir / args.bitmap_pattern.format(page=page)
        stream = args.dir / args.stream_pattern.format(page=page)
        with Image.open(bitmap) as image:
            width, height = image.size
        raw = stream.read_bytes()
        candidate = optimize_generic_stream(
            raw,
            width=width,
            height=height,
            encoder=args.encoder.resolve(),
            decoder=args.decoder.resolve(),
            try_template_1=args.template_1,
            timeout=600,
        )
        selected = candidate.data if candidate.data is not None else raw
        (args.out / f"{page:04d}.jb2").write_bytes(selected)
        record = {
            "page": page,
            "source_bytes": len(raw),
            "selected_bytes": len(selected),
            "delta_bytes": len(selected) - len(raw),
            "kind": candidate.kind,
            "reason": candidate.reason,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "pages": records,
        "source_bytes": sum(item["source_bytes"] for item in records),
        "selected_bytes": sum(item["selected_bytes"] for item in records),
        "delta_bytes": sum(item["delta_bytes"] for item in records),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("source_bytes", "selected_bytes", "delta_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
