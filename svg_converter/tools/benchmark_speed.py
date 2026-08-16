#!/usr/bin/env python3
"""Run repeatable PNG-to-SVG timing trials without writing the SVGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tlvector.core import VectorizerConfig, vectorize_array  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--colors", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--detail-colors", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    with Image.open(args.input) as image:
        source = image.convert("RGBA")
        width, height = source.size

    trials = []
    for colors in args.colors:
        runs = []
        last = None
        for _ in range(args.repeats):
            last = vectorize_array(
                source,
                VectorizerConfig(colors=colors, detail_colors=args.detail_colors),
                title=args.input.name,
            )
            runs.append(float(last.diagnostics["total_ms"]))
        assert last is not None
        trials.append({
            "colors": colors,
            "detail_colors": args.detail_colors,
            "median_ms": median(runs),
            "runs_ms": runs,
            "paths": last.diagnostics["paths"],
            "loops": last.diagnostics["loops"],
            "svg_bytes": last.diagnostics["svg_bytes"],
        })

    print(json.dumps({
        "input": str(args.input),
        "width": width,
        "height": height,
        "trials": trials,
    }, indent=2))


if __name__ == "__main__":
    main()
