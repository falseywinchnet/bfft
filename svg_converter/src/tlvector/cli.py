from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import VectorizerConfig, vectorize_png


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PNG to layered SVG contours")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--colors", type=int, default=12)
    parser.add_argument("--detail-colors", type=int, default=6)
    parser.add_argument("--coarse-side", type=int, default=160)
    parser.add_argument("--minimum-region", type=int, default=10)
    parser.add_argument("--simplify", type=float, default=0.85)
    parser.add_argument("--curve-tolerance", type=float, default=0.65)
    parser.add_argument("--seam-overlap", type=float, default=0.65)
    parser.add_argument(
        "--alpha-mode", choices=("auto", "preserve", "cutout"), default="auto",
        help="treat partial alpha as edge coverage, real translucency, or detect automatically",
    )
    parser.add_argument(
        "--no-trim", action="store_true",
        help="preserve fully transparent outer rows and columns",
    )
    parser.add_argument("--diagnostics", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or args.input.with_suffix(".svg")
    config = VectorizerConfig(
        colors=args.colors,
        detail_colors=args.detail_colors,
        coarse_side=args.coarse_side,
        minimum_region=args.minimum_region,
        simplify=args.simplify,
        curve_tolerance=args.curve_tolerance,
        seam_overlap=args.seam_overlap,
        alpha_mode=args.alpha_mode,
        trim_transparent=not args.no_trim,
    )
    result = vectorize_png(args.input, output, config)
    payload = json.dumps(result.diagnostics, indent=2, sort_keys=True)
    print(payload)
    if args.diagnostics:
        args.diagnostics.write_text(payload + "\n", encoding="utf-8")
    return 0
