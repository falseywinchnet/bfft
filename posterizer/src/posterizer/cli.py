"""Command-line interface for the perceptual posterizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import PosterizerConfig, posterize_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Posterize an image with optimal occupied-space OKLCH bifurcation"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colors", type=int, default=24)
    parser.add_argument("--method", choices=("oklch", "inherited"), default="oklch")
    parser.add_argument("--lightness-weight", type=float, default=1.0)
    parser.add_argument("--chroma-weight", type=float, default=1.0)
    parser.add_argument("--hue-weight", type=float, default=1.0)
    parser.add_argument("--node-separation", type=float, default=1.08)
    parser.add_argument("--minimum-leaf", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--minimum-island", type=int, default=6)
    parser.add_argument("--cleanup-rounds", type=int, default=1)
    parser.add_argument("--alpha-mode", choices=("auto", "cutout", "preserve"), default="auto")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--also-png", action="store_true")
    parser.add_argument("--also-svg", action="store_true")
    parser.add_argument("--also-svgz", action="store_true")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args(argv)
    config = PosterizerConfig(
        colors=args.colors,
        method=args.method,
        lightness_weight=args.lightness_weight,
        chroma_weight=args.chroma_weight,
        hue_weight=args.hue_weight,
        node_separation=args.node_separation,
        minimum_leaf=args.minimum_leaf,
        sample_limit=args.sample_limit,
        minimum_island=args.minimum_island,
        cleanup_rounds=args.cleanup_rounds,
        alpha_mode=args.alpha_mode,
        trim_transparent=not args.no_trim,
    )
    result = posterize_image(args.input, args.output, config)
    stem = args.output.with_suffix("")
    if args.also_png and args.output.suffix.lower() != ".png":
        result.save(stem.with_suffix(".png"))
    if args.also_svg and args.output.suffix.lower() != ".svg":
        result.save(stem.with_suffix(".svg"))
    if args.also_svgz and args.output.suffix.lower() != ".svgz":
        result.save(stem.with_suffix(".svgz"))
    text = json.dumps(result.diagnostics, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.diagnostics:
        args.diagnostics.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
