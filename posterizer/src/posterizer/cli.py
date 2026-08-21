"""Command-line interface for the perceptual posterizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import PosterizerConfig, posterize_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Posterize an image with saliency-balanced OKLCH bifurcation"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colors", type=int, default=24)
    parser.add_argument("--method", choices=("oklch", "inherited"), default="oklch")
    parser.add_argument("--lightness-weight", type=float, default=1.0)
    parser.add_argument("--chroma-weight", type=float, default=1.0)
    parser.add_argument("--hue-weight", type=float, default=1.0)
    parser.add_argument("--node-separation", type=float, default=1.08)
    parser.add_argument("--detail-priority", type=float, default=2.0)
    parser.add_argument("--population-exponent", type=float, default=0.65)
    parser.add_argument("--family-priority", type=float, default=1.0)
    parser.add_argument("--structure-radius", type=int, default=2)
    parser.add_argument("--structure-threshold", type=float, default=0.065)
    parser.add_argument("--texture-priority", type=float, default=0.25)
    parser.add_argument("--mixing-strength", type=float, default=0.0)
    parser.add_argument("--mixing-neighbors", type=int, default=3)
    parser.add_argument("--minimum-leaf", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--minimum-island", type=int, default=6)
    parser.add_argument("--cleanup-rounds", type=int, default=1)
    parser.add_argument("--alpha-mode", choices=("auto", "cutout", "preserve"), default="auto")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args(argv)
    config = PosterizerConfig(
        colors=args.colors,
        method=args.method,
        lightness_weight=args.lightness_weight,
        chroma_weight=args.chroma_weight,
        hue_weight=args.hue_weight,
        node_separation=args.node_separation,
        detail_priority=args.detail_priority,
        population_exponent=args.population_exponent,
        family_priority=args.family_priority,
        structure_radius=args.structure_radius,
        structure_threshold=args.structure_threshold,
        texture_priority=args.texture_priority,
        mixing_strength=args.mixing_strength,
        mixing_neighbors=args.mixing_neighbors,
        minimum_leaf=args.minimum_leaf,
        sample_limit=args.sample_limit,
        minimum_island=args.minimum_island,
        cleanup_rounds=args.cleanup_rounds,
        alpha_mode=args.alpha_mode,
        trim_transparent=not args.no_trim,
    )
    result = posterize_image(args.input, args.output, config)
    text = json.dumps(result.diagnostics, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.diagnostics:
        args.diagnostics.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
