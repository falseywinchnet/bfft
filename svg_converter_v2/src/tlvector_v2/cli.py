"""Command-line interfaces for converter v2 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .affine import build_v3_affine_svg
from .core import V2Config, vectorize_png_v2


def _write_diagnostics(payload: dict, destination: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if destination:
        destination.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Converter v2 compact rate-distortion PNG to SVG"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colors", type=int, default=128)
    parser.add_argument("--split-budget", type=int, default=96)
    parser.add_argument("--split-target-mse", type=float, default=20.0)
    parser.add_argument("--target-mse", type=float, default=30.0)
    parser.add_argument("--merge-maximum-area", type=int, default=32)
    parser.add_argument("--merge-rounds", type=int, default=2)
    parser.add_argument("--coarse-side", type=int, default=160)
    parser.add_argument("--minimum-region", type=int, default=10)
    parser.add_argument("--also-svgz", action="store_true")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args(argv)
    result = vectorize_png_v2(
        args.input,
        args.output,
        V2Config(
            colors=args.colors,
            split_budget=args.split_budget,
            split_target_mse=args.split_target_mse,
            final_target_mse=args.target_mse,
            merge_maximum_area=args.merge_maximum_area,
            merge_rounds=args.merge_rounds,
            coarse_side=args.coarse_side,
            minimum_region=args.minimum_region,
        ),
    )
    if args.also_svgz and args.output.suffix.lower() != ".svgz":
        args.output.with_suffix(".svgz").write_bytes(result.svgz)
    _write_diagnostics(result.diagnostics, args.diagnostics)
    return 0 if result.diagnostics["target_met"] else 2


def affine_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prototype rank-one SVG gradients on segmenting-v3 cells"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bfft-library")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--also-svgz", action="store_true")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args(argv)
    result = build_v3_affine_svg(
        args.input,
        bfft_library=args.bfft_library,
        threads=args.threads,
    )
    result.save(args.output)
    if args.also_svgz and args.output.suffix.lower() != ".svgz":
        args.output.with_suffix(".svgz").write_bytes(result.svgz)
    _write_diagnostics(result.diagnostics, args.diagnostics)
    return 0
