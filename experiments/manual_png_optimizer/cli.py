"""Command line interface for ownership-aware PNG optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import PNGConfig, compare_pngs, optimize_png


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manual-png", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    optimize = commands.add_parser("optimize", help="search measured PNG rate/distortion")
    optimize.add_argument("source", type=Path)
    optimize.add_argument("output", type=Path)
    optimize.add_argument("--target-bytes", type=int, default=0)
    optimize.add_argument("--minimum-ssim", type=float, default=0.0)
    optimize.add_argument("--colors", type=int, default=0, help="0 selects a short guided search")
    optimize.add_argument("--minimum-colors", type=int, default=8)
    optimize.add_argument(
        "--dither", choices=("none", "selective", "floyd", "auto"),
        default="none",
    )
    optimize.add_argument(
        "--quantizer",
        choices=(
            "auto", "edge-lloyd", "lloyd-rgb", "median-cut",
            "maximum-coverage", "fast-octree",
        ),
        default="auto",
    )
    optimize.add_argument("--lloyd-iterations", type=int, default=10)
    optimize.add_argument("--palette-edge-weight", type=float, default=1.5)
    optimize.add_argument("--palette-sample-limit", type=int, default=131072)
    optimize.add_argument("--palette-seed", type=int, default=508030340)
    optimize.add_argument("--diffusion-strength", type=float, default=0.9)
    optimize.add_argument("--diffusion-edge-barrier", type=float, default=3.0)
    optimize.add_argument(
        "--ownership-strength", type=float, default=-1.0,
        help="negative selects the automatic edge-gated ladder",
    )
    optimize.add_argument("--ownership-iterations", type=int, default=2)
    optimize.add_argument("--edge-protection", type=float, default=8.0)
    optimize.add_argument("--no-palette-transport", action="store_true")
    optimize.add_argument("--thorough", action="store_true")
    optimize.add_argument("--report", type=Path)

    lossless = commands.add_parser("lossless", help="optimize without changing decoded pixels")
    lossless.add_argument("source", type=Path)
    lossless.add_argument("output", type=Path)
    lossless.add_argument("--thorough", action="store_true")
    lossless.add_argument("--report", type=Path)

    compare = commands.add_parser("compare", help="measure a PNG against its reference")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)

    gui = commands.add_parser("gui", help="launch the Dear PyGui optimizer")
    gui.add_argument("source", type=Path, nargs="?")
    return parser


def _progress(current: int, total: int, message: str) -> None:
    print(f"[{current:>2}/{total:<2}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "gui":
        from .gui import run_gui

        run_gui(args.source)
        return 0
    if args.command == "compare":
        print(json.dumps(compare_pngs(args.reference, args.candidate), indent=2, sort_keys=True))
        return 0
    if args.command == "lossless":
        config = PNGConfig(
            lossless=True,
            filter_search="thorough" if args.thorough else "fast",
        )
    else:
        config = PNGConfig(
            target_bytes=max(0, args.target_bytes),
            minimum_ssim=max(0.0, min(1.0, args.minimum_ssim)),
            colors=args.colors,
            minimum_colors=args.minimum_colors,
            dither=args.dither,
            quantizer=args.quantizer,
            lloyd_iterations=args.lloyd_iterations,
            palette_edge_weight=args.palette_edge_weight,
            palette_sample_limit=args.palette_sample_limit,
            palette_seed=args.palette_seed,
            diffusion_strength=args.diffusion_strength,
            diffusion_edge_barrier=args.diffusion_edge_barrier,
            ownership_strength=args.ownership_strength,
            ownership_iterations=args.ownership_iterations,
            edge_protection=args.edge_protection,
            palette_transport=not args.no_palette_transport,
            filter_search="thorough" if args.thorough else "fast",
        )
    result = optimize_png(
        args.source, args.output, config=config, report=args.report, progress=_progress
    )
    print(json.dumps(result.report(), indent=2, sort_keys=True))
    return 0
