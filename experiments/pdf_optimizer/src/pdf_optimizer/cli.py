"""Command line interface for the PDF representation optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import OptimizationConfig, SignedPdfError, analyze_pdf, optimize_pdf, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-repr",
        description="Forensically analyze PDFs and retain the smallest lossless representation candidate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="inventory streams, images, fonts, and method applicability")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--password", default="")
    analyze.add_argument("--text-sample-pages", type=int, default=24)
    analyze.add_argument("--report", type=Path)

    optimize = commands.add_parser("optimize", help="search safe lossless representations")
    optimize.add_argument("source", type=Path)
    optimize.add_argument("output", type=Path)
    optimize.add_argument("--password", default="")
    optimize.add_argument("--report", type=Path)
    optimize.add_argument("--workers", type=int, default=max(1, min(4, __import__("os").cpu_count() or 1)))
    optimize.add_argument("--zopfli-iterations", type=int, default=8)
    optimize.add_argument("--no-zopfli", action="store_true")
    optimize.add_argument("--max-encoded-stream-mb", type=int, default=32)
    optimize.add_argument("--max-inflated-stream-mb", type=int, default=128)
    optimize.add_argument("--text-sample-pages", type=int, default=24)
    optimize.add_argument("--force-signed", action="store_true")
    optimize.add_argument("--jbig2-encoder", type=Path)
    optimize.add_argument("--jbig2-decoder", type=Path)
    optimize.add_argument("--jbig2-timeout-seconds", type=float, default=120.0)
    optimize.add_argument(
        "--jbig2-refined-symbols", action="store_true",
        help="try a private lossless refined-symbol dictionary on every eligible page",
    )
    optimize.add_argument("--jbig2-symbol-threshold", type=float, default=0.94)
    optimize.add_argument(
        "--jbig2-entropy-regions", action="store_true",
        help="terminal-measure top conditional-entropy splits into two exact generic regions",
    )
    optimize.add_argument("--jbig2-entropy-cell-size", type=int, default=128)
    optimize.add_argument("--jbig2-entropy-frontier", type=int, default=16)

    mrc = commands.add_parser(
        "mrc-optimize",
        help="opt in to measured foreground fields and mask-guided background deghosting",
    )
    mrc.add_argument("source", type=Path)
    mrc.add_argument("output", type=Path)
    mrc.add_argument("--password", default="")
    mrc.add_argument("--report", type=Path)
    mrc.add_argument("--jbig2-decoder", type=Path, required=True)
    mrc.add_argument("--mask-cache", type=Path)
    mrc.add_argument("--foreground-rmse", type=float, default=8.0)
    mrc.add_argument("--background-psnr", type=float, default=44.0)
    mrc.add_argument("--jbig2-timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "analyze":
        payload = analyze_pdf(
            args.source,
            password=args.password,
            text_sample_pages=args.text_sample_pages,
        )
    elif args.command == "mrc-optimize":
        from .mrc import MrcConfig, optimize_mrc_pdf, write_mrc_json

        result = optimize_mrc_pdf(
            args.source,
            args.output,
            config=MrcConfig(
                jbig2_decoder=args.jbig2_decoder,
                mask_cache=args.mask_cache,
                foreground_rmse_limit=args.foreground_rmse,
                background_outside_psnr=args.background_psnr,
                timeout_seconds=args.jbig2_timeout_seconds,
            ),
            password=args.password,
        )
        payload = result.to_dict()
        if args.report:
            write_mrc_json(args.report, result)
    else:
        config = OptimizationConfig(
            use_zopfli=not args.no_zopfli,
            zopfli_iterations=args.zopfli_iterations,
            workers=args.workers,
            max_encoded_stream_bytes=args.max_encoded_stream_mb * 1024 * 1024,
            max_inflated_stream_bytes=args.max_inflated_stream_mb * 1024 * 1024,
            text_sample_pages=args.text_sample_pages,
            force_signed=args.force_signed,
            jbig2_encoder=args.jbig2_encoder,
            jbig2_decoder=args.jbig2_decoder,
            jbig2_timeout_seconds=args.jbig2_timeout_seconds,
            jbig2_try_refined_symbols=args.jbig2_refined_symbols,
            jbig2_symbol_threshold=args.jbig2_symbol_threshold,
            jbig2_try_entropy_regions=args.jbig2_entropy_regions,
            jbig2_entropy_cell_size=args.jbig2_entropy_cell_size,
            jbig2_entropy_frontier=args.jbig2_entropy_frontier,
        )
        try:
            payload = optimize_pdf(
                args.source,
                args.output,
                config=config,
                password=args.password,
            ).to_dict()
        except SignedPdfError as exc:
            parser.error(str(exc))
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.report and args.command != "mrc-optimize":
        write_json(args.report, payload)
    return 0
