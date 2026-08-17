"""Command line interface for the manual JPEG laboratory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .core import (
    JPEGConfig,
    analyze_five_stages,
    decode,
    encode,
    image_metrics,
    infer_source_quality,
    load_rgb,
    optimize_jpeg,
    save_report,
    save_stage_images,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manual-jpeg", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="render all five JPEG stages")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--out", type=Path, default=Path("jpeg_stages"))
    analyze.add_argument("--quality", type=int, default=80)
    analyze.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=2)
    analyze.add_argument("--chroma-projection", type=float, default=0.0)
    analyze.add_argument("--luma-texture-shrink", type=float, default=0.0)
    analyze.add_argument("--phase-degrees", type=float, default=0.0)

    compare = commands.add_parser("compare", help="measure a JPEG against a decoded reference")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)

    dead_zone = commands.add_parser(
        "make-jldz", help="derive a balanced ownership dead-zone field for fused Jpegli"
    )
    dead_zone.add_argument("source", type=Path)
    dead_zone.add_argument("output", type=Path)
    dead_zone.add_argument("--quality", type=int)
    dead_zone.add_argument("--regions", type=int, default=256)
    dead_zone.add_argument("--minimum-region-blocks", type=int, default=24)
    dead_zone.add_argument("--strength", type=float, default=0.25)
    dead_zone.add_argument("--edge-protection", type=float, default=2.0)
    dead_zone.add_argument("--transport-lambda", type=float, default=0.0)
    dead_zone.add_argument("--frequency-weight", type=float, default=0.1)
    dead_zone.add_argument("--cross-region-weight", type=float, default=0.05)

    fusion = commands.add_parser(
        "jpegli-fuse", help="encode with balanced ownership and Jpegli terminal trellis"
    )
    fusion.add_argument("source", type=Path)
    fusion.add_argument("output", type=Path)
    fusion.add_argument("--quality", type=int, default=72)
    fusion.add_argument("--target-bytes", type=int, default=0)
    fusion.add_argument("--regions", type=int, default=256)
    fusion.add_argument("--minimum-region-blocks", type=int, default=24)
    fusion.add_argument("--field-strength", type=float, default=0.25)
    fusion.add_argument("--edge-protection", type=float, default=2.0)
    fusion.add_argument("--transport-lambda", type=float, default=0.002)
    fusion.add_argument("--frequency-weight", type=float, default=0.1)
    fusion.add_argument("--cross-region-weight", type=float, default=0.05)
    fusion.add_argument("--trellis-lambda", type=float, default=0.0695)
    fusion.add_argument("--ownership-weight", type=float, default=0.05)
    fusion.add_argument("--trellis-edge-weight", type=float, default=1.0)
    fusion.add_argument("--trellis-luma-weight", type=float, default=1.0)
    fusion.add_argument("--trellis-chroma-weight", type=float, default=1.0)
    fusion.add_argument("--quant-luma-tilt", type=float, default=-0.5)
    fusion.add_argument("--quant-chroma-tilt", type=float, default=0.0)

    optimize = commands.add_parser("optimize", help="search a measured JPEG frontier")
    optimize.add_argument("source", type=Path)
    optimize.add_argument("output", type=Path)
    optimize.add_argument("--target-bytes", type=int, default=29_200)
    optimize.add_argument("--exhaustive", action="store_true")
    optimize.add_argument("--report", type=Path)

    relax = commands.add_parser(
        "relax", help="solve the certified connection-valued relaxation"
    )
    relax.add_argument("source", type=Path)
    relax.add_argument("output", type=Path)
    relax.add_argument("--rate-lambda", type=float, default=2.0)
    relax.add_argument("--connection-lambda", type=float, default=1.0)
    relax.add_argument("--frame-mode", choices=("identity", "chroma", "full"), default="chroma")
    relax.add_argument("--cross-region-weight", type=float, default=0.05)
    relax.add_argument("--iterations", type=int, default=1000)
    relax.add_argument("--gap-tolerance", type=float, default=1e-6)
    relax.add_argument("--quality", type=int)
    relax.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=1)
    relax.add_argument("--report", type=Path)

    relax_search = commands.add_parser(
        "optimize-relax", help="sweep certified relaxations and select by actual JPEG rate/distortion"
    )
    relax_search.add_argument("source", type=Path)
    relax_search.add_argument("output", type=Path)
    relax_search.add_argument("--target-bytes", type=int, default=29_200)
    relax_search.add_argument("--rate-lambdas", default="0,0.25,0.5,1,1.5,2")
    relax_search.add_argument("--connection-lambdas", default="0,0.25,0.5,1")
    relax_search.add_argument("--frame-modes", default="identity,chroma,full")
    relax_search.add_argument("--iterations", type=int, default=500)
    relax_search.add_argument("--gap-tolerance", type=float, default=1e-5)

    spectral = commands.add_parser(
        "spectral-relax",
        help="solve the globally optimal three-channel connection relaxation and unrelax once",
    )
    spectral.add_argument("source", type=Path)
    spectral.add_argument("output", type=Path)
    spectral.add_argument("--rate-lambda", type=float, default=0.5)
    spectral.add_argument("--connection-lambda", type=float, default=0.5)
    spectral.add_argument("--cross-region-weight", type=float, default=0.05)
    spectral.add_argument("--iterations", type=int, default=500)
    spectral.add_argument("--gap-tolerance", type=float, default=1e-5)
    spectral.add_argument("--quality", type=int)
    spectral.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=1)
    spectral.add_argument("--report", type=Path)

    sdp = commands.add_parser(
        "sdp-relax",
        help="solve the channel-complete globally optimal bloom-region SDP relaxation",
    )
    sdp.add_argument("source", type=Path)
    sdp.add_argument("output", type=Path)
    sdp.add_argument("--maximum-regions", type=int, default=32)
    sdp.add_argument("--rate-lambda", type=float, default=0.5)
    sdp.add_argument("--connection-lambda", type=float, default=0.5)
    sdp.add_argument("--cross-region-weight", type=float, default=0.05)
    sdp.add_argument("--iterations", type=int, default=500)
    sdp.add_argument("--gap-tolerance", type=float, default=1e-5)
    sdp.add_argument("--sdp-tolerance", type=float, default=1e-7)
    sdp.add_argument("--solver", choices=("CLARABEL", "SCS"), default="CLARABEL")
    sdp.add_argument("--quality", type=int)
    sdp.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=1)
    sdp.add_argument("--report", type=Path)

    ownership = commands.add_parser(
        "ownership-relax",
        help="run phase-forest transport and globally optimal causal-tree bifurcation",
    )
    ownership.add_argument("source", type=Path)
    ownership.add_argument("output", type=Path)
    ownership.add_argument("--rate-lambda", type=float, default=0.0)
    ownership.add_argument("--branch-penalty", type=float, default=0.0)
    ownership.add_argument("--maximum-depth", type=int, default=8)
    ownership.add_argument("--minimum-atoms", type=int, default=128)
    ownership.add_argument("--maximum-condition", type=float, default=12.0)
    ownership.add_argument("--cross-region-weight", type=float, default=0.05)
    ownership.add_argument("--quality", type=int)
    ownership.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=1)
    ownership.add_argument("--report", type=Path)

    spatial_dct = commands.add_parser(
        "spatial-dct-relax",
        help="globally transport signed DCT ownership across space and frequency",
    )
    spatial_dct.add_argument("source", type=Path)
    spatial_dct.add_argument("output", type=Path)
    spatial_dct.add_argument("--transport-lambda", type=float, default=0.01)
    spatial_dct.add_argument("--frequency-weight", type=float, default=0.1)
    spatial_dct.add_argument("--cross-region-weight", type=float, default=0.05)
    spatial_dct.add_argument("--luma-mobility", type=float, default=1.0)
    spatial_dct.add_argument("--cb-mobility", type=float, default=1.0)
    spatial_dct.add_argument("--cr-mobility", type=float, default=1.0)
    spatial_dct.add_argument("--region-mode", choices=("balanced", "signature"), default="balanced")
    spatial_dct.add_argument("--regions", type=int, default=256)
    spatial_dct.add_argument("--minimum-region-blocks", type=int, default=24)
    spatial_dct.add_argument("--chroma-projection", type=float, default=0.0)
    spatial_dct.add_argument("--phase-degrees", type=float, default=0.0)
    spatial_dct.add_argument("--quality", type=int)
    spatial_dct.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=1)
    spatial_dct.add_argument("--report", type=Path)

    commands.add_parser("gui", help="launch the Dear PyGui inspector")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "gui":
        from .gui import run_gui
        run_gui()
        return 0
    if args.command == "analyze":
        config = JPEGConfig(
            quality=args.quality,
            subsampling=args.subsampling,
            chroma_projection=args.chroma_projection,
            luma_texture_shrink=args.luma_texture_shrink,
            phase_degrees=args.phase_degrees,
        )
        stages = analyze_five_stages(load_rgb(args.source), config)
        save_stage_images(stages, args.out)
        print(json.dumps({
            "source": str(args.source),
            "inferred_source_quality": infer_source_quality(args.source),
            "output": str(args.out),
            "stages": list(stages),
        }, indent=2))
        return 0
    if args.command == "compare":
        reference, candidate = load_rgb(args.reference), load_rgb(args.candidate)
        ssim, psnr, edge_psnr = image_metrics(reference, candidate)
        print(json.dumps({
            "reference": str(args.reference),
            "candidate": str(args.candidate),
            "reference_bytes": args.reference.stat().st_size,
            "candidate_bytes": args.candidate.stat().st_size,
            "ssim": ssim,
            "psnr_db": psnr,
            "edge_psnr_db": edge_psnr,
        }, indent=2))
        return 0
    if args.command == "make-jldz":
        from .balanced_regions import BalancedRegionConfig, balanced_bifurcation_regions
        from .core import rgb_to_ycc
        from .jpegli_bridge import ownership_dead_zone_field, write_jldz

        quality = args.quality or infer_source_quality(args.source)
        ycc = rgb_to_ycc(load_rgb(args.source))
        regions = balanced_bifurcation_regions(
            ycc,
            quality,
            BalancedRegionConfig(
                target_regions=args.regions,
                minimum_blocks=args.minimum_region_blocks,
            ),
        )
        transported = None
        transport_report = None
        if args.transport_lambda > 0.0:
            from .certified_relaxation import _coefficients
            from .spatial_dct_transport import (
                SpatialDCTTransportConfig,
                transport_spatial_dct,
            )

            result = transport_spatial_dct(
                _coefficients(ycc),
                regions.labels,
                quality,
                SpatialDCTTransportConfig(
                    transport_lambda=args.transport_lambda,
                    frequency_weight=args.frequency_weight,
                    cross_region_weight=args.cross_region_weight,
                ),
            )
            transported = result.coefficients
            transport_report = result.report()
        field = ownership_dead_zone_field(
            ycc, regions.labels, quality, strength=args.strength,
            edge_protection=args.edge_protection,
            transported_coefficients=transported,
        )
        write_jldz(args.output, field)
        print(json.dumps({
            "source": str(args.source),
            "output": str(args.output),
            "quality": quality,
            "strength": args.strength,
            "edge_protection": args.edge_protection,
            "field_shape": list(field.shape),
            "maximum_threshold": float(field.max()),
            "mean_ac_threshold": float(field[..., 1:].mean()),
            "regions": regions.report(),
            "spatial_frequency_transport": transport_report,
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "jpegli-fuse":
        from .jpegli_fusion import encode_jpegli_fusion

        report = encode_jpegli_fusion(
            args.source,
            args.output,
            quality=args.quality,
            target_bytes=args.target_bytes,
            regions=args.regions,
            minimum_region_blocks=args.minimum_region_blocks,
            field_strength=args.field_strength,
            edge_protection=args.edge_protection,
            transport_lambda=args.transport_lambda,
            frequency_weight=args.frequency_weight,
            cross_region_weight=args.cross_region_weight,
            trellis_lambda=args.trellis_lambda,
            ownership_weight=args.ownership_weight,
            trellis_edge_weight=args.trellis_edge_weight,
            trellis_luma_weight=args.trellis_luma_weight,
            trellis_chroma_weight=args.trellis_chroma_weight,
            quant_luma_tilt=args.quant_luma_tilt,
            quant_chroma_tilt=args.quant_chroma_tilt,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "relax":
        from .certified_relaxation import RelaxationConfig, relax_rgb

        quality = args.quality or infer_source_quality(args.source)
        relaxation = relax_rgb(
            load_rgb(args.source),
            RelaxationConfig(
                rate_lambda=args.rate_lambda,
                connection_lambda=args.connection_lambda,
                frame_mode=args.frame_mode,
                cross_region_weight=args.cross_region_weight,
                iterations=args.iterations,
                relative_gap_tolerance=args.gap_tolerance,
            ),
        )
        jpeg_config = JPEGConfig(quality=quality, subsampling=args.subsampling)
        data = encode(relaxation.rgb, jpeg_config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        ssim, psnr, edge_psnr = image_metrics(load_rgb(args.source), decode(data))
        report = {
            "source": str(args.source),
            "output": str(args.output),
            "source_bytes": args.source.stat().st_size,
            "output_bytes": len(data),
            "jpeg": {"quality": quality, "subsampling": args.subsampling},
            "ssim": ssim,
            "psnr_db": psnr,
            "edge_psnr_db": edge_psnr,
            "relaxation": relaxation.report(),
        }
        report_path = args.report or args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "optimize-relax":
        from .relaxation_sweep import search_relaxations

        def relaxation_progress(index, total, candidate):
            print(
                f"[{index:3d}/{total}] {candidate.size_bytes:6d} B "
                f"SSIM={candidate.ssim:.6f} gap={candidate.certificate['relative_gap']:.2e} "
                f"rate={candidate.config.rate_lambda:g} conn={candidate.config.connection_lambda:g} "
                f"frame={candidate.config.frame_mode}",
                file=sys.stderr,
            )

        report = search_relaxations(
            args.source,
            args.output,
            target_bytes=args.target_bytes,
            rate_lambdas=[float(value) for value in args.rate_lambdas.split(",") if value],
            connection_lambdas=[float(value) for value in args.connection_lambdas.split(",") if value],
            frame_modes=[value for value in args.frame_modes.split(",") if value],
            iterations=args.iterations,
            gap_tolerance=args.gap_tolerance,
            progress=relaxation_progress,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "spectral-relax":
        from .certified_relaxation import (
            RelaxationConfig,
            _coefficients,
            coefficients_to_rgb,
            solve_coefficients,
        )
        from .spectral_relaxation import solve_spectral_relaxation
        from .core import _region_labels, rgb_to_ycc

        reference = load_rgb(args.source)
        ycc = rgb_to_ycc(reference)
        labels, _ = _region_labels(ycc, 1.2, 0.58)
        source_coefficients = _coefficients(ycc)
        spectral_result = solve_spectral_relaxation(
            source_coefficients,
            labels,
            cross_region_weight=args.cross_region_weight,
        )
        inner_config = RelaxationConfig(
            rate_lambda=args.rate_lambda,
            connection_lambda=args.connection_lambda,
            frame_mode="identity",
            cross_region_weight=args.cross_region_weight,
            iterations=args.iterations,
            relative_gap_tolerance=args.gap_tolerance,
        )
        coefficients, inner_certificate = solve_coefficients(
            source_coefficients,
            labels,
            inner_config,
            fixed_frames=spectral_result.frames,
        )
        relaxed_rgb = coefficients_to_rgb(
            coefficients, labels.shape, reference.shape[:2]
        )
        quality = args.quality or infer_source_quality(args.source)
        data = encode(
            relaxed_rgb,
            JPEGConfig(quality=quality, subsampling=args.subsampling),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        ssim, psnr, edge_psnr = image_metrics(reference, decode(data))
        report = {
            "source": str(args.source),
            "output": str(args.output),
            "source_bytes": args.source.stat().st_size,
            "output_bytes": len(data),
            "ssim": ssim,
            "psnr_db": psnr,
            "edge_psnr_db": edge_psnr,
            "jpeg": {"quality": quality, "subsampling": args.subsampling},
            "global_spectral_relaxation": spectral_result.report(),
            "fixed_frame_inner_relaxation": {
                "config": inner_config.__dict__,
                **inner_certificate,
            },
            "proof_boundary": (
                "The spectral field is the attained global optimum of the "
                "joint three-section relaxation. Polar unrelaxation and the "
                "subsequent fixed-frame convex coefficient solve are reported "
                "separately and are not folded into that theorem."
            ),
        }
        report_path = args.report or args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "sdp-relax":
        from .certified_relaxation import (
            RelaxationConfig,
            _coefficients,
            coefficients_to_rgb,
            solve_coefficients,
        )
        from .sdp_relaxation import solve_sdp_relaxation
        from .core import _region_labels, rgb_to_ycc

        reference = load_rgb(args.source)
        ycc = rgb_to_ycc(reference)
        labels, _ = _region_labels(ycc, 1.2, 0.58)
        source_coefficients = _coefficients(ycc)
        sdp_result = solve_sdp_relaxation(
            source_coefficients,
            labels,
            maximum_regions=args.maximum_regions,
            cross_region_weight=args.cross_region_weight,
            solver=args.solver,
            tolerance=args.sdp_tolerance,
        )
        inner_config = RelaxationConfig(
            rate_lambda=args.rate_lambda,
            connection_lambda=args.connection_lambda,
            cross_region_weight=args.cross_region_weight,
            iterations=args.iterations,
            relative_gap_tolerance=args.gap_tolerance,
        )
        coefficients, inner_certificate = solve_coefficients(
            source_coefficients,
            labels,
            inner_config,
            fixed_frames=sdp_result.block_frames,
        )
        relaxed_rgb = coefficients_to_rgb(
            coefficients, labels.shape, reference.shape[:2]
        )
        quality = args.quality or infer_source_quality(args.source)
        data = encode(
            relaxed_rgb,
            JPEGConfig(quality=quality, subsampling=args.subsampling),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        ssim, psnr, edge_psnr = image_metrics(reference, decode(data))
        report = {
            "source": str(args.source),
            "output": str(args.output),
            "source_bytes": args.source.stat().st_size,
            "output_bytes": len(data),
            "ssim": ssim,
            "psnr_db": psnr,
            "edge_psnr_db": edge_psnr,
            "jpeg": {"quality": quality, "subsampling": args.subsampling},
            "global_channel_complete_sdp": sdp_result.report(),
            "fixed_frame_inner_relaxation": {
                "config": inner_config.__dict__,
                **inner_certificate,
            },
            "proof_boundary": (
                "The bloom-region Gram field is the certified global optimum "
                "of the locally channel-complete SDP relaxation. Rank-three "
                "rounding and the subsequent fixed-frame convex solve are "
                "reported separately."
            ),
        }
        report_path = args.report or args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "ownership-relax":
        from .certified_relaxation import _coefficients, coefficients_to_rgb
        from .ownership_bifurcation import BifurcationConfig, bifurcate_coefficients
        from .core import _region_labels, rgb_to_ycc

        reference = load_rgb(args.source)
        ycc = rgb_to_ycc(reference)
        labels, _ = _region_labels(ycc, 1.2, 0.58)
        source_coefficients = _coefficients(ycc)
        quality = args.quality or infer_source_quality(args.source)
        result = bifurcate_coefficients(
            source_coefficients,
            labels,
            quality,
            BifurcationConfig(
                rate_lambda=args.rate_lambda,
                branch_penalty=args.branch_penalty,
                maximum_depth=args.maximum_depth,
                minimum_atoms=args.minimum_atoms,
                maximum_condition=args.maximum_condition,
                cross_region_weight=args.cross_region_weight,
            ),
        )
        relaxed_rgb = coefficients_to_rgb(
            result.coefficients, labels.shape, reference.shape[:2]
        )
        data = encode(
            relaxed_rgb,
            JPEGConfig(quality=quality, subsampling=args.subsampling),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        ssim, psnr, edge_psnr = image_metrics(reference, decode(data))
        report = {
            "source": str(args.source),
            "output": str(args.output),
            "source_bytes": args.source.stat().st_size,
            "output_bytes": len(data),
            "ssim": ssim,
            "psnr_db": psnr,
            "edge_psnr_db": edge_psnr,
            "jpeg": {"quality": quality, "subsampling": args.subsampling},
            "ownership_bifurcation": result.report(),
            "proof_boundary": (
                "Phase transport, atom ownership, and inverse paths are exact. "
                "The Bellman value is the global optimum over every pruning "
                "of the generated Courant--Fischer/weighted-median causal tree."
            ),
        }
        report_path = args.report or args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "spatial-dct-relax":
        from .certified_relaxation import _coefficients, coefficients_to_rgb
        from .core import _region_labels, preprocess, rgb_to_ycc
        from .spatial_dct_transport import (
            SpatialDCTTransportConfig, transport_spatial_dct,
        )
        from .balanced_regions import (
            BalancedRegionConfig, balanced_bifurcation_regions,
        )

        reference = load_rgb(args.source)
        quality = args.quality or infer_source_quality(args.source)
        prepared = preprocess(
            reference,
            JPEGConfig(
                quality=quality,
                subsampling=args.subsampling,
                chroma_projection=args.chroma_projection,
                phase_degrees=args.phase_degrees,
            ),
        )
        ycc = rgb_to_ycc(prepared.rgb)
        region_report = None
        if args.region_mode == "balanced":
            balanced = balanced_bifurcation_regions(
                ycc,
                quality,
                BalancedRegionConfig(
                    target_regions=args.regions,
                    minimum_blocks=args.minimum_region_blocks,
                ),
            )
            labels = balanced.labels
            region_report = balanced.report()
        else:
            labels, _ = _region_labels(ycc, 1.2, 0.58)
        source_coefficients = _coefficients(ycc)
        result = transport_spatial_dct(
            source_coefficients,
            labels,
            quality,
            SpatialDCTTransportConfig(
                transport_lambda=args.transport_lambda,
                frequency_weight=args.frequency_weight,
                cross_region_weight=args.cross_region_weight,
                luma_mobility=args.luma_mobility,
                cb_mobility=args.cb_mobility,
                cr_mobility=args.cr_mobility,
            ),
        )
        transported_rgb = coefficients_to_rgb(
            result.coefficients, labels.shape, reference.shape[:2]
        )
        data = encode(
            transported_rgb,
            JPEGConfig(quality=quality, subsampling=args.subsampling),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        ssim, psnr, edge_psnr = image_metrics(reference, decode(data))
        report = {
            "source": str(args.source),
            "output": str(args.output),
            "source_bytes": args.source.stat().st_size,
            "output_bytes": len(data),
            "ssim": ssim,
            "psnr_db": psnr,
            "edge_psnr_db": edge_psnr,
            "jpeg": {"quality": quality, "subsampling": args.subsampling},
            "preprocess": {
                "chroma_projection": args.chroma_projection,
                "phase_degrees": args.phase_degrees,
            },
            "regions": {
                "mode": args.region_mode,
                "report": region_report,
            },
            "spatial_frequency_dct_transport": result.report(),
            "proof_boundary": (
                "Global optimality covers the continuous signed-mass "
                "redistribution on the fixed spatial/frequency ownership "
                "graph. JPEG quantization is the measured projection."
            ),
        }
        report_path = args.report or args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    def progress(index, total, candidate):
        if index == 1 or index == total or index % max(1, total // 20) == 0:
            metric = (
                "rate-probe" if math.isnan(candidate.ssim)
                else f"SSIM={candidate.ssim:.6f}"
            )
            print(
                f"[{index:4d}/{total}] {candidate.size_bytes:6d} B  "
                f"{metric}  q={candidate.config.quality} "
                f"sub={candidate.config.subsampling}",
                file=sys.stderr,
            )

    result = optimize_jpeg(
        args.source,
        args.output,
        target_bytes=args.target_bytes,
        exhaustive=args.exhaustive,
        progress=progress,
    )
    report_path = args.report or args.output.with_suffix(".json")
    save_report(result, report_path)
    print(json.dumps(result.report(), indent=2, sort_keys=True))
    return 0
