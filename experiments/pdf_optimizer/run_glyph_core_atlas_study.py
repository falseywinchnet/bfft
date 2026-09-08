#!/usr/bin/env python3
"""Terminal-measure an exact shared glyph-core atlas plus residual bitmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from pdf_optimizer.glyph_residual import (
    GlyphResidualConfig,
    canonicalize_masks_global_core,
    config_dict,
    decode_core_atlas,
    decode_core_corrections,
    decode_refined_core_atlas,
    decode_fixed_refinement_atlas,
    decode_split_refinement_atlas,
    serialize_core_atlas,
    serialize_core_corrections,
    serialize_fixed_refinement_atlas,
    serialize_split_refinement_atlas,
    subset_core_atlas,
)


def _bitmap(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _write_bitmap(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").convert("1").save(path)


def _run(command: list[str], *, cwd: Path | None = None, stdout: bool = False) -> bytes:
    result = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, timeout=600
    )
    return result.stdout if stdout else b""


def _generic_encode(encoder: Path, bitmap: Path) -> bytes:
    return _run(
        [str(encoder.resolve()), "-p", bitmap.name], cwd=bitmap.parent, stdout=True
    )


def _generic_roundtrip(decoder: Path, stream: Path, expected: np.ndarray, output: Path) -> None:
    _run(
        [
            str(decoder.resolve()), "-q", "-e", "-t", "pbm", "-o", str(output),
            str(stream),
        ]
    )
    if not np.array_equal(expected, _bitmap(output)):
        raise RuntimeError(f"generic stream failed exact round trip: {stream}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-repetitions", type=int, default=4)
    parser.add_argument("--spatial-distance", type=float, default=0.02)
    parser.add_argument("--descriptor-distance", type=float, default=0.30)
    parser.add_argument("--cartoon-floor", type=float, default=0.0)
    parser.add_argument(
        "--core-mode", choices=("intersection", "median"), default="intersection"
    )
    parser.add_argument(
        "--glyph-count-frontier",
        help="comma-separated most-frequent glyph-class counts to terminal-measure",
    )
    parser.add_argument(
        "--fast-cartoon",
        action="store_true",
        help="use the signed-distance field directly while screening byte viability",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    manifest = json.loads((args.assets / "manifest.json").read_text())
    metadata_values = list(manifest["pages"])
    sources = [
        _bitmap(args.assets / f"page-{int(item['page']):04d}" / "source-mask.pbm")
        for item in metadata_values
    ]
    config = GlyphResidualConfig(
        minimum_repetitions=args.minimum_repetitions,
        spatial_distance=args.spatial_distance,
        descriptor_distance=args.descriptor_distance,
    )
    splitter = (
        (lambda field, _passes: (field, np.zeros_like(field)))
        if args.fast_cartoon else None
    )
    result = canonicalize_masks_global_core(
        sources,
        config,
        meyer_splitter=splitter,
        cartoon_floor=args.cartoon_floor,
        core_mode=args.core_mode,
    )
    atlas = serialize_core_atlas(result)
    decoded_cores = decode_core_atlas(atlas)
    if any(
        not np.array_equal(expected, actual)
        for expected, actual in zip(result.core_masks, decoded_cores)
    ):
        raise RuntimeError("atlas serialization failed exact round trip")
    (args.out / "glyph-core-atlas.gca").write_bytes(atlas)
    corrections = serialize_core_corrections(result)
    decoded_refined = decode_refined_core_atlas(atlas, corrections)
    if any(
        not np.array_equal(source, refined | remainder)
        for source, refined, remainder in zip(
            result.source_masks,
            decoded_refined,
            result.remainder_masks,
        )
    ):
        raise RuntimeError("glyph-local correction stream failed exact round trip")
    (args.out / "glyph-core-corrections.gcc").write_bytes(corrections)
    fixed_refinement = serialize_fixed_refinement_atlas(result)
    decoded_fixed = decode_fixed_refinement_atlas(fixed_refinement)
    if any(
        not np.array_equal(source, refined | remainder)
        for source, refined, remainder in zip(
            result.source_masks, decoded_fixed, result.remainder_masks
        )
    ):
        raise RuntimeError("fixed refinement atlas failed exact round trip")
    (args.out / "glyph-core-fixed-refinement.gcf").write_bytes(fixed_refinement)
    split_refinement = serialize_split_refinement_atlas(result)
    decoded_split = decode_split_refinement_atlas(split_refinement)
    if any(
        not np.array_equal(source, refined | remainder)
        for source, refined, remainder in zip(
            result.source_masks, decoded_split, result.remainder_masks
        )
    ):
        raise RuntimeError("split refinement atlas failed exact round trip")
    (args.out / "glyph-core-split-refinement.gcs").write_bytes(split_refinement)

    page_records = []
    residual_total = 0
    remainder_total = 0
    source_generic_total = 0
    with tempfile.TemporaryDirectory(prefix="glyph-core-atlas-") as directory:
        work = Path(directory)
        for metadata, source, core, residual, remainder in zip(
            metadata_values, result.source_masks, result.core_masks,
            result.residual_masks, result.remainder_masks
        ):
            page = int(metadata["page"])
            core_path = args.out / f"core-{page:04d}.pbm"
            residual_path = args.out / f"residual-{page:04d}.pbm"
            remainder_path = args.out / f"remainder-{page:04d}.pbm"
            _write_bitmap(core_path, core)
            _write_bitmap(residual_path, residual)
            _write_bitmap(remainder_path, remainder)

            local_residual = work / f"residual-{page:04d}.pbm"
            local_source = work / f"source-{page:04d}.pbm"
            local_remainder = work / f"remainder-{page:04d}.pbm"
            _write_bitmap(local_residual, residual)
            _write_bitmap(local_source, source)
            _write_bitmap(local_remainder, remainder)
            residual_data = _generic_encode(args.encoder, local_residual)
            source_data = _generic_encode(args.encoder, local_source)
            remainder_data = _generic_encode(args.encoder, local_remainder)
            residual_stream = args.out / f"residual-{page:04d}.jb2"
            residual_stream.write_bytes(residual_data)
            source_stream = work / f"source-{page:04d}.jb2"
            source_stream.write_bytes(source_data)
            remainder_stream = args.out / f"remainder-{page:04d}.jb2"
            remainder_stream.write_bytes(remainder_data)
            _generic_roundtrip(
                args.decoder,
                residual_stream,
                residual,
                work / f"residual-decoded-{page:04d}.pbm",
            )
            _generic_roundtrip(
                args.decoder,
                source_stream,
                source,
                work / f"source-decoded-{page:04d}.pbm",
            )
            _generic_roundtrip(
                args.decoder,
                remainder_stream,
                remainder,
                work / f"remainder-decoded-{page:04d}.pbm",
            )
            residual_total += len(residual_data)
            remainder_total += len(remainder_data)
            source_generic_total += len(source_data)
            record = {
                "page": page,
                "source_embedded_bytes": int(metadata["mask_stream_bytes"]),
                "source_generic_bytes": len(source_data),
                "residual_generic_bytes": len(residual_data),
                "remainder_generic_bytes": len(remainder_data),
                "source_pixels": int(np.count_nonzero(source)),
                "core_pixels": int(np.count_nonzero(core)),
                "residual_pixels": int(np.count_nonzero(residual)),
                "remainder_pixels": int(np.count_nonzero(remainder)),
                "exact": bool(np.array_equal(source, core ^ residual)),
            }
            page_records.append(record)
            print(json.dumps(record), flush=True)

    source_embedded_total = sum(item["source_embedded_bytes"] for item in page_records)
    candidate_total = len(atlas) + residual_total
    local_candidate_total = len(atlas) + len(corrections) + remainder_total
    fixed_candidate_total = len(fixed_refinement) + remainder_total
    split_candidate_total = len(split_refinement) + remainder_total
    frontier_records = []
    if args.glyph_count_frontier:
        requested = sorted({
            int(value) for value in args.glyph_count_frontier.split(",")
            if value.strip() and int(value) > 0
        })
        with tempfile.TemporaryDirectory(prefix="glyph-core-frontier-") as directory:
            frontier_work = Path(directory)
            for glyph_count in requested:
                if glyph_count > len(result.glyphs):
                    continue
                subset = subset_core_atlas(result, range(glyph_count))
                subset_atlas = serialize_core_atlas(subset)
                subset_corrections = serialize_core_corrections(subset)
                subset_fixed = serialize_fixed_refinement_atlas(subset)
                subset_split = serialize_split_refinement_atlas(subset)
                subset_fixed_decoded = decode_fixed_refinement_atlas(subset_fixed)
                if any(
                    not np.array_equal(source, refined | remainder)
                    for source, refined, remainder in zip(
                        subset.source_masks,
                        subset_fixed_decoded,
                        subset.remainder_masks,
                    )
                ):
                    raise RuntimeError("frontier fixed refinement is not exact")
                subset_remainder_bytes = 0
                for metadata, remainder in zip(metadata_values, subset.remainder_masks):
                    page = int(metadata["page"])
                    target = frontier_work / f"remainder-{glyph_count:04d}-{page:04d}.pbm"
                    _write_bitmap(target, remainder)
                    subset_remainder_bytes += len(_generic_encode(args.encoder, target))
                total = len(subset_atlas) + len(subset_corrections) + subset_remainder_bytes
                record = {
                    "glyph_count": glyph_count,
                    "minimum_cluster_size": int(subset.cluster_sizes[-1]),
                    "occurrences": len(subset.occurrences),
                    "atlas_bytes": len(subset_atlas),
                    "correction_stream_bytes": len(subset_corrections),
                    "remainder_generic_bytes": subset_remainder_bytes,
                    "candidate_mask_bytes": total,
                    "delta_from_embedded_bytes": total - source_embedded_total,
                    "fixed_refinement_bytes": len(subset_fixed),
                    "fixed_candidate_mask_bytes": len(subset_fixed) + subset_remainder_bytes,
                    "fixed_delta_from_embedded_bytes": (
                        len(subset_fixed) + subset_remainder_bytes - source_embedded_total
                    ),
                    "split_refinement_bytes": len(subset_split),
                    "split_candidate_mask_bytes": len(subset_split) + subset_remainder_bytes,
                    "split_delta_from_embedded_bytes": (
                        len(subset_split) + subset_remainder_bytes - source_embedded_total
                    ),
                    "exact": True,
                }
                frontier_records.append(record)
                print(json.dumps({"frontier": record}), flush=True)
    report = {
        "method": "exact conservative global glyph-core atlas plus generic missing-ink residual",
        "config": config_dict(config),
        "cartoon_floor": args.cartoon_floor,
        "core_mode": args.core_mode,
        "fast_cartoon": args.fast_cartoon,
        "atlas_bytes": len(atlas),
        "residual_generic_bytes": residual_total,
        "candidate_mask_bytes": candidate_total,
        "correction_stream_bytes": len(corrections),
        "remainder_generic_bytes": remainder_total,
        "glyph_local_candidate_mask_bytes": local_candidate_total,
        "fixed_refinement_bytes": len(fixed_refinement),
        "fixed_candidate_mask_bytes": fixed_candidate_total,
        "fixed_delta_from_embedded_bytes": fixed_candidate_total - source_embedded_total,
        "split_refinement_bytes": len(split_refinement),
        "split_candidate_mask_bytes": split_candidate_total,
        "split_delta_from_embedded_bytes": split_candidate_total - source_embedded_total,
        "source_embedded_mask_bytes": source_embedded_total,
        "source_reencoded_generic_bytes": source_generic_total,
        "delta_from_embedded_bytes": candidate_total - source_embedded_total,
        "delta_from_reencoded_generic_bytes": candidate_total - source_generic_total,
        "glyph_local_delta_from_embedded_bytes": local_candidate_total - source_embedded_total,
        "saved_fraction_from_embedded": (
            (source_embedded_total - candidate_total) / source_embedded_total
            if source_embedded_total else 0.0
        ),
        "precodec_exact": all(item["exact"] for item in page_records),
        "atlas_roundtrip_exact": True,
        "diagnostics": result.diagnostics(),
        "glyph_count_frontier": frontier_records,
        "pages": page_records,
        "total_seconds": time.monotonic() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "atlas_bytes", "residual_generic_bytes", "candidate_mask_bytes",
            "correction_stream_bytes", "remainder_generic_bytes",
            "glyph_local_candidate_mask_bytes", "glyph_local_delta_from_embedded_bytes",
            "source_embedded_mask_bytes", "delta_from_embedded_bytes",
            "saved_fraction_from_embedded", "total_seconds",
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
