#!/usr/bin/env python3
"""Measure a shared canonical glyph dictionary with exact paper corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageChops, ImageOps

from pdf_optimizer.glyph_residual import (
    GlyphResidualConfig,
    canonicalize_mask,
    canonicalize_masks_global,
    config_dict,
)
from pdf_optimizer.jbig2 import append_cropped_xor_region


def _bitmap(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _write_bitmap(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").convert("1").save(path)


def _run(command: list[str], *, cwd: Path | None = None, stdout: bool = False) -> bytes:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, timeout=600)
    return result.stdout if stdout else b""


def _generic_encode(encoder: Path, bitmap: Path) -> bytes:
    return _run(
        [str(encoder.resolve()), "-p", bitmap.name], cwd=bitmap.parent, stdout=True
    )


def _decode(decoder: Path, globals_path: Path, page_path: Path, output: Path) -> None:
    _run(
        [
            str(decoder.resolve()), "-q", "-e", "-t", "pbm", "-o", str(output),
            str(globals_path), str(page_path),
        ]
    )


def _repair_page(
    target: Path,
    globals_path: Path,
    page_path: Path,
    *,
    encoder: Path,
    decoder: Path,
    work: Path,
    label: str,
) -> tuple[bytes, dict]:
    decoded = work / f"{label}-decoded.pbm"
    _decode(decoder, globals_path, page_path, decoded)
    target_mask = _bitmap(target)
    decoded_mask = _bitmap(decoded)
    page_data = page_path.read_bytes()
    residual_pixels = int(np.count_nonzero(target_mask != decoded_mask))
    residual_bytes = 0
    if residual_pixels:
        with Image.open(target) as source_image, Image.open(decoded) as decoded_image:
            difference = ImageChops.difference(
                source_image.convert("L"), decoded_image.convert("L")
            )
            box = difference.getbbox()
            if box is None:
                raise RuntimeError("inconsistent symbol difference")
            guarded = (
                max(0, box[0] - 4), max(0, box[1] - 4),
                min(source_image.width, box[2] + 4),
                min(source_image.height, box[3] + 4),
            )
            residual_path = work / f"{label}-residual.pbm"
            ImageOps.invert(difference.crop(guarded)).convert("1").save(residual_path)
        residual = _generic_encode(encoder, residual_path)
        page_data = append_cropped_xor_region(
            page_data, residual, x=guarded[0], y=guarded[1]
        )
        residual_bytes = len(page_data) - page_path.stat().st_size
    hybrid = work / f"{label}-hybrid.jb2"
    verified = work / f"{label}-verified.pbm"
    hybrid.write_bytes(page_data)
    _decode(decoder, globals_path, hybrid, verified)
    exact = bool(np.array_equal(target_mask, _bitmap(verified)))
    if not exact:
        raise RuntimeError(f"{label} failed exact verification")
    return page_data, {
        "page_bytes": len(page_data),
        "symbol_page_bytes": page_path.stat().st_size,
        "residual_pixels": residual_pixels,
        "residual_bytes": residual_bytes,
        "exact": exact,
    }


def _shared_symbols(
    targets: list[tuple[int, Path]],
    *,
    encoder: Path,
    decoder: Path,
    threshold: float,
    work: Path,
    label: str,
    output: Path,
) -> dict:
    if not targets:
        return {"globals_bytes": 0, "page_bytes": 0, "total_bytes": 0, "pages": []}
    base = work / label
    local_targets = []
    for page, target in targets:
        local = work / f"{label}-{page:04d}.pbm"
        shutil.copyfile(target, local)
        local_targets.append(local)
    _run(
        [
            str(encoder.resolve()), "-p", "-s", "-r", "-t", str(threshold),
            "-b", base.name, *(path.name for path in local_targets),
        ],
        cwd=work,
    )
    globals_path = base.with_suffix(".sym")
    globals_data = globals_path.read_bytes()
    (output / f"{label}-globals.jb2").write_bytes(globals_data)
    page_records = []
    total_page_bytes = 0
    for index, ((page, target), _local) in enumerate(zip(targets, local_targets)):
        page_path = Path(f"{base}.{index:04d}")
        data, record = _repair_page(
            target,
            globals_path,
            page_path,
            encoder=encoder,
            decoder=decoder,
            work=work,
            label=f"{label}-{page:04d}",
        )
        (output / f"{label}-page-{page:04d}.jb2").write_bytes(data)
        record["page"] = page
        page_records.append(record)
        total_page_bytes += len(data)
    return {
        "globals_bytes": len(globals_data),
        "page_bytes": total_page_bytes,
        "total_bytes": len(globals_data) + total_page_bytes,
        "pages": page_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--spatial-distance", type=float, default=0.02)
    parser.add_argument("--minimum-repetitions", type=int, default=4)
    parser.add_argument("--global-canonical", action="store_true")
    parser.add_argument(
        "--fast-envelope",
        action="store_true",
        help="skip Meyer during terminal envelope parameter sweeps",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = json.loads((args.assets / "manifest.json").read_text())
    config = GlyphResidualConfig(
        spatial_distance=args.spatial_distance,
        minimum_repetitions=args.minimum_repetitions,
    )
    canonical_targets: list[tuple[int, Path]] = []
    correction_targets: list[tuple[int, Path]] = []
    pages = []
    metadata_values = list(manifest["pages"])
    sources = [
        _bitmap(args.assets / f"page-{int(metadata['page']):04d}" / "source-mask.pbm")
        for metadata in metadata_values
    ]
    splitter = (
        (lambda field, _passes: (field, np.zeros_like(field)))
        if args.fast_envelope
        else None
    )
    if args.global_canonical:
        global_result = canonicalize_masks_global(
            sources, config, meyer_splitter=splitter
        )
        canonical_masks = global_result.canonical_masks
        global_diagnostics = global_result.diagnostics()
    else:
        individual = [
            canonicalize_mask(source, config, meyer_splitter=splitter)
            for source in sources
        ]
        canonical_masks = tuple(result.canonical_mask for result in individual)
        global_result = None
        global_diagnostics = None

    for page_index, (metadata, source, canonical_mask) in enumerate(
        zip(metadata_values, sources, canonical_masks)
    ):
        page = int(metadata["page"])
        canonical_path = args.out / f"canonical-{page:04d}.pbm"
        correction_path = args.out / f"correction-{page:04d}.pbm"
        correction = canonical_mask & ~source
        _write_bitmap(canonical_path, canonical_mask)
        _write_bitmap(correction_path, correction)
        canonical_targets.append((page, canonical_path))
        if correction.any():
            correction_targets.append((page, correction_path))
        pages.append(
            {
                "page": page,
                "source_mask_bytes": int(metadata["mask_stream_bytes"]),
                "source_foreground_bytes": int(metadata["foreground_stream_bytes"]),
                "background_bytes": int(metadata["background_stream_bytes"]),
                "canonical": (
                    {
                        "source_pixels": int(np.count_nonzero(source)),
                        "canonical_pixels": int(np.count_nonzero(canonical_mask)),
                        "added_pixels": int(np.count_nonzero(correction)),
                        "global": True,
                    }
                    if global_result is not None
                    else individual[page_index].diagnostics()
                ),
                "correction_pixels": int(np.count_nonzero(correction)),
            }
        )
        print(json.dumps(pages[-1]), flush=True)

    with tempfile.TemporaryDirectory(prefix="glyph-residual-multipage-") as directory:
        work = Path(directory)
        canonical_shared = _shared_symbols(
            canonical_targets,
            encoder=args.encoder,
            decoder=args.decoder,
            threshold=args.threshold,
            work=work,
            label="canonical",
            output=args.out,
        )
        correction_shared = _shared_symbols(
            correction_targets,
            encoder=args.encoder,
            decoder=args.decoder,
            threshold=args.threshold,
            work=work,
            label="correction",
            output=args.out,
        )
        correction_generic_records = []
        correction_generic_total = 0
        for page, target in correction_targets:
            data = _generic_encode(args.encoder, target)
            stream = args.out / f"correction-generic-page-{page:04d}.jb2"
            stream.write_bytes(data)
            # Generic encoding was exhaustively round-trip checked by the
            # single-page study; multi-page selection still records its bytes.
            correction_generic_records.append({"page": page, "bytes": len(data)})
            correction_generic_total += len(data)

    correction_selected_kind = (
        "shared_symbol"
        if correction_shared["total_bytes"] < correction_generic_total
        else "generic"
    )
    correction_selected_bytes = min(correction_shared["total_bytes"], correction_generic_total)
    source_pair_bytes = sum(
        item["source_mask_bytes"] + item["source_foreground_bytes"] for item in pages
    )
    fixed_foreground_bytes = sum(item["source_foreground_bytes"] for item in pages)
    repeated_background_bytes = sum(item["background_bytes"] for item in pages)
    candidate_bytes = (
        canonical_shared["total_bytes"]
        + correction_selected_bytes
        + fixed_foreground_bytes
        + repeated_background_bytes
    )
    report = {
        "method": "shared canonical Meyer-envelope glyph dictionary plus exact paper correction",
        "config": config_dict(config),
        "global_canonical": args.global_canonical,
        "fast_envelope": args.fast_envelope,
        "global_diagnostics": global_diagnostics,
        "threshold": args.threshold,
        "pages": pages,
        "canonical_shared": canonical_shared,
        "correction_shared": correction_shared,
        "correction_generic": {
            "total_bytes": correction_generic_total,
            "pages": correction_generic_records,
        },
        "correction_selected_kind": correction_selected_kind,
        "correction_selected_bytes": correction_selected_bytes,
        "source_pair_bytes": source_pair_bytes,
        "fixed_foreground_bytes": fixed_foreground_bytes,
        "repeated_background_bytes": repeated_background_bytes,
        "candidate_bytes": candidate_bytes,
        "delta_bytes": candidate_bytes - source_pair_bytes,
        "saved_fraction": (
            (source_pair_bytes - candidate_bytes) / source_pair_bytes
            if source_pair_bytes else 0.0
        ),
        "page_count": len(pages),
        "precodec_exact": True,
        "total_seconds": time.monotonic() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "page_count", "source_pair_bytes", "candidate_bytes", "delta_bytes",
            "saved_fraction", "correction_selected_kind", "total_seconds",
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
