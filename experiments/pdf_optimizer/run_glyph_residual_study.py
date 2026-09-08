#!/usr/bin/env python3
"""Terminal-measure canonical JBIG2 envelopes plus foreground residuals."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
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
    compose_mrc,
    config_dict,
    denoise_foreground_texture_family,
    edge_rmse,
    foreground_owned_composite,
    rmse,
)
from pdf_optimizer.jbig2 import append_cropped_xor_region


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def _run(command: list[str], *, stdout: bool = False, cwd: Path | None = None) -> bytes:
    completed = subprocess.run(
        command, check=True, capture_output=True, timeout=300, cwd=cwd
    )
    return completed.stdout if stdout else b""


def _generic_encode(encoder: Path, bitmap: Path) -> bytes:
    # The development Leptonica build fails absolute fopen paths but accepts
    # the same file relative to cwd.
    return _run(
        [str(encoder.resolve()), "-p", bitmap.name],
        stdout=True,
        cwd=bitmap.parent,
    )


def _decode(
    decoder: Path,
    output: Path,
    page_stream: Path,
    globals_stream: Path | None = None,
) -> None:
    inputs = [str(page_stream)] if globals_stream is None else [str(globals_stream), str(page_stream)]
    _run(
        [str(decoder), "-q", "-e", "-t", "pbm", "-o", str(output), *inputs]
    )


def _bitmap(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _write_bitmap(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").convert("1").save(path)


def _exact_generic(
    target: Path, encoder: Path, decoder: Path, work: Path, label: str
) -> tuple[bytes, dict]:
    data = _generic_encode(encoder, target)
    stream = work / f"{label}-generic.jb2"
    decoded = work / f"{label}-generic.pbm"
    stream.write_bytes(data)
    _decode(decoder, decoded, stream)
    exact = bool(np.array_equal(_bitmap(target), _bitmap(decoded)))
    if not exact:
        raise RuntimeError(f"generic {label} mask was not pixel exact")
    return data, {"kind": "generic", "page_bytes": len(data), "globals_bytes": 0, "total_bytes": len(data), "exact": True}


def _exact_symbol(
    target: Path,
    encoder: Path,
    decoder: Path,
    work: Path,
    *,
    threshold: float,
    label: str,
) -> tuple[bytes, bytes, dict]:
    base = work / f"{label}-symbol"
    local_target = work / f"{label}-target.pbm"
    shutil.copyfile(target, local_target)
    _run(
        [
            str(encoder.resolve()), "-p", "-s", "-r", "-t", str(threshold),
            "-b", base.name, local_target.name,
        ],
        cwd=work,
    )
    globals_path = base.with_suffix(".sym")
    page_path = Path(f"{base}.0000")
    decoded_path = work / f"{label}-symbol.pbm"
    _decode(decoder, decoded_path, page_path, globals_path)
    target_mask = _bitmap(target)
    decoded_mask = _bitmap(decoded_path)
    page_data = page_path.read_bytes()
    residual_pixels = int(np.count_nonzero(target_mask != decoded_mask))
    residual_bytes = 0
    residual_box = None
    if residual_pixels:
        with Image.open(target) as source_image, Image.open(decoded_path) as symbol_image:
            difference = ImageChops.difference(
                source_image.convert("L"), symbol_image.convert("L")
            )
            box = difference.getbbox()
            if box is None:
                raise RuntimeError("symbol comparison was inconsistent")
            residual_box = (
                max(0, box[0] - 4), max(0, box[1] - 4),
                min(source_image.width, box[2] + 4),
                min(source_image.height, box[3] + 4),
            )
            residual_path = work / f"{label}-residual.pbm"
            ImageOps.invert(difference.crop(residual_box)).convert("1").save(residual_path)
        residual = _generic_encode(encoder, residual_path)
        page_data = append_cropped_xor_region(
            page_data, residual, x=residual_box[0], y=residual_box[1]
        )
        residual_bytes = len(page_data) - page_path.stat().st_size

    hybrid_path = work / f"{label}-hybrid.jb2"
    verified_path = work / f"{label}-hybrid.pbm"
    hybrid_path.write_bytes(page_data)
    _decode(decoder, verified_path, hybrid_path, globals_path)
    exact = bool(np.array_equal(target_mask, _bitmap(verified_path)))
    if not exact:
        raise RuntimeError(f"symbol {label} hybrid was not pixel exact")
    globals_data = globals_path.read_bytes()
    return page_data, globals_data, {
        "kind": "refined_symbol",
        "threshold": threshold,
        "page_bytes": len(page_data),
        "globals_bytes": len(globals_data),
        "total_bytes": len(page_data) + len(globals_data),
        "symbol_page_bytes": page_path.stat().st_size,
        "residual_pixels": residual_pixels,
        "residual_segment_bytes": residual_bytes,
        "residual_box": residual_box,
        "exact": exact,
    }


def _encode_jpx(image: np.ndarray, rate: float) -> bytes:
    output = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB").save(
        output,
        format="JPEG2000",
        irreversible=True,
        quality_mode="rates",
        quality_layers=[float(rate)],
    )
    return output.getvalue()


def _decode_jpx(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _psnr(error: float, peak: float = 255.0) -> float:
    return 20.0 * math.log10(peak / max(error, 1e-12))


def _preview_box(mask: np.ndarray, width: int = 900, height: int = 600) -> tuple[int, int, int, int]:
    # Choose the densest coarse window rather than assuming text is centred.
    coarse = np.asarray(
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").resize(
            (max(1, mask.shape[1] // 32), max(1, mask.shape[0] // 32)),
            Image.Resampling.BOX,
        ),
        dtype=np.float64,
    )
    maximum = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    cy = int((maximum[0] + 0.5) * mask.shape[0] / coarse.shape[0])
    cx = int((maximum[1] + 0.5) * mask.shape[1] / coarse.shape[1])
    x0 = min(max(0, cx - width // 2), max(0, mask.shape[1] - width))
    y0 = min(max(0, cy - height // 2), max(0, mask.shape[0] - height))
    return x0, y0, min(mask.shape[1], x0 + width), min(mask.shape[0], y0 + height)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbol-threshold", type=float, default=0.92)
    parser.add_argument("--texture-gains", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--jpx-rates", default="50,100,200,400,800")
    parser.add_argument("--ink-rmse-limit", type=float, default=8.0)
    parser.add_argument("--edge-rmse-limit", type=float, default=18.0)
    parser.add_argument("--minimum-repetitions", type=int, default=4)
    parser.add_argument("--descriptor-distance", type=float, default=0.30)
    parser.add_argument("--spatial-distance", type=float, default=0.02)
    parser.add_argument(
        "--screen-clusters",
        action="store_true",
        help="terminal-encode each cluster's canonical and correction masks",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    metadata = json.loads((args.assets / "source.json").read_text())
    source_mask_path = args.assets / "source-mask.pbm"
    source_mask = _bitmap(source_mask_path)
    with Image.open(args.assets / "source-foreground.png") as image:
        source_foreground = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(args.assets / "source-background.png") as image:
        source_background = np.asarray(image.convert("RGB"), dtype=np.uint8)

    config = GlyphResidualConfig(
        minimum_repetitions=args.minimum_repetitions,
        descriptor_distance=args.descriptor_distance,
        spatial_distance=args.spatial_distance,
    )
    canonical = canonicalize_mask(source_mask, config)
    canonical_path = args.out / "canonical-mask.pbm"
    _write_bitmap(canonical_path, canonical.canonical_mask)
    source_composite, candidate_foreground, enlarged_background = foreground_owned_composite(
        source_background,
        source_foreground,
        source_mask,
        canonical.canonical_mask,
    )
    texture_family = denoise_foreground_texture_family(
        candidate_foreground,
        enlarged_background,
        canonical.canonical_mask,
        texture_gains=_csv_floats(args.texture_gains),
        virtual_passes=config.meyer_virtual_passes,
    )

    cluster_screen: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="glyph-residual-codecs-") as directory:
        work = Path(directory)
        source_generic_data, source_generic = _exact_generic(
            source_mask_path, args.encoder, args.decoder, work, "source"
        )
        if args.screen_clusters:
            for cluster_index, cluster in enumerate(canonical.clusters):
                isolated = canonicalize_mask(
                    source_mask,
                    config,
                    precomputed=canonical,
                    selected_cluster_indices=(cluster_index,),
                )
                isolated_path = work / f"cluster-{cluster_index:03d}.pbm"
                _write_bitmap(isolated_path, isolated.canonical_mask)
                isolated_mask_data, _isolated_mask_record = _exact_generic(
                    isolated_path,
                    args.encoder,
                    args.decoder,
                    work,
                    f"cluster-{cluster_index:03d}-mask",
                )
                isolated_correction = isolated.canonical_mask & ~source_mask
                if isolated_correction.any():
                    isolated_correction_path = work / f"cluster-{cluster_index:03d}-correction.pbm"
                    _write_bitmap(isolated_correction_path, isolated_correction)
                    isolated_correction_data, _isolated_correction_record = _exact_generic(
                        isolated_correction_path,
                        args.encoder,
                        args.decoder,
                        work,
                        f"cluster-{cluster_index:03d}-correction",
                    )
                else:
                    isolated_correction_data = b""
                combined = len(isolated_mask_data) + len(isolated_correction_data)
                cluster_screen.append(
                    {
                        "cluster": cluster_index,
                        "members": len(cluster),
                        "added_pixels": isolated.added_pixels,
                        "canonical_mask_bytes": len(isolated_mask_data),
                        "correction_mask_bytes": len(isolated_correction_data),
                        "combined_mask_bytes": combined,
                        "delta_from_source_reencoded_generic": combined - len(source_generic_data),
                    }
                )
                print(json.dumps(cluster_screen[-1]), flush=True)
        canonical_generic_data, canonical_generic = _exact_generic(
            canonical_path, args.encoder, args.decoder, work, "canonical"
        )
        canonical_symbol_page, canonical_symbol_globals, canonical_symbol = _exact_symbol(
            canonical_path,
            args.encoder,
            args.decoder,
            work,
            threshold=args.symbol_threshold,
            label="canonical",
        )
        mask_candidates = [
            (canonical_generic, canonical_generic_data, b""),
            (canonical_symbol, canonical_symbol_page, canonical_symbol_globals),
        ]
        selected_mask, selected_mask_data, selected_globals_data = min(
            mask_candidates, key=lambda item: item[0]["total_bytes"]
        )

        correction = canonical.canonical_mask & ~source_mask
        correction_path = args.out / "correction-mask.pbm"
        _write_bitmap(correction_path, correction)
        if correction.any():
            correction_generic_data, correction_generic = _exact_generic(
                correction_path, args.encoder, args.decoder, work, "correction"
            )
            correction_symbol_page, correction_symbol_globals, correction_symbol = _exact_symbol(
                correction_path,
                args.encoder,
                args.decoder,
                work,
                threshold=args.symbol_threshold,
                label="correction",
            )
            correction_candidates = [
                (correction_generic, correction_generic_data, b""),
                (correction_symbol, correction_symbol_page, correction_symbol_globals),
            ]
            selected_correction, selected_correction_data, selected_correction_globals = min(
                correction_candidates, key=lambda item: item[0]["total_bytes"]
            )
        else:
            correction_generic = correction_symbol = None
            selected_correction = {
                "kind": "empty",
                "page_bytes": 0,
                "globals_bytes": 0,
                "total_bytes": 0,
                "exact": True,
            }
            selected_correction_data = selected_correction_globals = b""

    foreground_records: list[dict] = []
    foreground_payloads: dict[tuple[float, float], bytes] = {}
    selected_composites: dict[tuple[float, float], np.ndarray] = {}
    for gain, foreground_value in texture_family.items():
        precodec_composite = compose_mrc(
            enlarged_background, foreground_value, canonical.canonical_mask
        )
        precodec_rmse = rmse(source_composite, precodec_composite)
        for rate in _csv_floats(args.jpx_rates):
            encoded = _encode_jpx(foreground_value, rate)
            decoded = _decode_jpx(encoded)
            composed = compose_mrc(enlarged_background, decoded, canonical.canonical_mask)
            full_rmse = rmse(source_composite, composed)
            ink_rmse = rmse(source_composite, composed, source_mask)
            envelope_rmse = rmse(source_composite, composed, canonical.canonical_mask)
            gradient_rmse = edge_rmse(source_composite, composed)
            record = {
                "texture_gain": gain,
                "jpx_rate": rate,
                "foreground_bytes": len(encoded),
                "mask_bytes": selected_mask["total_bytes"],
                "pair_bytes": len(encoded) + selected_mask["total_bytes"],
                "precodec_rmse": precodec_rmse,
                "full_rmse": full_rmse,
                "psnr": _psnr(full_rmse),
                "ink_rmse": ink_rmse,
                "ink_psnr": _psnr(ink_rmse),
                "envelope_rmse": envelope_rmse,
                "edge_rmse": gradient_rmse,
                "passes_quality_gate": bool(
                    ink_rmse <= args.ink_rmse_limit
                    and gradient_rmse <= args.edge_rmse_limit
                ),
            }
            foreground_records.append(record)
            foreground_payloads[(gain, rate)] = encoded
            selected_composites[(gain, rate)] = composed
            print(json.dumps(record), flush=True)

    qualified = [record for record in foreground_records if record["passes_quality_gate"]]
    if qualified:
        selected_foreground = min(
            qualified, key=lambda record: (record["pair_bytes"], record["ink_rmse"])
        )
    else:
        # Preserve the full negative result and a viewable best-quality
        # candidate instead of aborting before the report can be written.
        selected_foreground = min(
            foreground_records,
            key=lambda record: (record["ink_rmse"], record["edge_rmse"], record["pair_bytes"]),
        )
    key = (selected_foreground["texture_gain"], selected_foreground["jpx_rate"])
    (args.out / "candidate-foreground.jpx").write_bytes(foreground_payloads[key])
    (args.out / "candidate-mask.jb2").write_bytes(selected_mask_data)
    globals_path = args.out / "candidate-mask-globals.jb2"
    if selected_globals_data:
        globals_path.write_bytes(selected_globals_data)
    elif globals_path.exists():
        globals_path.unlink()
    correction_stream_path = args.out / "candidate-correction-mask.jb2"
    correction_globals_path = args.out / "candidate-correction-globals.jb2"
    if selected_correction_data:
        correction_stream_path.write_bytes(selected_correction_data)
    elif correction_stream_path.exists():
        correction_stream_path.unlink()
    if selected_correction_globals:
        correction_globals_path.write_bytes(selected_correction_globals)
    elif correction_globals_path.exists():
        correction_globals_path.unlink()

    preview = _preview_box(source_mask)
    source_preview = Image.fromarray(source_composite, "RGB").crop(preview)
    candidate_preview = Image.fromarray(selected_composites[key], "RGB").crop(preview)
    source_preview.save(args.out / "preview-source.png")
    candidate_preview.save(args.out / "preview-candidate.png")
    difference = np.abs(
        np.asarray(source_preview, dtype=np.int16) - np.asarray(candidate_preview, dtype=np.int16)
    )
    Image.fromarray(np.clip(difference * 8, 0, 255).astype(np.uint8), "RGB").save(
        args.out / "preview-error-x8.png"
    )

    source_pair_bytes = int(metadata["mask_stream_bytes"] + metadata["foreground_stream_bytes"])
    layered_pair_bytes = int(
        selected_mask["total_bytes"]
        + selected_correction["total_bytes"]
        + metadata["foreground_stream_bytes"]
        + metadata["background_stream_bytes"]
    )
    report = {
        "method": "eikonal/Fourier-circle/Meyer canonical glyph envelopes with foreground-owned residual",
        "source": metadata,
        "config": config_dict(config),
        "canonical": canonical.diagnostics(),
        "cluster_screen": cluster_screen,
        "cluster_screen_winners": [
            record for record in cluster_screen
            if record["delta_from_source_reencoded_generic"] < 0
        ],
        "mask_candidates": {
            "source_reencoded_generic": source_generic,
            "canonical_generic": canonical_generic,
            "canonical_refined_symbol": canonical_symbol,
            "selected": selected_mask,
        },
        "correction_mask_candidates": {
            "pixels": int(np.count_nonzero(correction)),
            "generic": correction_generic,
            "refined_symbol": correction_symbol,
            "selected": selected_correction,
        },
        "foreground_candidates": foreground_records,
        "selected_foreground": selected_foreground,
        "quality_gate_satisfied": bool(qualified),
        "source_pair_bytes": source_pair_bytes,
        "candidate_pair_bytes": selected_foreground["pair_bytes"],
        "pair_delta_bytes": selected_foreground["pair_bytes"] - source_pair_bytes,
        "pair_saved_fraction": (
            (source_pair_bytes - selected_foreground["pair_bytes"]) / source_pair_bytes
            if source_pair_bytes else 0.0
        ),
        "layered_pair_bytes": layered_pair_bytes,
        "layered_pair_delta_bytes": layered_pair_bytes - source_pair_bytes,
        "layered_pair_saved_fraction": (
            (source_pair_bytes - layered_pair_bytes) / source_pair_bytes
            if source_pair_bytes else 0.0
        ),
        "layered_precodec_exact": True,
        "precodec_ownership_exact": True,
        "preview_box": list(preview),
        "total_seconds": time.monotonic() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source_pair_bytes": report["source_pair_bytes"],
        "candidate_pair_bytes": report["candidate_pair_bytes"],
        "pair_delta_bytes": report["pair_delta_bytes"],
        "pair_saved_fraction": report["pair_saved_fraction"],
        "layered_pair_bytes": report["layered_pair_bytes"],
        "layered_pair_delta_bytes": report["layered_pair_delta_bytes"],
        "layered_pair_saved_fraction": report["layered_pair_saved_fraction"],
        "selected_mask": selected_mask,
        "selected_foreground": selected_foreground,
        "total_seconds": report["total_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
