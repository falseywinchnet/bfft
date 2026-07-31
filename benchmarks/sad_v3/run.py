#!/usr/bin/env python3
"""Run v3 fast/full on a SAD corpus and emit metrics plus rate estimates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT, ROOT / "experiments", ROOT / "viewer"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmarks.sad_v3.rate_model import estimate_v3_rate  # noqa: E402
from experiments.segmenting_v3 import (  # noqa: E402
    SegmentingV3Config,
    build_segmenting_v3,
)
from port_needed.fast_image_ops import resize  # noqa: E402


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _fit_to_side(image: np.ndarray, maximum_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    shape = (
        max(16, round(height * scale)),
        max(16, round(width * scale)),
    )
    if shape == (height, width):
        return image
    return np.clip(
        resize(image, shape, order=1, anti_aliasing=True),
        0.0,
        1.0,
    )


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def _psnr(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    mse = float(np.mean(np.square(reference - reconstruction)))
    return -10.0 * math.log10(max(mse, 1e-15))


def _linear_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> dict:
    reference_linear = _srgb_to_linear(reference)
    reconstruction_linear = _srgb_to_linear(reconstruction)
    output = {"psnr_db": _psnr(reference_linear, reconstruction_linear)}
    try:
        from skimage.metrics import structural_similarity

        output["ssim"] = float(structural_similarity(
            reference_linear,
            reconstruction_linear,
            data_range=1.0,
            channel_axis=2,
        ))
    except ImportError:
        output["ssim"] = None
    return output


def _png_bytes(image: np.ndarray) -> int:
    quantized = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(quantized, mode="RGB").save(
        stream,
        format="PNG",
        optimize=True,
    )
    return len(stream.getvalue())


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _benchmark_one(
    path: Path,
    dataset: str,
    mode: str,
    config: SegmentingV3Config,
    fast_side: int,
    save_reconstruction: Path | None,
) -> dict:
    native = _read_rgb(path)
    work = native if mode == "full" else _fit_to_side(native, fast_side)
    started = time.perf_counter()
    result = build_segmenting_v3(work, config)
    wall_ms = 1000.0 * (time.perf_counter() - started)
    reconstruction = result["reconstruction_rgb"]
    linear = _linear_metrics(work, reconstruction)
    rate = estimate_v3_rate(
        result,
        structural_ridges=config.structural_ridges,
        texture_ridges=config.nested_texture_ridges,
        graph_phase=config.texture_graph_phase,
    )
    output_png_bytes = _png_bytes(reconstruction)
    native_png_bytes = path.stat().st_size

    native_readout = reconstruction
    if reconstruction.shape != native.shape:
        native_readout = np.clip(
            resize(
                reconstruction,
                native.shape[:2],
                order=1,
                anti_aliasing=False,
            ),
            0.0,
            1.0,
        )
    native_linear = _linear_metrics(native, native_readout)

    if save_reconstruction is not None:
        save_reconstruction.mkdir(parents=True, exist_ok=True)
        quantized = np.clip(
            np.rint(reconstruction * 255.0),
            0,
            255,
        ).astype(np.uint8)
        Image.fromarray(quantized, mode="RGB").save(
            save_reconstruction / f"{path.stem}-{mode}.png")

    pixels = int(work.shape[0] * work.shape[1])
    return {
        "dataset": dataset,
        "image": path.name,
        "mode": mode,
        "fast_side": fast_side,
        "native_shape": list(native.shape[:2]),
        "processed_shape": list(work.shape[:2]),
        "native_equivalent_mode": bool(work.shape == native.shape),
        "model": result["model"],
        "cells": {
            "structural": len(result["centers"]),
            "texture": len(result["texture_centers"]),
            "texture_initial": len(result["texture_initial_centers"]),
            "texture_splits": result["texture_cleanup"]["split_count"],
            "texture_merges": result["texture_cleanup"]["merge_count"],
        },
        "metrics_processed": {
            "srgb_psnr_db": float(result["record"]["psnr"]),
            "linear_psnr_db": linear["psnr_db"],
            "linear_ssim": linear["ssim"],
        },
        "metrics_native_readout": {
            "linear_psnr_db": native_linear["psnr_db"],
            "linear_ssim": native_linear["ssim"],
        },
        "rate": {
            **rate,
            "reconstruction_png_bytes": output_png_bytes,
            "reconstruction_png_bpp": 8.0 * output_png_bytes / pixels,
            "source_png_bytes": native_png_bytes,
            "source_png_bpp_at_processed_pixels": (
                8.0 * native_png_bytes / pixels),
        },
        "timing_ms": {
            **result["timing"],
            "wall": wall_ms,
        },
    }


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def summarize(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        groups.setdefault(record["mode"], []).append(record)
    output = {}
    for mode, rows in sorted(groups.items()):
        output[mode] = {
            "images": len(rows),
            "native_equivalent_images": sum(
                row["native_equivalent_mode"] for row in rows),
            "mean_structural_cells": _mean([
                row["cells"]["structural"] for row in rows]),
            "mean_texture_cells": _mean([
                row["cells"]["texture"] for row in rows]),
            "mean_linear_psnr_db": _mean([
                row["metrics_processed"]["linear_psnr_db"] for row in rows]),
            "mean_linear_ssim": _mean([
                row["metrics_processed"]["linear_ssim"] for row in rows]),
            "mean_native_readout_linear_psnr_db": _mean([
                row["metrics_native_readout"]["linear_psnr_db"]
                for row in rows
            ]),
            "mean_native_readout_linear_ssim": _mean([
                row["metrics_native_readout"]["linear_ssim"]
                for row in rows
            ]),
            "mean_srgb_psnr_db": _mean([
                row["metrics_processed"]["srgb_psnr_db"] for row in rows]),
            "mean_estimated_stream_bpp": _mean([
                row["rate"]["estimated_stream_bpp"] for row in rows]),
            "mean_sad_layered_site_proxy_bpp": _mean([
                row["rate"]["sad_layered_site_proxy_bpp"] for row in rows]),
            "mean_reconstruction_png_bpp": _mean([
                row["rate"]["reconstruction_png_bpp"] for row in rows]),
            "mean_total_ms": _mean([
                row["timing_ms"]["total_ms"] for row in rows]),
            "median_total_ms": float(np.median([
                row["timing_ms"]["total_ms"] for row in rows])),
        }
    return output


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# V3 SAD-aligned benchmark summary",
        "",
        "| mode | images | native-equivalent | processed PSNR | "
        "native-readout PSNR | processed SSIM | native-readout SSIM | "
        "structural cells | texture cells | v3 estimated BPP | "
        "SAD-site proxy BPP | PNG BPP | total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|",
    ]
    for mode, row in sorted(summary.items()):
        lines.append(
            f"| {mode} | {row['images']} | "
            f"{row['native_equivalent_images']} | "
            f"{row['mean_linear_psnr_db']:.3f} | "
            f"{row['mean_native_readout_linear_psnr_db']:.3f} | "
            f"{row['mean_linear_ssim']:.5f} | "
            f"{row['mean_native_readout_linear_ssim']:.5f} | "
            f"{row['mean_structural_cells']:.1f} | "
            f"{row['mean_texture_cells']:.1f} | "
            f"{row['mean_estimated_stream_bpp']:.3f} | "
            f"{row['mean_sad_layered_site_proxy_bpp']:.3f} | "
            f"{row['mean_reconstruction_png_bpp']:.3f} | "
            f"{row['mean_total_ms']:.1f} |"
        )
    lines.extend([
        "",
        "The v3 rate is a declared fp16 parameter-plus-topology estimate, "
        "not an emitted codec stream. The SAD-site proxy applies SAD's "
        "16-byte site accounting to both v3 layers. PNG BPP is the actual "
        "lossless size of the quantized reconstruction and is not v3's rate.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("kodak", "div2k"),
        default="kodak",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "full", "both"),
        default="both",
    )
    parser.add_argument("--fast-side", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="regenerate summary files from an existing records.jsonl",
    )
    args = parser.parse_args()

    input_directory = (
        args.input
        if args.input is not None
        else HERE / "data" / (
            "kodak" if args.dataset == "kodak" else "div2k_sample")
    )
    output = (
        args.output
        if args.output is not None
        else HERE / "results" / args.dataset
    ).resolve()
    if args.summarize_only:
        jsonl = output / "records.jsonl"
        if not jsonl.is_file():
            raise SystemExit(f"No existing benchmark records: {jsonl}")
        records = [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        aggregate = summarize(records)
        (output / "summary.json").write_text(
            json.dumps(aggregate, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_markdown(aggregate, output / "summary.md")
        print(f"Summary regenerated in {output}")
        return 0

    paths = sorted(input_directory.resolve().glob("*.png"))
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        downloader = (
            "download_kodak.py"
            if args.dataset == "kodak"
            else "download_div2k_sample.py"
        )
        raise SystemExit(
            f"No PNG images found in {input_directory.resolve()}. "
            f"Run {downloader} first.")
    modes = ("fast", "full") if args.mode == "both" else (args.mode,)
    config = replace(
        SegmentingV3Config(),
        structural_topology="canonical_v2",
    )
    output.mkdir(parents=True, exist_ok=True)

    if not args.no_warmup:
        warm = _read_rgb(paths[0])
        warm = _fit_to_side(warm, min(args.fast_side, 160))
        warm_config = replace(
            config,
            structural_allocation_side=160,
            structural_safety_cells=4096,
            texture_safety_cells=8192,
        )
        print("warming native and Numba kernels ...", flush=True)
        build_segmenting_v3(warm, warm_config)

    manifest = {
        "created_unix": time.time(),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "dataset": (
            "Kodak lossless true color image suite"
            if args.dataset == "kodak"
            else "DIV2K validation sample"
        ),
        "config": asdict(config),
        "fast_side": args.fast_side,
        "modes": list(modes),
        "images": [path.name for path in paths],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    records = []
    jsonl = output / "records.jsonl"
    reconstruction_directory = (
        output / "reconstructions"
        if args.save_reconstructions
        else None
    )
    with jsonl.open("w", encoding="utf-8") as stream:
        for path in paths:
            for mode in modes:
                print(f"{path.name} [{mode}] ...", end=" ", flush=True)
                record = _benchmark_one(
                    path,
                    args.dataset,
                    mode,
                    config,
                    args.fast_side,
                    reconstruction_directory,
                )
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(
                    f"{record['metrics_processed']['linear_psnr_db']:.2f} dB, "
                    f"{record['rate']['estimated_stream_bpp']:.2f} est. BPP, "
                    f"{record['timing_ms']['total_ms']:.0f} ms",
                    flush=True,
                )

    aggregate = summarize(records)
    (output / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(aggregate, output / "summary.md")
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
