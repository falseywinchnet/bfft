#!/usr/bin/env python3
"""Invert JPEG quantization bins under the v3 structural geometry prior.

For every observed quantized coefficient q and quantizer Q, the unknown
pre-quantized coefficient lies approximately in

    Q (q - 1/2) <= c < Q (q + 1/2).

This experiment obtains a phase-correct cross-block geometry proposal from
segmenting v3 and projects that proposal into each observed interval in one
closed-form step.  It then distinguishes two questions:

1. Does the latent freedom inside the bins contain a visibly better image?
2. Can a standard JPEG express that latent image under the source byte budget?

The high-precision outputs answer the first question.  The rate-matched
half-quantizer outputs answer the second by pruning the least
geometry-supported coefficients until their actual progressive JPEG streams
fit the source byte count.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    import jpegio as jio
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        f"Install coefficient access with `{sys.executable} -m pip install jpegio`"
    ) from error

from experiments.jpeg_dct_geometry_reassembly import (  # noqa: E402
    _apply_budget_removals,
    _build_or_load_geometry,
    _coefficient_blocks,
    _image_metrics,
    _rank_budget_removals,
    _restore_component_phase_correct,
    _save_montage,
    _write_coefficients,
)


def _component_quantizers(jpeg) -> list[np.ndarray]:
    return [
        np.asarray(
            jpeg.quant_tables[component.quant_tbl_no], dtype=np.float64)
        for component in jpeg.comp_info
    ]


def _project_geometry_into_bins(
    original_blocks: list[np.ndarray],
    quantizers: list[np.ndarray],
    proposals: list[np.ndarray],
    strength: float,
) -> tuple[list[np.ndarray], dict]:
    """Return latent dequantized coefficients inside all observed bins."""
    latent = []
    clipped_count = 0
    moved_count = 0
    squared_motion = 0.0
    total = 0
    for blocks, quantizer, proposal in zip(
        original_blocks, quantizers, proposals
    ):
        center = (
            blocks.astype(np.float64) * quantizer[None, None, :, :]
        ).reshape(blocks.shape[0], blocks.shape[1], 64)
        half_width = 0.5 * quantizer.reshape(1, 1, 64)
        lower = center - half_width
        upper = center + half_width
        proposal_flat = proposal.reshape(center.shape)
        projected = np.clip(proposal_flat, lower, upper)
        clipped_count += int(np.count_nonzero(
            (proposal_flat < lower) | (proposal_flat > upper)))
        value = center + float(strength) * (projected - center)
        motion = value - center
        moved_count += int(np.count_nonzero(np.abs(motion) > 1e-12))
        squared_motion += float(np.sum(np.square(motion)))
        total += int(motion.size)
        latent.append(value.reshape(blocks.shape))
    return latent, {
        "clipped_proposal_coefficients": clipped_count,
        "moved_latent_coefficients": moved_count,
        "total_coefficients": total,
        "latent_motion_rms_dct_units": math.sqrt(
            squared_motion / max(total, 1)),
    }


def _quantize_latent(
    latent: list[np.ndarray],
    quantizers: list[np.ndarray],
) -> list[np.ndarray]:
    return [
        np.rint(value / table[None, None, :, :]).astype(np.int32)
        for value, table in zip(latent, quantizers)
    ]


def _center_coefficients(
    original_blocks: list[np.ndarray],
    source_quantizers: list[np.ndarray],
    destination_quantizers: list[np.ndarray],
) -> list[np.ndarray]:
    return [
        np.rint(
            blocks.astype(np.float64)
            * source[None, None, :, :]
            / destination[None, None, :, :]
        ).astype(np.int32)
        for blocks, source, destination in zip(
            original_blocks, source_quantizers, destination_quantizers)
    ]


def _byte_match(
    source: Path,
    destination: Path,
    candidate: list[np.ndarray],
    quantizers: list[np.ndarray],
    coherence: list[np.ndarray],
    byte_budget: int,
) -> tuple[list[np.ndarray], dict]:
    """Prune a fixed utility ordering until the real JPEG fits byte_budget."""
    ranked_component, ranked_index = _rank_budget_removals(
        candidate, quantizers, coherence)

    def write(count):
        projected = _apply_budget_removals(
            candidate, ranked_component, ranked_index, count)
        _write_coefficients(
            source,
            destination,
            projected,
            component_quantizers=quantizers,
        )
        return projected, destination.stat().st_size

    initial, initial_bytes = write(0)
    if initial_bytes <= byte_budget:
        return initial, {
            "unconstrained_bytes": initial_bytes,
            "removed_coefficients": 0,
            "bytes": initial_bytes,
        }

    low = 0
    high = min(
        len(ranked_index),
        max(256, 2 * (initial_bytes - byte_budget)),
    )
    high_value, high_bytes = write(high)
    while high_bytes > byte_budget and high < len(ranked_index):
        low = high
        high = min(
            len(ranked_index),
            high + max(512, 2 * (high_bytes - byte_budget)),
        )
        high_value, high_bytes = write(high)
    if high_bytes > byte_budget:
        raise RuntimeError("could not fit candidate under JPEG byte budget")

    best_value = high_value
    best_bytes = high_bytes
    best_count = high
    while low + 1 < high:
        middle = (low + high) // 2
        middle_value, middle_bytes = write(middle)
        if middle_bytes <= byte_budget:
            high = middle
            if byte_budget - middle_bytes < byte_budget - best_bytes:
                best_value = middle_value
                best_bytes = middle_bytes
                best_count = middle
        else:
            low = middle
    _write_coefficients(
        source,
        destination,
        best_value,
        component_quantizers=quantizers,
    )
    return best_value, {
        "unconstrained_bytes": initial_bytes,
        "removed_coefficients": best_count,
        "bytes": best_bytes,
    }


def _decode(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _difference_visual(
    control: np.ndarray,
    value: np.ndarray,
    destination: Path,
    gain: float = 8.0,
) -> None:
    difference = (
        127.5
        + float(gain) * (
            value.astype(np.float64) - control.astype(np.float64)
        )
    )
    Image.fromarray(
        np.clip(np.rint(difference), 0, 255).astype(np.uint8), "RGB"
    ).save(destination)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    jpeg = jio.read(str(source))
    original_blocks = [
        np.array(_coefficient_blocks(value), copy=True)
        for value in jpeg.coef_arrays
    ]
    source_quantizers = _component_quantizers(jpeg)
    labels, tensor, geometry_timing = _build_or_load_geometry(
        source,
        output / "structural_geometry.npz",
        args.allocation_side,
    )

    maximum_h = max(item.h_samp_factor for item in jpeg.comp_info)
    maximum_v = max(item.v_samp_factor for item in jpeg.comp_info)
    proposals = []
    coherence = []
    proposal_timing = []
    for component, blocks, quantizer in zip(
        jpeg.comp_info, original_blocks, source_quantizers
    ):
        phase = time.perf_counter()
        scale = (
            maximum_v / component.v_samp_factor,
            maximum_h / component.h_samp_factor,
        )
        proposal, local_coherence = _restore_component_phase_correct(
            blocks,
            quantizer,
            labels,
            tensor,
            scale,
            anisotropy=args.anisotropy,
            coherence_sigma=args.coherence_sigma,
        )
        proposals.append(proposal.reshape(blocks.shape))
        coherence.append(local_coherence)
        proposal_timing.append(
            1000.0 * (time.perf_counter() - phase))

    latent, projection = _project_geometry_into_bins(
        original_blocks,
        source_quantizers,
        proposals,
        args.strength,
    )
    exact_quantizers = [
        np.ones((8, 8), dtype=np.int32)
        for _ in source_quantizers
    ]
    exact_center = _center_coefficients(
        original_blocks, source_quantizers, exact_quantizers)
    exact_latent = _quantize_latent(latent, exact_quantizers)
    center_path = output / "precision_center_control.jpg"
    latent_path = output / "latent_interval_projection.jpg"
    _write_coefficients(
        source, center_path, exact_center, exact_quantizers)
    _write_coefficients(
        source, latent_path, exact_latent, exact_quantizers)

    rate_quantizers = [
        np.maximum(
            1,
            np.rint(table / float(args.precision_divisor)),
        ).astype(np.int32)
        for table in source_quantizers
    ]
    # A finer DC table expands every differential code and cannot be repaid
    # by pruning AC support at this bitrate. Keep the source DC precision and
    # spend the alternative representation only on within-bin AC placement.
    for source_table, rate_table in zip(
        source_quantizers, rate_quantizers
    ):
        rate_table[0, 0] = int(source_table[0, 0])
    # Cb and Cr share a table in the source and must remain identical.
    rate_quantizers[2] = rate_quantizers[1].copy()
    rate_center = _center_coefficients(
        original_blocks, source_quantizers, rate_quantizers)
    rate_latent = _quantize_latent(latent, rate_quantizers)
    rate_center_path = output / "rate_matched_center_control.jpg"
    rate_latent_path = output / "rate_matched_interval_projection.jpg"
    rate_center, center_budget = _byte_match(
        source,
        rate_center_path,
        rate_center,
        rate_quantizers,
        coherence,
        source.stat().st_size,
    )
    rate_latent, latent_budget = _byte_match(
        source,
        rate_latent_path,
        rate_latent,
        rate_quantizers,
        coherence,
        source.stat().st_size,
    )

    source_rgb = _decode(source)
    center_rgb = _decode(center_path)
    latent_rgb = _decode(latent_path)
    rate_center_rgb = _decode(rate_center_path)
    rate_latent_rgb = _decode(rate_latent_path)
    comparison = [
        ("original JPEG decode", source_rgb),
        ("precision center control", center_rgb),
        ("latent interval projection", latent_rgb),
        ("rate-matched finer-Q center", rate_center_rgb),
        ("rate-matched interval projection", rate_latent_rgb),
    ]
    _save_montage(comparison, output / "comparison.png")
    _difference_visual(
        center_rgb,
        latent_rgb,
        output / "latent_minus_center_x8.png",
    )
    _difference_visual(
        rate_center_rgb,
        rate_latent_rgb,
        output / "rate_latent_minus_control_x8.png",
    )

    report = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "strength": args.strength,
        "precision_divisor": args.precision_divisor,
        "projection": projection,
        "geometry_timing": geometry_timing,
        "proposal_component_ms": proposal_timing,
        "outputs": {
            "precision_center_control": {
                "path": str(center_path),
                "bytes": center_path.stat().st_size,
                **_image_metrics(source_rgb, center_rgb),
            },
            "latent_interval_projection": {
                "path": str(latent_path),
                "bytes": latent_path.stat().st_size,
                **_image_metrics(source_rgb, latent_rgb),
            },
            "rate_matched_center_control": {
                "path": str(rate_center_path),
                "budget": center_budget,
                **_image_metrics(source_rgb, rate_center_rgb),
            },
            "rate_matched_interval_projection": {
                "path": str(rate_latent_path),
                "budget": latent_budget,
                **_image_metrics(source_rgb, rate_latent_rgb),
            },
        },
        "elapsed_ms": 1000.0 * (time.perf_counter() - started),
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default="/Users/quentinkuttenkuler/Downloads/1500x500.jpeg",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments/out/jpeg_dct_interval_inverse"),
    )
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--precision-divisor", type=float, default=2.0)
    parser.add_argument("--allocation-side", type=int, default=768)
    parser.add_argument("--anisotropy", type=float, default=0.65)
    parser.add_argument("--coherence-sigma", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
