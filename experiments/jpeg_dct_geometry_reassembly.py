#!/usr/bin/env python3
"""Geometry-conditioned surgery on the native coefficients of a JPEG.

The experiment never re-encodes the source through RGB.  It reads the
quantized coefficient arrays, reconstructs each JPEG component on its native
sample lattice, and performs one phase-correct restriction/lift through the
structural quotient produced by segmenting v3.  That lifted field is projected
back into the source-aligned 8x8 DCTs.  Coherent coefficient energy is retained
while locally incoherent energy is relaxed toward the restored coefficients.
The resulting integers are written back with the source quantizers and
sampling factors under the original byte budget.

This is deliberately an experiment, not a JPEG restoration claim: without a
clean source, fidelity to the decoded JPEG is measurable but restoration
quality is only visually and structurally testable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    import jpegio as jio
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "jpegio is required for native coefficient access. Install it with "
        f"`{sys.executable} -m pip install jpegio`."
    ) from error

from experiments.segmenting_v3 import (  # noqa: E402
    SegmentingV3Config,
    build_segmenting_v3,
)
from port_needed.eikonal_lanczos import (  # noqa: E402
    eikonal_lanczos_resize,
)
from port_needed.fast_image_ops import gaussian_filter  # noqa: E402


def _dct_matrix() -> np.ndarray:
    matrix = np.empty((8, 8), dtype=np.float64)
    for frequency in range(8):
        scale = math.sqrt(1.0 / 8.0) if frequency == 0 else math.sqrt(2.0 / 8.0)
        for position in range(8):
            matrix[frequency, position] = scale * math.cos(
                math.pi * (2 * position + 1) * frequency / 16.0
            )
    return matrix


DCT8 = _dct_matrix()


def _coefficient_blocks(array: np.ndarray) -> np.ndarray:
    """View jpegio's tiled coefficient array as block_y, block_x, u, v."""
    height, width = array.shape
    if height % 8 or width % 8:
        raise ValueError("JPEG coefficient array is not block aligned")
    return array.reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3)


def _coefficient_tiles(blocks: np.ndarray) -> np.ndarray:
    """Return block_y, block_x, u, v coefficients in jpegio tiled form."""
    block_y, block_x, _, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(block_y * 8, block_x * 8)


def _half_average(fields: np.ndarray) -> np.ndarray:
    """Exact 2x2 box restriction for an HxWxC field, with edge extension."""
    height, width, _ = fields.shape
    padded = np.pad(
        fields,
        ((0, height % 2), (0, width % 2), (0, 0)),
        mode="edge",
    )
    return 0.25 * (
        padded[0::2, 0::2]
        + padded[1::2, 0::2]
        + padded[0::2, 1::2]
        + padded[1::2, 1::2]
    )


def _half_metric(
    labels: np.ndarray,
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    height, width = labels.shape
    y = np.minimum(2 * np.arange((height + 1) // 2) + 1, height - 1)
    x = np.minimum(2 * np.arange((width + 1) // 2) + 1, width - 1)
    low_labels = np.ascontiguousarray(labels[y[:, None], x[None, :]], np.int32)
    low_tensor = tuple(
        np.ascontiguousarray(
            _half_average(component[..., None])[..., 0], np.float64
        )
        for component in tensor
    )
    return low_labels, low_tensor


def _sample_metric_to_blocks(
    labels: np.ndarray,
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int],
    source_pixels_per_component_pixel: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample a full-resolution structural metric at component block centres."""
    block_y, block_x = shape
    scale_y, scale_x = source_pixels_per_component_pixel
    y = np.minimum(
        np.floor((np.arange(block_y) + 0.5) * 8.0 * scale_y).astype(np.intp),
        labels.shape[0] - 1,
    )
    x = np.minimum(
        np.floor((np.arange(block_x) + 0.5) * 8.0 * scale_x).astype(np.intp),
        labels.shape[1] - 1,
    )
    block_labels = np.ascontiguousarray(labels[y[:, None], x[None, :]], np.int32)
    block_tensor = tuple(
        np.ascontiguousarray(component[y[:, None], x[None, :]], np.float64)
        for component in tensor
    )
    return block_labels, block_tensor


def _sample_metric_to_component(
    labels: np.ndarray,
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int],
    source_pixels_per_component_pixel: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample the structural quotient on a JPEG component's native lattice."""
    height, width = shape
    scale_y, scale_x = source_pixels_per_component_pixel
    y = np.minimum(
        np.floor((np.arange(height) + 0.5) * scale_y).astype(np.intp),
        labels.shape[0] - 1,
    )
    x = np.minimum(
        np.floor((np.arange(width) + 0.5) * scale_x).astype(np.intp),
        labels.shape[1] - 1,
    )
    component_labels = np.ascontiguousarray(
        labels[y[:, None], x[None, :]], np.int32)
    component_tensor = tuple(
        np.ascontiguousarray(component[y[:, None], x[None, :]], np.float64)
        for component in tensor
    )
    return component_labels, component_tensor


def _source_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _build_or_load_geometry(
    source: Path,
    cache: Path,
    allocation_side: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    rgb = _source_rgb(source)
    if cache.exists():
        stored = np.load(cache)
        if tuple(stored["labels"].shape) == tuple(rgb.shape[:2]):
            labels = np.ascontiguousarray(stored["labels"], np.int32)
            tensor = tuple(
                np.ascontiguousarray(stored[name], np.float64)
                for name in ("xx", "xy", "yy")
            )
            return labels, tensor, {"cache": True, "total_ms": 0.0}

    config = SegmentingV3Config(
        structural_topology="canonical_v2",
        structural_allocation_side=int(allocation_side),
        structural_safety_cells=32768,
        structural_flow_sweeps=1,
        structural_characteristic_passes=1,
        meyer_sweeps=1,
        texture_model="parent_ridges",
        texture_cleanup=False,
        texture_graph_phase=False,
        texture_dirichlet_envelope=False,
        texture_coordinates=0,
        threads=4,
    )
    result = build_segmenting_v3(rgb, config)
    labels = np.ascontiguousarray(result["labels"], np.int32)
    geometry = result["cartoon_geometry"]
    tensor = tuple(
        np.ascontiguousarray(geometry[name], np.float64)
        for name in ("boundary_xx", "boundary_xy", "boundary_yy")
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        labels=labels,
        xx=tensor[0],
        xy=tensor[1],
        yy=tensor[2],
    )
    timing = dict(result["timing"])
    timing["cache"] = False
    return labels, tensor, timing


def _restore_component_modes(
    quantized_blocks: np.ndarray,
    quantizer: np.ndarray,
    block_labels: np.ndarray,
    block_tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    anisotropy: float,
    coherence_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return restored mode fields and their local signed coherence."""
    block_y, block_x, _, _ = quantized_blocks.shape
    dequantized = (
        quantized_blocks.astype(np.float64)
        * quantizer[None, None, :, :]
    ).reshape(block_y, block_x, 64)
    restored = np.empty_like(dequantized)
    coherence = np.empty_like(dequantized)
    low_labels, low_tensor = _half_metric(block_labels, block_tensor)

    # Three independent DCT modes share one fused resampling call.
    for first in range(0, 64, 3):
        stop = min(first + 3, 64)
        active = dequantized[..., first:stop]
        width = stop - first
        if width < 3:
            active = np.pad(active, ((0, 0), (0, 0), (0, 3 - width)))
        restricted = np.ascontiguousarray(_half_average(active), np.float64)
        lifted = eikonal_lanczos_resize(
            restricted,
            (block_y, block_x),
            low_labels,
            low_tensor,
            anisotropy=float(anisotropy),
            clamp_range=True,
        )
        restored[..., first:stop] = lifted[..., :width]

        active_channels = np.moveaxis(active[..., :width], -1, 0)
        smooth_signed = gaussian_filter(active_channels, coherence_sigma)
        smooth_magnitude = gaussian_filter(
            np.abs(active_channels), coherence_sigma)
        local_coherence = (
            np.abs(smooth_signed) / np.maximum(smooth_magnitude, 1e-12)
        )
        coherence[..., first:stop] = np.moveaxis(
            np.clip(local_coherence, 0.0, 1.0), 0, -1)
    return restored, coherence


def _blocks_to_component(
    quantized_blocks: np.ndarray,
    quantizer: np.ndarray,
) -> np.ndarray:
    dequantized = (
        quantized_blocks.astype(np.float64)
        * quantizer[None, None, :, :]
    )
    spatial_blocks = np.einsum(
        "ui,abuv,vj->abij",
        DCT8,
        dequantized,
        DCT8,
        optimize=True,
    )
    block_y, block_x = quantized_blocks.shape[:2]
    return (
        spatial_blocks.transpose(0, 2, 1, 3)
        .reshape(block_y * 8, block_x * 8)
        + 128.0
    )


def _component_to_dequantized_blocks(component: np.ndarray) -> np.ndarray:
    height, width = component.shape
    spatial_blocks = (
        component.reshape(height // 8, 8, width // 8, 8)
        .transpose(0, 2, 1, 3)
        - 128.0
    )
    return np.einsum(
        "ui,abij,vj->abuv",
        DCT8,
        spatial_blocks,
        DCT8,
        optimize=True,
    )


def _mode_coherence(
    quantized_blocks: np.ndarray,
    quantizer: np.ndarray,
    sigma: float,
) -> np.ndarray:
    block_y, block_x = quantized_blocks.shape[:2]
    modes = (
        quantized_blocks.astype(np.float64)
        * quantizer[None, None, :, :]
    ).reshape(block_y, block_x, 64)
    channels = np.moveaxis(modes, -1, 0)
    smooth_signed = gaussian_filter(channels, sigma)
    smooth_magnitude = gaussian_filter(np.abs(channels), sigma)
    return np.moveaxis(
        np.clip(
            np.abs(smooth_signed) / np.maximum(smooth_magnitude, 1e-12),
            0.0,
            1.0,
        ),
        0,
        -1,
    )


def _restore_component_phase_correct(
    quantized_blocks: np.ndarray,
    quantizer: np.ndarray,
    labels: np.ndarray,
    tensor: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_pixels_per_component_pixel: tuple[float, float],
    *,
    anisotropy: float,
    coherence_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict/lift on the native sample lattice, then return to block DCT."""
    component = _blocks_to_component(quantized_blocks, quantizer)
    metric = _sample_metric_to_component(
        labels,
        tensor,
        component.shape,
        source_pixels_per_component_pixel,
    )
    low_labels, low_tensor = _half_metric(*metric)
    restricted = _half_average(np.repeat(component[..., None], 3, axis=2))
    restored_component = eikonal_lanczos_resize(
        np.ascontiguousarray(restricted, np.float64),
        component.shape,
        low_labels,
        low_tensor,
        anisotropy=float(anisotropy),
        clamp_range=True,
    )[..., 0]
    restored = _component_to_dequantized_blocks(restored_component).reshape(
        quantized_blocks.shape[0], quantized_blocks.shape[1], 64)
    coherence = _mode_coherence(
        quantized_blocks, quantizer, coherence_sigma)
    return restored, coherence


def _relaxed_quantized_coefficients(
    original_blocks: np.ndarray,
    quantizer: np.ndarray,
    restored: np.ndarray,
    coherence: np.ndarray,
    *,
    strength: float,
    dc_fraction: float,
    frequency_power: float,
    coherence_power: float,
    deadzone: float,
) -> np.ndarray:
    block_y, block_x, _, _ = original_blocks.shape
    original = (
        original_blocks.astype(np.float64)
        * quantizer[None, None, :, :]
    ).reshape(block_y, block_x, 64)
    frequency = np.fromiter(
        (
            math.hypot(u, v) / math.hypot(7.0, 7.0)
            for u in range(8) for v in range(8)
        ),
        dtype=np.float64,
        count=64,
    )
    spectral_weight = (
        float(dc_fraction)
        + (1.0 - float(dc_fraction))
        * np.power(frequency, float(frequency_power))
    )
    incoherence = 1.0 - np.power(
        np.clip(coherence, 0.0, 1.0), float(coherence_power))
    alpha = np.clip(
        float(strength)
        * spectral_weight[None, None, :]
        * incoherence,
        0.0,
        1.0,
    )
    candidate = original + alpha * (restored - original)
    quantizer_flat = quantizer.reshape(64).astype(np.float64)
    change = np.abs(candidate - original)
    candidate = np.where(
        change >= float(deadzone) * quantizer_flat[None, None, :],
        candidate,
        original,
    )
    quantized = np.rint(
        candidate / quantizer_flat[None, None, :]
    ).astype(np.int32)
    return quantized.reshape(block_y, block_x, 8, 8)


def _write_coefficients(
    source: Path,
    destination: Path,
    component_blocks: list[np.ndarray],
    component_quantizers: list[np.ndarray] | None = None,
) -> None:
    jpeg = jio.read(str(source))
    for index, blocks in enumerate(component_blocks):
        jpeg.coef_arrays[index][:] = _coefficient_tiles(blocks)
    if component_quantizers is not None:
        assigned = {}
        for component, quantizer in zip(
            jpeg.comp_info, component_quantizers
        ):
            table_index = int(component.quant_tbl_no)
            table = np.asarray(quantizer, dtype=np.int32)
            previous = assigned.get(table_index)
            if previous is not None and not np.array_equal(previous, table):
                raise ValueError(
                    "components sharing a JPEG quantizer received "
                    "different replacement tables"
                )
            assigned[table_index] = table
            jpeg.quant_tables[table_index][:] = table
    destination.parent.mkdir(parents=True, exist_ok=True)
    jpeg.write(str(destination))
    with Image.open(source) as source_image:
        icc_profile = source_image.info.get("icc_profile")
    if icc_profile:
        _insert_icc_profile(destination, icc_profile)


def _insert_icc_profile(path: Path, profile: bytes) -> None:
    """Insert standard APP2 ICC chunks after SOI without touching entropy data."""
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("coefficient writer did not produce a JPEG")
    maximum_chunk = 65519
    chunks = [
        profile[offset:offset + maximum_chunk]
        for offset in range(0, len(profile), maximum_chunk)
    ]
    markers = bytearray()
    for sequence, chunk in enumerate(chunks, start=1):
        payload = b"ICC_PROFILE\x00" + bytes((sequence, len(chunks))) + chunk
        markers += b"\xff\xe2"
        markers += (len(payload) + 2).to_bytes(2, "big")
        markers += payload
    path.write_bytes(data[:2] + markers + data[2:])


def _rank_budget_removals(
    candidate: list[np.ndarray],
    quantizers: list[np.ndarray],
    coherence: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Rank removable non-DC coefficients from least to most useful."""
    utilities = []
    locations = []
    frequency = np.fromiter(
        (
            math.hypot(u, v) / math.hypot(7.0, 7.0)
            for u in range(8) for v in range(8)
        ),
        dtype=np.float64,
        count=64,
    ).reshape(1, 1, 8, 8)
    for component, (values, table, local_coherence) in enumerate(zip(
        candidate, quantizers, coherence
    )):
        mode_coherence = local_coherence.reshape(values.shape)
        utility = (
            (0.25 + 0.75 * mode_coherence)
            * np.abs(values).astype(np.float64)
            * table[None, None, :, :]
            / (1.0 + 2.0 * frequency)
        )
        selectable = values != 0
        selectable[..., 0, 0] = False
        flat_indices = np.flatnonzero(selectable)
        utilities.append(utility.ravel()[flat_indices])
        locations.append((
            np.full(flat_indices.size, component, dtype=np.int16),
            flat_indices,
        ))
    all_utility = np.concatenate(utilities)
    all_component = np.concatenate([item[0] for item in locations])
    all_index = np.concatenate([item[1] for item in locations])
    order = np.argsort(all_utility, kind="stable")
    return all_component[order], all_index[order]


def _apply_budget_removals(
    candidate: list[np.ndarray],
    ranked_component: np.ndarray,
    ranked_index: np.ndarray,
    count: int,
) -> list[np.ndarray]:
    output = [
        np.ascontiguousarray(value).copy()
        for value in candidate
    ]
    remove_component = ranked_component[:count]
    remove_index = ranked_index[:count]
    for component in range(len(output)):
        selected = remove_index[remove_component == component]
        flat = output[component].reshape(-1)
        flat[selected] = 0
    return output


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )


def _image_metrics(reference: np.ndarray, value: np.ndarray) -> dict:
    reference = reference.astype(np.float64)
    value = value.astype(np.float64)
    mse = float(np.mean(np.square(reference - value)))
    y = _luminance(value)
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    block_x = np.arange(7, dx.shape[1], 8)
    block_y = np.arange(7, dy.shape[0], 8)
    other_x = np.setdiff1d(np.arange(dx.shape[1]), block_x)
    other_y = np.setdiff1d(np.arange(dy.shape[0]), block_y)
    boundary = 0.5 * (
        float(np.mean(dx[:, block_x]))
        + float(np.mean(dy[block_y, :]))
    )
    interior = 0.5 * (
        float(np.mean(dx[:, other_x]))
        + float(np.mean(dy[other_y, :]))
    )
    laplacian = (
        -4.0 * y
        + np.roll(y, 1, axis=0)
        + np.roll(y, -1, axis=0)
        + np.roll(y, 1, axis=1)
        + np.roll(y, -1, axis=1)
    )
    gradient = np.hypot(
        0.5 * (np.roll(y, -1, axis=1) - np.roll(y, 1, axis=1)),
        0.5 * (np.roll(y, -1, axis=0) - np.roll(y, 1, axis=0)),
    )
    flat = gradient <= np.quantile(gradient, 0.40)
    return {
        "decoded_mse_from_source": mse,
        "decoded_psnr_from_source_db": (
            -10.0 * math.log10(max(mse / (255.0 * 255.0), 1e-15))
        ),
        "block_boundary_mean_abs_delta": boundary,
        "nonboundary_mean_abs_delta": interior,
        "block_boundary_ratio": boundary / max(interior, 1e-12),
        "flat_region_laplacian_rms": float(
            np.sqrt(np.mean(np.square(laplacian[flat])))),
    }


def _coefficient_metrics(
    original: list[np.ndarray],
    candidate: list[np.ndarray],
) -> dict:
    changed = 0
    total = 0
    original_nonzero = 0
    candidate_nonzero = 0
    squared_delta = 0.0
    for before, after in zip(original, candidate):
        delta = after.astype(np.float64) - before.astype(np.float64)
        changed += int(np.count_nonzero(delta))
        total += int(delta.size)
        original_nonzero += int(np.count_nonzero(before))
        candidate_nonzero += int(np.count_nonzero(after))
        squared_delta += float(np.sum(np.square(delta)))
    return {
        "changed_coefficients": changed,
        "total_coefficients": total,
        "changed_fraction": changed / max(total, 1),
        "original_nonzero_coefficients": original_nonzero,
        "candidate_nonzero_coefficients": candidate_nonzero,
        "quantized_coefficient_delta_rms": math.sqrt(
            squared_delta / max(total, 1)),
    }


def _save_montage(
    images: list[tuple[str, np.ndarray]],
    destination: Path,
) -> None:
    source_height, source_width = images[0][1].shape[:2]
    preview_width = 900
    preview_height = round(source_height * preview_width / source_width)
    crop_box = (
        round(0.24 * source_width),
        round(0.12 * source_height),
        round(0.93 * source_width),
        round(0.68 * source_height),
    )
    crop_width = 900
    crop_height = round(
        (crop_box[3] - crop_box[1])
        * crop_width / (crop_box[2] - crop_box[0])
    )
    label_height = 30
    canvas = Image.new(
        "RGB",
        (
            preview_width,
            len(images) * (label_height + preview_height + crop_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    y = 0
    for name, array in images:
        image = Image.fromarray(array.astype(np.uint8), "RGB")
        draw.text((8, y + 7), name, fill="black")
        y += label_height
        preview = image.resize(
            (preview_width, preview_height), Image.Resampling.LANCZOS)
        canvas.paste(preview, (0, y))
        y += preview_height
        crop = image.crop(crop_box).resize(
            (crop_width, crop_height), Image.Resampling.NEAREST)
        canvas.paste(crop, (0, y))
        y += crop_height
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def run(args: argparse.Namespace) -> dict:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)

    started = time.perf_counter()
    jpeg = jio.read(str(source))
    original_blocks = [
        np.array(_coefficient_blocks(array), copy=True)
        for array in jpeg.coef_arrays
    ]
    quantizers = [
        np.asarray(
            jpeg.quant_tables[component.quant_tbl_no], dtype=np.float64)
        for component in jpeg.comp_info
    ]
    labels, tensor, geometry_timing = _build_or_load_geometry(
        source,
        output / "structural_geometry.npz",
        args.allocation_side,
    )

    maximum_h = max(component.h_samp_factor for component in jpeg.comp_info)
    maximum_v = max(component.v_samp_factor for component in jpeg.comp_info)
    restored_components = []
    coherence_components = []
    component_timing = []
    for index, (component, blocks, quantizer) in enumerate(zip(
        jpeg.comp_info, original_blocks, quantizers
    )):
        phase = time.perf_counter()
        scale = (
            maximum_v / component.v_samp_factor,
            maximum_h / component.h_samp_factor,
        )
        if args.operator == "phase_correct":
            restored, coherence = _restore_component_phase_correct(
                blocks,
                quantizer,
                labels,
                tensor,
                scale,
                anisotropy=args.anisotropy,
                coherence_sigma=args.coherence_sigma,
            )
        else:
            metric = _sample_metric_to_blocks(
                labels, tensor, blocks.shape[:2], scale)
            restored, coherence = _restore_component_modes(
                blocks,
                quantizer,
                *metric,
                anisotropy=args.anisotropy,
                coherence_sigma=args.coherence_sigma,
            )
        restored_components.append(restored)
        coherence_components.append(coherence)
        component_timing.append({
            "component": index,
            "block_shape": list(blocks.shape[:2]),
            "milliseconds": 1000.0 * (time.perf_counter() - phase),
        })

    source_decoded = np.asarray(
        Image.open(source).convert("RGB"), dtype=np.uint8)
    identity_path = output / "identity_coefficient_roundtrip.jpg"
    _write_coefficients(source, identity_path, original_blocks)

    report = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "dimensions": [jpeg.image_width, jpeg.image_height],
        "progressive": bool(jpeg.progressive_mode),
        "operator": args.operator,
        "sampling_factors": [
            [item.h_samp_factor, item.v_samp_factor]
            for item in jpeg.comp_info
        ],
        "quantization_tables": [
            table.astype(int).tolist() for table in jpeg.quant_tables
        ],
        "geometry_timing": geometry_timing,
        "component_timing": component_timing,
        "variants": [],
    }
    montage_images = [("original JPEG decode", source_decoded)]
    for strength in args.strength:
        unbudgeted_blocks = []
        for blocks, quantizer, restored, coherence in zip(
            original_blocks,
            quantizers,
            restored_components,
            coherence_components,
        ):
            unbudgeted_blocks.append(_relaxed_quantized_coefficients(
                blocks,
                quantizer,
                restored,
                coherence,
                strength=strength,
                dc_fraction=args.dc_fraction,
                frequency_power=args.frequency_power,
                coherence_power=args.coherence_power,
                deadzone=args.deadzone,
            ))
        tag = f"s{strength:.2f}".replace(".", "p")
        path = output / f"geometry_dct_{tag}.jpg"
        original_nonzero = sum(
            int(np.count_nonzero(value)) for value in original_blocks)
        unbudgeted_nonzero = sum(
            int(np.count_nonzero(value)) for value in unbudgeted_blocks)
        minimum_removals = (
            max(unbudgeted_nonzero - original_nonzero, 0)
            if args.rate_neutral else 0
        )
        ranked_component, ranked_index = _rank_budget_removals(
            unbudgeted_blocks, quantizers, coherence_components)

        def write_with_removals(count):
            projected = _apply_budget_removals(
                unbudgeted_blocks, ranked_component, ranked_index, count)
            _write_coefficients(source, path, projected)
            return projected, path.stat().st_size

        candidate_blocks, candidate_bytes = write_with_removals(
            minimum_removals)
        byte_removals = minimum_removals
        if args.byte_neutral and candidate_bytes > source.stat().st_size:
            low = minimum_removals
            high = min(
                len(ranked_index),
                low + max(128, 2 * (candidate_bytes - source.stat().st_size)),
            )
            high_blocks, high_bytes = write_with_removals(high)
            while high_bytes > source.stat().st_size and high < len(ranked_index):
                low = high
                high = min(
                    len(ranked_index),
                    high + max(256, 2 * (high_bytes - source.stat().st_size)),
                )
                high_blocks, high_bytes = write_with_removals(high)
            best_blocks = high_blocks
            best_bytes = high_bytes
            best_count = high
            while low + 1 < high:
                middle = (low + high) // 2
                middle_blocks, middle_bytes = write_with_removals(middle)
                if middle_bytes <= source.stat().st_size:
                    high = middle
                    if (
                        source.stat().st_size - middle_bytes
                        < source.stat().st_size - best_bytes
                    ):
                        best_blocks = middle_blocks
                        best_bytes = middle_bytes
                        best_count = middle
                else:
                    low = middle
            candidate_blocks = best_blocks
            candidate_bytes = best_bytes
            byte_removals = best_count
            _write_coefficients(source, path, candidate_blocks)
        rate_budget = {
            "nonzero_enabled": bool(args.rate_neutral),
            "byte_enabled": bool(args.byte_neutral),
            "nonzero_budget": original_nonzero,
            "minimum_nonzero_removals": minimum_removals,
            "total_budget_removals": byte_removals,
        }
        decoded = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        record = {
            "strength": strength,
            "path": str(path),
            "bytes": candidate_bytes,
            "byte_delta": candidate_bytes - source.stat().st_size,
            "rate_budget": rate_budget,
            **_coefficient_metrics(original_blocks, candidate_blocks),
            **_image_metrics(source_decoded, decoded),
        }
        report["variants"].append(record)
        montage_images.append((f"geometry DCT strength {strength:.2f}", decoded))

    identity_decoded = np.asarray(
        Image.open(identity_path).convert("RGB"), dtype=np.uint8)
    report["identity_control"] = {
        "path": str(identity_path),
        "bytes": identity_path.stat().st_size,
        "byte_delta": identity_path.stat().st_size - source.stat().st_size,
        **_image_metrics(source_decoded, identity_decoded),
    }
    report["elapsed_ms"] = 1000.0 * (time.perf_counter() - started)
    _save_montage(montage_images, output / "comparison.png")
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
        default=str(ROOT / "experiments/out/jpeg_dct_geometry"),
    )
    parser.add_argument(
        "--strength",
        type=float,
        action="append",
        default=None,
        help="relaxation strength; repeat for multiple variants",
    )
    parser.add_argument("--allocation-side", type=int, default=768)
    parser.add_argument(
        "--operator",
        choices=("phase_correct", "mode_plane"),
        default="phase_correct",
    )
    parser.add_argument("--anisotropy", type=float, default=0.65)
    parser.add_argument("--coherence-sigma", type=float, default=1.0)
    parser.add_argument("--dc-fraction", type=float, default=0.0)
    parser.add_argument("--frequency-power", type=float, default=0.70)
    parser.add_argument("--coherence-power", type=float, default=2.0)
    parser.add_argument("--deadzone", type=float, default=0.35)
    parser.add_argument(
        "--rate-neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do not increase the total nonzero coefficient support",
    )
    parser.add_argument(
        "--byte-neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prune least-useful modes until entropy payload fits source bytes",
    )
    arguments = parser.parse_args()
    if arguments.strength is None:
        arguments.strength = [0.12, 0.20, 0.35]
    return arguments


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
