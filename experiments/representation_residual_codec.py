#!/usr/bin/env python3
"""Exact base-plus-correction codec experiment for transport representations.

The experiment does not propose a production bitstream.  It answers the rate
question first:

    source = deterministic_base + exact_modular_correction

Three compact, completely decodable packet families are compared:

* PNG-style adaptive row filters followed by DEFLATE;
* a SharpIQ-like finite bank of causal 2-D predictors, selected per tile;
* JPEG-style separable 8x8 DCT packets, made lossless by a PNG correction.

Run as a script to build the canonical transport-cell representation for gallery
images and emit a CSV/JSON rate ledger plus diagnostic figures.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "viewer"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


PNG_HEADER = struct.Struct(">4sIIBB")
MODEL_HEADER = struct.Struct(">4sIIBBH")
DCT_HEADER = struct.Struct(">4sIIBBB")
CELL_HEADER = struct.Struct(">4sIIBIB")
COMPACT_GEOMETRY_HEADER = struct.Struct(">4sIIBH")

JPEG_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

JPEG_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def _zigzag_indices() -> np.ndarray:
    order = []
    for diagonal in range(15):
        entries = [
            (row, diagonal - row)
            for row in range(8)
            if 0 <= diagonal - row < 8
        ]
        if diagonal % 2 == 0:
            entries.reverse()
        order.extend(entries)
    return np.asarray([row * 8 + column for row, column in order], dtype=np.intp)


ZIGZAG = _zigzag_indices()
INVERSE_ZIGZAG = np.argsort(ZIGZAG)


def as_u8_rgb(image: np.ndarray) -> np.ndarray:
    """Canonical codec input: contiguous HxWx3 sRGB bytes."""

    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = value[..., :3]
    if np.issubdtype(value.dtype, np.floating):
        scale = 255.0 if value.max(initial=0.0) <= 1.5 else 1.0
        value = np.rint(np.clip(value * scale, 0.0, 255.0))
    return np.ascontiguousarray(np.clip(value, 0, 255), dtype=np.uint8)


def _as_u8_channels(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3:
        raise ValueError("byte image must have shape HxW or HxWxC")
    return np.ascontiguousarray(np.clip(value, 0, 255), dtype=np.uint8)


def modular_difference(source: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return the exact byte-group correction ``source - base (mod 256)``."""

    source = as_u8_rgb(source)
    base = as_u8_rgb(base)
    if source.shape != base.shape:
        raise ValueError("source and base shapes differ")
    return ((source.astype(np.uint16) - base.astype(np.uint16)) & 255).astype(
        np.uint8
    )


def apply_modular_difference(base: np.ndarray, correction: np.ndarray) -> np.ndarray:
    base = as_u8_rgb(base)
    correction = as_u8_rgb(correction)
    if base.shape != correction.shape:
        raise ValueError("base and correction shapes differ")
    return ((base.astype(np.uint16) + correction.astype(np.uint16)) & 255).astype(
        np.uint8
    )


def _paeth(left: np.ndarray, up: np.ndarray, upper_left: np.ndarray) -> np.ndarray:
    left_i = left.astype(np.int16)
    up_i = up.astype(np.int16)
    upper_left_i = upper_left.astype(np.int16)
    estimate = left_i + up_i - upper_left_i
    dl = np.abs(estimate - left_i)
    du = np.abs(estimate - up_i)
    dul = np.abs(estimate - upper_left_i)
    return np.where(
        (dl <= du) & (dl <= dul),
        left,
        np.where(du <= dul, up, upper_left),
    )


def _signed_byte_cost(filtered: np.ndarray) -> int:
    signed = filtered.astype(np.int16)
    signed[signed >= 128] -= 256
    return int(np.sum(np.abs(signed), dtype=np.int64))


def png_predictive_encode(image: np.ndarray, level: int = 9) -> bytes:
    """Encode one adaptive PNG filter byte per row followed by DEFLATE."""

    image = _as_u8_channels(image)
    height, width, channels = image.shape
    flat = image.reshape(height, width * channels)
    previous = np.zeros(width * channels, dtype=np.uint8)
    rows = bytearray()
    bpp = channels
    for row in flat:
        left = np.zeros_like(row)
        left[bpp:] = row[:-bpp]
        upper_left = np.zeros_like(row)
        upper_left[bpp:] = previous[:-bpp]
        candidates = (
            row,
            ((row.astype(np.int16) - left.astype(np.int16)) & 255).astype(np.uint8),
            ((row.astype(np.int16) - previous.astype(np.int16)) & 255).astype(np.uint8),
            (
                (
                    row.astype(np.int16)
                    - ((left.astype(np.int16) + previous.astype(np.int16)) // 2)
                )
                & 255
            ).astype(np.uint8),
            (
                (
                    row.astype(np.int16)
                    - _paeth(left, previous, upper_left).astype(np.int16)
                )
                & 255
            ).astype(np.uint8),
        )
        choice = min(range(5), key=lambda index: _signed_byte_cost(candidates[index]))
        rows.append(choice)
        rows.extend(candidates[choice].tobytes())
        previous = row
    header = PNG_HEADER.pack(b"RPNG", height, width, channels, level)
    return header + zlib.compress(bytes(rows), level)


def png_predictive_decode(packet: bytes) -> np.ndarray:
    magic, height, width, channels, _ = PNG_HEADER.unpack_from(packet)
    if magic != b"RPNG":
        raise ValueError("not an RPNG packet")
    raw = memoryview(zlib.decompress(packet[PNG_HEADER.size:]))
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError("RPNG payload length mismatch")
    output = np.zeros((height, stride), dtype=np.uint8)
    cursor = 0
    for y in range(height):
        kind = int(raw[cursor])
        cursor += 1
        filtered = np.frombuffer(raw[cursor:cursor + stride], dtype=np.uint8)
        cursor += stride
        previous = output[y - 1] if y else np.zeros(stride, dtype=np.uint8)
        for x in range(stride):
            left = int(output[y, x - channels]) if x >= channels else 0
            up = int(previous[x])
            upper_left = int(previous[x - channels]) if x >= channels else 0
            if kind == 0:
                prediction = 0
            elif kind == 1:
                prediction = left
            elif kind == 2:
                prediction = up
            elif kind == 3:
                prediction = (left + up) // 2
            elif kind == 4:
                estimate = left + up - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - up),
                    abs(estimate - upper_left),
                )
                prediction = (left, up, upper_left)[distances.index(min(distances))]
            else:
                raise ValueError(f"unknown RPNG filter {kind}")
            output[y, x] = (int(filtered[x]) + prediction) & 255
    return output.reshape(height, width, channels)


def standard_png_encode(image: np.ndarray) -> bytes:
    """Encode a conventional optimized RGB PNG for an external baseline."""

    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(as_u8_rgb(image)).save(
        output,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return output.getvalue()


def standard_png_decode(packet: bytes) -> np.ndarray:
    from PIL import Image

    return as_u8_rgb(np.asarray(Image.open(io.BytesIO(packet)).convert("RGB")))


def _predictor_value(
    kind: int,
    decoded: np.ndarray,
    y: int,
    x: int,
    channel: int,
) -> int:
    left = int(decoded[y, x - 1, channel]) if x else 0
    left2 = int(decoded[y, x - 2, channel]) if x > 1 else left
    up = int(decoded[y - 1, x, channel]) if y else 0
    up2 = int(decoded[y - 2, x, channel]) if y > 1 else up
    upper_left = int(decoded[y - 1, x - 1, channel]) if x and y else 0
    if kind == 0:
        return 0
    if kind == 1:
        return left
    if kind == 2:
        return (2 * left - left2) & 255
    if kind == 3:
        return up
    if kind == 4:
        return (2 * up - up2) & 255
    if kind == 5:
        return (left + up - upper_left) & 255
    if kind == 6:
        estimate = left + up - upper_left
        values = (left, up, upper_left)
        return values[min(range(3), key=lambda i: abs(estimate - values[i]))]
    raise ValueError(f"unknown finite predictor {kind}")


def finite_predictor_encode(
    image: np.ndarray,
    tile_size: int = 32,
    level: int = 9,
) -> bytes:
    """Select one of seven causal predictors per tile and colour component."""

    image = _as_u8_channels(image)
    height, width, channels = image.shape
    tiles_y = math.ceil(height / tile_size)
    tiles_x = math.ceil(width / tile_size)
    predictors = np.empty((tiles_y, tiles_x, channels), dtype=np.uint8)
    residual = np.empty_like(image)

    # All candidates depend only on already-known source samples, so selection
    # is independent for every tile while decode remains one causal raster pass.
    for ty in range(tiles_y):
        y0, y1 = ty * tile_size, min((ty + 1) * tile_size, height)
        for tx in range(tiles_x):
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, width)
            for channel in range(channels):
                best_kind = 0
                best_cost = None
                best_values = None
                for kind in range(7):
                    values = np.empty((y1 - y0, x1 - x0), dtype=np.uint8)
                    for yy, y in enumerate(range(y0, y1)):
                        for xx, x in enumerate(range(x0, x1)):
                            prediction = _predictor_value(
                                kind, image, y, x, channel
                            )
                            values[yy, xx] = (
                                int(image[y, x, channel]) - prediction
                            ) & 255
                    cost = _signed_byte_cost(values)
                    if best_cost is None or cost < best_cost:
                        best_kind, best_cost, best_values = kind, cost, values
                predictors[ty, tx, channel] = best_kind
                residual[y0:y1, x0:x1, channel] = best_values

    raw = predictors.tobytes() + residual.tobytes()
    return MODEL_HEADER.pack(
        b"RMOD", height, width, channels, level, tile_size
    ) + zlib.compress(raw, level)


def finite_predictor_decode(packet: bytes) -> np.ndarray:
    magic, height, width, channels, _, tile_size = MODEL_HEADER.unpack_from(packet)
    if magic != b"RMOD":
        raise ValueError("not an RMOD packet")
    raw = zlib.decompress(packet[MODEL_HEADER.size:])
    tiles_y = math.ceil(height / tile_size)
    tiles_x = math.ceil(width / tile_size)
    predictor_bytes = tiles_y * tiles_x * channels
    predictors = np.frombuffer(
        raw[:predictor_bytes], dtype=np.uint8
    ).reshape(tiles_y, tiles_x, channels)
    residual = np.frombuffer(
        raw[predictor_bytes:], dtype=np.uint8
    ).reshape(height, width, channels)
    decoded = np.zeros_like(residual)
    for y in range(height):
        ty = y // tile_size
        for x in range(width):
            tx = x // tile_size
            for channel in range(channels):
                prediction = _predictor_value(
                    int(predictors[ty, tx, channel]),
                    decoded,
                    y,
                    x,
                    channel,
                )
                decoded[y, x, channel] = (
                    int(residual[y, x, channel]) + prediction
                ) & 255
    return decoded


def _signed_planes(source: np.ndarray, base: np.ndarray) -> np.ndarray:
    source = as_u8_rgb(source).astype(np.int16)
    base = as_u8_rgb(base).astype(np.int16)
    difference = source - base
    mapped = np.where(
        difference >= 0,
        2 * difference,
        -2 * difference - 1,
    ).astype(np.uint16)
    return np.concatenate(
        ((mapped & 255).astype(np.uint8), (mapped >> 8).astype(np.uint8)),
        axis=2,
    )


def _apply_signed_planes(base: np.ndarray, planes: np.ndarray) -> np.ndarray:
    base = as_u8_rgb(base)
    planes = _as_u8_channels(planes)
    channels = base.shape[2]
    if planes.shape != (*base.shape[:2], 2 * channels):
        raise ValueError("signed correction plane shape mismatch")
    mapped = (
        planes[..., :channels].astype(np.uint16)
        | (planes[..., channels:].astype(np.uint16) << 8)
    )
    difference = np.where(
        (mapped & 1) == 0,
        mapped // 2,
        -(mapped // 2) - 1,
    ).astype(np.int16)
    restored = base.astype(np.int16) + difference
    if np.any((restored < 0) | (restored > 255)):
        raise ValueError("signed correction reconstructs outside byte range")
    return restored.astype(np.uint8)


def signed_png_encode(source: np.ndarray, base: np.ndarray) -> bytes:
    return b"RSPG" + png_predictive_encode(_signed_planes(source, base))


def signed_png_decode(base: np.ndarray, packet: bytes) -> np.ndarray:
    if packet[:4] != b"RSPG":
        raise ValueError("not an RSPG packet")
    return _apply_signed_planes(base, png_predictive_decode(packet[4:]))


def signed_finite_encode(source: np.ndarray, base: np.ndarray) -> bytes:
    return b"RSFM" + finite_predictor_encode(_signed_planes(source, base))


def signed_finite_decode(base: np.ndarray, packet: bytes) -> np.ndarray:
    if packet[:4] != b"RSFM":
        raise ValueError("not an RSFM packet")
    return _apply_signed_planes(base, finite_predictor_decode(packet[4:]))


def _cell_order(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    height, width = labels.shape
    y, x = np.mgrid[:height, :width]
    # Alternating rows avoid a full-width jump at every scanline.  Stable
    # label-first sorting then gives a deterministic local walk inside each
    # canonical cell without transmitting an ordering.
    snake = y * width + np.where((y & 1) == 0, x, width - 1 - x)
    order = np.lexsort((snake.ravel(), labels.ravel()))
    cells = int(labels.max(initial=-1)) + 1
    counts = np.bincount(labels.ravel(), minlength=cells)
    return order, np.concatenate(([0], np.cumsum(counts)))


def _zigzag_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    return np.where(values >= 0, 2 * values, -2 * values - 1).astype(np.uint32)


def _varint_cost(values: np.ndarray) -> int:
    mapped = _zigzag_array(values)
    return int(np.sum(
        1
        + (mapped >= (1 << 7))
        + (mapped >= (1 << 14))
        + (mapped >= (1 << 21)),
        dtype=np.int64,
    ))


def _predict_sequence(values: np.ndarray, kind: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    predicted = np.zeros_like(values)
    if kind == 1 and len(values) > 1:
        predicted[1:] = values[:-1]
    elif kind == 2:
        if len(values) > 1:
            predicted[1] = values[0]
        if len(values) > 2:
            predicted[2:] = 2 * values[1:-1] - values[:-2]
    elif kind != 0:
        raise ValueError(f"unknown cell predictor {kind}")
    return values - predicted


def cell_residual_encode(
    source: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    level: int = 9,
) -> bytes:
    """Encode residuals in canonical-cell order with order-0/1/2 models."""

    source = as_u8_rgb(source)
    base = as_u8_rgb(base)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != source.shape[:2]:
        raise ValueError("label map shape differs from image")
    order, offsets = _cell_order(labels)
    cells = len(offsets) - 1
    difference = (
        source.astype(np.int16) - base.astype(np.int16)
    ).reshape(-1, 3)[order].astype(np.int32)
    kinds = np.zeros((cells, 3), dtype=np.uint8)
    coded = bytearray()
    for cell in range(cells):
        begin, end = int(offsets[cell]), int(offsets[cell + 1])
        for channel in range(3):
            values = difference[begin:end, channel]
            candidates = [_predict_sequence(values, kind) for kind in range(3)]
            kind = min(range(3), key=lambda index: _varint_cost(candidates[index]))
            kinds[cell, channel] = kind
            for value in candidates[kind]:
                _write_signed(coded, int(value))
    payload = kinds.tobytes() + bytes(coded)
    return CELL_HEADER.pack(
        b"RCEL",
        source.shape[0],
        source.shape[1],
        source.shape[2],
        cells,
        level,
    ) + zlib.compress(payload, level)


def cell_residual_decode(
    base: np.ndarray,
    labels: np.ndarray,
    packet: bytes,
) -> np.ndarray:
    magic, height, width, channels, cells, _ = CELL_HEADER.unpack_from(packet)
    if magic != b"RCEL" or channels != 3:
        raise ValueError("not an RGB RCEL packet")
    base = as_u8_rgb(base)
    labels = np.asarray(labels, dtype=np.int32)
    if base.shape != (height, width, channels) or labels.shape != (height, width):
        raise ValueError("RCEL geometry mismatch")
    order, offsets = _cell_order(labels)
    if len(offsets) - 1 != cells:
        raise ValueError("RCEL cell count mismatch")
    payload = memoryview(zlib.decompress(packet[CELL_HEADER.size:]))
    kind_bytes = cells * channels
    kinds = np.frombuffer(
        payload[:kind_bytes], dtype=np.uint8
    ).reshape(cells, channels)
    cursor = kind_bytes
    difference = np.zeros((height * width, channels), dtype=np.int32)
    ordered = difference[order]
    for cell in range(cells):
        begin, end = int(offsets[cell]), int(offsets[cell + 1])
        for channel in range(channels):
            kind = int(kinds[cell, channel])
            for position in range(begin, end):
                residual, cursor = _read_signed(payload, cursor)
                if kind == 0 or position == begin:
                    prediction = 0
                elif kind == 1 or position == begin + 1:
                    prediction = int(ordered[position - 1, channel])
                elif kind == 2:
                    prediction = (
                        2 * int(ordered[position - 1, channel])
                        - int(ordered[position - 2, channel])
                    )
                else:
                    raise ValueError(f"unknown RCEL predictor {kind}")
                ordered[position, channel] = prediction + residual
    if cursor != len(payload):
        raise ValueError("trailing RCEL residual bytes")
    difference[order] = ordered
    restored = base.astype(np.int32) + difference.reshape(height, width, channels)
    if np.any((restored < 0) | (restored > 255)):
        raise ValueError("RCEL reconstructs outside byte range")
    return restored.astype(np.uint8)


def _nearest_site_labels(
    height: int,
    width: int,
    positions: np.ndarray,
) -> np.ndarray:
    """Integer squared-distance ownership with stable lowest-ID tie breaking."""

    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions must be nonempty Nx2 integer coordinates")
    y, x = np.mgrid[:height, :width]
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(np.int64)
    labels = np.empty(len(pixels), dtype=np.int32)
    # Bound the temporary distance matrix while retaining a vectorized exact
    # argmin. np.argmin supplies the specified lowest-site-ID tie break.
    chunk = max(1, 8_000_000 // len(positions))
    for begin in range(0, len(pixels), chunk):
        sample = pixels[begin:begin + chunk]
        distance = (
            (sample[:, None, 0] - positions[None, :, 0]) ** 2
            + (sample[:, None, 1] - positions[None, :, 1]) ** 2
        )
        labels[begin:begin + len(sample)] = np.argmin(distance, axis=1)
    return labels.reshape(height, width)


def compact_geometry_encode(
    source: np.ndarray,
    canonical_labels: np.ndarray,
    canonical_centers: np.ndarray,
    site_count: int,
    selection: str = "area",
    level: int = 9,
) -> bytes:
    """Serialize a small Euclidean site partition and its constant RGB readout.

    ``selection`` is an encoder-only choice. ``area`` takes the largest
    canonical cells, ``mass`` samples equal population quantiles in raster
    order, and ``farthest`` performs a small weighted farthest-point scan.
    The packet itself contains only quantized integer site positions and
    fitted byte RGB means. Decode does not need the source, selection rule,
    transport metric, or canonical label raster.
    """

    source = as_u8_rgb(source)
    labels = np.asarray(canonical_labels, dtype=np.int32)
    centers = np.asarray(canonical_centers, dtype=np.float64)
    if labels.shape != source.shape[:2]:
        raise ValueError("canonical label map shape differs from source")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("canonical centers must have shape Nx2")
    available = int(labels.max(initial=-1)) + 1
    if available <= 0 or len(centers) < available:
        raise ValueError("canonical centers do not cover canonical labels")
    requested = min(max(int(site_count), 1), available)
    counts = np.bincount(labels.ravel(), minlength=available)
    height, width = source.shape[:2]
    candidates = np.column_stack((
        np.rint(centers[:available, 0] * (width - 1)),
        np.rint(centers[:available, 1] * (height - 1)),
    ))
    if selection == "area":
        chosen = np.argsort(counts, kind="stable")[-requested:]
    elif selection == "mass":
        raster = np.lexsort((candidates[:, 0], candidates[:, 1]))
        cumulative = np.cumsum(counts[raster])
        targets = (np.arange(requested) + 0.5) * cumulative[-1] / requested
        quantiles = raster[np.minimum(
            np.searchsorted(cumulative, targets), available - 1
        )]
        chosen_list = list(dict.fromkeys(map(int, quantiles)))
        chosen_set = set(chosen_list)
        for index in np.argsort(counts, kind="stable")[::-1]:
            if len(chosen_list) >= requested:
                break
            if int(index) not in chosen_set:
                chosen_list.append(int(index))
                chosen_set.add(int(index))
        chosen = np.asarray(chosen_list, dtype=np.int64)
    elif selection == "farthest":
        chosen_list = [int(np.argmax(counts))]
        distance = np.sum(
            (candidates - candidates[chosen_list[0]]) ** 2, axis=1
        ).astype(np.float64)
        weight = np.sqrt(np.maximum(counts, 1))
        while len(chosen_list) < requested:
            score = distance * weight
            score[chosen_list] = -1.0
            index = int(np.argmax(score))
            chosen_list.append(index)
            distance = np.minimum(
                distance,
                np.sum((candidates - candidates[index]) ** 2, axis=1),
            )
        chosen = np.asarray(chosen_list, dtype=np.int64)
    else:
        raise ValueError(f"unknown compact site selection {selection}")
    positions = candidates[chosen]
    positions = np.unique(
        np.clip(positions, 0, 65535).astype(np.uint16),
        axis=0,
    )
    if len(positions) > 65535:
        raise ValueError("compact geometry exceeds 65535 sites")
    compact_labels = _nearest_site_labels(height, width, positions)
    compact_count = len(positions)
    population = np.maximum(
        np.bincount(compact_labels.ravel(), minlength=compact_count), 1
    )
    colours = np.stack([
        np.bincount(
            compact_labels.ravel(),
            weights=source[..., channel].ravel(),
            minlength=compact_count,
        ) / population
        for channel in range(3)
    ], axis=1)
    colours = np.clip(np.rint(colours), 0, 255).astype(np.uint8)
    payload = positions.astype("<u2", copy=False).tobytes() + colours.tobytes()
    return COMPACT_GEOMETRY_HEADER.pack(
        b"RCGT", height, width, 3, compact_count
    ) + zlib.compress(payload, level)


def compact_geometry_decode(packet: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic RGB base and regenerated ownership labels."""

    magic, height, width, channels, sites = COMPACT_GEOMETRY_HEADER.unpack_from(
        packet
    )
    if magic != b"RCGT" or channels != 3 or sites == 0:
        raise ValueError("not an RGB RCGT packet")
    payload = zlib.decompress(packet[COMPACT_GEOMETRY_HEADER.size:])
    position_bytes = sites * 4
    expected = position_bytes + sites * channels
    if len(payload) != expected:
        raise ValueError("RCGT payload length mismatch")
    positions = np.frombuffer(
        payload[:position_bytes], dtype="<u2"
    ).reshape(sites, 2)
    colours = np.frombuffer(
        payload[position_bytes:], dtype=np.uint8
    ).reshape(sites, channels)
    labels = _nearest_site_labels(height, width, positions)
    return colours[labels], labels


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


def _quality_table(base: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((base * scale + 50.0) / 100.0), 1, 255)


def _rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack(
        (
            0.299 * r + 0.587 * g + 0.114 * b,
            -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0,
            0.5 * r - 0.418688 * g - 0.081312 * b + 128.0,
        ),
        axis=-1,
    )


def _ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    y = image[..., 0]
    cb = image[..., 1] - 128.0
    cr = image[..., 2] - 128.0
    return np.stack(
        (
            y + 1.402 * cr,
            y - 0.344136 * cb - 0.714136 * cr,
            y + 1.772 * cb,
        ),
        axis=-1,
    )


def _write_varuint(output: bytearray, value: int) -> None:
    value = int(value)
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def _read_varuint(data: memoryview, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = int(data[cursor])
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise ValueError("oversized varint")


def _write_signed(output: bytearray, value: int) -> None:
    mapped = 2 * value if value >= 0 else -2 * value - 1
    _write_varuint(output, mapped)


def _read_signed(data: memoryview, cursor: int) -> tuple[int, int]:
    mapped, cursor = _read_varuint(data, cursor)
    value = mapped // 2 if mapped % 2 == 0 else -(mapped // 2) - 1
    return value, cursor


def dct_encode(image: np.ndarray, quality: int = 75, level: int = 9) -> bytes:
    """Encode a JPEG-style separable 8x8 DCT coefficient stream."""

    image = as_u8_rgb(image)
    height, width, channels = image.shape
    padded_h = math.ceil(height / 8) * 8
    padded_w = math.ceil(width / 8) * 8
    ycbcr = _rgb_to_ycbcr(image)
    padded = np.pad(
        ycbcr,
        ((0, padded_h - height), (0, padded_w - width), (0, 0)),
        mode="edge",
    )
    qtables = np.stack(
        (
            _quality_table(JPEG_LUMA, quality),
            _quality_table(JPEG_CHROMA, quality),
            _quality_table(JPEG_CHROMA, quality),
        )
    )
    coefficients = np.empty(
        (padded_h // 8, padded_w // 8, channels, 64), dtype=np.int32
    )
    for by in range(padded_h // 8):
        for bx in range(padded_w // 8):
            for channel in range(channels):
                block = (
                    padded[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8, channel]
                    - 128.0
                )
                transformed = DCT8 @ block @ DCT8.T
                quantized = np.rint(transformed / qtables[channel]).astype(np.int32)
                coefficients[by, bx, channel] = quantized.ravel()[ZIGZAG]

    coded = bytearray()
    previous_dc = np.zeros(channels, dtype=np.int32)
    for by in range(coefficients.shape[0]):
        for bx in range(coefficients.shape[1]):
            for channel in range(channels):
                values = coefficients[by, bx, channel]
                dc = int(values[0])
                _write_signed(coded, dc - int(previous_dc[channel]))
                previous_dc[channel] = dc
                run = 0
                for value in values[1:]:
                    value = int(value)
                    if value == 0:
                        run += 1
                    else:
                        _write_varuint(coded, run + 1)
                        _write_signed(coded, value)
                        run = 0
                _write_varuint(coded, 0)
    return DCT_HEADER.pack(
        b"RDCT", height, width, channels, int(quality), level
    ) + zlib.compress(bytes(coded), level)


def dct_decode(packet: bytes) -> np.ndarray:
    magic, height, width, channels, quality, _ = DCT_HEADER.unpack_from(packet)
    if magic != b"RDCT":
        raise ValueError("not an RDCT packet")
    padded_h = math.ceil(height / 8) * 8
    padded_w = math.ceil(width / 8) * 8
    blocks_y, blocks_x = padded_h // 8, padded_w // 8
    data = memoryview(zlib.decompress(packet[DCT_HEADER.size:]))
    cursor = 0
    coefficients = np.zeros((blocks_y, blocks_x, channels, 64), dtype=np.int32)
    previous_dc = np.zeros(channels, dtype=np.int32)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            for channel in range(channels):
                delta, cursor = _read_signed(data, cursor)
                dc = int(previous_dc[channel]) + delta
                previous_dc[channel] = dc
                coefficients[by, bx, channel, 0] = dc
                position = 1
                while True:
                    marker, cursor = _read_varuint(data, cursor)
                    if marker == 0:
                        break
                    position += marker - 1
                    if position >= 64:
                        raise ValueError("DCT AC run exceeds block")
                    value, cursor = _read_signed(data, cursor)
                    coefficients[by, bx, channel, position] = value
                    position += 1
    if cursor != len(data):
        raise ValueError("trailing DCT coefficient bytes")

    qtables = np.stack(
        (
            _quality_table(JPEG_LUMA, quality),
            _quality_table(JPEG_CHROMA, quality),
            _quality_table(JPEG_CHROMA, quality),
        )
    )
    ycbcr = np.empty((padded_h, padded_w, channels), dtype=np.float64)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            for channel in range(channels):
                raster = coefficients[by, bx, channel][INVERSE_ZIGZAG].reshape(8, 8)
                block = DCT8.T @ (raster * qtables[channel]) @ DCT8 + 128.0
                ycbcr[
                    by * 8:(by + 1) * 8,
                    bx * 8:(bx + 1) * 8,
                    channel,
                ] = block
    rgb = _ycbcr_to_rgb(ycbcr[:height, :width])
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _verified_row(
    method: str,
    source: np.ndarray,
    base_packet: bytes,
    decoded_base: np.ndarray,
    correction_method: str = "modular",
    **fields,
) -> dict:
    if correction_method == "modular":
        correction_packet = png_predictive_encode(
            modular_difference(source, decoded_base)
        )
        reconstructed = apply_modular_difference(
            decoded_base, png_predictive_decode(correction_packet)
        )
    elif correction_method == "png":
        correction_packet = signed_png_encode(source, decoded_base)
        reconstructed = signed_png_decode(decoded_base, correction_packet)
    elif correction_method == "finite":
        correction_packet = signed_finite_encode(source, decoded_base)
        reconstructed = signed_finite_decode(decoded_base, correction_packet)
    else:
        raise ValueError(f"unknown correction method {correction_method}")
    exact = np.array_equal(reconstructed, source)
    if not exact:
        raise AssertionError(f"{method} failed exact reconstruction")
    return {
        "method": method,
        "base_bytes": len(base_packet),
        "correction_bytes": len(correction_packet),
        "total_bytes": len(base_packet) + len(correction_packet),
        "exact": exact,
        **fields,
    }


def benchmark_pair(
    source: np.ndarray,
    support: np.ndarray,
    qualities: tuple[int, ...] = (50, 75, 90),
    labels: np.ndarray | None = None,
    centers: np.ndarray | None = None,
    compact_site_counts: tuple[int, ...] = (
        2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512
    ),
    compact_selections: tuple[str, ...] = ("area", "mass", "farthest"),
) -> tuple[list[dict], dict]:
    """Return exact rate rows and the geometry-only break-even ledger."""

    source = as_u8_rgb(source)
    support = as_u8_rgb(support)
    if source.shape != support.shape:
        raise ValueError("source and support shapes differ")
    raw_bytes = int(source.size)

    source_png = png_predictive_encode(source)
    if not np.array_equal(png_predictive_decode(source_png), source):
        raise AssertionError("source PNG packet is not exact")
    rows = [{
        "method": "source_png",
        "base_bytes": len(source_png),
        "correction_bytes": 0,
        "total_bytes": len(source_png),
        "exact": True,
    }]
    source_standard_png = standard_png_encode(source)
    if not np.array_equal(standard_png_decode(source_standard_png), source):
        raise AssertionError("standard source PNG is not exact")
    rows.append({
        "method": "source_standard_png",
        "base_bytes": len(source_standard_png),
        "correction_bytes": 0,
        "total_bytes": len(source_standard_png),
        "exact": True,
    })

    support_png = png_predictive_encode(support)
    rows.append(_verified_row(
        "support_png+modular_png",
        source,
        support_png,
        png_predictive_decode(support_png),
    ))
    support_standard_png = standard_png_encode(support)
    standard_delta_png = standard_png_encode(
        modular_difference(source, support)
    )
    standard_exact = np.array_equal(
        apply_modular_difference(
            standard_png_decode(support_standard_png),
            standard_png_decode(standard_delta_png),
        ),
        source,
    )
    if not standard_exact:
        raise AssertionError("standard support PNG split is not exact")
    rows.append({
        "method": "support_standard_png+modular_standard_png",
        "base_bytes": len(support_standard_png),
        "correction_bytes": len(standard_delta_png),
        "total_bytes": len(support_standard_png) + len(standard_delta_png),
        "exact": True,
    })
    rows.append(_verified_row(
        "support_png+signed_png",
        source,
        support_png,
        png_predictive_decode(support_png),
        correction_method="png",
    ))

    source_model = finite_predictor_encode(source)
    if not np.array_equal(finite_predictor_decode(source_model), source):
        raise AssertionError("source finite-model packet is not exact")
    rows.append({
        "method": "source_finite_predictor",
        "base_bytes": len(source_model),
        "correction_bytes": 0,
        "total_bytes": len(source_model),
        "exact": True,
    })
    support_model = finite_predictor_encode(support)
    rows.append(_verified_row(
        "support_finite+signed_finite",
        source,
        support_model,
        finite_predictor_decode(support_model),
        correction_method="finite",
    ))

    correction_to_support_png = png_predictive_encode(
        modular_difference(source, support)
    )
    correction_to_support_signed = signed_png_encode(source, support)
    correction_to_support_model = signed_finite_encode(source, support)
    break_even = {
        "raw_bytes": raw_bytes,
        "source_png_bytes": len(source_png),
        "source_standard_png_bytes": len(source_standard_png),
        "source_finite_bytes": len(source_model),
        "support_png_bytes": len(support_png),
        "support_standard_png_bytes": len(support_standard_png),
        "support_finite_bytes": len(support_model),
        "correction_to_exact_support_png_bytes": len(correction_to_support_png),
        "correction_to_exact_support_signed_bytes": len(correction_to_support_signed),
        "correction_to_exact_support_finite_bytes": len(correction_to_support_model),
        "geometry_budget_vs_source_png": len(source_png) - len(correction_to_support_png),
        "geometry_budget_vs_source_finite": (
            len(source_model) - len(correction_to_support_model)
        ),
    }
    best_direct = min(len(source_png), len(source_standard_png), len(source_model))
    break_even["best_direct_lossless_bytes"] = best_direct
    break_even["geometry_budget_vs_best_direct_raster"] = (
        best_direct - len(correction_to_support_png)
    )

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int32)
        cell_packet = cell_residual_encode(source, support, labels)
        cell_exact = np.array_equal(
            cell_residual_decode(support, labels, cell_packet), source
        )
        if not cell_exact:
            raise AssertionError("cell residual packet is not exact")
        rows.append({
            "method": "geometry_base+cell_residual",
            "base_bytes": 0,
            "correction_bytes": len(cell_packet),
            "total_bytes": len(cell_packet),
            "exact": True,
            "geometry_bytes_included": False,
        })
        break_even["correction_to_exact_support_cell_bytes"] = len(cell_packet)
        break_even["geometry_budget_vs_source_png_cell_order"] = (
            len(source_png) - len(cell_packet)
        )
        break_even["geometry_budget_vs_best_direct_cell_order"] = (
            best_direct - len(cell_packet)
        )

    if labels is not None and centers is not None:
        compact_candidates = []
        available = int(np.max(labels, initial=-1)) + 1
        for requested in compact_site_counts:
            if int(requested) > available:
                continue
            for selection in compact_selections:
                geometry_packet = compact_geometry_encode(
                    source,
                    labels,
                    centers,
                    int(requested),
                    selection=selection,
                )
                compact_base, compact_labels = compact_geometry_decode(
                    geometry_packet
                )
                correction_packet = cell_residual_encode(
                    source, compact_base, compact_labels
                )
                exact = np.array_equal(
                    cell_residual_decode(
                        compact_base, compact_labels, correction_packet
                    ),
                    source,
                )
                if not exact:
                    raise AssertionError("compact geometry codec is not exact")
                actual_sites = int(np.max(compact_labels)) + 1
                compact_candidates.append({
                    "requested_sites": int(requested),
                    "selection": selection,
                    "sites": actual_sites,
                    "geometry_bytes": len(geometry_packet),
                    "correction_bytes": len(correction_packet),
                    "total_bytes": (
                        len(geometry_packet) + len(correction_packet)
                    ),
                })
        if compact_candidates:
            best_compact = min(
                compact_candidates, key=lambda item: item["total_bytes"]
            )
            rows.append({
                "method": (
                    f"compact_geometry_{best_compact['sites']}"
                    "+cell_residual"
                ),
                "base_bytes": best_compact["geometry_bytes"],
                "correction_bytes": best_compact["correction_bytes"],
                "total_bytes": best_compact["total_bytes"],
                "exact": True,
                "geometry_bytes_included": True,
                "sites": best_compact["sites"],
                "selection": best_compact["selection"],
            })
            break_even["compact_geometry_candidates"] = compact_candidates
            break_even["compact_geometry_best_delta_vs_direct"] = (
                best_compact["total_bytes"] - best_direct
            )

    for quality in qualities:
        # The geometry already provides ``support``.  Encode only a centered,
        # half-resolution view of its signed residual with the separable DCT,
        # then carry an exact modular tail.  The 2:1 mapping covers the complete
        # byte difference range [-255, 255]; any DCT/mapping loss is repaired by
        # the verified tail.
        difference = (
            source.astype(np.int16) - support.astype(np.int16)
        )
        residual_proxy = np.rint(
            (difference.astype(np.float64) + 255.0) * 0.5
        ).astype(np.uint8)
        residual_dct = dct_encode(residual_proxy, quality)
        decoded_proxy = dct_decode(residual_dct).astype(np.int16)
        approximate_difference = 2 * decoded_proxy - 255
        predicted_source = np.clip(
            support.astype(np.int16) + approximate_difference,
            0,
            255,
        ).astype(np.uint8)
        residual_row = _verified_row(
            f"geometry_base+residual_dct_q{quality}+tail_png",
            source,
            residual_dct,
            predicted_source,
            quality=quality,
            geometry_bytes_included=False,
        )
        rows.append(residual_row)
        break_even[f"geometry_budget_residual_dct_q{quality}"] = (
            len(source_png) - residual_row["total_bytes"]
        )
        break_even[f"geometry_budget_vs_best_direct_residual_dct_q{quality}"] = (
            best_direct - residual_row["total_bytes"]
        )

        source_dct = dct_encode(source, quality)
        rows.append(_verified_row(
            f"source_dct_q{quality}+modular_png",
            source,
            source_dct,
            dct_decode(source_dct),
            quality=quality,
        ))
        support_dct = dct_encode(support, quality)
        rows.append(_verified_row(
            f"support_dct_q{quality}+modular_png",
            source,
            support_dct,
            dct_decode(support_dct),
            quality=quality,
        ))

    for row in rows:
        row["raw_bytes"] = raw_bytes
        row["bits_per_pixel"] = 8.0 * row["total_bytes"] / (
            source.shape[0] * source.shape[1]
        )
        row["ratio_to_raw"] = raw_bytes / row["total_bytes"]
        row["delta_vs_source_png"] = row["total_bytes"] - len(source_png)
        row["delta_vs_best_direct"] = row["total_bytes"] - best_direct
    return rows, break_even


def _fit_longest(image: np.ndarray, side: int) -> np.ndarray:
    from PIL import Image

    image = as_u8_rgb(image)
    if side <= 0 or max(image.shape[:2]) <= side:
        return image.astype(np.float64) / 255.0
    scale = side / max(image.shape[:2])
    size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    resized = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64) / 255.0


def _diagnostic_figure(
    source: np.ndarray,
    support: np.ndarray,
    rows: list[dict],
    path: Path,
    title: str,
    support_label: str = "transport support",
) -> None:
    import matplotlib.pyplot as plt

    residual = modular_difference(source, support).astype(np.int16)
    residual[residual >= 128] -= 256
    gain = 4.0
    residual_view = np.clip(0.5 + gain * residual / 255.0, 0.0, 1.0)
    labels = [
        row["method"] + (
            "  [+ geometry bytes]"
            if row.get("geometry_bytes_included") is False
            else ""
        )
        for row in rows
    ]
    totals = [row["total_bytes"] / 1024.0 for row in rows]
    colours = [
        "#2b8cbe" if label.startswith("source") else "#7bccc4"
        for label in labels
    ]

    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.1))
    for axis, image, label in zip(
        (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
         figure.add_subplot(grid[0, 2])),
        (source, support, residual_view),
        ("source", support_label, "exact correction ×4 + 0.5"),
    ):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    axis = figure.add_subplot(grid[1, :])
    axis.barh(np.arange(len(labels)), totals, color=colours)
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel(
        "coded KiB (every row is exact; [+ geometry bytes] rows exclude G)"
    )
    axis.grid(axis="x", alpha=0.25)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_gallery(
    keys: list[str],
    output: Path,
    work_side: int,
    allocation_side: int,
    qualities: tuple[int, ...],
) -> list[dict]:
    import gallery
    from port_needed import SegmentingConfig, build_segmenting_representation

    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries = []
    for key in keys:
        source = _fit_longest(gallery.load(key), work_side)
        config = SegmentingConfig(
            allocation_max_side=allocation_side,
            curvature_limited_density=True,
            soft_support_passes=16,
            soft_support_coupling=0.8,
            soft_support_colour_percentile=60.0,
        )
        started = time.perf_counter()
        result = build_segmenting_representation(source, config)
        elapsed = time.perf_counter() - started
        support = result["record"]["rgb"]
        rows, break_even = benchmark_pair(
            source,
            support,
            qualities,
            labels=result["labels"],
            centers=result["centers"],
        )
        for row in rows:
            row.update({
                "image": key,
                "width": source.shape[1],
                "height": source.shape[0],
                "cells": len(result["centers"]),
                "representation_seconds": elapsed,
                "psnr": float(result["record"]["psnr"]),
            })
        all_rows.extend(rows)
        summary = {
            "image": key,
            "shape": list(source.shape),
            "cells": len(result["centers"]),
            "representation_seconds": elapsed,
            "psnr": float(result["record"]["psnr"]),
            "break_even": break_even,
            "rows": rows,
        }
        summaries.append(summary)
        _diagnostic_figure(
            as_u8_rgb(source),
            as_u8_rgb(support),
            rows,
            output / f"{key}.png",
            f"{gallery.describe(key)['label']} — representation + exact correction",
        )
        compact_candidates = break_even.get("compact_geometry_candidates", [])
        if compact_candidates:
            compact_best = min(
                compact_candidates, key=lambda item: item["total_bytes"]
            )
            geometry_packet = compact_geometry_encode(
                source,
                result["labels"],
                result["centers"],
                compact_best["requested_sites"],
                selection=compact_best["selection"],
            )
            compact_base, compact_labels = compact_geometry_decode(
                geometry_packet
            )
            correction_packet = cell_residual_encode(
                as_u8_rgb(source), compact_base, compact_labels
            )
            if not np.array_equal(
                cell_residual_decode(
                    compact_base, compact_labels, correction_packet
                ),
                as_u8_rgb(source),
            ):
                raise AssertionError("saved compact packets are not exact")
            (output / f"{key}.rcgt").write_bytes(geometry_packet)
            (output / f"{key}.rcel").write_bytes(correction_packet)
            _diagnostic_figure(
                as_u8_rgb(source),
                compact_base,
                rows,
                output / f"{key}-compact.png",
                (
                    f"{gallery.describe(key)['label']} — complete compact "
                    "geometry codec"
                ),
                support_label=(
                    f"decoded compact G ({compact_best['sites']} sites, "
                    f"{compact_best['selection']}, {len(geometry_packet)} B)"
                ),
            )
        print(
            f"{key}: {source.shape[1]}x{source.shape[0]}, "
            f"{len(result['centers'])} cells, {elapsed:.2f}s, "
            "geometry budget vs best direct "
            f"{break_even['geometry_budget_vs_best_direct_raster']:+d} B"
        )
        for row in rows:
            print(
                f"  {row['method']:36s} {row['total_bytes']:8d} B "
                f"({row['delta_vs_best_direct']:+8d} vs best direct)"
            )

    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    if all_rows:
        with (output / "rates.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(
                key for row in all_rows for key in row
            ))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gallery",
        nargs="+",
        default=["pikachu", "coffee", "camera"],
        help="gallery keys to build and compare",
    )
    parser.add_argument("--work-side", type=int, default=384)
    parser.add_argument("--allocation-side", type=int, default=384)
    parser.add_argument("--quality", type=int, nargs="+", default=[50, 75, 90])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/representation_residual_codec",
    )
    parser.add_argument(
        "--decode-compact",
        nargs=2,
        type=Path,
        metavar=("GEOMETRY_RCGT", "CORRECTION_RCEL"),
        help="decode one saved compact packet pair instead of running the study",
    )
    parser.add_argument(
        "--decode-output",
        type=Path,
        help="PNG destination for --decode-compact",
    )
    args = parser.parse_args()
    if args.decode_compact:
        from PIL import Image

        geometry_path, correction_path = args.decode_compact
        base, labels = compact_geometry_decode(geometry_path.read_bytes())
        source = cell_residual_decode(
            base, labels, correction_path.read_bytes()
        )
        destination = (
            args.decode_output
            if args.decode_output is not None
            else geometry_path.with_suffix(".decoded.png")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(source).save(destination)
        print(destination)
        return 0
    run_gallery(
        args.gallery,
        args.output,
        args.work_side,
        args.allocation_side,
        tuple(args.quality),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
