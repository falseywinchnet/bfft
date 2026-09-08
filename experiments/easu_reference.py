"""CPU reference port of AMD FSR 1.0 EASU supplied for comparison."""

from __future__ import annotations

import math

import numpy as np


Array = np.ndarray
TAP_OFFSETS = {
    "b": (0, -1), "c": (1, -1),
    "e": (-1, 0), "f": (0, 0), "g": (1, 0), "h": (2, 0),
    "i": (-1, 1), "j": (0, 1), "k": (1, 1), "l": (2, 1),
    "n": (0, 2), "o": (1, 2),
}


def _as_color(image: Array) -> tuple[Array, bool, bool]:
    value = np.asarray(image, dtype=np.float64)
    scalar = value.ndim == 2
    singleton = value.ndim == 3 and value.shape[2] == 1
    if scalar:
        value = np.repeat(value[..., None], 3, axis=2)
    elif singleton:
        value = np.repeat(value, 3, axis=2)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("EASU input must be HxW, HxWx1, or HxWxC with C >= 3")
    return value, scalar, singleton


def _restore_channels(value: Array, scalar: bool, singleton: bool) -> Array:
    if scalar:
        return value[..., 0]
    if singleton:
        return value[..., :1]
    return value


def _fetch(image: Array, x: int, y: int) -> Array:
    height, width, _ = image.shape
    return image[
        int(np.clip(y, 0, height - 1)),
        int(np.clip(x, 0, width - 1)),
    ]


def _set_direction(
    direction: Array,
    length: float,
    weight: float,
    la: float,
    lb: float,
    lc: float,
    ld: float,
    le: float,
) -> float:
    dc, cb = ld - lc, lc - lb
    denominator = max(abs(dc), abs(cb))
    dx = ld - lb
    direction[0] += dx * weight
    lx = 0.0 if denominator == 0.0 else min(1.0, abs(dx) / denominator)
    length += lx * lx * weight

    ec, ca = le - lc, lc - la
    denominator = max(abs(ec), abs(ca))
    dy = le - la
    direction[1] += dy * weight
    ly = 0.0 if denominator == 0.0 else min(1.0, abs(dy) / denominator)
    return length + ly * ly * weight


def easu_at_source_coordinate(image: Array, x: float, y: float) -> Array:
    """Evaluate EASU at one source-pixel coordinate on a color field."""

    point = np.array((x, y), dtype=np.float64)
    base = np.floor(point).astype(int)
    fraction = point - base
    taps = {
        name: _fetch(image, base[0] + dx, base[1] + dy)
        for name, (dx, dy) in TAP_OFFSETS.items()
    }
    luma = {
        name: float(0.5 * value[0] + value[1] + 0.5 * value[2])
        for name, value in taps.items()
    }

    direction = np.zeros(2, dtype=np.float64)
    edge_length = 0.0
    weights = (
        (1.0 - fraction[0]) * (1.0 - fraction[1]),
        fraction[0] * (1.0 - fraction[1]),
        (1.0 - fraction[0]) * fraction[1],
        fraction[0] * fraction[1],
    )
    arguments = (
        ("b", "e", "f", "g", "j"),
        ("c", "f", "g", "h", "k"),
        ("f", "i", "j", "k", "n"),
        ("g", "j", "k", "l", "o"),
    )
    for weight, names in zip(weights, arguments):
        edge_length = _set_direction(
            direction, edge_length, weight, *(luma[name] for name in names)
        )
    norm_squared = float(direction @ direction)
    if norm_squared < 1.0 / 32768.0:
        direction[:] = (1.0, 0.0)
    else:
        direction /= math.sqrt(norm_squared)
    edge_length = (0.5 * edge_length) ** 2
    stretch = 1.0 / max(abs(direction[0]), abs(direction[1]))
    length_scale = np.array((
        1.0 + (stretch - 1.0) * edge_length,
        1.0 - 0.5 * edge_length,
    ))
    lobe = 0.5 + ((0.25 - 0.04) - 0.5) * edge_length
    clip_radius_squared = 1.0 / lobe

    color = np.zeros(image.shape[2], dtype=np.float64)
    total_weight = 0.0
    for name, (dx, dy) in TAP_OFFSETS.items():
        offset = np.array((dx, dy), dtype=np.float64) - fraction
        vector = np.array((
            offset[0] * direction[0] + offset[1] * direction[1],
            -offset[0] * direction[1] + offset[1] * direction[0],
        )) * length_scale
        distance_squared = min(float(vector @ vector), clip_radius_squared)
        wb = (2.0 / 5.0) * distance_squared - 1.0
        wa = lobe * distance_squared - 1.0
        wb = (25.0 / 16.0) * wb * wb - ((25.0 / 16.0) - 1.0)
        weight = wb * wa * wa
        color += taps[name] * weight
        total_weight += weight
    filtered = color / total_weight
    nearest = np.stack((taps["f"], taps["g"], taps["j"], taps["k"]))
    return np.minimum(nearest.max(axis=0), np.maximum(nearest.min(axis=0), filtered))


def easu_resize(image: Array, scale: float = 2.0) -> Array:
    """Conventional pixel-center EASU resampling at any positive scale."""

    field, scalar, singleton = _as_color(image)
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    height, width, channels = field.shape
    out_height = int(round(height * scale))
    out_width = int(round(width * scale))
    output = np.empty((out_height, out_width, channels), dtype=np.float64)
    sx, sy = width / out_width, height / out_height
    for oy in range(out_height):
        y = oy * sy + 0.5 * sy - 0.5
        for ox in range(out_width):
            x = ox * sx + 0.5 * sx - 0.5
            output[oy, ox] = easu_at_source_coordinate(field, x, y)
    return _restore_channels(output, scalar, singleton)


def easu_nested_2x(image: Array) -> Array:
    """Evaluate EASU on the nested ``(2H-1)x(2W-1)`` comparison lattice."""

    field, scalar, singleton = _as_color(image)
    height, width, channels = field.shape
    output = np.empty((2 * height - 1, 2 * width - 1, channels), dtype=np.float64)
    for oy in range(output.shape[0]):
        for ox in range(output.shape[1]):
            output[oy, ox] = easu_at_source_coordinate(
                field, ox / 2.0, oy / 2.0
            )
    return _restore_channels(output, scalar, singleton)
