#!/usr/bin/env python3
"""Pure mathematical references for the Meyer structural gate.

The native periodic-FACR gate is a sum of four oriented responses.  For an
orientation ``q`` its unnormalised response is

    r_q(f) = abs(D_q K_q^3 B_q^3 f),

where ``B`` is a long-axis periodic box average, ``K`` is a narrow periodic
three-tap average on the transverse axis, and ``D`` is a forward difference.
All three operators are circulant convolutions on the image torus.

This file provides three deliberately independent formulations:

``gate_staged``
    Literal seven-stage definition: three boxes, three transverse taps, and
    one difference for each orientation.

``gate_collapsed``
    Algebraic definition.  The three boxes are convolved into one FIR, the
    three transverse taps are convolved into one FIR, and the difference is
    folded into the transverse FIR whenever both use the same direction.

``gate_ring``
    Streaming definition.  Each repeated filter is evaluated as a cascade of
    small ring-buffer states along the periodic direction cycles.  It retains
    the O(1)-per-pixel moving-sum box recurrence without allocating a full
    image for every stage.

The collapsed and ring implementations are references for native work.  They
are not intended to be fast in Python; their purpose is to prove operator
equivalence on arbitrary periodic image shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Direction = tuple[int, int]


@dataclass(frozen=True)
class OrientedGate:
    """One separable oriented response in the four-direction gate."""

    long: Direction
    cross: Direction
    difference: Direction
    box_radius: int
    cross_side: float


ORIENTATIONS = (
    OrientedGate((0, 1), (1, 0), (1, 0), 3, 0.125),
    OrientedGate((1, 0), (0, 1), (0, 1), 3, 0.125),
    OrientedGate((1, 1), (1, -1), (1, 1), 2, 0.0625),
    OrientedGate((1, -1), (1, 1), (1, -1), 2, 0.0625),
)


def _as_image(image: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(image, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 2:
        raise ValueError("the gate expects a two-dimensional image at least 2x2")
    return value


def _shift(value: np.ndarray, direction: Direction, offset: int) -> np.ndarray:
    """Return ``value[p + offset*direction]`` with periodic boundaries."""

    dy, dx = direction
    return np.roll(value, (-offset * dy, -offset * dx), axis=(0, 1))


def directional_fir(
    value: np.ndarray,
    direction: Direction,
    coefficients: Iterable[float],
    origin: int,
) -> np.ndarray:
    """Apply a one-dimensional FIR along a periodic lattice direction.

    Coefficient ``coefficients[k]`` multiplies the sample at directional
    offset ``k-origin``.  ``origin`` therefore records the zero-offset index
    and lets this routine represent asymmetric difference kernels too.
    """

    source = _as_image(value)
    taps = np.asarray(tuple(coefficients), dtype=np.float64)
    if taps.ndim != 1 or taps.size < 1 or not 0 <= origin < taps.size:
        raise ValueError("invalid FIR coefficients or origin")
    result = np.zeros_like(source)
    for index, coefficient in enumerate(taps):
        result += coefficient * _shift(source, direction, index - origin)
    return result


def compose_fir(
    first: tuple[np.ndarray, int], second: tuple[np.ndarray, int]
) -> tuple[np.ndarray, int]:
    """Compose two same-direction FIRs, retaining their offset origin."""

    a, oa = first
    b, ob = second
    return np.convolve(a, b), oa + ob


def box_kernel(radius: int) -> tuple[np.ndarray, int]:
    if radius < 1:
        raise ValueError("box radius must be positive")
    return np.full(2 * radius + 1, 1.0 / (2 * radius + 1)), radius


def cross_kernel(side: float) -> tuple[np.ndarray, int]:
    if not 0.0 <= side <= 0.5:
        raise ValueError("three-tap side weight must be in [0, 0.5]")
    return np.array((side, 1.0 - 2.0 * side, side)), 1


def difference_kernel() -> tuple[np.ndarray, int]:
    # D f[p] = f[p+d] - f[p].
    return np.array((-1.0, 1.0)), 0


def _power_kernel(kernel: tuple[np.ndarray, int], exponent: int) -> tuple[np.ndarray, int]:
    result = (np.array((1.0,)), 0)
    for _ in range(exponent):
        result = compose_fir(result, kernel)
    return result


def response_staged(image: np.ndarray, orientation: OrientedGate) -> np.ndarray:
    """Literal ``D K^3 B^3`` response before absolute value."""

    value = _as_image(image)
    box = box_kernel(orientation.box_radius)
    cross = cross_kernel(orientation.cross_side)
    for _ in range(3):
        value = directional_fir(value, orientation.long, *box)
    for _ in range(3):
        value = directional_fir(value, orientation.cross, *cross)
    return directional_fir(value, orientation.difference, *difference_kernel())


def response_collapsed(image: np.ndarray, orientation: OrientedGate) -> np.ndarray:
    """Algebraically collapsed form of ``D K^3 B^3``.

    When the difference is transverse (the two axial orientations), it is
    folded into the seven-tap ``K^3`` kernel, producing one eight-tap FIR.
    When the difference is longitudinal (the diagonal orientations), it is
    folded into ``B^3``.  The latter composite is also the exact identity

        D B_R^3 = B_R^2 (D B_R),

    whose ``D B_R`` factor has only two nonzero endpoint coefficients.
    """

    value = _as_image(image)
    long3 = _power_kernel(box_kernel(orientation.box_radius), 3)
    cross3 = _power_kernel(cross_kernel(orientation.cross_side), 3)
    difference = difference_kernel()
    if orientation.difference == orientation.cross:
        value = directional_fir(value, orientation.long, *long3)
        return directional_fir(
            value,
            orientation.cross,
            *compose_fir(cross3, difference),
        )
    if orientation.difference == orientation.long:
        value = directional_fir(
            value,
            orientation.long,
            *compose_fir(long3, difference),
        )
        return directional_fir(value, orientation.cross, *cross3)
    # The generic commuting form is retained for future orientations.
    value = directional_fir(value, orientation.long, *long3)
    value = directional_fir(value, orientation.cross, *cross3)
    return directional_fir(value, orientation.difference, *difference)


class _MovingAverage:
    """Causal moving average used as one delayed centered-box stage."""

    def __init__(self, width: int):
        self.values = np.zeros(width, dtype=np.float64)
        self.position = 0
        self.total = 0.0
        self.inverse = 1.0 / width

    def push(self, value: float) -> float:
        old = self.values[self.position]
        self.values[self.position] = value
        self.position += 1
        if self.position == self.values.size:
            self.position = 0
        self.total += value - old
        return self.total * self.inverse


class _ThreeTap:
    """Causal ring state corresponding to one centered three-tap stage."""

    def __init__(self, side: float):
        self.values = np.zeros(3, dtype=np.float64)
        self.position = 0
        self.side = side
        self.center = 1.0 - 2.0 * side

    def push(self, value: float) -> float:
        self.values[self.position] = value
        self.position = (self.position + 1) % 3
        newest = self.values[(self.position - 1) % 3]
        middle = self.values[(self.position - 2) % 3]
        oldest = self.values[self.position]
        return self.side * oldest + self.center * middle + self.side * newest


def _stream_periodic_cascade(
    line: np.ndarray,
    stages: list[_MovingAverage] | list[_ThreeTap],
    radius: int,
) -> np.ndarray:
    """Stream a centered FIR cascade over an infinitely periodic line."""

    source = np.ascontiguousarray(line, dtype=np.float64)
    length = source.size
    memory = 2 * radius * len(stages)
    latency = radius * len(stages)
    result = np.empty_like(source)
    for time in range(memory + length):
        value = float(source[time % length])
        for stage in stages:
            value = stage.push(value)
        if time >= memory:
            result[(time - latency) % length] = value
    return result


def _direction_cycles(shape: tuple[int, int], direction: Direction):
    """Yield every disjoint periodic cycle of a lattice direction."""

    height, width = shape
    dy, dx = direction
    visited = np.zeros(shape, dtype=bool)
    for seed in range(height * width):
        sy, sx = divmod(seed, width)
        if visited[sy, sx]:
            continue
        cycle: list[tuple[int, int]] = []
        y, x = sy, sx
        while not visited[y, x]:
            visited[y, x] = True
            cycle.append((y, x))
            y = (y + dy) % height
            x = (x + dx) % width
        yield cycle


def _ring_box_three(
    image: np.ndarray, direction: Direction, radius: int
) -> np.ndarray:
    result = np.empty_like(image)
    width = 2 * radius + 1
    for cycle in _direction_cycles(image.shape, direction):
        line = np.array([image[y, x] for y, x in cycle])
        filtered = _stream_periodic_cascade(
            line, [_MovingAverage(width) for _ in range(3)], radius
        )
        for (y, x), value in zip(cycle, filtered):
            result[y, x] = value
    return result


def _ring_cross_three(
    image: np.ndarray, direction: Direction, side: float
) -> np.ndarray:
    result = np.empty_like(image)
    for cycle in _direction_cycles(image.shape, direction):
        line = np.array([image[y, x] for y, x in cycle])
        filtered = _stream_periodic_cascade(
            line, [_ThreeTap(side) for _ in range(3)], 1
        )
        for (y, x), value in zip(cycle, filtered):
            result[y, x] = value
    return result


def response_ring(image: np.ndarray, orientation: OrientedGate) -> np.ndarray:
    """Ring-buffer cascade equivalent to the literal staged response."""

    value = _ring_box_three(
        _as_image(image), orientation.long, orientation.box_radius
    )
    value = _ring_cross_three(value, orientation.cross, orientation.cross_side)
    return _shift(value, orientation.difference, 1) - value


def normalize_gate(raw: np.ndarray) -> np.ndarray:
    """Apply the native mean scale and sixth-power soft tail selection."""

    value = _as_image(raw)
    scale = max(1.6 * float(np.mean(value)), 1e-12)
    ratio = value / scale
    square = ratio * ratio
    base = square / (1.0 + square)
    base2 = base * base
    return base2 * base2 * base2


def _gate(image: np.ndarray, response) -> tuple[np.ndarray, np.ndarray]:
    source = _as_image(image)
    raw = np.zeros_like(source)
    for orientation in ORIENTATIONS:
        raw += np.abs(response(source, orientation))
    return normalize_gate(raw), raw


def gate_staged(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normalised_gate, raw_four_direction_response)``."""

    return _gate(image, response_staged)


def gate_collapsed(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the algebraically collapsed gate and raw response."""

    return _gate(image, response_collapsed)


def gate_ring(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ring-buffer gate and raw response."""

    return _gate(image, response_ring)


__all__ = [
    "ORIENTATIONS",
    "OrientedGate",
    "box_kernel",
    "compose_fir",
    "cross_kernel",
    "difference_kernel",
    "directional_fir",
    "gate_collapsed",
    "gate_ring",
    "gate_staged",
    "normalize_gate",
    "response_collapsed",
    "response_ring",
    "response_staged",
]
