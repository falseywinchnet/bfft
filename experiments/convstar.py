"""Bounded-work FIR realization of Compact Ordered Nodal Variation.

CONV* keeps the formal CONV ledger and signed-fibre admission, realizes the
compact jet as FIR arithmetic, solves admission by its exact scalar KKT
threshold, and fuses current integration with fixed polyphase synthesis.
It is the compiled form of the same operator, not a second interpolation law.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


Array = np.ndarray
_EPS = np.finfo(np.float64).eps

# Each row maps (y_{i-2},...,y_{i+3}) to one raw interior current.
FUSED_CURRENT_KERNELS = np.array((
    (4, -32, 0, 32, -4, 0),
    (3, -16, -30, 48, -5, 0),
    (-7, 39, -130, 130, -39, 7),
    (0, 5, -48, 30, 16, -3),
    (0, 4, -32, 0, 32, -4),
), dtype=np.float64) / 240.0

def _local_two_jet(lines: Array) -> tuple[Array, Array]:
    """Fourth-order interior jet with the formal second-order closure."""

    source = np.asarray(lines, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 5:
        raise ValueError("lines must have shape (N,C) with N >= 5")
    first = np.empty_like(source)
    second = np.empty_like(source)
    first[0] = (-3.0 * source[0] + 4.0 * source[1] - source[2]) / 2.0
    first[1] = (source[2] - source[0]) / 2.0
    first[-2] = (source[-1] - source[-3]) / 2.0
    first[-1] = (3.0 * source[-1] - 4.0 * source[-2] + source[-3]) / 2.0
    second[0] = 2.0 * source[0] - 5.0 * source[1] + 4.0 * source[2] - source[3]
    second[1] = source[0] - 2.0 * source[1] + source[2]
    second[-2] = source[-3] - 2.0 * source[-2] + source[-1]
    second[-1] = (
        2.0 * source[-1] - 5.0 * source[-2]
        + 4.0 * source[-3] - source[-4]
    )
    first[2:-2] = (
        source[:-4] - 8.0 * source[1:-3]
        + 8.0 * source[3:-1] - source[4:]
    ) / 12.0
    second[2:-2] = (
        -source[4:] + 16.0 * source[3:-1] - 30.0 * source[2:-2]
        + 16.0 * source[1:-3] - source[:-4]
    ) / 12.0
    return first, second


def raw_current_jet_bank(lines: Array) -> tuple[Array, Array]:
    """Construct all five raw currents from the compact FIR jet bank."""

    source = np.asarray(lines, dtype=np.float64)
    first, second = _local_two_jet(source)
    delta = np.diff(source, axis=0)
    raw = np.empty((source.shape[0] - 1, 5, source.shape[1]), dtype=np.float64)
    raw[:, 0] = first[:-1] / 5.0
    raw[:, 1] = first[:-1] / 5.0 + second[:-1] / 20.0
    raw[:, 2] = (
        delta - (2.0 / 5.0) * (first[:-1] + first[1:])
        + (second[1:] - second[:-1]) / 20.0
    )
    raw[:, 3] = first[1:] / 5.0 - second[1:] / 20.0
    raw[:, 4] = first[1:] / 5.0
    return raw, delta


def raw_current_fused_bank(lines: Array) -> tuple[Array, Array]:
    """Use the exact six-tap five-current bank on interior cells.

    The formal outer and near-outer closure is retained verbatim.  It is
    cheaper in NumPy to form the two jet channels everywhere, so this routine
    exists to test convolution-backend choices rather than as the CPU default.
    """

    source = np.asarray(lines, dtype=np.float64)
    raw, delta = raw_current_jet_bank(source)
    if source.shape[0] >= 6:
        windows = np.stack(
            [source[offset : offset + source.shape[0] - 5] for offset in range(6)],
            axis=1,
        )
        raw[2 : source.shape[0] - 3] = np.einsum(
            "jk,ikc->ijc", FUSED_CURRENT_KERNELS, windows, optimize=True
        )
    return raw, delta


def _fill_zero_signs(delta: Array) -> Array:
    """Propagate the last witnessed sign, backfilling only the leading run."""

    filled = np.sign(np.asarray(delta, dtype=np.float64))
    for index in range(1, filled.shape[0]):
        missing = filled[index] == 0.0
        filled[index, missing] = filled[index - 1, missing]
    for index in range(filled.shape[0] - 2, -1, -1):
        missing = filled[index] == 0.0
        filled[index, missing] = filled[index + 1, missing]
    return filled


def _ordered_sign_ledger_reference(raw: Array, delta: Array) -> Array:
    """Scalar-component oracle for the exact segmented causal scan."""

    intervals, order, components = raw.shape
    if order != 5 or delta.shape != (intervals, components):
        raise ValueError("inconsistent raw-current and secant arrays")
    flat_raw = raw.reshape(intervals * order, components)
    coarse = _fill_zero_signs(delta)
    ledger = np.zeros_like(flat_raw)
    for component in range(components):
        signs = coarse[:, component]
        if not np.any(signs):
            continue
        transition_knots = np.flatnonzero(signs[:-1] != signs[1:]) + 1
        boundaries: list[tuple[int, float]] = []
        previous = 0
        for knot in transition_knots:
            centre = order * int(knot)
            candidates = np.arange(
                max(previous + 1, centre - order + 1),
                min(order * intervals, centre + order),
            )
            local = np.arange(
                max(0, centre - order), min(order * intervals, centre + order)
            )
            value = flat_raw[local, component]
            expected = np.where(
                local[None, :] < candidates[:, None],
                signs[knot - 1],
                signs[knot],
            )
            cost = np.sum(
                np.where(expected * value[None, :] < 0.0, value[None, :] ** 2, 0.0),
                axis=1,
            )
            boundary = int(candidates[int(np.argmin(cost))])
            boundaries.append((boundary, float(signs[knot])))
            previous = boundary
        sequence = np.full(intervals * order, signs[0], dtype=np.float64)
        for boundary, new_sign in boundaries:
            sequence[boundary:] = new_sign
        ledger[:, component] = sequence
    return ledger.reshape(intervals, order, components)


def ordered_sign_ledger(raw: Array, delta: Array) -> Array:
    """Return the exact sign ledger with knot-wise component batching.

    The formal recurrence is causal only in transition-knot order.  Components
    at one knot are independent, so their nine possible boundary placements
    and ten-sample contradiction costs are evaluated as one tensor.  This
    removes the Python component loop without changing candidate order,
    tie-breaking, or the previous-boundary constraint.
    """

    intervals, order, components = raw.shape
    if order != 5 or delta.shape != (intervals, components):
        raise ValueError("inconsistent raw-current and secant arrays")
    flat_raw = raw.reshape(intervals * order, components)
    coarse = _fill_zero_signs(delta)
    total = intervals * order
    previous = np.zeros(components, dtype=np.int64)
    transition = coarse[:-1] != coarse[1:]
    candidate_offsets = np.arange(-order + 1, order, dtype=np.int64)
    boundary_marker = np.zeros((total, components), dtype=bool)
    boundary_value = np.zeros((total, components), dtype=np.float64)

    for knot in range(1, intervals):
        active = np.flatnonzero(transition[knot - 1])
        if not active.size:
            continue
        centre = order * knot
        local = np.arange(
            max(0, centre - order), min(total, centre + order),
            dtype=np.int64,
        )
        candidates = centre + candidate_offsets[None, :]
        candidates = np.broadcast_to(
            candidates, (active.size, candidate_offsets.size))
        valid = (
            (candidates > previous[active, None])
            & (candidates >= max(1, centre - order + 1))
            & (candidates < min(total, centre + order))
        )
        value = flat_raw[local[:, None], active[None, :]].T
        expected = np.where(
            local[None, None, :] < candidates[:, :, None],
            coarse[knot - 1, active, None, None],
            coarse[knot, active, None, None],
        )
        contradiction = expected * value[:, None, :] < 0.0
        cost = np.sum(
            np.where(contradiction, value[:, None, :] ** 2, 0.0), axis=2)
        cost[~valid] = np.inf
        selected = np.argmin(cost, axis=1)
        boundary = candidates[np.arange(active.size), selected]
        previous[active] = boundary
        boundary_marker[boundary, active] = True
        boundary_value[boundary, active] = coarse[knot, active]

    # Each component's chosen boundaries are strictly increasing.  Record the
    # sparse changes above, then perform one causal prefix fill instead of
    # rewriting every remaining row after every transition.
    row_coordinate = np.arange(total, dtype=np.int64)[:, None]
    last_boundary = np.maximum.accumulate(
        np.where(boundary_marker, row_coordinate, -1), axis=0)
    gathered = np.take_along_axis(
        boundary_value, np.maximum(last_boundary, 0), axis=0)
    ledger = np.where(
        last_boundary >= 0,
        gathered,
        np.broadcast_to(coarse[0], (total, components)),
    )
    return ledger.reshape(intervals, order, components)


def project_signed_fibres(
    raw: Array,
    signs: Array,
    delta: Array,
    *,
    chunk_size: int = 8192,
) -> Array:
    """Project each signed fibre by its exact scalar KKT breakpoint.

    In oriented coordinates the problem is

    ``min ||x - s*r||^2, x >= 0, <s,x> = delta``.

    Hence ``x_i = max(s_i (r_i - lambda), 0)``. The five raw currents are
    exactly the breakpoints of this monotone scalar equation, so sorting them
    leaves six possible affine intervals and determines the unique current.
    """

    intervals, order, components = raw.shape
    if order != 5 or signs.shape != raw.shape or delta.shape != (
            intervals, components):
        raise ValueError("inconsistent projection arrays")
    flat_raw = raw.transpose(0, 2, 1).reshape(-1, order)
    flat_sign = signs.transpose(0, 2, 1).reshape(-1, order)
    flat_delta = delta.reshape(-1)
    result = np.zeros_like(flat_raw)
    valid_rows = np.flatnonzero(np.any(flat_sign != 0.0, axis=1))
    position = np.arange(order, dtype=np.int64)[None, :]
    interval_index = np.arange(order + 1, dtype=np.int64)

    for start in range(0, valid_rows.size, chunk_size):
        rows = valid_rows[start : start + chunk_size]
        value = flat_raw[rows]
        orientation = flat_sign[rows]
        permutation = np.argsort(value, axis=1, kind="stable")
        breakpoint = np.take_along_axis(value, permutation, axis=1)
        ordered_sign = np.take_along_axis(
            orientation, permutation, axis=1)
        active = (
            ((ordered_sign[:, None, :] > 0.0)
             & (position[:, None, :] >= interval_index[None, :, None]))
            | ((ordered_sign[:, None, :] < 0.0)
               & (position[:, None, :] < interval_index[None, :, None]))
        )
        active_count = np.sum(active, axis=2)
        numerator = (
            np.sum(np.where(active, breakpoint[:, None, :], 0.0), axis=2)
            - flat_delta[rows, None]
        )
        multiplier = np.divide(
            numerator,
            active_count,
            out=np.zeros_like(numerator),
            where=active_count > 0,
        )
        lower = np.concatenate((
            np.full((rows.size, 1), -np.inf), breakpoint,
        ), axis=1)
        upper = np.concatenate((
            breakpoint, np.full((rows.size, 1), np.inf),
        ), axis=1)
        scale = np.maximum(
            np.max(np.abs(breakpoint), axis=1, keepdims=True),
            np.abs(multiplier),
        )
        tolerance = 256.0 * _EPS * np.maximum(scale, 1.0)
        feasible = (
            (active_count > 0)
            & (multiplier >= lower - tolerance)
            & (multiplier <= upper + tolerance)
        )
        if np.any(~np.any(feasible, axis=1)):
            raise RuntimeError("signed current fibre has no KKT interval")
        selected_interval = np.argmax(feasible, axis=1)
        selected_multiplier = multiplier[
            np.arange(rows.size), selected_interval]
        oriented = np.maximum(
            orientation * (value - selected_multiplier[:, None]), 0.0)
        result[rows] = orientation * oriented
    return result.reshape(intervals, components, order).transpose(0, 2, 1)


@lru_cache(maxsize=None)
def polyphase_tail_weights(scale: int) -> Array:
    """Tail-integrated quintic Bernstein weights for inserted phases."""

    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be positive")
    tails = np.empty((max(0, scale - 1), 5), dtype=np.float64)
    for phase in range(1, scale):
        u = phase / scale
        basis = np.array([
            math.comb(5, degree) * u**degree * (1.0 - u) ** (5 - degree)
            for degree in range(6)
        ])
        tails[phase - 1] = [np.sum(basis[index + 1 :]) for index in range(5)]
    return tails


def synthesize_polyphase(source: Array, current: Array, scale: int) -> Array:
    """Fuse current integration and fixed-scale Bernstein synthesis."""

    source = np.asarray(source, dtype=np.float64)
    intervals, order, components = current.shape
    if source.shape != (intervals + 1, components) or order != 5:
        raise ValueError("inconsistent source and current arrays")
    output = np.empty((intervals * scale + 1, components), dtype=np.float64)
    output[::scale] = source
    tails = polyphase_tail_weights(scale)
    if scale > 1:
        inserted = source[:-1, None, :] + np.einsum(
            "ikc,pk->ipc", current, tails, optimize=True
        )
        for phase in range(1, scale):
            output[phase::scale] = inserted[:, phase - 1]
    return output


def refine_lines(
    lines: Array,
    scale: int = 2,
    *,
    front_end: str = "jet",
    chunk_size: int = 8192,
) -> Array:
    """Apply one complete compiled CONV* factor to a batch of lines."""

    source = np.asarray(lines, dtype=np.float64)
    if front_end == "jet":
        raw, delta = raw_current_jet_bank(source)
    elif front_end == "fused":
        raw, delta = raw_current_fused_bank(source)
    else:
        raise ValueError("front_end must be 'jet' or 'fused'")
    signs = ordered_sign_ledger(raw, delta)
    current = project_signed_fibres(raw, signs, delta, chunk_size=chunk_size)
    return synthesize_polyphase(source, current, scale)


def refine_axis(
    values: Array,
    axis: int,
    scale: int = 2,
    *,
    front_end: str = "jet",
    chunk_size: int = 8192,
) -> Array:
    """Batch every line/component along one Cartesian factor."""

    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    source_shape = moved.shape
    lines = moved.reshape(source_shape[0], -1)
    refined = refine_lines(
        lines, scale, front_end=front_end, chunk_size=chunk_size
    )
    restored = refined.reshape((refined.shape[0],) + source_shape[1:])
    return np.moveaxis(restored, 0, axis)


def convstar_resize(
    samples: Array,
    scale: int = 2,
    *,
    front_end: str = "jet",
    chunk_size: int = 8192,
) -> Array:
    """Horizontal-then-vertical fixed-scale CONV* composition."""

    field = np.asarray(samples, dtype=np.float64)
    if field.ndim not in (2, 3):
        raise ValueError("samples must be HxW or HxWxC")
    along_x = refine_axis(
        field, 1, scale, front_end=front_end, chunk_size=chunk_size
    )
    return refine_axis(
        along_x, 0, scale, front_end=front_end, chunk_size=chunk_size
    )
