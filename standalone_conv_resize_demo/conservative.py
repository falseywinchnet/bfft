"""Conservative dyadic support geometry for the standalone CONV demo.

Raster samples are interpreted as cell averages.  Restriction is the exact
mean of each 2x2 child block.  Synthesis transports the horizontal, vertical,
and mixed centered moments predicted by the admitted CONV profile, then
intersects their common ray with the witnessed face-current capacities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .backend import (
        admit_moments_2d, conv_four_child_moment_atlas, conv_moment_axis,
    )
except ImportError:  # Standalone scripts place this directory on sys.path.
    from backend import (
        admit_moments_2d, conv_four_child_moment_atlas, conv_moment_axis,
    )


Array = np.ndarray


@dataclass(frozen=True)
class BlockMoments:
    mean: Array
    horizontal: Array
    vertical: Array
    mixed: Array


def restrict_2x2(fine: Array) -> Array:
    """Return exact dyadic parent-cell averages."""

    value = np.asarray(fine, dtype=np.float32)
    if value.ndim not in (2, 3) or value.shape[0] % 2 or value.shape[1] % 2:
        raise ValueError("restriction requires an even HxW or HxWxC raster")
    return np.float32(0.25) * (
        value[0::2, 0::2] + value[0::2, 1::2]
        + value[1::2, 0::2] + value[1::2, 1::2]
    )


def _filled_sign_axis(difference: Array, axis: int) -> Array:
    """Fill zero face signs from the nearest preceding witnessed current."""

    sign = np.sign(np.asarray(difference, dtype=np.float32))
    moved = np.moveaxis(sign, axis, 0).copy()
    for index in range(1, moved.shape[0]):
        zero = moved[index] == 0
        moved[index][zero] = moved[index - 1][zero]
    for index in range(moved.shape[0] - 2, -1, -1):
        zero = moved[index] == 0
        moved[index][zero] = moved[index + 1][zero]
    return np.moveaxis(moved, 0, axis)


def _cell_lineage_sign(proposal: Array, face_sign: Array, axis: int) -> Array:
    """Assign each block moment to one adjacent witnessed sign lineage."""

    raw = np.moveaxis(np.asarray(proposal, dtype=np.float32), axis, 0)
    sign = np.moveaxis(np.asarray(face_sign, dtype=np.float32), axis, 0)
    result = np.empty_like(raw)
    result[0] = sign[0]
    result[-1] = sign[-1]
    left = sign[:-1]
    right = sign[1:]
    middle = raw[1:-1]
    result[1:-1] = np.where(
        left == right,
        left,
        np.where(middle * right > middle * left, right, left),
    )
    return np.moveaxis(result, 0, axis)


def proposal_2d(coarse: Array) -> BlockMoments:
    """Predict the three centered half-cell moments of every parent cell."""

    value = np.asarray(coarse, dtype=np.float32)
    if value.ndim not in (2, 3) or min(value.shape[:2]) < 5:
        raise ValueError("moment prediction requires at least 5x5 parent cells")
    horizontal = conv_moment_axis(value, 1)
    vertical = conv_moment_axis(value, 0)
    mixed = conv_moment_axis(horizontal, 0)
    return BlockMoments(value.copy(), horizontal, vertical, mixed)


def _face_capacity(
    delta: Array,
    orientation: Array,
    residual_a: Array,
    residual_b: Array,
) -> Array:
    """Return the exact common ray capacity left by one split face."""

    base = np.abs(delta)
    consumed = (
        np.maximum(-orientation * residual_a, np.float32(0.0))
        + np.maximum(-orientation * residual_b, np.float32(0.0))
    )
    capacity = np.ones_like(base, dtype=np.float32)
    active = (orientation != 0) & (consumed > base) & (consumed > 0)
    ratio = np.divide(base, consumed, out=np.ones_like(base), where=active)
    capacity[active] = np.nextafter(
        ratio[active], np.float32(0.0), dtype=np.float32
    )
    epsilon = np.finfo(np.float32).eps
    gamma8 = np.float32((8.0 * epsilon) / (1.0 - 8.0 * epsilon))
    cancellation_error = gamma8 * (np.abs(residual_a) + np.abs(residual_b))
    unsupported = (orientation == 0) & (
        np.abs(residual_a + residual_b) > cancellation_error
    )
    capacity[unsupported] = 0
    return capacity


def admit_2d_reference(coarse: Array, proposal: BlockMoments) -> BlockMoments:
    """Admit a tensor-moment proposal to the complete face-current polytope."""

    value = np.asarray(coarse, dtype=np.float32)
    scalar = value.ndim == 2
    c = value[..., None] if scalar else value
    px = np.asarray(proposal.horizontal, dtype=np.float32)
    py = np.asarray(proposal.vertical, dtype=np.float32)
    pxy = np.asarray(proposal.mixed, dtype=np.float32)
    if scalar:
        px, py, pxy = px[..., None], py[..., None], pxy[..., None]
    if px.shape != c.shape or py.shape != c.shape or pxy.shape != c.shape:
        raise ValueError("all proposed moments must have the coarse shape")

    sx = _filled_sign_axis(np.diff(c, axis=1), 1)
    sy = _filled_sign_axis(np.diff(c, axis=0), 0)
    tx = _cell_lineage_sign(px, sx, 1)
    ty = _cell_lineage_sign(py, sy, 0)
    ax = np.maximum(tx * px, np.float32(0.0))
    ay = np.maximum(ty * py, np.float32(0.0))
    qx = tx * ax
    qy = ty * ay
    qxy = np.sign(pxy) * np.minimum(np.abs(pxy), np.minimum(ax, ay))
    alpha = np.ones_like(c, dtype=np.float32)

    # Each coarse horizontal face splits into upper and lower fine faces.
    delta = c[:, 1:] - c[:, :-1]
    orientation = sx
    upper = _face_capacity(
        delta, orientation,
        -qx[:, :-1] + qy[:, :-1] + qxy[:, :-1],
        -qx[:, 1:] - qy[:, 1:] + qxy[:, 1:],
    )
    lower = _face_capacity(
        delta, orientation,
        -qx[:, :-1] - qy[:, :-1] - qxy[:, :-1],
        -qx[:, 1:] + qy[:, 1:] - qxy[:, 1:],
    )
    horizontal_cap = np.minimum(upper, lower)
    alpha[:, :-1] = np.minimum(alpha[:, :-1], horizontal_cap)
    alpha[:, 1:] = np.minimum(alpha[:, 1:], horizontal_cap)

    # Each coarse vertical face splits into left and right fine faces.
    delta = c[1:] - c[:-1]
    orientation = sy
    left = _face_capacity(
        delta, orientation,
        qx[:-1] - qy[:-1] + qxy[:-1],
        -qx[1:] - qy[1:] + qxy[1:],
    )
    right = _face_capacity(
        delta, orientation,
        -qx[:-1] - qy[:-1] - qxy[:-1],
        qx[1:] - qy[1:] - qxy[1:],
    )
    vertical_cap = np.minimum(left, right)
    alpha[:-1] = np.minimum(alpha[:-1], vertical_cap)
    alpha[1:] = np.minimum(alpha[1:], vertical_cap)

    admitted = BlockMoments(c, alpha * qx, alpha * qy, alpha * qxy)
    if not scalar:
        return admitted
    return BlockMoments(*(item[..., 0] for item in admitted.__dict__.values()))


def admit_2d(coarse: Array, proposal: BlockMoments) -> BlockMoments:
    """Native complete face-current admission."""

    value = np.asarray(coarse, dtype=np.float32)
    horizontal, vertical, mixed = admit_moments_2d(
        value, proposal.horizontal, proposal.vertical, proposal.mixed
    )
    return BlockMoments(value.copy(), horizontal, vertical, mixed)


def moments_2d(coarse: Array) -> BlockMoments:
    """Return admitted horizontal, vertical, and mixed child moments."""

    proposed = proposal_2d(coarse)
    return admit_2d(proposed.mean, proposed)


def synthesize(moments: BlockMoments) -> Array:
    """Invert the normalized 2x2 mean/moment transform."""

    c = np.asarray(moments.mean, dtype=np.float32)
    qx = np.asarray(moments.horizontal, dtype=np.float32)
    qy = np.asarray(moments.vertical, dtype=np.float32)
    qxy = np.asarray(moments.mixed, dtype=np.float32)
    if qx.shape != c.shape or qy.shape != c.shape or qxy.shape != c.shape:
        raise ValueError("all block moments must have the same shape")
    fine = np.empty((2 * c.shape[0], 2 * c.shape[1]) + c.shape[2:], np.float32)
    fine[0::2, 0::2] = c - qx - qy + qxy
    fine[0::2, 1::2] = c + qx - qy - qxy
    fine[1::2, 0::2] = c - qx + qy - qxy
    fine[1::2, 1::2] = c + qx + qy + qxy
    return fine


def zero_detail_synthesis_reference(coarse: Array) -> Array:
    """Decomposed oracle for one conservative zero-detail synthesis level."""

    return synthesize(moments_2d(coarse))


def zero_detail_synthesis(coarse: Array) -> Array:
    """Fused native four-child moment atlas for one dyadic level."""

    return conv_four_child_moment_atlas(coarse)


def dyadic_levels(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> int:
    """Return the common exact dyadic depth or raise for incompatible grids."""

    sh, sw = map(int, source_shape)
    th, tw = map(int, target_shape)
    if th < 5 or tw < 5 or sh < th or sw < tw or sh % th or sw % tw:
        raise ValueError("target must be an exact dyadic reduction with 5x5 support")
    rh, rw = sh // th, sw // tw
    if rh != rw or rh < 1 or rh & (rh - 1):
        raise ValueError("both axes must use the same power-of-two reduction")
    return rh.bit_length() - 1


def restrict_to(fine: Array, target_shape: tuple[int, int]) -> tuple[Array, int]:
    """Conservatively restrict to an exact dyadic target."""

    current = np.asarray(fine, dtype=np.float32)
    levels = dyadic_levels(current.shape[:2], target_shape)
    for _ in range(levels):
        current = restrict_2x2(current)
    return current, levels


def zero_detail_expand(coarse: Array, levels: int) -> Array:
    """Apply matched zero-detail synthesis from coarsest to finest."""

    current = np.asarray(coarse, dtype=np.float32)
    for _ in range(int(levels)):
        current = zero_detail_synthesis(current)
    return current


def matched_cycle(fine: Array, target_shape: tuple[int, int]) -> tuple[Array, Array]:
    """Return exact conservative coarse state and its matched reconstruction."""

    coarse, levels = restrict_to(fine, target_shape)
    return coarse, zero_detail_expand(coarse, levels)
