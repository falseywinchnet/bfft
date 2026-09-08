"""Bounded-work forms implied by the current all-grid CONV equations.

The paper's all-source partition is cardinal on the source lattice.  Its
support inertias therefore vanish identically, so every stated chart reduces
to the reverse Cartesian factor order.  This module exposes that exact
collapse and a permutation-covariant local tensor weight for testing the one
remaining design question without the obsolete sparse-owner machinery.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "standalone_conv_resize_demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from backend import (  # noqa: E402
    _basin_average_axis,
    conv_evaluate_profile,
    conv_oriented_chord_blend,
    conv_resize,
    conv_two_order_synthesis,
    linear_resize,
    q1_order_blend,
)
from experiments.convstar import _local_two_jet  # noqa: E402


Array = np.ndarray


def _variation_jet(values: Array) -> tuple[Array, Array]:
    """Apply CONV's declared first-jet stencil on both Cartesian axes."""

    field = np.asarray(values, dtype=np.float64)
    height, width, channels = field.shape
    gx_lines = np.moveaxis(field, 1, 0).reshape(width, -1)
    gy_lines = field.reshape(height, -1)
    gx = _local_two_jet(gx_lines)[0].reshape(
        (width, height, channels)
    ).transpose(1, 0, 2)
    gy = _local_two_jet(gy_lines)[0].reshape(field.shape)
    return gx, gy


def nodal_current_geometry(values: Array) -> tuple[Array, Array, Array, Array]:
    """Return ``Gxx,Gxy,Gyy`` and its parameter-free order coordinate.

    Diagonalizing the 2-by-2 tensor in the formal expression cancels
    algebraically, leaving ``beta = Gyy / (Gxx + Gyy)``.  The cross term is
    retained in the return value for metric construction, but neither it nor
    an eigendecomposition is needed for the order coordinate itself.  The
    zero-current convention is the symmetric value one half.
    """

    field = np.asarray(values, dtype=np.float64)
    if field.ndim == 2:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 5:
        raise ValueError("distilled CONV requires an HxW or HxWxC raster")
    gx, gy = _variation_jet(field)
    gxx = np.sum(gx * gx, axis=2)
    gxy = np.sum(gx * gy, axis=2)
    gyy = np.sum(gy * gy, axis=2)
    trace = gxx + gyy
    beta = np.divide(
        gyy,
        trace,
        out=np.full_like(trace, 0.5),
        where=trace > 0.0,
    )
    return gxx, gxy, gyy, np.clip(beta, 0.0, 1.0)


def reverse_conv_resize(values: Array, target_shape: tuple[int, int]) -> Array:
    """Vertical-then-horizontal cardinal CONV synthesis."""

    field = np.asarray(values, dtype=np.float32)
    transposed = np.swapaxes(field, 0, 1)
    resized = conv_resize(
        transposed, (int(target_shape[1]), int(target_shape[0]))
    )
    return np.swapaxes(resized, 0, 1)


def symmetric_conv_synthesis(values: Array, target_shape: tuple[int, int]) -> Array:
    """The permutation-covariant mean of the two declared factor orders."""

    forward = np.asarray(conv_resize(values, target_shape), dtype=np.float64)
    reverse = np.asarray(reverse_conv_resize(values, target_shape), dtype=np.float64)
    return (0.5 * (forward + reverse)).astype(np.float32)


def distilled_conv_synthesis(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Two-order CONV synthesis with a local tensor-covariant partition."""

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    beta_source = nodal_current_geometry(field)[3].astype(np.float32)
    return conv_two_order_synthesis(field, target, beta_source)


def distilled_conv_resize(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Basin analysis on reduction and bounded-work synthesis on enlargement."""

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5 or min(target) < 5:
        raise ValueError("distilled CONV requires at least five sites per axis")
    output = field
    if target[0] < output.shape[0]:
        output = _basin_average_axis(output, target[0], 0)
    if target[1] < output.shape[1]:
        output = _basin_average_axis(output, target[1], 1)
    if output.shape[:2] == target:
        return np.asarray(output, dtype=np.float32)
    return distilled_conv_synthesis(output, target)


def _rectangle_sums(values: Array, radius_left: int, radius_right: int) -> Array:
    """Sum one nodal field over each cell's clipped support rectangle."""

    field = np.asarray(values, dtype=np.float64)
    height, width = field.shape
    integral = np.pad(field, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    iy, ix = np.indices((height - 1, width - 1))
    top = np.maximum(0, iy - radius_left)
    left = np.maximum(0, ix - radius_left)
    bottom = np.minimum(height - 1, iy + radius_right)
    right = np.minimum(width - 1, ix + radius_right)
    return (
        integral[bottom + 1, right + 1]
        - integral[top, right + 1]
        - integral[bottom + 1, left]
        + integral[top, left]
    )


def _cell_transport_geometry(
    values: Array,
    *,
    jet_support: bool,
) -> tuple[Array, Array, Array]:
    """Return coherence and the unoriented tangent on every source cell."""

    gxx, gxy, gyy, _ = nodal_current_geometry(values)
    if jet_support:
        # The interior current for cell i reads nodes i-2,...,i+3.  The 2-D
        # tensor uses exactly the Cartesian product of those declared reads.
        xx = _rectangle_sums(gxx, 2, 3)
        xy = _rectangle_sums(gxy, 2, 3)
        yy = _rectangle_sums(gyy, 2, 3)
    else:
        xx = gxx[:-1, :-1] + gxx[1:, :-1] + gxx[:-1, 1:] + gxx[1:, 1:]
        xy = gxy[:-1, :-1] + gxy[1:, :-1] + gxy[:-1, 1:] + gxy[1:, 1:]
        yy = gyy[:-1, :-1] + gyy[1:, :-1] + gyy[:-1, 1:] + gyy[1:, 1:]
    trace = xx + yy
    gap = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy * xy, 0.0))
    coherence = np.divide(
        gap, trace, out=np.zeros_like(gap), where=trace > 0.0
    )
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    # angle is the principal normal; its quarter turn is the tangent.
    tangent_x = -np.sin(angle)
    tangent_y = np.cos(angle)
    return np.clip(coherence, 0.0, 1.0), tangent_x, tangent_y


def _sample_cell_field_bilinear(field: Array, x: Array, y: Array) -> Array:
    """Evaluate a cell-centred scalar field by its affine cell extension."""

    values = np.asarray(field, dtype=np.float64)
    # Cell (j,i) is centred at source coordinate (i+1/2,j+1/2).
    qx = np.clip(np.asarray(x, dtype=np.float64) - 0.5, 0.0, values.shape[1] - 1.0)
    qy = np.clip(np.asarray(y, dtype=np.float64) - 0.5, 0.0, values.shape[0] - 1.0)
    x0 = np.floor(qx).astype(np.intp)
    y0 = np.floor(qy).astype(np.intp)
    x1 = np.minimum(x0 + 1, values.shape[1] - 1)
    y1 = np.minimum(y0 + 1, values.shape[0] - 1)
    ux = qx - x0
    uy = qy - y0
    return (
        (1.0 - ux) * (1.0 - uy) * values[y0, x0]
        + ux * (1.0 - uy) * values[y0, x1]
        + (1.0 - ux) * uy * values[y1, x0]
        + ux * uy * values[y1, x1]
    )


def _transported_cell_geometry(
    mass_cell: Array,
    tangent_x_cell: Array,
    tangent_y_cell: Array,
    x: Array,
    y: Array,
) -> tuple[Array, Array, Array]:
    """Continuously extend an unoriented tensor before taking its frame.

    The normalized deviatoric tensor has coordinates

        q = chi cos(2 theta),  r = chi sin(2 theta),

    where theta is the principal-normal angle and chi is coherence.  These
    coordinates are sign-free and lie in the unit disk.  Affine extension is
    therefore bounded and axis-covariant.  Recovering the polar form after
    extension transports both the line field and its admission without
    interpolating eigenvector signs or selecting source-cell owners.
    """

    mass = np.asarray(mass_cell, dtype=np.float64)
    tx = np.asarray(tangent_x_cell, dtype=np.float64)
    ty = np.asarray(tangent_y_cell, dtype=np.float64)
    # n=(ty,-tx), hence cos(2 theta)=ty^2-tx^2 and
    # sin(2 theta)=-2 tx ty.
    q_cell = mass * (ty * ty - tx * tx)
    r_cell = -2.0 * mass * tx * ty
    q = _sample_cell_field_bilinear(q_cell, x, y)
    r = _sample_cell_field_bilinear(r_cell, x, y)
    transported_mass = np.minimum(np.sqrt(q * q + r * r), 1.0)
    angle = 0.5 * np.arctan2(r, q)
    return transported_mass, -np.sin(angle), np.cos(angle)


def _sample_profile_boundaries(
    values: Array,
    x: Array,
    y: Array,
    horizontal: Array,
) -> Array:
    """Batch arbitrary admitted-profile reads on source rows or columns."""

    source = np.asarray(values, dtype=np.float32)
    scalar = source.ndim == 2
    if scalar:
        source = source[..., None]
    query_x = np.asarray(x, dtype=np.float64).reshape(-1)
    query_y = np.asarray(y, dtype=np.float64).reshape(-1)
    use_row = np.asarray(horizontal, dtype=bool).reshape(-1)
    output = np.empty((query_x.size, source.shape[2]), dtype=np.float32)
    for row_mode in (True, False):
        selected = np.flatnonzero(use_row == row_mode)
        if row_mode:
            fixed = np.clip(
                np.rint(query_y[selected]).astype(np.intp),
                0, source.shape[0] - 1,
            )
            positions = query_x[selected]
        else:
            fixed = np.clip(
                np.rint(query_x[selected]).astype(np.intp),
                0, source.shape[1] - 1,
            )
            positions = query_y[selected]
        for coordinate in np.unique(fixed):
            local = np.flatnonzero(fixed == coordinate)
            index = selected[local]
            profile = (
                source[coordinate]
                if row_mode else source[:, coordinate]
            )
            output[index] = conv_evaluate_profile(
                profile, positions[local].astype(np.float32)
            )
    return output[:, 0] if scalar else output


def oriented_chord_conv_synthesis(
    values: Array,
    target_shape: tuple[int, int],
    *,
    jet_tensor: bool = True,
    jet_chord: bool = True,
    continuous_geometry: bool = False,
    native: bool = True,
) -> Array:
    """Bounded source-support transport compiled from the CONV two-jet.

    This is the candidate fast core.  Each target has one measured tangent,
    two exits from a declared source-support rectangle, two admitted 1-D
    profile values, and one positive chord average.  Coherence is the exact
    directional mass and mixes that chord with Cartesian CONV.
    """

    source = np.asarray(values, dtype=np.float32)
    scalar = source.ndim == 2
    if scalar:
        source_field = source[..., None]
    else:
        source_field = source
    if source_field.ndim != 3 or min(source_field.shape[:2]) < 5:
        raise ValueError("oriented CONV requires an HxW or HxWxC raster")
    target = tuple(map(int, target_shape))
    baseline = symmetric_conv_synthesis(source, target)
    if native and jet_tensor and not jet_chord and not continuous_geometry:
        return conv_oriented_chord_blend(source, baseline, target)
    mass_cell, tx_cell, ty_cell = _cell_transport_geometry(
        source_field, jet_support=jet_tensor
    )

    target_y = np.linspace(0.0, source_field.shape[0] - 1.0, target[0])
    target_x = np.linspace(0.0, source_field.shape[1] - 1.0, target[1])
    gy, gx = np.meshgrid(target_y, target_x, indexing="ij")
    iy = np.minimum(np.floor(gy).astype(np.intp), source_field.shape[0] - 2)
    ix = np.minimum(np.floor(gx).astype(np.intp), source_field.shape[1] - 2)
    if continuous_geometry:
        mass, tx, ty = _transported_cell_geometry(
            mass_cell, tx_cell, ty_cell, gx, gy
        )
    else:
        tx = tx_cell[iy, ix]
        ty = ty_cell[iy, ix]
        mass = mass_cell[iy, ix]
    if jet_chord:
        left = np.maximum(0, ix - 2).astype(np.float64)
        right = np.minimum(source_field.shape[1] - 1, ix + 3).astype(np.float64)
        top = np.maximum(0, iy - 2).astype(np.float64)
        bottom = np.minimum(source_field.shape[0] - 1, iy + 3).astype(np.float64)
    else:
        left, right = ix.astype(np.float64), (ix + 1).astype(np.float64)
        top, bottom = iy.astype(np.float64), (iy + 1).astype(np.float64)

    tolerance = 128.0 * np.finfo(np.float64).eps

    def exit_data(direction: float) -> tuple[Array, Array, Array, Array]:
        dx = direction * tx
        dy = direction * ty
        x_numerator = np.where(dx > tolerance, right - gx, left - gx)
        y_numerator = np.where(dy > tolerance, bottom - gy, top - gy)
        x_distance = np.divide(
            x_numerator, dx,
            out=np.full_like(dx, np.inf), where=np.abs(dx) > tolerance,
        )
        y_distance = np.divide(
            y_numerator, dy,
            out=np.full_like(dy, np.inf), where=np.abs(dy) > tolerance,
        )
        horizontal = y_distance <= x_distance
        distance = np.minimum(x_distance, y_distance)
        return (
            gx + distance * dx,
            gy + distance * dy,
            horizontal,
            distance,
        )

    backward_x, backward_y, backward_horizontal, backward_length = exit_data(-1.0)
    forward_x, forward_y, forward_horizontal, forward_length = exit_data(1.0)
    backward = _sample_profile_boundaries(
        source, backward_x, backward_y, backward_horizontal
    )
    forward = _sample_profile_boundaries(
        source, forward_x, forward_y, forward_horizontal
    )
    denominator = backward_length + forward_length
    degenerate = denominator <= tolerance
    denominator = np.where(degenerate, 1.0, denominator)
    nested = (
        (np.abs(gx - np.rint(gx)) <= tolerance)
        & (np.abs(gy - np.rint(gy)) <= tolerance)
    )
    mass = np.where(degenerate | nested, 0.0, mass)
    if scalar:
        chord = (
            forward_length * backward.reshape(target)
            + backward_length * forward.reshape(target)
        ) / denominator
        result = (1.0 - mass) * baseline + mass * chord
    else:
        backward = backward.reshape(target + (source_field.shape[2],))
        forward = forward.reshape(target + (source_field.shape[2],))
        chord = (
            forward_length[..., None] * backward
            + backward_length[..., None] * forward
        ) / denominator[..., None]
        result = (1.0 - mass[..., None]) * baseline + mass[..., None] * chord
    return np.asarray(result, dtype=np.float32)


def _trace_continuous_characteristic(
    mass_cell: Array,
    tangent_x_cell: Array,
    tangent_y_cell: Array,
    x: Array,
    y: Array,
    left: Array,
    right: Array,
    top: Array,
    bottom: Array,
    direction: float,
    *,
    step: float = 0.125,
) -> tuple[Array, Array, Array, Array, Array]:
    """Trace the continuous unoriented tensor line to a support boundary."""

    px = np.asarray(x, dtype=np.float64).copy()
    py = np.asarray(y, dtype=np.float64).copy()
    _, initial_tx, initial_ty = _transported_cell_geometry(
        mass_cell, tangent_x_cell, tangent_y_cell, px, py
    )
    dx = direction * initial_tx
    dy = direction * initial_ty
    length = np.zeros_like(px)
    mass_integral = np.zeros_like(px)
    active = np.ones_like(px, dtype=bool)
    epsilon = 1.0e-12
    max_steps = 96

    for _ in range(max_steps):
        if not np.any(active):
            break
        mass, tx, ty = _transported_cell_geometry(
            mass_cell, tangent_x_cell, tangent_y_cell, px, py
        )
        reverse = tx * dx + ty * dy < 0.0
        tx = np.where(reverse, -tx, tx)
        ty = np.where(reverse, -ty, ty)
        dx, dy = tx, ty
        x_distance = np.where(
            dx > epsilon,
            (right - px) / dx,
            np.where(dx < -epsilon, (left - px) / dx, np.inf),
        )
        y_distance = np.where(
            dy > epsilon,
            (bottom - py) / dy,
            np.where(dy < -epsilon, (top - py) / dy, np.inf),
        )
        boundary_distance = np.maximum(np.minimum(x_distance, y_distance), 0.0)
        ds = np.minimum(step, boundary_distance)
        ds = np.where(active, ds, 0.0)
        px += ds * dx
        py += ds * dy
        length += ds
        mass_integral += ds * mass
        reached = active & (boundary_distance <= step + epsilon)
        active &= ~reached
    failed = active.copy()

    # Snap the numerically reached exit to its nearest exact support face.
    face_distance = np.stack((np.abs(px - left), np.abs(px - right),
                              np.abs(py - top), np.abs(py - bottom)), axis=0)
    face = np.argmin(face_distance, axis=0)
    px = np.where(face == 0, left, np.where(face == 1, right, px))
    py = np.where(face == 2, top, np.where(face == 3, bottom, py))
    horizontal = face >= 2
    average_mass = np.divide(
        mass_integral, length, out=np.zeros_like(length), where=length > epsilon
    )
    # A negative sentinel records a closed or trapped line: it carries no
    # two-exit chord measure on this compact support.
    average_mass = np.where(failed, -1.0, average_mass)
    return px, py, horizontal, length, average_mass


def characteristic_transport_conv_synthesis(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Transport on compact integral curves of the continuous tensor field."""

    source = np.asarray(values, dtype=np.float32)
    scalar = source.ndim == 2
    source_field = source[..., None] if scalar else source
    if source_field.ndim != 3 or min(source_field.shape[:2]) < 5:
        raise ValueError("characteristic CONV requires an HxW or HxWxC raster")
    target = tuple(map(int, target_shape))
    baseline = symmetric_conv_synthesis(source, target)
    mass_cell, tx_cell, ty_cell = _cell_transport_geometry(
        source_field, jet_support=True
    )
    target_y = np.linspace(0.0, source_field.shape[0] - 1.0, target[0])
    target_x = np.linspace(0.0, source_field.shape[1] - 1.0, target[1])
    gy, gx = np.meshgrid(target_y, target_x, indexing="ij")
    iy = np.minimum(np.floor(gy).astype(np.intp), source_field.shape[0] - 2)
    ix = np.minimum(np.floor(gx).astype(np.intp), source_field.shape[1] - 2)
    left = np.maximum(0, ix - 2).astype(np.float64)
    right = np.minimum(source_field.shape[1] - 1, ix + 3).astype(np.float64)
    top = np.maximum(0, iy - 2).astype(np.float64)
    bottom = np.minimum(source_field.shape[0] - 1, iy + 3).astype(np.float64)

    backward = _trace_continuous_characteristic(
        mass_cell, tx_cell, ty_cell, gx, gy, left, right, top, bottom, -1.0
    )
    forward = _trace_continuous_characteristic(
        mass_cell, tx_cell, ty_cell, gx, gy, left, right, top, bottom, 1.0
    )
    bx, by, bhorizontal, backward_length, backward_mass = backward
    fx, fy, fhorizontal, forward_length, forward_mass = forward
    backward_value = _sample_profile_boundaries(source, bx, by, bhorizontal)
    forward_value = _sample_profile_boundaries(source, fx, fy, fhorizontal)
    denominator = backward_length + forward_length
    tolerance = 128.0 * np.finfo(np.float64).eps
    denominator = np.where(denominator > tolerance, denominator, 1.0)
    valid_connection = (backward_mass >= 0.0) & (forward_mass >= 0.0)
    transported_mass = (
        backward_length * np.maximum(backward_mass, 0.0)
        + forward_length * np.maximum(forward_mass, 0.0)
    ) / denominator
    nested = (
        (np.abs(gx - np.rint(gx)) <= tolerance)
        & (np.abs(gy - np.rint(gy)) <= tolerance)
    )
    transported_mass = np.where(
        nested | ~valid_connection,
        0.0,
        np.clip(transported_mass, 0.0, 1.0),
    )
    if scalar:
        chord = (
            forward_length * backward_value.reshape(target)
            + backward_length * forward_value.reshape(target)
        ) / denominator
        result = (1.0 - transported_mass) * baseline + transported_mass * chord
    else:
        backward_value = backward_value.reshape(target + (source_field.shape[2],))
        forward_value = forward_value.reshape(target + (source_field.shape[2],))
        chord = (
            forward_length[..., None] * backward_value
            + backward_length[..., None] * forward_value
        ) / denominator[..., None]
        result = (
            (1.0 - transported_mass[..., None]) * baseline
            + transported_mass[..., None] * chord
        )
    return np.asarray(result, dtype=np.float32)


def convstar_bounded_resize(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Current bounded-work CONV* candidate for the interactive demo."""

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5 or min(target) < 5:
        raise ValueError("CONV* requires at least five sites per axis")
    output = field
    if target[0] < output.shape[0]:
        output = _basin_average_axis(output, target[0], 0)
    if target[1] < output.shape[1]:
        output = _basin_average_axis(output, target[1], 1)
    if output.shape[:2] == target:
        return np.asarray(output, dtype=np.float32)
    return oriented_chord_conv_synthesis(
        output, target, jet_tensor=True, jet_chord=False
    )


def convstar_tangent_transport_resize(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Experimental continuous-tensor transport for the bounded CONV core."""

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5 or min(target) < 5:
        raise ValueError("CONV* requires at least five sites per axis")
    output = field
    if target[0] < output.shape[0]:
        output = _basin_average_axis(output, target[0], 0)
    if target[1] < output.shape[1]:
        output = _basin_average_axis(output, target[1], 1)
    if output.shape[:2] == target:
        return np.asarray(output, dtype=np.float32)
    return oriented_chord_conv_synthesis(
        output,
        target,
        jet_tensor=True,
        jet_chord=False,
        continuous_geometry=True,
        native=False,
    )


def convstar_characteristic_transport_resize(
    values: Array,
    target_shape: tuple[int, int],
) -> Array:
    """Experimental compact curved-characteristic form."""

    field = np.asarray(values, dtype=np.float32)
    target = tuple(map(int, target_shape))
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 5 or min(target) < 5:
        raise ValueError("CONV* requires at least five sites per axis")
    output = field
    if target[0] < output.shape[0]:
        output = _basin_average_axis(output, target[0], 0)
    if target[1] < output.shape[1]:
        output = _basin_average_axis(output, target[1], 1)
    if output.shape[:2] == target:
        return np.asarray(output, dtype=np.float32)
    return characteristic_transport_conv_synthesis(output, target)
