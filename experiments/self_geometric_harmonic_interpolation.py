"""Range-preserving cosine-Hermite characteristic interpolation.

The active operator accepts only samples and an integer nested-grid scale.
It measures a determinant-one Riemannian metric from the source lattice,
forms a DCT-I pseudospectral derivative proposal on every source line,
retracts each proposal into the shared monotone cubic-Hermite admissible set, and
transports those profiles on the measured characteristics.  A positive
harmonic fill supplies the isotropic and unavailable-characteristic limit.
No metric, orientation, spectral cutoff, phase parameter, owner field, or
test truth is accepted as input.

The module retains the component constructions and rejected alternatives as
named ablations.  ``cosine_hermite_characteristic_resize`` is the integrated
operator used by the benchmark and interactive comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.fft import dct, dctn, dst


Array = np.ndarray


def _as_field(samples: Array) -> tuple[Array, bool]:
    value = np.asarray(samples, dtype=np.float64)
    scalar = value.ndim == 2
    if scalar:
        value = value[..., None]
    if value.ndim != 3 or min(value.shape[:2]) < 2:
        raise ValueError("samples must have shape HxW or HxWxC with H,W >= 2")
    if not np.all(np.isfinite(value)):
        raise ValueError("samples must be finite")
    return np.ascontiguousarray(value), scalar


def _binomial3(value: Array) -> Array:
    kernel = np.array((0.25, 0.5, 0.25), dtype=np.float64)
    result = convolve1d(value, kernel, axis=0, mode="reflect")
    return convolve1d(result, kernel, axis=1, mode="reflect")


def _gradient(value: Array) -> tuple[Array, Array]:
    gy = np.gradient(value, axis=0, edge_order=1)
    gx = np.gradient(value, axis=1, edge_order=1)
    return gx, gy


def internally_measured_metric(samples: Array, *, persistent: bool = True) -> Array:
    """Return one determinant-one SPD metric at every source sample.

    A one-lattice-step symmetric binomial observation is fixed by the source
    lattice.  Cross-scale agreement suppresses a direction that reverses under
    that observation.  The tensor is normalized by its local trace plus the
    domain mean trace, so it is invariant to a common multiplication of all
    value channels and tends continuously to the identity as evidence becomes
    weak relative to the image as a whole.

        J = A_h sum_c a grad(f_c) grad(f_c)^T
        B = I + J / (tr(J) + mean(tr(J)))
        M = B / sqrt(det(B)).

    The unit coefficient is the base lattice precision; there is no
    anisotropy gain or coherence threshold.
    """

    field, _ = _as_field(samples)
    gx, gy = _gradient(field)
    if persistent:
        smooth = _binomial3(field)
        sx, sy = _gradient(smooth)
        numerator = np.sum(gx * sx + gy * sy, axis=2)
        raw_energy = np.sum(gx * gx + gy * gy, axis=2)
        smooth_energy = np.sum(sx * sx + sy * sy, axis=2)
        denominator = np.sqrt(raw_energy * smooth_energy)
        agreement = np.divide(
            np.maximum(numerator, 0.0),
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        agreement = np.clip(agreement, 0.0, 1.0)
    else:
        agreement = np.ones(field.shape[:2], dtype=np.float64)

    jxx = _binomial3(agreement * np.sum(gx * gx, axis=2))
    jxy = _binomial3(agreement * np.sum(gx * gy, axis=2))
    jyy = _binomial3(agreement * np.sum(gy * gy, axis=2))
    trace = np.maximum(jxx + jyy, 0.0)
    mean_trace = float(np.mean(trace))
    if mean_trace == 0.0:
        identity = np.eye(2, dtype=np.float64)
        return np.broadcast_to(identity, field.shape[:2] + (2, 2)).copy()

    normalizer = trace + mean_trace
    bxx = 1.0 + jxx / normalizer
    bxy = jxy / normalizer
    byy = 1.0 + jyy / normalizer
    determinant = bxx * byy - bxy * bxy
    if float(np.min(determinant)) <= 0.0:
        raise RuntimeError("internally measured tensor was not SPD")
    root = np.sqrt(determinant)
    metric = np.empty(field.shape[:2] + (2, 2), dtype=np.float64)
    metric[..., 0, 0] = bxx / root
    metric[..., 0, 1] = bxy / root
    metric[..., 1, 0] = bxy / root
    metric[..., 1, 1] = byy / root
    return metric


def _measured_normal_tensor(samples: Array, *, persistent: bool = True) -> Array:
    """Return the uncollapsed positive tensor measured by the metric stage."""

    field, _ = _as_field(samples)
    gx, gy = _gradient(field)
    if persistent:
        smooth = _binomial3(field)
        sx, sy = _gradient(smooth)
        numerator = np.sum(gx * sx + gy * sy, axis=2)
        denominator = np.sqrt(
            np.sum(gx * gx + gy * gy, axis=2)
            * np.sum(sx * sx + sy * sy, axis=2)
        )
        agreement = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        agreement = np.clip(agreement, 0.0, 1.0)
    else:
        agreement = np.ones(field.shape[:2], dtype=np.float64)
    tensor = np.empty(field.shape[:2] + (2, 2), dtype=np.float64)
    tensor[..., 0, 0] = _binomial3(agreement * np.sum(gx * gx, axis=2))
    tensor[..., 0, 1] = _binomial3(agreement * np.sum(gx * gy, axis=2))
    tensor[..., 1, 0] = tensor[..., 0, 1]
    tensor[..., 1, 1] = _binomial3(agreement * np.sum(gy * gy, axis=2))
    return tensor


def _directional_coefficients(diffusion: Array) -> tuple[float, float, float, int]:
    """Exact nonnegative 8-neighbour decomposition for condition <= 2."""

    a = float(diffusion[0, 0])
    b = float(diffusion[0, 1])
    c = float(diffusion[1, 1])
    diagonal = abs(b)
    horizontal = a - diagonal
    vertical = c - diagonal
    tolerance = 256.0 * np.finfo(float).eps * max(a, c, diagonal, 1.0)
    if horizontal < -tolerance or vertical < -tolerance:
        raise RuntimeError("metric left the monotone radius-one tensor cone")
    return max(horizontal, 0.0), max(vertical, 0.0), diagonal, (1 if b >= 0.0 else -1)


def _cell_boundary(corners: Array, scale: int) -> Array:
    result = np.zeros((scale + 1, scale + 1, corners.shape[1]), dtype=np.float64)
    coordinate = np.arange(scale + 1, dtype=np.float64) / scale
    for index, t in enumerate(coordinate):
        result[0, index] = (1.0 - t) * corners[0] + t * corners[1]
        result[scale, index] = (1.0 - t) * corners[3] + t * corners[2]
        result[index, 0] = (1.0 - t) * corners[0] + t * corners[3]
        result[index, scale] = (1.0 - t) * corners[1] + t * corners[2]
    return result


def _harmonic_cell(corners: Array, metric: Array, scale: int) -> Array:
    values = _cell_boundary(corners, scale)
    return _harmonic_cell_from_boundary(values, metric)


def _harmonic_cell_from_boundary(values: Array, metric: Array) -> Array:
    """Fill one cell from already transported, range-preserving traces."""

    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("cell boundary must have shape (scale+1)^2xC")
    scale = values.shape[0] - 1
    if scale == 1:
        return values
    diffusion = np.linalg.inv(metric)
    horizontal, vertical, diagonal, diagonal_sign = _directional_coefficients(diffusion)
    directions = (
        (1, 0, horizontal),
        (0, 1, vertical),
        (1, diagonal_sign, diagonal),
    )
    interior = [(x, y) for y in range(1, scale) for x in range(1, scale)]
    row_of = {point: row for row, point in enumerate(interior)}
    matrix = np.zeros((len(interior), len(interior)), dtype=np.float64)
    right = np.zeros((len(interior), values.shape[2]), dtype=np.float64)
    center = 2.0 * sum(coefficient for _, _, coefficient in directions)
    for point, row in row_of.items():
        matrix[row, row] = center
        x, y = point
        for dx, dy, coefficient in directions:
            if coefficient == 0.0:
                continue
            for sign in (-1, 1):
                neighbor = (x + sign * dx, y + sign * dy)
                if neighbor in row_of:
                    matrix[row, row_of[neighbor]] -= coefficient
                else:
                    nx, ny = neighbor
                    if not (0 <= nx <= scale and 0 <= ny <= scale):
                        raise RuntimeError("radius-one stencil escaped its source cell")
                    right[row] += coefficient * values[ny, nx]
    solution = np.linalg.solve(matrix, right)
    for point, row in row_of.items():
        values[point[1], point[0]] = solution[row]
    return values


def harmonic_resize(
    samples: Array,
    scale: int,
    *,
    adaptive: bool = True,
    persistent: bool = True,
) -> Array:
    """Refine a nested Cartesian lattice by positive harmonic cells."""

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    if adaptive:
        metrics = internally_measured_metric(field, persistent=persistent)
    else:
        metrics = np.broadcast_to(
            np.eye(2, dtype=np.float64), (height, width, 2, 2)
        )
    output = np.empty(
        ((height - 1) * scale + 1, (width - 1) * scale + 1, channels),
        dtype=np.float64,
    )
    for iy in range(height - 1):
        for ix in range(width - 1):
            corners = np.stack((
                field[iy, ix],
                field[iy, ix + 1],
                field[iy + 1, ix + 1],
                field[iy + 1, ix],
            ))
            metric = np.mean(
                metrics[iy : iy + 2, ix : ix + 2], axis=(0, 1)
            )
            metric /= np.sqrt(np.linalg.det(metric))
            cell = _harmonic_cell(corners, metric, scale)
            y0, x0 = iy * scale, ix * scale
            output[y0 : y0 + scale + 1, x0 : x0 + scale + 1] = cell
    return output[..., 0] if scalar else output


def transported_harmonic_resize(
    samples: Array,
    scale: int,
    *,
    adaptive: bool = True,
    persistent: bool = True,
) -> Array:
    """Move monotone source-line profiles, then harmonically fill cells.

    PCHIP transports one-dimensional value jets along every observed source
    row and column.  It is shape preserving on each source interval: the
    transported edge trace remains between its two endpoint samples.  The
    Riemannian harmonic solve then fills the cell from those four traces.
    Thus the construction can change profile shape without signed value
    weights, while retaining the cellwise maximum principle.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    source_x = np.arange(width, dtype=np.float64)
    source_y = np.arange(height, dtype=np.float64)
    target_x = np.arange(out_width, dtype=np.float64) / scale
    target_y = np.arange(out_height, dtype=np.float64) / scale
    traces = np.full((out_height, out_width, channels), np.nan, dtype=np.float64)

    for iy in range(height):
        traces[iy * scale] = PchipInterpolator(
            source_x, field[iy], axis=0
        )(target_x)
    for ix in range(width):
        traces[:, ix * scale] = PchipInterpolator(
            source_y, field[:, ix], axis=0
        )(target_y)

    if adaptive:
        metrics = internally_measured_metric(field, persistent=persistent)
    else:
        metrics = np.broadcast_to(
            np.eye(2, dtype=np.float64), (height, width, 2, 2)
        )
    for iy in range(height - 1):
        for ix in range(width - 1):
            y0, x0 = iy * scale, ix * scale
            boundary = np.zeros((scale + 1, scale + 1, channels), dtype=np.float64)
            boundary[0] = traces[y0, x0 : x0 + scale + 1]
            boundary[scale] = traces[y0 + scale, x0 : x0 + scale + 1]
            boundary[:, 0] = traces[y0 : y0 + scale + 1, x0]
            boundary[:, scale] = traces[y0 : y0 + scale + 1, x0 + scale]
            metric = np.mean(
                metrics[iy : iy + 2, ix : ix + 2], axis=(0, 1)
            )
            metric /= np.sqrt(np.linalg.det(metric))
            cell = _harmonic_cell_from_boundary(boundary, metric)
            traces[y0 : y0 + scale + 1, x0 : x0 + scale + 1] = cell
    if np.any(~np.isfinite(traces)):
        raise RuntimeError("transported harmonic grid was not completely filled")
    return traces[..., 0] if scalar else traces


def variation_lineage_harmonic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Fill Riemannian cells from variation-diminishing boundary currents."""

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    target_x = np.arange(out_width, dtype=np.float64) / scale
    target_y = np.arange(out_height, dtype=np.float64) / scale
    traces = np.full((out_height, out_width, channels), np.nan, dtype=np.float64)
    row_profiles = [_variation_lineage_profile(field[row])
                    for row in range(height)]
    column_profiles = [_variation_lineage_profile(field[:, column])
                       for column in range(width)]
    for row, profile in enumerate(row_profiles):
        traces[row * scale] = np.stack([
            _evaluate_variation_lineage_profile(profile, x) for x in target_x
        ])
    for column, profile in enumerate(column_profiles):
        traces[:, column * scale] = np.stack([
            _evaluate_variation_lineage_profile(profile, y) for y in target_y
        ])

    metrics = internally_measured_metric(field, persistent=persistent)
    for iy in range(height - 1):
        for ix in range(width - 1):
            y0, x0 = iy * scale, ix * scale
            boundary = np.zeros((scale + 1, scale + 1, channels), dtype=np.float64)
            boundary[0] = traces[y0, x0 : x0 + scale + 1]
            boundary[scale] = traces[y0 + scale, x0 : x0 + scale + 1]
            boundary[:, 0] = traces[y0 : y0 + scale + 1, x0]
            boundary[:, scale] = traces[y0 : y0 + scale + 1, x0 + scale]
            metric = np.mean(metrics[iy : iy + 2, ix : ix + 2], axis=(0, 1))
            metric /= np.sqrt(np.linalg.det(metric))
            traces[y0 : y0 + scale + 1, x0 : x0 + scale + 1] = (
                _harmonic_cell_from_boundary(boundary, metric)
            )
    if np.any(~np.isfinite(traces)):
        raise RuntimeError("variation-lineage grid was not completely filled")
    return traces[..., 0] if scalar else traces


def _cell_directional_chord(
    row_profiles: list[object],
    column_profiles: list[object],
    iy: int,
    ix: int,
    u: float,
    v: float,
    tangent: Array,
    evaluator: object | None = None,
) -> Array:
    """Evaluate the positive two-exit measure of one cell chord."""

    if evaluator is None:
        evaluator = _evaluate_variation_lineage_profile
    tx, ty = map(float, tangent)
    tolerance = 128.0 * np.finfo(float).eps

    def exit_distance(direction: float) -> float:
        candidates: list[float] = []
        dx, dy = direction * tx, direction * ty
        if dx > tolerance:
            candidates.append((1.0 - u) / dx)
        elif dx < -tolerance:
            candidates.append(-u / dx)
        if dy > tolerance:
            candidates.append((1.0 - v) / dy)
        elif dy < -tolerance:
            candidates.append(-v / dy)
        positive = [distance for distance in candidates if distance >= 0.0]
        if not positive:
            raise RuntimeError("nonzero tangent did not meet the cell boundary")
        return min(positive)

    backward = exit_distance(-1.0)
    forward = exit_distance(1.0)

    def boundary_value(x: float, y: float) -> Array:
        if abs(y) <= tolerance:
            return evaluator(
                row_profiles[iy], ix + min(max(x, 0.0), 1.0)
            )
        if abs(y - 1.0) <= tolerance:
            return evaluator(
                row_profiles[iy + 1], ix + min(max(x, 0.0), 1.0)
            )
        if abs(x) <= tolerance:
            return evaluator(
                column_profiles[ix], iy + min(max(y, 0.0), 1.0)
            )
        if abs(x - 1.0) <= tolerance:
            return evaluator(
                column_profiles[ix + 1], iy + min(max(y, 0.0), 1.0)
            )
        raise RuntimeError("computed chord exit is not on the cell boundary")

    left = boundary_value(u - backward * tx, v - backward * ty)
    right = boundary_value(u + forward * tx, v + forward * ty)
    return (forward * left + backward * right) / (backward + forward)


def directional_variation_measure_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Refine through the harmonic measure of the measured tangent generator.

    If the eigenvalues of the measured normal tensor are
    ``lambda_min <= lambda_max`` and ``t`` is the tangent orthogonal to its
    principal normal, the induced diffusion generator has the exact split

        D = lambda_min I + (lambda_max - lambda_min) t t^T.

    Its trace therefore assigns masses ``2*lambda_min`` and
    ``lambda_max-lambda_min`` to isotropic harmonic exit and rank-one chordal
    exit, respectively.  Both are positive boundary measures of one
    generator.  The construction accepts no anisotropy gain, threshold, or
    externally supplied direction.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    row_profiles = [_variation_lineage_profile(field[row])
                    for row in range(height)]
    column_profiles = [_variation_lineage_profile(field[:, column])
                       for column in range(width)]
    normal_tensor = _measured_normal_tensor(field, persistent=persistent)
    output = np.empty((out_height, out_width, channels), dtype=np.float64)

    for iy in range(height - 1):
        for ix in range(width - 1):
            y0, x0 = iy * scale, ix * scale
            boundary = np.zeros((scale + 1, scale + 1, channels), dtype=np.float64)
            for sx in range(scale + 1):
                u = sx / scale
                boundary[0, sx] = _evaluate_variation_lineage_profile(
                    row_profiles[iy], ix + u
                )
                boundary[scale, sx] = _evaluate_variation_lineage_profile(
                    row_profiles[iy + 1], ix + u
                )
            for sy in range(scale + 1):
                v = sy / scale
                boundary[sy, 0] = _evaluate_variation_lineage_profile(
                    column_profiles[ix], iy + v
                )
                boundary[sy, scale] = _evaluate_variation_lineage_profile(
                    column_profiles[ix + 1], iy + v
                )
            isotropic = _harmonic_cell_from_boundary(boundary, np.eye(2))
            tensor = np.mean(
                normal_tensor[iy : iy + 2, ix : ix + 2], axis=(0, 1)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
            total_mass = float(eigenvalues[0] + eigenvalues[1])
            if total_mass <= 0.0:
                cell = isotropic
            else:
                isotropic_mass = 2.0 * max(float(eigenvalues[0]), 0.0)
                directional_mass = max(
                    float(eigenvalues[1] - eigenvalues[0]), 0.0
                )
                denominator = isotropic_mass + directional_mass
                tangent = np.array(
                    (-eigenvectors[1, 1], eigenvectors[0, 1]),
                    dtype=np.float64,
                )
                cell = isotropic.copy()
                if directional_mass > 0.0:
                    for sy in range(1, scale):
                        for sx in range(1, scale):
                            chord = _cell_directional_chord(
                                row_profiles,
                                column_profiles,
                                iy,
                                ix,
                                sx / scale,
                                sy / scale,
                                tangent,
                            )
                            cell[sy, sx] = (
                                isotropic_mass * isotropic[sy, sx]
                                + directional_mass * chord
                            ) / denominator
            output[y0 : y0 + scale + 1, x0 : x0 + scale + 1] = cell
    return output[..., 0] if scalar else output


def atomic_variation_measure_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Retain the measured tangent atoms instead of their second moment."""

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    row_profiles = [_variation_lineage_profile(field[row])
                    for row in range(height)]
    column_profiles = [_variation_lineage_profile(field[:, column])
                       for column in range(width)]
    gx, gy = _gradient(field)
    if persistent:
        smooth = _binomial3(field)
        sx, sy = _gradient(smooth)
        numerator = gx * sx + gy * sy
        denominator = np.sqrt((gx * gx + gy * gy) * (sx * sx + sy * sy))
        agreement = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        agreement = np.clip(agreement, 0.0, 1.0)
    else:
        agreement = np.ones_like(gx)
    output = np.empty((out_height, out_width, channels), dtype=np.float64)

    for iy in range(height - 1):
        for ix in range(width - 1):
            y0, x0 = iy * scale, ix * scale
            boundary = np.zeros((scale + 1, scale + 1, channels), dtype=np.float64)
            for sx_index in range(scale + 1):
                u = sx_index / scale
                boundary[0, sx_index] = _evaluate_variation_lineage_profile(
                    row_profiles[iy], ix + u
                )
                boundary[scale, sx_index] = _evaluate_variation_lineage_profile(
                    row_profiles[iy + 1], ix + u
                )
            for sy_index in range(scale + 1):
                v = sy_index / scale
                boundary[sy_index, 0] = _evaluate_variation_lineage_profile(
                    column_profiles[ix], iy + v
                )
                boundary[sy_index, scale] = _evaluate_variation_lineage_profile(
                    column_profiles[ix + 1], iy + v
                )
            cell = _harmonic_cell_from_boundary(boundary, np.eye(2))
            atoms: list[tuple[float, Array]] = []
            for source_y in (iy, iy + 1):
                for source_x in (ix, ix + 1):
                    for channel in range(channels):
                        normal = np.array(
                            (gx[source_y, source_x, channel],
                             gy[source_y, source_x, channel]),
                            dtype=np.float64,
                        )
                        mass = float(
                            agreement[source_y, source_x, channel]
                            * np.dot(normal, normal)
                        )
                        if mass > 0.0:
                            tangent = np.array((-normal[1], normal[0]))
                            tangent /= np.linalg.norm(tangent)
                            atoms.append((mass, tangent))
            total_mass = sum(mass for mass, _ in atoms)
            if total_mass > 0.0:
                for sy_index in range(1, scale):
                    for sx_index in range(1, scale):
                        value = np.zeros(channels, dtype=np.float64)
                        for mass, tangent in atoms:
                            value += mass * _cell_directional_chord(
                                row_profiles,
                                column_profiles,
                                iy,
                                ix,
                                sx_index / scale,
                                sy_index / scale,
                                tangent,
                            )
                        cell[sy_index, sx_index] = value / total_mass
            output[y0 : y0 + scale + 1, x0 : x0 + scale + 1] = cell
    return output[..., 0] if scalar else output


def characteristic_transport_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Align source-line profiles along the measured metric tangent.

    The smaller-eigenvalue eigenvector of the internally measured metric is
    the local isophote tangent.  Through every target point, that tangent is
    traced in both Cartesian charts: to the two vertical source lines and to
    the two horizontal source lines surrounding the cell.  Shape-preserving
    source-line profiles supply the endpoint values.  The two transported
    estimates are combined by the squared tangent components, which are the
    diagonal entries of the rank-one tangent projector.  This is a
    semi-Lagrangian value transport, not a signed reconstruction kernel.

    Isotropic regions continuously return to the transported harmonic field.
    The blend weight is kappa(M)-1.  It lies in [0,1] because the fixed metric
    construction has 1 <= kappa(M) <= 2; no coherence threshold or fitted
    gain is introduced.  Characteristics that meet the image boundary before
    meeting both required source lines use the harmonic field, which keeps
    the boundary condition exact.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    source_x = np.arange(width, dtype=np.float64)
    source_y = np.arange(height, dtype=np.float64)
    row_profiles = [
        PchipInterpolator(source_x, field[iy], axis=0)
        for iy in range(height)
    ]
    column_profiles = [
        PchipInterpolator(source_y, field[:, ix], axis=0)
        for ix in range(width)
    ]
    baseline = transported_harmonic_resize(
        field, scale, adaptive=True, persistent=persistent
    )
    metrics = internally_measured_metric(field, persistent=persistent)
    output = np.asarray(baseline, dtype=np.float64).copy()

    tolerance = 128.0 * np.finfo(float).eps
    for iy in range(height - 1):
        for ix in range(width - 1):
            metric = np.mean(metrics[iy : iy + 2, ix : ix + 2], axis=(0, 1))
            eigenvalues, eigenvectors = np.linalg.eigh(metric)
            tangent = eigenvectors[:, 0]
            condition = float(eigenvalues[1] / eigenvalues[0])
            transport_weight = min(max(condition - 1.0, 0.0), 1.0)
            if transport_weight <= tolerance:
                continue
            tx, ty = float(tangent[0]), float(tangent[1])
            for sy in range(scale + 1):
                v = sy / scale
                gy = iy + v
                oy = iy * scale + sy
                for sx in range(scale + 1):
                    u = sx / scale
                    gx = ix + u
                    ox = ix * scale + sx
                    transported_sum = np.zeros(channels, dtype=np.float64)
                    transported_mass = 0.0
                    if abs(tx) > tolerance:
                        slope = ty / tx
                        left_y = gy - u * slope
                        right_y = gy + (1.0 - u) * slope
                        if (
                            0.0 <= left_y <= height - 1.0
                            and 0.0 <= right_y <= height - 1.0
                        ):
                            left = column_profiles[ix](left_y)
                            right = column_profiles[ix + 1](right_y)
                            horizontal_chart = (1.0 - u) * left + u * right
                            chart_mass = tx * tx
                            transported_sum += chart_mass * horizontal_chart
                            transported_mass += chart_mass
                    if abs(ty) > tolerance:
                        slope = tx / ty
                        top_x = gx - v * slope
                        bottom_x = gx + (1.0 - v) * slope
                        if (
                            0.0 <= top_x <= width - 1.0
                            and 0.0 <= bottom_x <= width - 1.0
                        ):
                            top = row_profiles[iy](top_x)
                            bottom = row_profiles[iy + 1](bottom_x)
                            vertical_chart = (1.0 - v) * top + v * bottom
                            chart_mass = ty * ty
                            transported_sum += chart_mass * vertical_chart
                            transported_mass += chart_mass
                    if transported_mass > 0.0:
                        admitted_weight = transport_weight * transported_mass
                        output[oy, ox] = (
                            (1.0 - admitted_weight) * output[oy, ox]
                            + transport_weight * transported_sum
                        )
    return output[..., 0] if scalar else output


def directional_measure_transport_resize(
    samples: Array,
    scale: int,
) -> Array:
    """Transport through the full observed tangent measure in each cell.

    An SPD structure tensor is the second moment of a directional measure and
    consequently collapses simultaneous directions.  This lift retains every
    channel gradient at the four cell vertices as an atom on the unoriented
    tangent circle.  Every atom generates both Cartesian characteristic
    charts used by :func:`characteristic_transport_resize`; their convex sum
    therefore represents crossings without choosing a component count.

    Cross-scale gradient agreement supplies atom mass.  The transported share
    is the local persistent energy divided by itself plus the domain-mean
    persistent energy.  Both quantities are observations already used by the
    metric construction.  No orientation threshold, rank decision, or fitted
    gain is present.  All value combinations remain convex.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    source_x = np.arange(width, dtype=np.float64)
    source_y = np.arange(height, dtype=np.float64)
    row_profiles = [
        PchipInterpolator(source_x, field[iy], axis=0)
        for iy in range(height)
    ]
    column_profiles = [
        PchipInterpolator(source_y, field[:, ix], axis=0)
        for ix in range(width)
    ]
    baseline = transported_harmonic_resize(field, scale, adaptive=True)
    output = np.asarray(baseline, dtype=np.float64).copy()

    gx, gy = _gradient(field)
    smooth = _binomial3(field)
    sx, sy = _gradient(smooth)
    numerator = np.sum(gx * sx + gy * sy, axis=2)
    raw_energy = np.sum(gx * gx + gy * gy, axis=2)
    smooth_energy = np.sum(sx * sx + sy * sy, axis=2)
    denominator = np.sqrt(raw_energy * smooth_energy)
    agreement = np.divide(
        np.maximum(numerator, 0.0),
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    atom_energy = agreement[..., None] * (gx * gx + gy * gy)
    mean_energy = float(np.mean(np.sum(atom_energy, axis=2)))
    if mean_energy == 0.0:
        return baseline[..., 0] if scalar else baseline

    tolerance = 128.0 * np.finfo(float).eps
    for iy in range(height - 1):
        for ix in range(width - 1):
            cell_energy = atom_energy[iy : iy + 2, ix : ix + 2]
            total_energy = float(np.sum(cell_energy))
            if total_energy <= tolerance * mean_energy:
                continue
            local_mean_energy = total_energy / 4.0
            transport_weight = local_mean_energy / (
                local_mean_energy + mean_energy
            )
            atoms = []
            for cy in range(2):
                for cx in range(2):
                    for channel in range(channels):
                        mass = float(cell_energy[cy, cx, channel])
                        if mass <= tolerance * total_energy:
                            continue
                        normal_x = float(gx[iy + cy, ix + cx, channel])
                        normal_y = float(gy[iy + cy, ix + cx, channel])
                        norm = float(np.hypot(normal_x, normal_y))
                        if norm <= 0.0:
                            continue
                        atoms.append((
                            mass / total_energy,
                            -normal_y / norm,
                            normal_x / norm,
                        ))
            for sy_index in range(scale + 1):
                v = sy_index / scale
                global_y = iy + v
                oy = iy * scale + sy_index
                for sx_index in range(scale + 1):
                    u = sx_index / scale
                    global_x = ix + u
                    ox = ix * scale + sx_index
                    transported_sum = np.zeros(channels, dtype=np.float64)
                    transported_mass = 0.0
                    for atom_mass, tx, ty in atoms:
                        if abs(tx) > tolerance:
                            slope = ty / tx
                            left_y = global_y - u * slope
                            right_y = global_y + (1.0 - u) * slope
                            if (
                                0.0 <= left_y <= height - 1.0
                                and 0.0 <= right_y <= height - 1.0
                            ):
                                left = column_profiles[ix](left_y)
                                right = column_profiles[ix + 1](right_y)
                                chart_mass = atom_mass * tx * tx
                                transported_sum += chart_mass * (
                                    (1.0 - u) * left + u * right
                                )
                                transported_mass += chart_mass
                        if abs(ty) > tolerance:
                            slope = tx / ty
                            top_x = global_x - v * slope
                            bottom_x = global_x + (1.0 - v) * slope
                            if (
                                0.0 <= top_x <= width - 1.0
                                and 0.0 <= bottom_x <= width - 1.0
                            ):
                                top = row_profiles[iy](top_x)
                                bottom = row_profiles[iy + 1](bottom_x)
                                chart_mass = atom_mass * ty * ty
                                transported_sum += chart_mass * (
                                    (1.0 - v) * top + v * bottom
                                )
                                transported_mass += chart_mass
                    if transported_mass > 0.0:
                        admitted_weight = transport_weight * transported_mass
                        output[oy, ox] = (
                            (1.0 - admitted_weight) * output[oy, ox]
                            + transport_weight * transported_sum
                        )
    return output[..., 0] if scalar else output


def lanczos_resize(samples: Array, scale: int, radius: int = 3) -> Array:
    """Nested-grid separable normalized Lanczos reference."""

    field, scalar = _as_field(samples)

    def matrix(count: int) -> Array:
        coordinates = np.arange((count - 1) * scale + 1, dtype=np.float64) / scale
        result = np.zeros((coordinates.size, count), dtype=np.float64)
        source = np.arange(count, dtype=np.float64)
        for row, coordinate in enumerate(coordinates):
            distance = coordinate - source
            admitted = np.abs(distance) < radius
            weight = np.zeros(count, dtype=np.float64)
            weight[admitted] = (
                np.sinc(distance[admitted])
                * np.sinc(distance[admitted] / radius)
            )
            total = float(np.sum(weight))
            if abs(total) <= 1.0e-14:
                weight[int(np.clip(round(coordinate), 0, count - 1))] = 1.0
                total = 1.0
            result[row] = weight / total
        return result

    wy = matrix(field.shape[0])
    wx = matrix(field.shape[1])
    intermediate = np.einsum("yh,hwc->ywc", wy, field)
    output = np.einsum("xw,ywc->yxc", wx, intermediate)
    return output[..., 0] if scalar else output


def lanczos_resample_to_shape(
    samples: Array,
    output_shape: tuple[int, int],
    radius: int = 3,
) -> Array:
    """Endpoint-aligned Lanczos resampling with widened downscale support."""

    field, scalar = _as_field(samples)
    out_height, out_width = map(int, output_shape)
    if min(out_height, out_width) < 2 or radius < 1:
        raise ValueError("output dimensions must be >= 2 and radius positive")

    def matrix(source_count: int, target_count: int) -> Array:
        coordinate = np.linspace(0.0, source_count - 1.0, target_count)
        source = np.arange(source_count, dtype=np.float64)
        contraction = min(
            1.0, (target_count - 1.0) / (source_count - 1.0)
        )
        result = np.zeros((target_count, source_count), dtype=np.float64)
        for row, point in enumerate(coordinate):
            distance = (point - source) * contraction
            admitted = np.abs(distance) < radius
            weight = np.zeros(source_count, dtype=np.float64)
            weight[admitted] = (
                np.sinc(distance[admitted])
                * np.sinc(distance[admitted] / radius)
            )
            total = float(np.sum(weight))
            if abs(total) <= 1.0e-14:
                weight[int(np.clip(round(point), 0, source_count - 1))] = 1.0
                total = 1.0
            result[row] = weight / total
        return result

    wy = matrix(field.shape[0], out_height)
    wx = matrix(field.shape[1], out_width)
    intermediate = np.einsum("yh,hwc->ywc", wy, field)
    output = np.einsum("xw,ywc->yxc", wx, intermediate)
    return output[..., 0] if scalar else output


def bilinear_resample_to_shape(
    samples: Array,
    output_shape: tuple[int, int],
) -> Array:
    """Endpoint-aligned separable bilinear resampling to an arbitrary shape."""

    field, scalar = _as_field(samples)
    out_height, out_width = map(int, output_shape)
    if min(out_height, out_width) < 2:
        raise ValueError("output dimensions must be >= 2")

    def matrix(source_count: int, target_count: int) -> Array:
        coordinate = np.linspace(0.0, source_count - 1.0, target_count)
        lower = np.floor(coordinate).astype(int)
        upper = np.minimum(lower + 1, source_count - 1)
        fraction = coordinate - lower
        result = np.zeros((target_count, source_count), dtype=np.float64)
        rows = np.arange(target_count)
        result[rows, lower] += 1.0 - fraction
        result[rows, upper] += fraction
        return result

    wy = matrix(field.shape[0], out_height)
    wx = matrix(field.shape[1], out_width)
    intermediate = np.einsum("yh,hwc->ywc", wy, field)
    output = np.einsum("xw,ywc->yxc", wx, intermediate)
    return output[..., 0] if scalar else output


def cosine_phase_resize(samples: Array, scale: int) -> Array:
    """Exact nested-grid continuation in the lattice cosine basis.

    This is the phase-only positive control.  The DCT-I basis is fixed by the
    finite source interval and imposes its natural even reflection at the
    boundary.  Every admitted source-lattice phase is continued exactly; no
    cutoff or window is selected.  The operator is linear and cardinal, but
    it has no transport geometry and therefore retains the usual spectral
    ringing risk at fronts.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, _channels = field.shape

    def evaluation_matrix(count: int) -> Array:
        intervals = count - 1
        target = np.arange(intervals * scale + 1, dtype=np.float64) / scale
        frequency = np.arange(count, dtype=np.float64)
        weights = np.full(count, 2.0, dtype=np.float64)
        weights[[0, -1]] = 1.0
        return (
            np.cos(np.pi * target[:, None] * frequency[None] / intervals)
            * weights[None]
            / (2.0 * intervals)
        )

    coefficients = dctn(field, type=1, axes=(0, 1), norm=None)
    ey = evaluation_matrix(height)
    ex = evaluation_matrix(width)
    output = np.einsum("yk,klc,xl->yxc", ey, coefficients, ex)
    return output[..., 0] if scalar else output


def local_phase_frame_resize(
    samples: Array,
    scale: int,
    frame_size: int,
) -> Array:
    """Continue every coefficient of an exact local phase frame.

    Square-root periodic Hann packets with half-window translation form a
    constant-overlap-add frame.  Inside each packet, every discrete Fourier
    coefficient is continued by its exact phase action

        c_k(x) = c_k(0) exp(i <k, x>).

    Thus the packet atoms are local constant-metric Eikonal solutions.  The
    frame-size argument is deliberately exposed: this is an ablation used to
    determine whether a single packet scale can be principled.  It is not yet
    the closed interpolation operator.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    size = int(frame_size)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if size < 4 or size % 2:
        raise ValueError("frame_size must be an even integer at least four")
    height, width, channels = field.shape
    if size > min(height + size, width + size):
        raise ValueError("frame_size is incompatible with the source")
    hop = size // 2
    margin = size // 2
    base_height = height + 2 * margin
    base_width = width + 2 * margin
    extra_height = (hop - (base_height - size) % hop) % hop
    extra_width = (hop - (base_width - size) % hop) % hop
    padded = np.pad(
        field,
        ((margin, margin + extra_height),
         (margin, margin + extra_width),
         (0, 0)),
        mode="reflect",
    )

    source_position = np.arange(size, dtype=np.float64)
    source_window_1d = np.sqrt(np.maximum(
        0.5 - 0.5 * np.cos(2.0 * np.pi * source_position / size),
        0.0,
    ))
    source_window = source_window_1d[:, None] * source_window_1d[None]
    fine_position = np.arange(size * scale, dtype=np.float64) / scale
    frequency = np.fft.fftfreq(size)
    evaluation = np.exp(
        2j * np.pi * fine_position[:, None] * frequency[None]
    ) / np.sqrt(size)
    window_coefficients = np.fft.fft2(source_window, norm="ortho")
    fine_window = np.einsum(
        "tk,kl,ul->tu",
        evaluation,
        window_coefficients,
        evaluation,
    ).real

    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    output = np.zeros((out_height, out_width, channels), dtype=np.float64)
    normalization = np.zeros((out_height, out_width), dtype=np.float64)
    for padded_y in range(0, padded.shape[0] - size + 1, hop):
        for padded_x in range(0, padded.shape[1] - size + 1, hop):
            patch = padded[
                padded_y : padded_y + size,
                padded_x : padded_x + size,
            ]
            coefficients = np.fft.fft2(
                patch * source_window[..., None],
                axes=(0, 1),
                norm="ortho",
            )
            continued = np.einsum(
                "tk,klc,ul->tuc",
                evaluation,
                coefficients,
                evaluation,
            ).real
            continued *= fine_window[..., None]
            start_y = (padded_y - margin) * scale
            start_x = (padded_x - margin) * scale
            local_y0 = max(0, -start_y)
            local_x0 = max(0, -start_x)
            local_y1 = min(size * scale, out_height - start_y)
            local_x1 = min(size * scale, out_width - start_x)
            if local_y0 >= local_y1 or local_x0 >= local_x1:
                continue
            target_y = slice(start_y + local_y0, start_y + local_y1)
            target_x = slice(start_x + local_x0, start_x + local_x1)
            local_y = slice(local_y0, local_y1)
            local_x = slice(local_x0, local_x1)
            output[target_y, target_x] += continued[local_y, local_x]
            normalization[target_y, target_x] += (
                fine_window[local_y, local_x] ** 2
            )
    if float(np.min(normalization)) <= 0.0:
        raise RuntimeError("local phase frame did not cover the target grid")
    output /= normalization[..., None]
    return output[..., 0] if scalar else output


def local_eikonal_phase_frame_resize(
    samples: Array,
    scale: int,
    frame_size: int,
) -> Array:
    """Continue local packets in the frozen Riemannian Eikonal basis.

    For the mean determinant-one metric M of a packet, the local phase atoms

        psi_k(x) = exp(i (M^(-1/2)k)^T M^(1/2) (x-x0))

    use the metric-transformed sampling lattice and its reciprocal lattice.
    Their phase pairing is therefore exactly x^T k.  This function is an
    explicit gauge-invariance check: a frozen metric cannot change a complete
    local Fourier frame when both vectors and covectors are transformed
    correctly.  Any genuine Eikonal phase effect must come from a spatially
    varying connection or from a declared spectral restriction.

    Like :func:`local_phase_frame_resize`, the exposed packet size makes this
    an ablation rather than the final scale-complete phase operator.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    size = int(frame_size)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if size < 4 or size % 2:
        raise ValueError("frame_size must be an even integer at least four")
    height, width, channels = field.shape
    hop = size // 2
    margin = size // 2
    base_height = height + 2 * margin
    base_width = width + 2 * margin
    extra_height = (hop - (base_height - size) % hop) % hop
    extra_width = (hop - (base_width - size) % hop) % hop
    padding = (
        (margin, margin + extra_height),
        (margin, margin + extra_width),
    )
    padded = np.pad(
        field,
        (padding[0], padding[1], (0, 0)),
        mode="reflect",
    )
    metrics = internally_measured_metric(field, persistent=True)
    padded_metrics = np.pad(
        metrics,
        (padding[0], padding[1], (0, 0), (0, 0)),
        mode="reflect",
    )

    source_position = np.arange(size, dtype=np.float64)
    source_window_1d = np.sqrt(np.maximum(
        0.5 - 0.5 * np.cos(2.0 * np.pi * source_position / size),
        0.0,
    ))
    source_window = source_window_1d[:, None] * source_window_1d[None]
    sy, sx = np.meshgrid(source_position, source_position, indexing="ij")
    source_coordinates = np.stack((sx.ravel(), sy.ravel()), axis=1)
    fine_position = np.arange(size * scale, dtype=np.float64) / scale
    fy, fx = np.meshgrid(fine_position, fine_position, indexing="ij")
    fine_coordinates = np.stack((fx.ravel(), fy.ravel()), axis=1)
    angular_frequency = 2.0 * np.pi * np.fft.fftfreq(size)
    qy, qx = np.meshgrid(
        angular_frequency, angular_frequency, indexing="ij"
    )
    lattice_covectors = np.stack((qx.ravel(), qy.ravel()), axis=1)

    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    output = np.zeros((out_height, out_width, channels), dtype=np.float64)
    normalization = np.zeros((out_height, out_width), dtype=np.float64)
    for padded_y in range(0, padded.shape[0] - size + 1, hop):
        for padded_x in range(0, padded.shape[1] - size + 1, hop):
            patch_metric = np.mean(
                padded_metrics[
                    padded_y : padded_y + size,
                    padded_x : padded_x + size,
                ],
                axis=(0, 1),
            )
            eigenvalues, eigenvectors = np.linalg.eigh(patch_metric)
            square_root = (
                eigenvectors
                @ np.diag(np.sqrt(eigenvalues))
                @ eigenvectors.T
            )
            reciprocal_covectors = (
                lattice_covectors @ np.linalg.inv(square_root)
            )
            source_chart = source_coordinates @ square_root
            fine_chart = fine_coordinates @ square_root
            source_basis = np.exp(
                1j * source_chart @ reciprocal_covectors.T
            )
            fine_basis = np.exp(
                1j * fine_chart @ reciprocal_covectors.T
            )
            window_coefficients = np.linalg.solve(
                source_basis, source_window.ravel()
            )
            fine_window = (
                fine_basis @ window_coefficients
            ).real.reshape(size * scale, size * scale)
            patch = padded[
                padded_y : padded_y + size,
                padded_x : padded_x + size,
            ]
            coefficients = np.linalg.solve(
                source_basis,
                (patch * source_window[..., None]).reshape(
                    size * size, channels
                ),
            )
            continued = (
                fine_basis @ coefficients
            ).real.reshape(size * scale, size * scale, channels)
            continued *= fine_window[..., None]
            start_y = (padded_y - margin) * scale
            start_x = (padded_x - margin) * scale
            local_y0 = max(0, -start_y)
            local_x0 = max(0, -start_x)
            local_y1 = min(size * scale, out_height - start_y)
            local_x1 = min(size * scale, out_width - start_x)
            if local_y0 >= local_y1 or local_x0 >= local_x1:
                continue
            target_y = slice(start_y + local_y0, start_y + local_y1)
            target_x = slice(start_x + local_x0, start_x + local_x1)
            local_y = slice(local_y0, local_y1)
            local_x = slice(local_x0, local_x1)
            output[target_y, target_x] += continued[local_y, local_x]
            normalization[target_y, target_x] += (
                fine_window[local_y, local_x] ** 2
            )
    if float(np.min(normalization)) <= 0.0:
        raise RuntimeError("local Eikonal phase frame did not cover the target")
    output /= normalization[..., None]
    return output[..., 0] if scalar else output


def _bruun_rfft_pairs(values: Array) -> tuple[Array, Array]:
    """Batched real FFT in BFFT's native ``(Re, -Im)`` pair state.

    This is the NumPy reference form of ``phase_fft_real`` and the shipped
    ``pair_reduce_cs`` cells.  It intentionally never constructs a complex
    number.  Axis zero must have power-of-two length.
    """

    source = np.asarray(values, dtype=np.float64)
    length = source.shape[0]
    if length < 4 or length & (length - 1):
        raise ValueError("Bruun pair transform requires power-of-two length >= 4")
    trailing = source.shape[1:]
    dc = source.reshape(length, -1).copy()
    nyquist = None
    a: dict[int, Array] = {}
    b: dict[int, Array] = {}
    width = length
    extent = 1
    while width > 1:
        half = width // 2
        even = slice(0, half)
        odd = slice(half, width)
        next_dc = dc[even] + dc[odd]
        next_nyquist = dc[even] - dc[odd]
        next_a: dict[int, Array] = {}
        next_b: dict[int, Array] = {}
        if extent >= 2:
            next_a[extent // 2] = nyquist[even].copy()
            next_b[extent // 2] = nyquist[odd].copy()
        for diagonal in range(1, extent // 2):
            theta = np.pi * diagonal / extent
            cosine = float(np.cos(theta))
            sine = float(np.sin(theta))
            rotated_real = cosine * a[diagonal][odd] - sine * b[diagonal][odd]
            rotated_pair = sine * a[diagonal][odd] + cosine * b[diagonal][odd]
            next_a[diagonal] = a[diagonal][even] + rotated_real
            next_b[diagonal] = b[diagonal][even] + rotated_pair
            next_a[extent - diagonal] = a[diagonal][even] - rotated_real
            next_b[extent - diagonal] = rotated_pair - b[diagonal][even]
        dc, nyquist, a, b = (
            next_dc,
            next_nyquist,
            next_a,
            next_b,
        )
        width = half
        extent *= 2
    real = np.zeros((length // 2 + 1, dc.shape[1]), dtype=np.float64)
    negative_imaginary = np.zeros_like(real)
    real[0] = dc[0]
    real[-1] = nyquist[0]
    for diagonal in range(1, length // 2):
        real[diagonal] = a[diagonal][0]
        negative_imaginary[diagonal] = b[diagonal][0]
    return (
        real.reshape((length // 2 + 1,) + trailing),
        negative_imaginary.reshape((length // 2 + 1,) + trailing),
    )


def _bruun_phase_evaluate(values: Array, positions: Array) -> Array:
    """Evaluate a real periodic field by rotations of its Bruun pairs."""

    source = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    length = source.shape[0]
    real, negative_imaginary = _bruun_rfft_pairs(source)
    trailing = source.shape[1:]
    real = real.reshape(length // 2 + 1, -1)
    negative_imaginary = negative_imaginary.reshape(length // 2 + 1, -1)
    output = (
        real[0][None]
        + np.cos(np.pi * positions)[:, None] * real[-1][None]
    )
    if length > 2:
        frequency = np.arange(1, length // 2, dtype=np.float64)
        angle = 2.0 * np.pi * positions[:, None] * frequency[None] / length
        # Re((a - i b) exp(i angle)) = a cos(angle) + b sin(angle).
        output += 2.0 * (
            np.cos(angle) @ real[1:-1]
            + np.sin(angle) @ negative_imaginary[1:-1]
        )
    output /= length
    return output.reshape((positions.size,) + trailing)


def _bruun_even_detrended_profile(
    values: Array,
) -> tuple[Array, Array, Array, Array]:
    """Bruun pairs for an even extension after removing the endpoint chord."""

    source = np.asarray(values, dtype=np.float64)
    intervals = source.shape[0] - 1
    if intervals < 2 or intervals & (intervals - 1):
        raise ValueError("profile interval count must be a power of two")
    coordinate = np.arange(intervals + 1, dtype=np.float64) / intervals
    chord = (
        (1.0 - coordinate).reshape((-1,) + (1,) * (source.ndim - 1)) * source[0]
        + coordinate.reshape((-1,) + (1,) * (source.ndim - 1)) * source[-1]
    )
    residual = source - chord
    extension = np.concatenate((residual, residual[-2:0:-1]), axis=0)
    real, pair = _bruun_rfft_pairs(extension)
    return real, pair, source[0].copy(), source[-1].copy()


def _bruun_evaluate_profile(
    profile: tuple[Array, Array, Array, Array],
    position: float,
) -> Array:
    """Evaluate one detrended even Bruun profile at a scalar coordinate."""

    real, pair, left, right = profile
    length = 2 * (real.shape[0] - 1)
    intervals = length // 2
    t = float(position)
    frequency = np.arange(1, length // 2, dtype=np.float64)
    output = real[0].copy()
    output += np.cos(np.pi * t) * real[-1]
    if frequency.size:
        angle = 2.0 * np.pi * t * frequency / length
        reshape = (frequency.size,) + (1,) * (real.ndim - 1)
        output += 2.0 * np.sum(
            np.cos(angle).reshape(reshape) * real[1:-1]
            + np.sin(angle).reshape(reshape) * pair[1:-1],
            axis=0,
        )
    output /= length
    fraction = t / intervals
    return output + (1.0 - fraction) * left + fraction * right


def _bruun_limited_profile(values: Array) -> tuple[Array, Array]:
    """Return samples and Bruun-derived nodal derivatives.

    The endpoint chord is removed before differentiating the even Bruun
    continuation.  Restoring its constant derivative makes affine profiles
    exact.  The derivatives remain proposals here; the interval evaluator
    projects each adjacent pair into a sufficient monotone-Hermite set.
    """

    source = np.asarray(values, dtype=np.float64)
    profile = _bruun_even_detrended_profile(source)
    real, pair, left, right = profile
    length = 2 * (real.shape[0] - 1)
    intervals = length // 2
    positions = np.arange(intervals + 1, dtype=np.float64)
    frequency = np.arange(1, length // 2, dtype=np.float64)
    derivative = np.zeros_like(source, dtype=np.float64)
    derivative -= (
        np.pi
        * np.sin(np.pi * positions).reshape(
            (positions.size,) + (1,) * (source.ndim - 1)
        )
        * real[-1]
    ) / length
    if frequency.size:
        angle = (
            2.0
            * np.pi
            * positions[:, None]
            * frequency[None]
            / length
        )
        angular_frequency = 2.0 * np.pi * frequency / length
        spectral = (
            -np.sin(angle).reshape(
                (positions.size, frequency.size)
                + (1,) * (source.ndim - 1)
            )
            * real[1:-1][None]
            + np.cos(angle).reshape(
                (positions.size, frequency.size)
                + (1,) * (source.ndim - 1)
            )
            * pair[1:-1][None]
        )
        derivative += (
            2.0
            * np.sum(
                angular_frequency.reshape(
                    (1, frequency.size) + (1,) * (source.ndim - 1)
                )
                * spectral,
                axis=1,
            )
            / length
        )
    derivative += (right - left) / intervals
    return source.copy(), derivative


def _evaluate_limited_bruun_profile(
    profile: tuple[Array, Array],
    position: float,
) -> Array:
    """Evaluate the Bruun phase jet after a fixed monotonicity projection.

    For one interval with secant ``d``, normalize the proposed endpoint
    derivatives as ``(alpha,beta)=(m0/d,m1/d)``.  Their Euclidean projection
    onto ``alpha,beta >= 0`` and ``alpha**2 + beta**2 <= 9`` is unique.  This
    Fritsch--Carlson disk is a sufficient monotonicity set for the cubic
    Hermite segment.  Thus the phase state supplies the jet, while the
    observed endpoint order supplies a hard admissibility condition; no
    threshold, blend gain, or content-selected method is introduced.
    """

    source, derivatives = profile
    t = float(position)
    interval = min(max(int(np.floor(t)), 0), source.shape[0] - 2)
    u = min(max(t - interval, 0.0), 1.0)
    y0 = source[interval]
    y1 = source[interval + 1]
    secant = y1 - y0
    m0 = derivatives[interval].copy()
    m1 = derivatives[interval + 1].copy()
    zero = secant == 0.0
    alpha = np.divide(m0, secant, out=np.zeros_like(m0), where=~zero)
    beta = np.divide(m1, secant, out=np.zeros_like(m1), where=~zero)
    alpha = np.maximum(alpha, 0.0)
    beta = np.maximum(beta, 0.0)
    radius = np.hypot(alpha, beta)
    contraction = np.divide(
        3.0,
        radius,
        out=np.ones_like(radius),
        where=radius > 3.0,
    )
    alpha *= contraction
    beta *= contraction
    m0 = alpha * secant
    m1 = beta * secant
    h00 = (2.0 * u - 3.0) * u * u + 1.0
    h10 = ((u - 2.0) * u + 1.0) * u
    h01 = (-2.0 * u + 3.0) * u * u
    h11 = (u - 1.0) * u * u
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def _retract_admissible_hermite_jet(source: Array, proposed: Array) -> Array:
    """Explicitly retract one shared nodal jet into all interval disks.

    A nodal derivative survives the sign projection only when it agrees with
    every incident nonzero secant.  Each interval then supplies the radial
    Fritsch--Carlson contraction needed by its two endpoint derivatives.  A
    node receives the smallest contraction supplied by its incident
    intervals.  Decreasing either coordinate cannot leave a previously
    satisfied quarter disk, so every interval remains monotone and the one
    shared derivative makes the complete profile C1.
    """

    source = np.asarray(source, dtype=np.float64)
    derivative = np.asarray(proposed, dtype=np.float64).copy()
    if source.shape != derivative.shape:
        raise ValueError("source and proposed jet must have equal shape")
    secant = np.diff(source, axis=0)
    for node in range(source.shape[0]):
        incident = []
        if node > 0:
            incident.append(secant[node - 1])
        if node < source.shape[0] - 1:
            incident.append(secant[node])
        admitted = np.ones_like(derivative[node], dtype=bool)
        for slope in incident:
            admitted &= slope != 0.0
            admitted &= derivative[node] * slope >= 0.0
        if len(incident) == 2:
            admitted &= incident[0] * incident[1] > 0.0
        derivative[node] = np.where(admitted, derivative[node], 0.0)

    factor = np.ones_like(derivative)
    for interval, delta in enumerate(secant):
        alpha = np.divide(
            derivative[interval],
            delta,
            out=np.zeros_like(delta),
            where=delta != 0.0,
        )
        beta = np.divide(
            derivative[interval + 1],
            delta,
            out=np.zeros_like(delta),
            where=delta != 0.0,
        )
        radius = np.hypot(alpha, beta)
        contraction = np.divide(
            3.0,
            radius,
            out=np.ones_like(radius),
            where=radius > 3.0,
        )
        factor[interval] = np.minimum(factor[interval], contraction)
        factor[interval + 1] = np.minimum(
            factor[interval + 1], contraction
        )
    return derivative * factor


def _bruun_c1_limited_profile(values: Array) -> tuple[Array, Array]:
    """Legacy Bruun factorization of the retracted spectral Hermite jet."""

    source, proposed = _bruun_limited_profile(values)
    return source, _retract_admissible_hermite_jet(source, proposed)


def _evaluate_c1_hermite_profile(
    profile: tuple[Array, Array], position: float
) -> Array:
    """Evaluate a cubic Hermite segment from an already admissible C1 jet."""

    source, derivative = profile
    t = float(position)
    interval = min(max(int(np.floor(t)), 0), source.shape[0] - 2)
    u = min(max(t - interval, 0.0), 1.0)
    y0 = source[interval]
    y1 = source[interval + 1]
    m0 = derivative[interval]
    m1 = derivative[interval + 1]
    h00 = (2.0 * u - 3.0) * u * u + 1.0
    h10 = ((u - 2.0) * u + 1.0) * u
    h01 = (-2.0 * u + 3.0) * u * u
    h11 = (u - 1.0) * u * u
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


# Retained for the experiment tests that compare the superseded factorization.
_evaluate_c1_bruun_profile = _evaluate_c1_hermite_profile


def _cosine_spectral_jet(values: Array) -> tuple[Array, Array]:
    """Return the DCT-I pseudospectral nodal derivative proposal.

    The endpoint chord is removed, the residual is continued evenly, and the
    derivative of its unique cosine interpolant is evaluated at the source
    nodes.  DCT-I and DST-I implement the two exact finite sums.  Unlike the
    Bruun factorization, this definition is valid for every interval count.
    """

    source = np.asarray(values, dtype=np.float64)
    intervals = source.shape[0] - 1
    if intervals < 2:
        raise ValueError("profile must contain at least two intervals")
    coordinate = np.arange(intervals + 1, dtype=np.float64) / intervals
    reshape = (intervals + 1,) + (1,) * (source.ndim - 1)
    chord = (
        (1.0 - coordinate).reshape(reshape) * source[0]
        + coordinate.reshape(reshape) * source[-1]
    )
    coefficient = dct(source - chord, type=1, axis=0)
    derivative = np.broadcast_to(
        (source[-1] - source[0]) / intervals, source.shape
    ).copy()
    if intervals > 1:
        frequency = np.arange(1, intervals, dtype=np.float64)
        weighted = coefficient[1:intervals] * frequency.reshape(
            (intervals - 1,) + (1,) * (source.ndim - 1)
        )
        derivative[1:intervals] -= (
            np.pi
            * dst(weighted, type=1, axis=0)
            / (2.0 * intervals * intervals)
        )
    return source.copy(), derivative


def _project_signed_bezier_increment(
    proposed: Array,
    signs: Array,
    integral: Array,
) -> Array:
    """Project three control-edge increments onto one signed mass fibre.

    For each scalar component this returns the unique Euclidean projection
    onto

        sum_j c_j = integral,       signs_j * c_j >= 0.

    The set is a closed convex slice of an orthant.  The cubic and quintic
    currents used here have three and five coordinates, respectively, so all
    nonempty faces can be enumerated directly.
    """

    raw = np.asarray(proposed, dtype=np.float64)
    orientation = np.asarray(signs, dtype=np.float64)
    total = np.asarray(integral, dtype=np.float64)
    degree = raw.shape[0]
    if degree < 1 or degree > 5 or orientation.shape != raw.shape:
        raise ValueError("Bezier increment and sign arrays must have 1..5 rows")
    flat_raw = raw.reshape(degree, -1)
    flat_sign = orientation.reshape(degree, -1)
    flat_total = np.broadcast_to(total, raw.shape[1:]).reshape(-1)
    result = np.empty_like(flat_raw)
    for component in range(flat_raw.shape[1]):
        target = flat_sign[:, component] * flat_raw[:, component]
        signed_sum = flat_sign[:, component]
        best = None
        best_error = np.inf
        for mask in range(1, 1 << degree):
            active = np.array(
                tuple(bool(mask & (1 << index)) for index in range(degree))
            )
            coefficient = signed_sum[active]
            candidate = np.zeros(degree, dtype=np.float64)
            lagrange = (
                float(np.dot(coefficient, target[active]))
                - flat_total[component]
            ) / float(np.dot(coefficient, coefficient))
            candidate[active] = target[active] - lagrange * coefficient
            if np.min(candidate[active]) < -64.0 * np.finfo(float).eps:
                continue
            candidate = np.maximum(candidate, 0.0)
            if abs(float(np.dot(signed_sum, candidate)) - flat_total[component]) > (
                256.0
                * np.finfo(float).eps
                * max(1.0, abs(float(flat_total[component])))
            ):
                continue
            error = float(np.sum((candidate - target) ** 2))
            if error < best_error:
                best = candidate
                best_error = error
        if best is None:
            raise RuntimeError("signed Bezier mass fibre is unexpectedly empty")
        result[:, component] = flat_sign[:, component] * best
    return result.reshape(raw.shape)


def _admit_ordered_variation(raw: Array, secant: Array) -> Array:
    """Admit Bezier differential coefficients into observed sign order."""

    intervals, order = raw.shape[:2]
    flat_raw = raw.reshape(intervals * order, -1)
    flat_secant = secant.reshape(intervals, -1)
    admitted = np.empty_like(flat_raw)
    for component in range(flat_raw.shape[1]):
        differences = flat_secant[:, component]
        coarse_sign = np.sign(differences)
        for interval in range(intervals):
            if coarse_sign[interval] != 0.0:
                continue
            left = next(
                (coarse_sign[index] for index in range(interval - 1, -1, -1)
                 if coarse_sign[index] != 0.0),
                0.0,
            )
            right = next(
                (coarse_sign[index] for index in range(interval + 1, intervals)
                 if coarse_sign[index] != 0.0),
                0.0,
            )
            coarse_sign[interval] = left if left != 0.0 else right
        if not np.any(coarse_sign):
            admitted[:, component] = 0.0
            continue

        first = int(np.flatnonzero(coarse_sign)[0])
        coarse_sign[:first] = coarse_sign[first]
        boundary: list[tuple[int, float]] = []
        previous_location = 0
        for knot in range(1, intervals):
            left_sign = coarse_sign[knot - 1]
            right_sign = coarse_sign[knot]
            if left_sign == right_sign:
                continue
            centre = order * knot
            candidates = range(
                max(previous_location + 1, centre - order + 1),
                min(order * intervals, centre + order),
            )
            local_start = max(0, centre - order)
            local_stop = min(order * intervals, centre + order)
            best_boundary = centre
            best_cost = np.inf
            for candidate in candidates:
                indices = np.arange(local_start, local_stop)
                expected = np.where(indices < candidate, left_sign, right_sign)
                wrong = expected * flat_raw[indices, component] < 0.0
                cost = float(np.sum(flat_raw[indices[wrong], component] ** 2))
                if cost < best_cost:
                    best_boundary = candidate
                    best_cost = cost
            boundary.append((best_boundary, right_sign))
            previous_location = best_boundary

        sign_sequence = np.full(order * intervals, coarse_sign[first])
        for location, new_sign in boundary:
            sign_sequence[location:] = new_sign
        for interval in range(intervals):
            admitted[
                order * interval : order * interval + order, component
            ] = _project_signed_bezier_increment(
                flat_raw[
                    order * interval : order * interval + order, component
                ],
                sign_sequence[order * interval : order * interval + order],
                differences[interval],
            )
    return admitted.reshape(raw.shape)


def _variation_lineage_profile(values: Array) -> tuple[Array, Array]:
    """Return a cardinal cubic whose differential has witnessed sign order.

    The detrended cosine interpolant supplies only a differential proposal.
    Its Hermite form is converted to three Bezier control-edge increments per
    source interval.  The signs of observed first differences define the
    ordered variation lineages.  At each witnessed sign transition the
    boundary is allowed to move across either adjacent interval; its unique
    location is the one with least squared disagreement with the proposed
    differential.  Each interval is then projected onto the corresponding
    signed mass fibre.

    The returned increments sum exactly to every observed first difference.
    Their concatenated sign sequence has no more changes than the observed
    first differences.  Because the quadratic Bernstein basis is totally
    positive, the synthesized derivative cannot have more sign changes than
    that control-edge sequence.  Existing turning points may move between
    samples, but an additional alternating variation pair cannot be created.
    """

    source, derivative = _cosine_spectral_jet(values)
    intervals = source.shape[0] - 1
    secant = np.diff(source, axis=0)
    raw = np.empty((intervals, 3) + source.shape[1:], dtype=np.float64)
    raw[:, 0] = derivative[:-1] / 3.0
    raw[:, 2] = derivative[1:] / 3.0
    raw[:, 1] = secant - raw[:, 0] - raw[:, 2]
    increments = _admit_ordered_variation(raw, secant)
    return source.copy(), increments


def _evaluate_variation_lineage_profile(
    profile: tuple[Array, Array], position: float
) -> Array:
    """Integrate one admitted Bezier differential at a source coordinate."""

    source, increment = profile
    t = float(position)
    interval = min(max(int(np.floor(t)), 0), source.shape[0] - 2)
    u = min(max(t - interval, 0.0), 1.0)
    p0 = source[interval]
    p1 = p0 + increment[interval, 0]
    p2 = p1 + increment[interval, 1]
    p3 = p2 + increment[interval, 2]
    one_minus = 1.0 - u
    return (
        one_minus**3 * p0
        + 3.0 * one_minus**2 * u * p1
        + 3.0 * one_minus * u**2 * p2
        + u**3 * p3
    )


def _local_fourth_order_2jet(values: Array) -> tuple[Array, Array, Array]:
    """Return the minimal compact 2-jet with fourth-order interior precision."""

    source = np.asarray(values, dtype=np.float64)
    if source.shape[0] < 5:
        raise ValueError("a fourth-order 2-jet requires at least five nodes")
    first = np.empty_like(source)
    second = np.empty_like(source)
    first[0] = (-3.0 * source[0] + 4.0 * source[1] - source[2]) / 2.0
    first[1] = (source[2] - source[0]) / 2.0
    first[-2] = (source[-1] - source[-3]) / 2.0
    first[-1] = (3.0 * source[-1] - 4.0 * source[-2] + source[-3]) / 2.0
    second[0] = (
        2.0 * source[0] - 5.0 * source[1]
        + 4.0 * source[2] - source[3]
    )
    second[1] = source[0] - 2.0 * source[1] + source[2]
    second[-2] = source[-3] - 2.0 * source[-2] + source[-1]
    second[-1] = (
        2.0 * source[-1] - 5.0 * source[-2]
        + 4.0 * source[-3] - source[-4]
    )
    first[2:-2] = (
        source[:-4]
        - 8.0 * source[1:-3]
        + 8.0 * source[3:-1]
        - source[4:]
    ) / 12.0
    second[2:-2] = (
        -source[4:]
        + 16.0 * source[3:-1]
        - 30.0 * source[2:-2]
        + 16.0 * source[1:-3]
        - source[:-4]
    ) / 12.0
    return source.copy(), first, second


def _quintic_variation_lineage_profile(values: Array) -> tuple[Array, Array]:
    """Lift the compact 2-jet into an ordered quintic differential current."""

    source, first, second = _local_fourth_order_2jet(values)
    secant = np.diff(source, axis=0)
    intervals = source.shape[0] - 1
    control = np.empty((intervals, 6) + source.shape[1:], dtype=np.float64)
    control[:, 0] = source[:-1]
    control[:, 1] = source[:-1] + first[:-1] / 5.0
    control[:, 2] = source[:-1] + 2.0 * first[:-1] / 5.0 + second[:-1] / 20.0
    control[:, 5] = source[1:]
    control[:, 4] = source[1:] - first[1:] / 5.0
    control[:, 3] = source[1:] - 2.0 * first[1:] / 5.0 + second[1:] / 20.0
    raw = np.diff(control, axis=1)
    return source, _admit_ordered_variation(raw, secant)


def _evaluate_quintic_variation_lineage_profile(
    profile: tuple[Array, Array], position: float
) -> Array:
    """Integrate one admitted quintic Bernstein differential."""

    source, increment = profile
    t = float(position)
    interval = min(max(int(np.floor(t)), 0), source.shape[0] - 2)
    u = min(max(t - interval, 0.0), 1.0)
    control = np.concatenate((
        source[interval][None],
        source[interval][None] + np.cumsum(increment[interval], axis=0),
    ), axis=0)
    for _ in range(5):
        control = (1.0 - u) * control[:-1] + u * control[1:]
    return control[0]


def _refine_profile_axis(
    values: Array,
    axis: int,
    scale: int,
    profile_builder: object,
    evaluator: object,
) -> Array:
    """Apply one cardinal source-line current to a complete array axis."""

    source = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    target_length = (source.shape[0] - 1) * scale + 1
    positions = np.arange(target_length, dtype=np.float64) / scale
    profile = profile_builder(source)
    refined = np.stack([evaluator(profile, position) for position in positions])
    return np.moveaxis(refined, 0, axis)


def tensor_product_quintic_variation_resize(samples: Array, scale: int) -> Array:
    """Compose the ordered quintic current on the two Cartesian factors."""

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    along_x = _refine_profile_axis(
        field,
        1,
        scale,
        _quintic_variation_lineage_profile,
        _evaluate_quintic_variation_lineage_profile,
    )
    output = _refine_profile_axis(
        along_x,
        0,
        scale,
        _quintic_variation_lineage_profile,
        _evaluate_quintic_variation_lineage_profile,
    )
    return output[..., 0] if scalar else output


def quintic_variation_lifting_analysis(
    values: Array, axis: int = 0
) -> tuple[Array, Array]:
    """Split a dyadic line into canon anchors and exact odd-site details."""

    fine = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    if fine.shape[0] < 9 or fine.shape[0] % 2 != 1:
        raise ValueError("lifting analysis requires an odd axis of length at least 9")
    coarse = fine[::2].copy()
    profile = _quintic_variation_lineage_profile(coarse)
    predicted = np.stack([
        _evaluate_quintic_variation_lineage_profile(profile, index + 0.5)
        for index in range(coarse.shape[0] - 1)
    ])
    detail = fine[1::2] - predicted
    return np.moveaxis(coarse, 0, axis), np.moveaxis(detail, 0, axis)


def quintic_variation_lifting_synthesis(
    coarse: Array, detail: Array, axis: int = 0
) -> Array:
    """Invert :func:`quintic_variation_lifting_analysis` exactly in arithmetic."""

    anchors = np.moveaxis(np.asarray(coarse, dtype=np.float64), axis, 0)
    residual = np.moveaxis(np.asarray(detail, dtype=np.float64), axis, 0)
    expected = (anchors.shape[0] - 1,) + anchors.shape[1:]
    if anchors.shape[0] < 5 or residual.shape != expected:
        raise ValueError("detail shape must contain one value between each anchor")
    profile = _quintic_variation_lineage_profile(anchors)
    predicted = np.stack([
        _evaluate_quintic_variation_lineage_profile(profile, index + 0.5)
        for index in range(anchors.shape[0] - 1)
    ])
    fine = np.empty((2 * anchors.shape[0] - 1,) + anchors.shape[1:])
    fine[::2] = anchors
    fine[1::2] = predicted + residual
    return np.moveaxis(fine, 0, axis)


def riemannian_product_variation_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Compose product-current resolution with the full tangent generator.

    The isotropic component is the rank-preserving Cartesian product current.
    The anisotropic component is the positive two-exit measure of the
    principal tangent.  Their masses are fixed by the canonical generator
    decomposition, so the construction remains one operator as the measured
    tensor passes continuously between isotropic and rank-one geometry.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    product = tensor_product_quintic_variation_resize(field, scale)
    product, _ = _as_field(product)
    row_profiles = [_quintic_variation_lineage_profile(field[row])
                    for row in range(field.shape[0])]
    column_profiles = [_quintic_variation_lineage_profile(field[:, column])
                       for column in range(field.shape[1])]
    normal_tensor = _measured_normal_tensor(field, persistent=persistent)
    output = product.copy()
    for iy in range(field.shape[0] - 1):
        for ix in range(field.shape[1] - 1):
            tensor = np.mean(
                normal_tensor[iy : iy + 2, ix : ix + 2], axis=(0, 1)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
            denominator = float(eigenvalues[0] + eigenvalues[1])
            if denominator <= 0.0:
                continue
            directional_mass = max(
                float(eigenvalues[1] - eigenvalues[0]), 0.0
            )
            weight = directional_mass / denominator
            if weight == 0.0:
                continue
            tangent = np.array(
                (-eigenvectors[1, 1], eigenvectors[0, 1]), dtype=np.float64
            )
            for sy in range(1, scale):
                oy = iy * scale + sy
                for sx in range(1, scale):
                    ox = ix * scale + sx
                    chord = _cell_directional_chord(
                        row_profiles,
                        column_profiles,
                        iy,
                        ix,
                        sx / scale,
                        sy / scale,
                        tangent,
                        _evaluate_quintic_variation_lineage_profile,
                    )
                    output[oy, ox] = (
                        (1.0 - weight) * product[oy, ox] + weight * chord
                    )
    return output[..., 0] if scalar else output


def _natural_cubic_variational_jet(values: Array) -> tuple[Array, Array]:
    """Return the minimum-bending-energy interpolatory nodal jet.

    For unit-spaced samples, the natural cubic spline derivatives solve one
    fixed tridiagonal system.  The leading axis is the source line; every
    trailing scalar component is solved in the same forward/backward sweep.
    """

    source = np.asarray(values, dtype=np.float64)
    nodes = source.shape[0]
    if nodes < 3:
        raise ValueError("profile must contain at least two intervals")
    secant = np.diff(source, axis=0)
    right = np.empty_like(source)
    right[0] = 3.0 * secant[0]
    right[-1] = 3.0 * secant[-1]
    right[1:-1] = 3.0 * (secant[:-1] + secant[1:])

    upper = np.empty(nodes - 1, dtype=np.float64)
    work = right.copy()
    upper[0] = 0.5
    work[0] *= 0.5
    for node in range(1, nodes - 1):
        denominator = 4.0 - upper[node - 1]
        upper[node] = 1.0 / denominator
        work[node] = (work[node] - work[node - 1]) / denominator
    denominator = 2.0 - upper[-1]
    work[-1] = (work[-1] - work[-2]) / denominator

    derivative = np.empty_like(source)
    derivative[-1] = work[-1]
    for node in range(nodes - 2, -1, -1):
        derivative[node] = work[node] - upper[node] * derivative[node + 1]
    return source.copy(), derivative


def _sbp42_jet(values: Array) -> tuple[Array, Array]:
    """Return the fourth-interior/second-boundary diagonal-norm SBP jet."""

    source = np.asarray(values, dtype=np.float64)
    nodes = source.shape[0]
    if nodes < 3:
        raise ValueError("profile must contain at least two intervals")
    if nodes < 8:
        derivative = np.gradient(source, axis=0, edge_order=1)
        return source.copy(), derivative

    derivative = np.empty_like(source)
    closures = (
        np.array((-24.0 / 17.0, 59.0 / 34.0, -4.0 / 17.0, -3.0 / 34.0)),
        np.array((-0.5, 0.0, 0.5)),
        np.array((4.0 / 43.0, -59.0 / 86.0, 0.0, 59.0 / 86.0, -4.0 / 43.0)),
        np.array((3.0 / 98.0, 0.0, -59.0 / 98.0, 0.0, 32.0 / 49.0, -4.0 / 49.0)),
    )
    for node, coefficient in enumerate(closures):
        derivative[node] = np.tensordot(
            coefficient, source[: coefficient.size], axes=(0, 0)
        )
        derivative[-1 - node] = -np.tensordot(
            coefficient, source[-coefficient.size :][::-1], axes=(0, 0)
        )
    for node in range(4, nodes - 4):
        derivative[node] = (
            source[node - 2]
            - 8.0 * source[node - 1]
            + 8.0 * source[node + 1]
            - source[node + 2]
        ) / 12.0
    return source.copy(), derivative


def _batched_retracted_profile_fields(
    field: Array, proposal: object
) -> tuple[Array, Array]:
    """Build horizontal and vertical admitted jets with two batched calls."""

    column_source, column_proposal = proposal(field)
    column_derivative = _retract_admissible_hermite_jet(
        column_source, column_proposal
    )
    row_source = np.moveaxis(field, 1, 0)
    row_source, row_proposal = proposal(row_source)
    row_derivative = np.moveaxis(
        _retract_admissible_hermite_jet(row_source, row_proposal), 0, 1
    )
    return row_derivative, column_derivative


def _project_admissible_hermite_jet(
    source: Array,
    proposed: Array,
    *,
    diagnostics: bool = False,
) -> Array | tuple[Array, dict[str, float | int]]:
    """Hilbert-project a nodal jet onto the shared monotone-Hermite set.

    Dykstra's cyclic projections converge to the Euclidean projection onto an
    intersection of closed convex sets.  Each interval set is either the
    origin pair for a zero secant or a signed quarter disk of radius
    ``3*abs(secant)``.  Its individual Euclidean projection is closed form.
    The stopping scale is derived only from binary64 roundoff; failure to
    satisfy the primal and fixed-point certificates raises an error.
    """

    value = np.asarray(source, dtype=np.float64)
    proposal = np.asarray(proposed, dtype=np.float64)
    if value.shape != proposal.shape or value.shape[0] < 3:
        raise ValueError("source and proposed jet must have equal profile shape")
    original_shape = value.shape
    value = value.reshape(value.shape[0], -1)
    proposal = proposal.reshape(proposal.shape[0], -1)
    secant = np.diff(value, axis=0)
    projected = proposal.copy()
    correction = np.zeros((secant.shape[0], 2, value.shape[1]), dtype=np.float64)
    magnitude = max(
        1.0,
        float(np.max(np.abs(value))),
        float(np.max(np.abs(proposal))),
        float(np.max(np.abs(secant))),
    )
    tolerance = 256.0 * np.finfo(np.float64).eps * magnitude
    maximum_sweeps = 100000
    fixed_point_residual = np.inf
    feasibility_residual = np.inf
    for sweep in range(1, maximum_sweeps + 1):
        previous = projected.copy()
        for interval, delta in enumerate(secant):
            candidate = np.stack(
                (projected[interval], projected[interval + 1]), axis=0
            ) + correction[interval]
            zero = delta == 0.0
            sign = np.where(delta < 0.0, -1.0, 1.0)
            oriented = candidate * sign[None]
            oriented = np.maximum(oriented, 0.0)
            radius = 3.0 * np.abs(delta)
            norm = np.hypot(oriented[0], oriented[1])
            contraction = np.divide(
                radius,
                norm,
                out=np.ones_like(radius),
                where=norm > radius,
            )
            admitted = oriented * contraction[None]
            admitted[:, zero] = 0.0
            admitted *= sign[None]
            correction[interval] = candidate - admitted
            projected[interval] = admitted[0]
            projected[interval + 1] = admitted[1]

        fixed_point_residual = float(np.max(np.abs(projected - previous)))
        feasibility_residual = 0.0
        for interval, delta in enumerate(secant):
            if np.any(delta == 0.0):
                zero = delta == 0.0
                feasibility_residual = max(
                    feasibility_residual,
                    float(np.max(np.abs(projected[interval][zero]), initial=0.0)),
                    float(np.max(
                        np.abs(projected[interval + 1][zero]), initial=0.0
                    )),
                )
            nonzero = delta != 0.0
            if np.any(nonzero):
                oriented0 = projected[interval][nonzero] * np.sign(delta[nonzero])
                oriented1 = projected[interval + 1][nonzero] * np.sign(
                    delta[nonzero]
                )
                radius = 3.0 * np.abs(delta[nonzero])
                feasibility_residual = max(
                    feasibility_residual,
                    float(np.max(-oriented0, initial=0.0)),
                    float(np.max(-oriented1, initial=0.0)),
                    float(np.max(
                        np.hypot(oriented0, oriented1) - radius,
                        initial=0.0,
                    )),
                )
        if (
            fixed_point_residual <= tolerance
            and feasibility_residual <= tolerance
        ):
            break
    else:
        raise RuntimeError(
            "admissible Hermite projection did not converge: "
            f"fixed-point residual {fixed_point_residual:.17g}, "
            f"feasibility residual {feasibility_residual:.17g}, "
            f"binary64 tolerance {tolerance:.17g}"
        )
    information: dict[str, float | int] = {
        "sweeps": sweep,
        "fixed_point_residual": fixed_point_residual,
        "feasibility_residual": feasibility_residual,
        "binary64_tolerance": tolerance,
    }
    projected = projected.reshape(original_shape)
    return (projected, information) if diagnostics else projected


def _projected_cosine_hermite_profile(values: Array) -> tuple[Array, Array]:
    source, proposed = _cosine_spectral_jet(values)
    return source, _project_admissible_hermite_jet(source, proposed)


def _cosine_hermite_profile(values: Array) -> tuple[Array, Array]:
    """Cosine-pseudospectral jet with the explicit admissible retraction."""

    source, proposed = _cosine_spectral_jet(values)
    return source, _retract_admissible_hermite_jet(source, proposed)


def _characteristic_from_profiles(
    field: Array,
    scalar: bool,
    scale: int,
    persistent: bool,
    row_profiles: list[object],
    column_profiles: list[object],
    evaluate: object,
    *,
    continuous_metric: bool = False,
    unit_transport: bool = False,
) -> Array:
    """Shared Riemannian characteristic transport for profile evaluators."""

    height, width, channels = field.shape
    baseline = transported_harmonic_resize(
        field, scale, adaptive=True, persistent=persistent
    )
    output = np.asarray(baseline, dtype=np.float64).copy()
    metrics = internally_measured_metric(field, persistent=persistent)
    tolerance = 128.0 * np.finfo(float).eps
    for iy in range(height - 1):
        for ix in range(width - 1):
            frozen_metric = np.mean(
                metrics[iy : iy + 2, ix : ix + 2], axis=(0, 1)
            )
            frozen_eigensystem = (
                None if continuous_metric else np.linalg.eigh(frozen_metric)
            )
            for sy in range(scale + 1):
                v = sy / scale
                global_y = iy + v
                oy = iy * scale + sy
                for sx in range(scale + 1):
                    u = sx / scale
                    global_x = ix + u
                    ox = ix * scale + sx
                    if continuous_metric:
                        metric = (
                            (1.0 - u) * (1.0 - v) * metrics[iy, ix]
                            + u * (1.0 - v) * metrics[iy, ix + 1]
                            + (1.0 - u) * v * metrics[iy + 1, ix]
                            + u * v * metrics[iy + 1, ix + 1]
                        )
                    else:
                        metric = frozen_metric
                    if frozen_eigensystem is None:
                        eigenvalues, eigenvectors = np.linalg.eigh(metric)
                    else:
                        eigenvalues, eigenvectors = frozen_eigensystem
                    tangent = eigenvectors[:, 0]
                    condition = float(eigenvalues[1] / eigenvalues[0])
                    transport_weight = min(max(condition - 1.0, 0.0), 1.0)
                    if transport_weight <= tolerance:
                        continue
                    if unit_transport:
                        transport_weight = 1.0
                    tx, ty = float(tangent[0]), float(tangent[1])
                    transported_sum = np.zeros(channels, dtype=np.float64)
                    transported_mass = 0.0
                    if abs(tx) > tolerance:
                        slope = ty / tx
                        left_y = global_y - u * slope
                        right_y = global_y + (1.0 - u) * slope
                        if (
                            0.0 <= left_y <= height - 1.0
                            and 0.0 <= right_y <= height - 1.0
                        ):
                            left = evaluate(column_profiles[ix], left_y)
                            right = evaluate(column_profiles[ix + 1], right_y)
                            chart_mass = tx * tx
                            transported_sum += chart_mass * (
                                (1.0 - u) * left + u * right
                            )
                            transported_mass += chart_mass
                    if abs(ty) > tolerance:
                        slope = tx / ty
                        top_x = global_x - v * slope
                        bottom_x = global_x + (1.0 - v) * slope
                        if (
                            0.0 <= top_x <= width - 1.0
                            and 0.0 <= bottom_x <= width - 1.0
                        ):
                            top = evaluate(row_profiles[iy], top_x)
                            bottom = evaluate(row_profiles[iy + 1], bottom_x)
                            chart_mass = ty * ty
                            transported_sum += chart_mass * (
                                (1.0 - v) * top + v * bottom
                            )
                            transported_mass += chart_mass
                    if transported_mass > 0.0:
                        admitted_weight = transport_weight * transported_mass
                        output[oy, ox] = (
                            (1.0 - admitted_weight) * output[oy, ox]
                            + transport_weight * transported_sum
                        )
    return output[..., 0] if scalar else output


def bruun_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Spatial characteristic transport with Bruun endpoint phase.

    The Riemannian tangent decides only where a target characteristic meets
    the surrounding source lines.  Each required fractional endpoint value
    is then evaluated in a real Bruun cosine basis on that complete source
    line.  Removing and restoring the endpoint chord makes the phase
    continuation exactly affine-reproducing.  No phase statistic influences
    the spatial metric and no spatial metric deforms the Bruun basis.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, channels = field.shape
    if (height - 1) & (height - 2) or (width - 1) & (width - 2):
        raise ValueError("source interval counts must be powers of two")
    row_profiles = [
        _bruun_even_detrended_profile(field[iy]) for iy in range(height)
    ]
    column_profiles = [
        _bruun_even_detrended_profile(field[:, ix]) for ix in range(width)
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _bruun_evaluate_profile,
    )


def bruun_phase_jet_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Transport a Bruun phase jet on Riemannian characteristics.

    Bruun analysis proposes the first derivative of each complete source
    line.  On every source interval, the two proposed derivatives are
    projected into the fixed Fritsch--Carlson disk before a cubic Hermite
    endpoint value is evaluated.  The value transport, metric, tangent, and
    isotropic limit are otherwise identical to
    :func:`characteristic_transport_resize`.

    This is one deterministic operator.  It does not classify a location as
    an edge or a texture and it does not switch interpolators.  It continues
    phase through a jet wherever that continuation is consistent with the
    order of the two observed endpoint values, and contracts only the part of
    the jet that would contradict that order.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, _channels = field.shape
    if (height - 1) & (height - 2) or (width - 1) & (width - 2):
        raise ValueError("source interval counts must be powers of two")
    row_profiles = [_bruun_limited_profile(field[iy]) for iy in range(height)]
    column_profiles = [
        _bruun_limited_profile(field[:, ix]) for ix in range(width)
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_limited_bruun_profile,
    )


def bruun_phase_jet_continuous_metric_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Phase-jet transport using the continuous sampled SPD field.

    The measured quadratic forms at the four source-cell corners are combined
    with bilinear barycentric weights at each target point.  A positive
    weighted sum of SPD quadratic forms is SPD, and the construction is
    covariant under a common change of Cartesian coordinates.  The local
    tangent and anisotropy are taken from that target form rather than from
    one cell-wide average.  The characteristic is still straight; this
    function isolates metric continuity from curved-path integration.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, _channels = field.shape
    if (height - 1) & (height - 2) or (width - 1) & (width - 2):
        raise ValueError("source interval counts must be powers of two")
    row_profiles = [_bruun_limited_profile(field[iy]) for iy in range(height)]
    column_profiles = [
        _bruun_limited_profile(field[:, ix]) for ix in range(width)
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_limited_bruun_profile,
        continuous_metric=True,
    )


def bruun_phase_jet_c1_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Characteristic transport with one admissible Bruun jet per node."""

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, _channels = field.shape
    if (height - 1) & (height - 2) or (width - 1) & (width - 2):
        raise ValueError("source interval counts must be powers of two")
    row_profiles = [
        _bruun_c1_limited_profile(field[iy]) for iy in range(height)
    ]
    column_profiles = [
        _bruun_c1_limited_profile(field[:, ix]) for ix in range(width)
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_c1_hermite_profile,
    )


def projected_cosine_hermite_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Riemannian transport of projected cosine-spectral Hermite profiles.

    This is the formal replacement candidate for the Bruun phase-jet layer.
    DCT-I supplies the derivative of the unique detrended even trigonometric
    interpolant.  The derivative vector is Hilbert-projected onto the closed
    convex intersection of shared monotone cubic-Hermite jets.  The resulting C1
    profiles are transported by the unchanged internally measured SPD field.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, _channels = field.shape
    row_profiles = [
        _projected_cosine_hermite_profile(field[iy]) for iy in range(height)
    ]
    column_profiles = [
        _projected_cosine_hermite_profile(field[:, ix]) for ix in range(width)
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_c1_hermite_profile,
    )


def cosine_hermite_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Riemannian transport of shape-constrained cosine-Hermite profiles.

    This is the active formal reconstruction stack.  DCT-I differentiation
    proposes one global nodal jet for each detrended source line.  The fixed
    shared-node retraction admits that jet into the sufficient monotone cubic
    Hermite set, after which the unique C1 piecewise cubic is evaluated on
    the SPD characteristics.  No complex state, phase parameter, spectral
    cutoff, or power-of-two source dimension is required.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width, _channels = field.shape
    row_profiles = [_cosine_hermite_profile(field[iy]) for iy in range(height)]
    column_profiles = [
        _cosine_hermite_profile(field[:, ix]) for ix in range(width)
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_c1_hermite_profile,
    )


def variation_lineage_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
    unit_transport: bool = False,
) -> Array:
    """Transport a single ordered differential current per source line.

    Unlike the monotone-Hermite operator, this representation does not pin a
    witnessed turning point to a source node.  Unlike an unrestricted signed
    reconstruction kernel, its Bernstein differential cannot synthesize an
    additional alternating variation pair.  Source values enter synthesis
    only as integration anchors for that one differential current.

    ``unit_transport`` is retained only as a diagnostic of the metric mixing
    law; it does not alter the measured tangent or the variation current.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    row_profiles = [_variation_lineage_profile(field[row])
                    for row in range(field.shape[0])]
    column_profiles = [_variation_lineage_profile(field[:, column])
                       for column in range(field.shape[1])]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_variation_lineage_profile,
        unit_transport=unit_transport,
    )


def _batched_jet_characteristic_resize(
    samples: Array,
    scale: int,
    persistent: bool,
    proposal: object,
) -> Array:
    """Shared comparison path for batched source-line jet proposals."""

    field, scalar = _as_field(samples)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    row_derivative, column_derivative = _batched_retracted_profile_fields(
        field, proposal
    )
    row_profiles = [
        (field[row], row_derivative[row]) for row in range(field.shape[0])
    ]
    column_profiles = [
        (field[:, column], column_derivative[:, column])
        for column in range(field.shape[1])
    ]
    return _characteristic_from_profiles(
        field,
        scalar,
        scale,
        persistent,
        row_profiles,
        column_profiles,
        _evaluate_c1_hermite_profile,
    )


def batched_cosine_hermite_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Algebraically unchanged cosine jet with axis-batched transforms."""

    return _batched_jet_characteristic_resize(
        samples, scale, persistent, _cosine_spectral_jet
    )


def compact_spline_hermite_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Natural-cubic variational jet with the unchanged admissible transport."""

    return _batched_jet_characteristic_resize(
        samples, scale, persistent, _natural_cubic_variational_jet
    )


def sbp42_hermite_characteristic_resize(
    samples: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Strictly local SBP(2,4) jet with the unchanged admissible transport."""

    return _batched_jet_characteristic_resize(
        samples, scale, persistent, _sbp42_jet
    )


def local_bruun_phase_frame_resize(
    samples: Array,
    scale: int,
    frame_size: int,
) -> Array:
    """Local phase continuation using only real Bruun pair rotations.

    This is the real-arithmetic counterpart of the local Fourier packet
    ablation.  The phase state is ``(a,b)=(Re,-Im)`` and every fractional
    advance is a two-plane Givens rotation.  Power-of-two packet sizes are
    required by the exact Bruun tree; no complex analysis or synthesis is
    used anywhere in this operator.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    size = int(frame_size)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if size < 4 or size & (size - 1):
        raise ValueError("frame_size must be a power of two at least four")
    height, width, channels = field.shape
    hop = size // 2
    margin = size // 2
    base_height = height + 2 * margin
    base_width = width + 2 * margin
    extra_height = (hop - (base_height - size) % hop) % hop
    extra_width = (hop - (base_width - size) % hop) % hop
    padded = np.pad(
        field,
        ((margin, margin + extra_height),
         (margin, margin + extra_width),
         (0, 0)),
        mode="reflect",
    )
    source_position = np.arange(size, dtype=np.float64)
    source_window_1d = np.sqrt(np.maximum(
        0.5 - 0.5 * np.cos(2.0 * np.pi * source_position / size),
        0.0,
    ))
    source_window = source_window_1d[:, None] * source_window_1d[None]
    fine_position = np.arange(size * scale, dtype=np.float64) / scale

    def continue_2d(value: Array) -> Array:
        along_x = _bruun_phase_evaluate(
            np.moveaxis(value, 1, 0), fine_position
        )
        along_x = np.moveaxis(along_x, 0, 1)
        return _bruun_phase_evaluate(along_x, fine_position)

    fine_window = continue_2d(source_window)
    out_height = (height - 1) * scale + 1
    out_width = (width - 1) * scale + 1
    output = np.zeros((out_height, out_width, channels), dtype=np.float64)
    normalization = np.zeros((out_height, out_width), dtype=np.float64)
    for padded_y in range(0, padded.shape[0] - size + 1, hop):
        for padded_x in range(0, padded.shape[1] - size + 1, hop):
            patch = padded[
                padded_y : padded_y + size,
                padded_x : padded_x + size,
            ]
            continued = continue_2d(patch * source_window[..., None])
            continued *= fine_window[..., None]
            start_y = (padded_y - margin) * scale
            start_x = (padded_x - margin) * scale
            local_y0 = max(0, -start_y)
            local_x0 = max(0, -start_x)
            local_y1 = min(size * scale, out_height - start_y)
            local_x1 = min(size * scale, out_width - start_x)
            if local_y0 >= local_y1 or local_x0 >= local_x1:
                continue
            target_y = slice(start_y + local_y0, start_y + local_y1)
            target_x = slice(start_x + local_x0, start_x + local_x1)
            local_y = slice(local_y0, local_y1)
            local_x = slice(local_x0, local_x1)
            output[target_y, target_x] += continued[local_y, local_x]
            normalization[target_y, target_x] += (
                fine_window[local_y, local_x] ** 2
            )
    if float(np.min(normalization)) <= 0.0:
        raise RuntimeError("local Bruun phase frame did not cover the target")
    output /= normalization[..., None]
    return output[..., 0] if scalar else output


def _expand_real_rfft_pairs(real: Array, pair: Array, length: int) -> tuple[Array, Array]:
    """Expand a real signal's half-spectrum pair planes to all frequencies."""

    full_real = np.empty((length,) + real.shape[1:], dtype=np.float64)
    full_pair = np.empty_like(full_real)
    full_real[: length // 2 + 1] = real
    full_pair[: length // 2 + 1] = pair
    for frequency in range(1, length // 2):
        full_real[length - frequency] = real[frequency]
        full_pair[length - frequency] = -pair[frequency]
    return full_real, full_pair


def _bruun_rfft2_pairs(values: Array) -> tuple[Array, Array]:
    """Two-dimensional RFFT as real Bruun ``(Re,-Im)`` planes."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 2 or source.shape[0] != source.shape[1]:
        raise ValueError("Bruun packet must be square")
    size = source.shape[0]
    along_x_real, along_x_pair = _bruun_rfft_pairs(
        np.moveaxis(source, 1, 0)
    )
    along_x_real = np.moveaxis(along_x_real, 0, 1)
    along_x_pair = np.moveaxis(along_x_pair, 0, 1)
    ar_half, ab_half = _bruun_rfft_pairs(along_x_real)
    br_half, bb_half = _bruun_rfft_pairs(along_x_pair)
    ar, ab = _expand_real_rfft_pairs(ar_half, ab_half, size)
    br, bb = _expand_real_rfft_pairs(br_half, bb_half, size)
    # DFT_y(a - i b) = (Ar - Bb) - i(Ab + Br).
    return ar - bb, ab + br


def bruun_connection_phase_resize(
    samples: Array,
    scale: int,
    frame_size: int,
) -> Array:
    """Transport local phase connections in BFFT's real pair state.

    A square-root-Hann packet is analyzed at every source site.  For every
    spatial frequency pair, neighboring packet cross-products measure one
    horizontal and vertical phase advance.  The coefficient field is
    demodulated by that measured connection, interpolated on the spatial
    lattice, remodulated at the target, and evaluated at the packet center.

    All phase arithmetic is real: analysis uses Bruun cells, cross-products
    use the two pair planes, and transport uses Givens rotations.  The
    spatial coefficient interpolation here is isotropic bilinear; it is kept
    separate deliberately so the phase experiment does not borrow an edge
    detector or an external metric.
    """

    field, scalar = _as_field(samples)
    scale = int(scale)
    size = int(frame_size)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if size < 4 or size & (size - 1):
        raise ValueError("frame_size must be a power of two at least four")
    height, width, channels = field.shape
    center = size // 2
    padded = np.pad(
        field,
        ((center, center), (center, center), (0, 0)),
        mode="reflect",
    )
    position = np.arange(size, dtype=np.float64)
    window_1d = np.sqrt(np.maximum(
        0.5 - 0.5 * np.cos(2.0 * np.pi * position / size),
        0.0,
    ))
    window = window_1d[:, None] * window_1d[None]
    frequency_x = size // 2 + 1
    pair_real = np.empty(
        (height, width, size, frequency_x, channels), dtype=np.float64
    )
    pair_b = np.empty_like(pair_real)
    for iy in range(height):
        for ix in range(width):
            patch = padded[iy : iy + size, ix : ix + size]
            pair_real[iy, ix], pair_b[iy, ix] = _bruun_rfft2_pairs(
                patch * window[..., None]
            )

    def connection(axis: int) -> tuple[Array, Array]:
        if axis == 1:
            left_a, right_a = pair_real[:, :-1], pair_real[:, 1:]
            left_b, right_b = pair_b[:, :-1], pair_b[:, 1:]
        else:
            left_a, right_a = pair_real[:-1], pair_real[1:]
            left_b, right_b = pair_b[:-1], pair_b[1:]
        cross_real = np.sum(
            right_a * left_a + right_b * left_b,
            axis=(0, 1, 4),
        )
        cross_imaginary = np.sum(
            right_a * left_b - right_b * left_a,
            axis=(0, 1, 4),
        )
        magnitude = np.hypot(cross_real, cross_imaginary)
        cosine = np.divide(
            cross_real,
            magnitude,
            out=np.ones_like(cross_real),
            where=magnitude > 0.0,
        )
        sine = np.divide(
            cross_imaginary,
            magnitude,
            out=np.zeros_like(cross_imaginary),
            where=magnitude > 0.0,
        )
        return cosine, sine

    cosine_x, sine_x = connection(1)
    cosine_y, sine_y = connection(0)
    phase_x = np.arctan2(sine_x, cosine_x)
    phase_y = np.arctan2(sine_y, cosine_y)
    source_y = np.arange(height, dtype=np.float64)[:, None, None, None]
    source_x = np.arange(width, dtype=np.float64)[None, :, None, None]
    source_phase = (
        source_x * phase_x[None, None]
        + source_y * phase_y[None, None]
    )
    source_cosine = np.cos(source_phase)[..., None]
    source_sine = np.sin(source_phase)[..., None]
    # (a,b) for c exp(-i phase).
    demodulated_a = (
        source_cosine * pair_real - source_sine * pair_b
    )
    demodulated_b = (
        source_sine * pair_real + source_cosine * pair_b
    )
    coefficient_shape = (size, frequency_x, channels)
    interpolated_a = harmonic_resize(
        demodulated_a.reshape(height, width, -1),
        scale,
        adaptive=False,
    ).reshape(
        (height - 1) * scale + 1,
        (width - 1) * scale + 1,
        *coefficient_shape,
    )
    interpolated_b = harmonic_resize(
        demodulated_b.reshape(height, width, -1),
        scale,
        adaptive=False,
    ).reshape(interpolated_a.shape)

    target_y = (
        np.arange(interpolated_a.shape[0], dtype=np.float64)[:, None, None, None]
        / scale
    )
    target_x = (
        np.arange(interpolated_a.shape[1], dtype=np.float64)[None, :, None, None]
        / scale
    )
    target_phase = (
        target_x * phase_x[None, None]
        + target_y * phase_y[None, None]
    )
    target_cosine = np.cos(target_phase)[..., None]
    target_sine = np.sin(target_phase)[..., None]
    # (a,b) for c exp(+i phase).
    reconstructed_a = (
        target_cosine * interpolated_a + target_sine * interpolated_b
    )
    reconstructed_b = (
        -target_sine * interpolated_a + target_cosine * interpolated_b
    )

    ky = np.arange(size, dtype=np.float64)[:, None]
    kx = np.arange(frequency_x, dtype=np.float64)[None]
    center_angle = 2.0 * np.pi * center * (ky + kx) / size
    frequency_weight = np.full((1, frequency_x), 2.0, dtype=np.float64)
    frequency_weight[0, 0] = 1.0
    frequency_weight[0, -1] = 1.0
    cosine_weight = frequency_weight * np.cos(center_angle) / (size * size)
    sine_weight = frequency_weight * np.sin(center_angle) / (size * size)
    output = np.einsum(
        "...nkc,nk->...c", reconstructed_a, cosine_weight
    ) + np.einsum(
        "...nkc,nk->...c", reconstructed_b, sine_weight
    )
    return output[..., 0] if scalar else output


def positive_restrict(refined: Array, scale: int) -> Array:
    """Anti-alias by the scale-determined triangular kernel, then sample."""

    field, scalar = _as_field(refined)
    offsets = np.arange(-(scale - 1), scale, dtype=np.float64)
    kernel = (scale - np.abs(offsets)) / float(scale * scale)
    filtered = convolve1d(field, kernel, axis=0, mode="reflect")
    filtered = convolve1d(filtered, kernel, axis=1, mode="reflect")
    result = filtered[::scale, ::scale]
    return result[..., 0] if scalar else result


def riemannian_harmonic_adjoint_restrict(
    refined: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Restrict by the mass-normalized adjoint of harmonic transport.

    The fine field supplies the same determinant-one SPD geometry used by the
    prolongation.  In each coarse cell, the four positive Riemannian harmonic
    coordinates form a row-stochastic prolongation.  Transposing those
    coordinates transports uniform fine-grid mass back to the coarse sites;
    division by received mass makes every restriction row sum exactly one.
    The result is positive, constant preserving, globally range preserving,
    and contains no bandwidth, gain, or support parameter beyond ``scale``.
    """

    field, scalar = _as_field(refined)
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    fine_height, fine_width, channels = field.shape
    if (fine_height - 1) % scale or (fine_width - 1) % scale:
        raise ValueError("fine interval counts must be divisible by scale")
    coarse_height = (fine_height - 1) // scale + 1
    coarse_width = (fine_width - 1) // scale + 1
    if min(coarse_height, coarse_width) < 2:
        raise ValueError("restriction must contain at least one coarse cell")

    fine_metric = internally_measured_metric(field, persistent=persistent)
    result = _harmonic_adjoint_restrict_with_metric(field, fine_metric, scale)
    return result[..., 0] if scalar else result


def _harmonic_adjoint_restrict_with_metric(
    field: Array,
    fine_metric: Array,
    scale: int,
) -> Array:
    """Apply harmonic adjoint weights for an already measured fine metric."""

    field, _ = _as_field(field)
    fine_metric = np.asarray(fine_metric, dtype=np.float64)
    fine_height, fine_width, channels = field.shape
    if fine_metric.shape != (fine_height, fine_width, 2, 2):
        raise ValueError("fine metric must have shape HxWx2x2")
    if (fine_height - 1) % scale or (fine_width - 1) % scale:
        raise ValueError("fine interval counts must be divisible by scale")
    coarse_height = (fine_height - 1) // scale + 1
    coarse_width = (fine_width - 1) // scale + 1
    cell_coordinates = np.empty(
        (coarse_height - 1, coarse_width - 1, scale + 1, scale + 1, 4),
        dtype=np.float64,
    )
    corner_basis = np.eye(4, dtype=np.float64)
    for iy in range(coarse_height - 1):
        y0 = iy * scale
        for ix in range(coarse_width - 1):
            x0 = ix * scale
            metric = np.mean(
                fine_metric[y0 : y0 + scale + 1, x0 : x0 + scale + 1],
                axis=(0, 1),
            )
            metric /= np.sqrt(np.linalg.det(metric))
            cell_coordinates[iy, ix] = _harmonic_cell(
                corner_basis, metric, scale
            )

    accumulated = np.zeros(
        (coarse_height, coarse_width, channels), dtype=np.float64
    )
    received_mass = np.zeros((coarse_height, coarse_width), dtype=np.float64)
    for y in range(fine_height):
        iy = min(y // scale, coarse_height - 2)
        sy = y - iy * scale
        for x in range(fine_width):
            ix = min(x // scale, coarse_width - 2)
            sx = x - ix * scale
            weight = cell_coordinates[iy, ix, sy, sx]
            corners = (
                (iy, ix),
                (iy, ix + 1),
                (iy + 1, ix + 1),
                (iy + 1, ix),
            )
            for coefficient, (cy, cx) in zip(weight, corners):
                accumulated[cy, cx] += coefficient * field[y, x]
                received_mass[cy, cx] += coefficient
    if np.any(received_mass <= 0.0):
        raise RuntimeError("harmonic adjoint left a coarse site without mass")
    return accumulated / received_mass[..., None]


def riemannian_transport_restrict(
    refined: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """Reverse the harmonic/characteristic blend used by prolongation.

    The harmonic branch is its positive mass-normalized adjoint.  The
    characteristic branch is the cardinal pullback to the nested sites.  At
    each coarse site they are combined by the prolongation's same intrinsic
    transport coefficient ``tau = kappa(M) - 1``.  Since the measured metric
    has ``1 <= kappa(M) <= 2``, this is a convex blend with no added parameter.
    """

    field, scalar = _as_field(refined)
    scale = int(scale)
    harmonic = riemannian_harmonic_adjoint_restrict(
        field, scale, persistent=persistent
    )
    harmonic, _ = _as_field(harmonic)
    cardinal = field[::scale, ::scale]
    metric = internally_measured_metric(field, persistent=persistent)[
        ::scale, ::scale
    ]
    eigenvalues = np.linalg.eigvalsh(metric)
    condition = eigenvalues[..., 1] / eigenvalues[..., 0]
    transport = np.clip(condition - 1.0, 0.0, 1.0)[..., None]
    result = (1.0 - transport) * harmonic + transport * cardinal
    return result[..., 0] if scalar else result


def riemannian_lifting_restrict(
    refined: Array,
    scale: int,
    *,
    persistent: bool = True,
) -> Array:
    """One-stage transport lifting restriction for the active prolongation.

    Cardinal pullback supplies the coarse primal field.  The active
    prolongation predicts the fine field, and the residual is returned once
    through the mass-normalized harmonic adjoint measured in the original
    fine-field geometry.  Each updated coarse value is projected onto the
    componentwise range of the fine sites in its exact adjoint support.  The
    subsequent positive prolongation therefore closes a local no-new-extrema
    chain.  This is a direct predict/update lifting step, not a fitted
    iteration.
    """

    field, scalar = _as_field(refined)
    scale = int(scale)
    if (field.shape[0] - 1) % scale or (field.shape[1] - 1) % scale:
        raise ValueError("fine interval counts must be divisible by scale")
    cardinal = field[::scale, ::scale].copy()
    prediction = cosine_hermite_characteristic_resize(
        cardinal, scale, persistent=persistent
    )
    prediction, _ = _as_field(prediction)
    fine_metric = internally_measured_metric(field, persistent=persistent)
    correction = _harmonic_adjoint_restrict_with_metric(
        field - prediction, fine_metric, scale
    )
    updated = cardinal + correction
    lower = np.empty_like(cardinal)
    upper = np.empty_like(cardinal)
    for coarse_y in range(cardinal.shape[0]):
        center_y = coarse_y * scale
        y0 = max(0, center_y - scale)
        y1 = min(field.shape[0], center_y + scale + 1)
        for coarse_x in range(cardinal.shape[1]):
            center_x = coarse_x * scale
            x0 = max(0, center_x - scale)
            x1 = min(field.shape[1], center_x + scale + 1)
            support = field[y0:y1, x0:x1]
            lower[coarse_y, coarse_x] = np.min(support, axis=(0, 1))
            upper[coarse_y, coarse_x] = np.max(support, axis=(0, 1))
    result = np.minimum(upper, np.maximum(lower, updated))
    return result[..., 0] if scalar else result


def _local_range_overshoot(reconstruction: Array, samples: Array, scale: int) -> float:
    output, _ = _as_field(reconstruction)
    source, _ = _as_field(samples)
    worst = 0.0
    for y in range(output.shape[0]):
        iy = min(y // scale, source.shape[0] - 2)
        for x in range(output.shape[1]):
            ix = min(x // scale, source.shape[1] - 2)
            corners = source[iy : iy + 2, ix : ix + 2]
            low = np.min(corners, axis=(0, 1))
            high = np.max(corners, axis=(0, 1))
            worst = max(
                worst,
                float(np.max(low - output[y, x], initial=0.0)),
                float(np.max(output[y, x] - high, initial=0.0)),
            )
    return worst


def audit(reconstruction: Array, truth: Array, samples: Array, scale: int) -> dict[str, float]:
    estimate, _ = _as_field(reconstruction)
    target, _ = _as_field(truth)
    source, _ = _as_field(samples)
    nested = estimate[::scale, ::scale]
    returned, _ = _as_field(positive_restrict(estimate, scale))
    interior = tuple(slice(1, -1) for _ in range(2)) + (slice(None),)
    return {
        "mse": float(np.mean((estimate - target) ** 2)),
        "maximum_sample_error": float(np.max(np.abs(nested - source))),
        "maximum_local_range_overshoot": _local_range_overshoot(
            estimate, source, scale
        ),
        "positive_round_trip_mse_interior": float(
            np.mean((returned[interior] - source[interior]) ** 2)
        ),
        "positive_round_trip_linf_interior": float(
            np.max(np.abs(returned[interior] - source[interior]))
        ),
        "minimum": float(np.min(estimate)),
        "maximum": float(np.max(estimate)),
    }


def analytic_scene(name: str, side: int) -> Array:
    axis = np.linspace(0.0, 1.0, side)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    if name == "affine":
        return 0.12 + 0.31 * x + 0.27 * y
    if name == "oblique_edge":
        theta = np.deg2rad(31.0)
        normal = (x - 0.52) * np.cos(theta) + (y - 0.47) * np.sin(theta)
        return 0.5 + 0.42 * np.tanh(normal / 0.035)
    if name == "oblique_carrier":
        theta = np.deg2rad(37.0)
        phase = x * np.cos(theta) + y * np.sin(theta)
        envelope = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
        return 0.5 + 0.32 * envelope * np.sin(2.0 * np.pi * 4.25 * phase)
    if name == "curved_edge":
        radius = np.hypot(x - 0.47, y - 0.53)
        return 0.5 + 0.4 * np.tanh((radius - 0.27) / 0.03)
    if name == "crossing":
        first = np.sin(2.0 * np.pi * (3.0 * x + 1.0 * y))
        second = np.sin(2.0 * np.pi * (-1.0 * x + 3.0 * y))
        return 0.5 + 0.18 * (first + second)
    if name == "super_nyquist_carrier":
        theta = np.deg2rad(37.0)
        phase = x * np.cos(theta) + y * np.sin(theta)
        envelope = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
        return 0.5 + 0.32 * envelope * np.sin(2.0 * np.pi * 20.0 * phase)
    if name == "fine_checker":
        fine_x = x * (side - 1)
        fine_y = y * (side - 1)
        return 0.5 + 0.45 * np.cos(np.pi * fine_x) * np.cos(np.pi * fine_y)
    raise ValueError(f"unknown analytic scene {name!r}")


def oriented_scene(kind: str, side: int, angle_degrees: float) -> Array:
    axis = np.linspace(0.0, 1.0, side)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    theta = np.deg2rad(float(angle_degrees))
    phase = x * np.cos(theta) + y * np.sin(theta)
    if kind == "edge":
        normal = (x - 0.5) * np.cos(theta) + (y - 0.5) * np.sin(theta)
        return 0.5 + 0.42 * np.tanh(normal / 0.035)
    if kind == "carrier":
        envelope = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
        return 0.5 + 0.32 * envelope * np.sin(2.0 * np.pi * 4.25 * phase)
    raise ValueError(f"unknown oriented scene {kind!r}")


def _angle_sweep(source_side: int, scale: int) -> dict[str, object]:
    truth_side = (source_side - 1) * scale + 1
    angles = tuple(float(value) for value in np.linspace(0.0, 90.0, 13))
    result: dict[str, object] = {}
    for kind in ("edge", "carrier"):
        rows = []
        for angle in angles:
            truth = oriented_scene(kind, truth_side, angle)
            samples = truth[::scale, ::scale]
            bilinear = harmonic_resize(samples, scale, adaptive=False)
            adaptive = harmonic_resize(samples, scale, adaptive=True)
            transported = transported_harmonic_resize(samples, scale, adaptive=True)
            characteristic = characteristic_transport_resize(samples, scale)
            directional_measure = directional_measure_transport_resize(
                samples, scale
            )
            lanczos = lanczos_resize(samples, scale)
            bilinear_mse = float(np.mean((bilinear - truth) ** 2))
            adaptive_mse = float(np.mean((adaptive - truth) ** 2))
            transported_mse = float(np.mean((transported - truth) ** 2))
            characteristic_mse = float(np.mean((characteristic - truth) ** 2))
            directional_measure_mse = float(np.mean(
                (directional_measure - truth) ** 2
            ))
            lanczos_mse = float(np.mean((lanczos - truth) ** 2))
            rows.append({
                "angle_degrees": angle,
                "adaptive_over_bilinear_mse": adaptive_mse / max(bilinear_mse, 1e-300),
                "transported_over_bilinear_mse": transported_mse / max(bilinear_mse, 1e-300),
                "characteristic_over_bilinear_mse": characteristic_mse / max(bilinear_mse, 1e-300),
                "directional_measure_over_bilinear_mse": directional_measure_mse / max(bilinear_mse, 1e-300),
                "lanczos_over_bilinear_mse": lanczos_mse / max(bilinear_mse, 1e-300),
                "adaptive_local_overshoot": _local_range_overshoot(
                    adaptive, samples, scale
                ),
                "transported_local_overshoot": _local_range_overshoot(
                    transported, samples, scale
                ),
                "characteristic_local_overshoot": _local_range_overshoot(
                    characteristic, samples, scale
                ),
                "directional_measure_local_overshoot": _local_range_overshoot(
                    directional_measure, samples, scale
                ),
                "lanczos_local_overshoot": _local_range_overshoot(
                    lanczos, samples, scale
                ),
            })
        ratios = [float(row["adaptive_over_bilinear_mse"]) for row in rows]
        transported_ratios = [
            float(row["transported_over_bilinear_mse"]) for row in rows
        ]
        characteristic_ratios = [
            float(row["characteristic_over_bilinear_mse"]) for row in rows
        ]
        directional_measure_ratios = [
            float(row["directional_measure_over_bilinear_mse"]) for row in rows
        ]
        result[kind] = {
            "rows": rows,
            "adaptive_wins": sum(ratio < 1.0 for ratio in ratios),
            "adaptive_losses": sum(ratio > 1.0 for ratio in ratios),
            "best_adaptive_ratio": min(ratios),
            "worst_adaptive_ratio": max(ratios),
            "transported_wins": sum(ratio < 1.0 for ratio in transported_ratios),
            "transported_losses": sum(ratio > 1.0 for ratio in transported_ratios),
            "best_transported_ratio": min(transported_ratios),
            "worst_transported_ratio": max(transported_ratios),
            "characteristic_wins": sum(
                ratio < 1.0 for ratio in characteristic_ratios
            ),
            "characteristic_losses": sum(
                ratio > 1.0 for ratio in characteristic_ratios
            ),
            "best_characteristic_ratio": min(characteristic_ratios),
            "worst_characteristic_ratio": max(characteristic_ratios),
            "directional_measure_wins": sum(
                ratio < 1.0 for ratio in directional_measure_ratios
            ),
            "directional_measure_losses": sum(
                ratio > 1.0 for ratio in directional_measure_ratios
            ),
            "best_directional_measure_ratio": min(directional_measure_ratios),
            "worst_directional_measure_ratio": max(directional_measure_ratios),
        }
    return result


def _grey_panel(value: Array, label: str, enlargement: int = 4) -> Image.Image:
    shown = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(np.rint(255.0 * shown).astype(np.uint8), mode="L").convert("RGB")
    image = image.resize(
        (image.width * enlargement, image.height * enlargement),
        Image.Resampling.NEAREST,
    )
    panel = Image.new("RGB", (image.width, image.height + 24), "white")
    panel.paste(image, (0, 24))
    ImageDraw.Draw(panel).text((5, 5), label, fill="black")
    return panel


def render_examples(directory: Path, source_side: int, scale: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    truth_side = (source_side - 1) * scale + 1
    for name in ("oblique_edge", "oblique_carrier", "curved_edge", "crossing"):
        truth = analytic_scene(name, truth_side)
        samples = truth[::scale, ::scale]
        methods = (
            ("truth", truth),
            ("bilinear", harmonic_resize(samples, scale, adaptive=False)),
            ("self metric", harmonic_resize(samples, scale, adaptive=True)),
            ("transported", transported_harmonic_resize(samples, scale)),
            ("two-chart", characteristic_transport_resize(samples, scale)),
            ("Lanczos-3", lanczos_resize(samples, scale)),
        )
        panels = [_grey_panel(value, label) for label, value in methods]
        canvas = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "white",
        )
        cursor = 0
        for panel in panels:
            canvas.paste(panel, (cursor, 0))
            cursor += panel.width
        canvas.save(directory / f"{name}.png")


def run_benchmark(source_side: int = 17, scale: int = 4) -> dict[str, object]:
    truth_side = (source_side - 1) * scale + 1
    scenes = ("affine", "oblique_edge", "oblique_carrier", "curved_edge", "crossing")
    result: dict[str, object] = {
        "input_contract": ["samples", "target nested-grid scale"],
        "source_side": source_side,
        "scale": scale,
        "truth_side": truth_side,
        "scenes": {},
        "angle_sweep": _angle_sweep(source_side, scale),
    }
    rng = np.random.default_rng(20260826)
    for name in scenes:
        truth = analytic_scene(name, truth_side)
        samples = truth[::scale, ::scale]
        reconstructions = {
            "bilinear": harmonic_resize(samples, scale, adaptive=False),
            "self_metric_raw": harmonic_resize(
                samples, scale, adaptive=True, persistent=False
            ),
            "self_metric_persistent": harmonic_resize(
                samples, scale, adaptive=True, persistent=True
            ),
            "transported_harmonic": transported_harmonic_resize(
                samples, scale, adaptive=True, persistent=True
            ),
            "characteristic_transport": characteristic_transport_resize(
                samples, scale, persistent=True
            ),
            "directional_measure_transport": directional_measure_transport_resize(
                samples, scale
            ),
            "lanczos3": lanczos_resize(samples, scale, radius=3),
        }
        records = {
            method: audit(reconstruction, truth, samples, scale)
            for method, reconstruction in reconstructions.items()
        }
        noise = rng.choice((-1.0, 1.0), size=samples.shape) * 1.0e-7
        for method, adaptive, persistent in (
            ("bilinear", False, True),
            ("self_metric_raw", True, False),
            ("self_metric_persistent", True, True),
        ):
            perturbed = harmonic_resize(
                samples + noise,
                scale,
                adaptive=adaptive,
                persistent=persistent,
            )
            records[method]["empirical_linf_perturbation_gain"] = float(
                np.max(np.abs(perturbed - reconstructions[method]))
                / np.max(np.abs(noise))
            )
        perturbed_transport = transported_harmonic_resize(
            samples + noise, scale, adaptive=True, persistent=True
        )
        records["transported_harmonic"]["empirical_linf_perturbation_gain"] = float(
            np.max(np.abs(
                perturbed_transport - reconstructions["transported_harmonic"]
            )) / np.max(np.abs(noise))
        )
        for method, call in (
            ("characteristic_transport", characteristic_transport_resize),
            ("directional_measure_transport", directional_measure_transport_resize),
        ):
            perturbed = call(samples + noise, scale)
            records[method]["empirical_linf_perturbation_gain"] = float(
                np.max(np.abs(perturbed - reconstructions[method]))
                / np.max(np.abs(noise))
            )
        metric = internally_measured_metric(samples, persistent=True)
        eigenvalues = np.linalg.eigvalsh(metric)
        records["self_metric_persistent"]["maximum_metric_condition"] = float(
            np.max(eigenvalues[..., 1] / eigenvalues[..., 0])
        )
        result["scenes"][name] = records
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-side", type=int, default=17)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/self_geometric_harmonic_interpolation.json"),
    )
    parser.add_argument("--render-dir", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(args.source_side, args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.render_dir is not None:
        render_examples(args.render_dir, args.source_side, args.scale)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
