"""Direct inverse-map reconstruction and positive warped-pixel admission."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from experiments.conv_distilled_core import nodal_current_geometry
from standalone_conv_resize_demo.backend import (
    conv_evaluate_lines,
    conv_evaluate_profile,
)

from .geometry import ProjectiveMap
from .synthetic import endpoint_basin_bounds


Array = np.ndarray


def _as_field(values: Array) -> tuple[Array, bool]:
    field = np.asarray(values, dtype=np.float32)
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    if field.ndim != 3 or min(field.shape[:2]) < 5:
        raise ValueError("warp reconstruction requires HxW or HxWxC, H,W >= 5")
    return field, scalar


def target_nodes(shape: tuple[int, int]) -> tuple[Array, Array]:
    y = np.linspace(0.0, 1.0, int(shape[0]))
    x = np.linspace(0.0, 1.0, int(shape[1]))
    return np.meshgrid(x, y, indexing="xy")


def source_validity_mask(
    transform: ProjectiveMap,
    target_shape: tuple[int, int],
    source_shape: tuple[int, int],
    *,
    support_radius: float = 3.0,
) -> Array:
    """Queries whose complete compact source support lies inside the raster."""

    x, y = target_nodes(target_shape)
    px, py = transform.map(x, y)
    margin_x = support_radius / max(1, int(source_shape[1]) - 1)
    margin_y = support_radius / max(1, int(source_shape[0]) - 1)
    return (
        (px >= margin_x) & (px <= 1.0 - margin_x)
        & (py >= margin_y) & (py <= 1.0 - margin_y)
    )


def _sample_q1(values: Array, px: Array, py: Array) -> Array:
    field = np.asarray(values, dtype=np.float64)
    height, width = field.shape[:2]
    x = np.clip(px, 0.0, 1.0) * (width - 1)
    y = np.clip(py, 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    ux, uy = x - x0, y - y0
    suffix = (None,) * (field.ndim - 2)
    ux = ux[(...,) + suffix]
    uy = uy[(...,) + suffix]
    return (
        (1.0 - ux) * (1.0 - uy) * field[y0, x0]
        + ux * (1.0 - uy) * field[y0, x1]
        + (1.0 - ux) * uy * field[y1, x0]
        + ux * uy * field[y1, x1]
    )


def _sample_lanczos(values: Array, px: Array, py: Array, radius: int = 3) -> Array:
    field, scalar = _as_field(values)
    height, width, channels = field.shape
    output = np.empty(px.shape + (channels,), dtype=np.float64)
    for index in np.ndindex(px.shape):
        x = float(px[index]) * (width - 1)
        y = float(py[index]) * (height - 1)
        ix = np.arange(max(0, int(np.floor(x)) - radius + 1),
                       min(width, int(np.floor(x)) + radius + 1))
        iy = np.arange(max(0, int(np.floor(y)) - radius + 1),
                       min(height, int(np.floor(y)) + radius + 1))
        wx = np.sinc(x - ix) * np.sinc((x - ix) / radius)
        wy = np.sinc(y - iy) * np.sinc((y - iy) / radius)
        wx /= np.sum(wx)
        wy /= np.sum(wy)
        output[index] = np.einsum(
            "i,j,ijc->c", wy, wx, field[np.ix_(iy, ix)], optimize=True
        )
    return output[..., 0] if scalar else output


def _conv_factor_point(field: Array, x: float, y: float, order: str) -> Array:
    """Evaluate one of the two exact Cartesian CONV factor orders."""

    height, width, channels = field.shape
    sx = np.float32(np.clip(x, 0.0, 1.0) * (width - 1))
    sy = np.float32(np.clip(y, 0.0, 1.0) * (height - 1))
    if order == "xy":
        lines = np.moveaxis(field, 1, 0).reshape(width, height * channels)
        horizontal = conv_evaluate_lines(
            lines, np.full(height * channels, sx, dtype=np.float32)
        ).reshape(height, channels)
        return np.asarray(conv_evaluate_profile(horizontal, np.array([sy]))[0])
    if order == "yx":
        lines = field.reshape(height, width * channels)
        vertical = conv_evaluate_lines(
            lines, np.full(width * channels, sy, dtype=np.float32)
        ).reshape(width, channels)
        return np.asarray(conv_evaluate_profile(vertical, np.array([sx]))[0])
    raise ValueError("factor order must be 'xy' or 'yx'")


def _sample_conv(values: Array, px: Array, py: Array) -> Array:
    field, scalar = _as_field(values)
    beta_nodes = nodal_current_geometry(field)[3]
    beta = _sample_q1(beta_nodes, px, py)
    output = np.empty(px.shape + (field.shape[2],), dtype=np.float64)
    for index in np.ndindex(px.shape):
        xy = _conv_factor_point(field, float(px[index]), float(py[index]), "xy")
        yx = _conv_factor_point(field, float(px[index]), float(py[index]), "yx")
        output[index] = (1.0 - beta[index]) * xy + beta[index] * yx
    return output[..., 0] if scalar else output


def direct_warp(
    values: Array,
    transform: ProjectiveMap,
    target_shape: tuple[int, int],
    *,
    method: str = "conv",
) -> Array:
    """Evaluate one reconstruction through one direct inverse map."""

    x, y = target_nodes(target_shape)
    px, py = transform.map(x, y)
    if method == "conv":
        return np.asarray(_sample_conv(values, px, py), dtype=np.float32)
    if method == "lanczos3":
        return np.asarray(_sample_lanczos(values, px, py, 3), dtype=np.float32)
    if method == "bilinear":
        return np.asarray(_sample_q1(values, px, py), dtype=np.float32)
    raise ValueError(f"unknown warp method {method!r}")


def sample_inverse_warp(
    values: Array,
    transform: ProjectiveMap,
    target_x: Array,
    target_y: Array,
    *,
    method: str="conv",
) -> Array:
    """Evaluate reconstruction at arbitrary target-space coordinates."""

    qx,qy=np.broadcast_arrays(
        np.asarray(target_x,dtype=np.float64),np.asarray(target_y,dtype=np.float64)
    )
    px,py=transform.map(qx,qy)
    if method=="conv":
        return np.asarray(_sample_conv(values,px,py),dtype=np.float32)
    if method=="lanczos3":
        return np.asarray(_sample_lanczos(values,px,py,3),dtype=np.float32)
    if method=="bilinear":
        return np.asarray(_sample_q1(values,px,py),dtype=np.float32)
    raise ValueError(f"unknown warp method {method!r}")


@lru_cache(maxsize=None)
def _gauss_rule(order: int) -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    # Normalized positive weights on [-1,1].
    return nodes.astype(np.float64), (0.5 * weights).astype(np.float64)


def positive_warped_basin_average(
    values: Array,
    transform: ProjectiveMap,
    target_shape: tuple[int, int],
    *,
    method: str = "conv",
    quadrature_order: int = 3,
) -> Array:
    """Average direct reconstruction over target Voronoi basins.

    Tensor Gauss--Legendre weights are strictly positive and sum to one.
    Hence every reported basin value is a convex combination of direct
    reconstruction values: constants and the admitted value range are
    preserved independently of quadrature accuracy.
    """

    target = tuple(map(int, target_shape))
    x_bounds = endpoint_basin_bounds(target[1])
    y_bounds = endpoint_basin_bounds(target[0])
    nodes, weights = _gauss_rule(int(quadrature_order))
    field, scalar = _as_field(values)
    output = np.zeros(target + (field.shape[2],), dtype=np.float64)
    for jy in range(target[0]):
        y_mid = 0.5 * (y_bounds[jy] + y_bounds[jy + 1])
        y_half = 0.5 * (y_bounds[jy + 1] - y_bounds[jy])
        qy = y_mid + y_half * nodes
        for ix in range(target[1]):
            x_mid = 0.5 * (x_bounds[ix] + x_bounds[ix + 1])
            x_half = 0.5 * (x_bounds[ix + 1] - x_bounds[ix])
            qx = x_mid + x_half * nodes
            yy, xx = np.meshgrid(qy, qx, indexing="ij")
            px, py = transform.map(xx, yy)
            if method == "conv":
                samples = _sample_conv(field, px, py)
            elif method == "lanczos3":
                samples = _sample_lanczos(field, px, py, 3)
            elif method == "bilinear":
                samples = _sample_q1(field, px, py)
            else:
                raise ValueError(f"unknown warp method {method!r}")
            output[jy, ix] = np.einsum(
                "i,j,ijc->c", weights, weights, samples, optimize=True
            )
    return output[..., 0].astype(np.float32) if scalar else output.astype(np.float32)


def adaptive_positive_warped_basin_average(
    values: Array,
    transform: ProjectiveMap,
    target_shape: tuple[int, int],
    *,
    method: str = "conv",
    tolerance: float = 2.0e-5,
    maximum_depth: int = 4,
) -> tuple[Array, dict[str, Array | int | float]]:
    """Adaptive positive target-space integration for projective footprints.

    A leaf uses the positive two-point tensor Gauss rule.  Its estimator is
    the difference from the equally weighted union of its four child rules.
    Accepted output always uses the children, so every coefficient remains
    positive and the full coefficient sum is exactly one up to floating-point
    summation.  The estimator is an implementation convergence diagnostic,
    not a formal quadrature-error bound.
    """

    target = tuple(map(int, target_shape))
    x_bounds = endpoint_basin_bounds(target[1])
    y_bounds = endpoint_basin_bounds(target[0])
    field, scalar = _as_field(values)
    nodes, weights = _gauss_rule(2)
    output = np.empty(target + (field.shape[2],), dtype=np.float64)
    estimator = np.empty(target, dtype=np.float64)
    depths = np.empty(target, dtype=np.int16)
    leaf_count = 0

    def sample_rectangle(x0: float, x1: float, y0: float, y1: float) -> Array:
        x = 0.5*(x0+x1) + 0.5*(x1-x0)*nodes
        y = 0.5*(y0+y1) + 0.5*(y1-y0)*nodes
        yy, xx = np.meshgrid(y, x, indexing="ij")
        px, py = transform.map(xx, yy)
        if method == "conv":
            samples = _sample_conv(field, px, py)
        elif method == "lanczos3":
            samples = _sample_lanczos(field, px, py, 3)
        elif method == "bilinear":
            samples = _sample_q1(field, px, py)
        else:
            raise ValueError(f"unknown warp method {method!r}")
        return np.einsum("i,j,ijc->c", weights, weights, samples, optimize=True)

    def integrate(
        x0: float, x1: float, y0: float, y1: float, depth: int
    ) -> tuple[Array, float, int]:
        nonlocal leaf_count
        coarse = sample_rectangle(x0, x1, y0, y1)
        xm, ym = 0.5*(x0+x1), 0.5*(y0+y1)
        boxes = ((x0,xm,y0,ym), (xm,x1,y0,ym),
                 (x0,xm,ym,y1), (xm,x1,ym,y1))
        child_values = [sample_rectangle(*box) for box in boxes]
        fine = 0.25 * sum(child_values)
        error = float(np.max(np.abs(fine-coarse)))
        if error <= tolerance or depth >= maximum_depth:
            leaf_count += 4
            return fine, error, depth
        values_out = []
        errors = []
        used_depth = depth
        for box in boxes:
            value, local_error, local_depth = integrate(*box, depth+1)
            values_out.append(value)
            errors.append(local_error)
            used_depth = max(used_depth, local_depth)
        return 0.25 * sum(values_out), max(errors), used_depth

    for row in range(target[0]):
        for column in range(target[1]):
            value, error, depth = integrate(
                x_bounds[column], x_bounds[column+1],
                y_bounds[row], y_bounds[row+1], 0,
            )
            output[row, column] = value
            estimator[row, column] = error
            depths[row, column] = depth
    result = output[..., 0] if scalar else output
    return result.astype(np.float32), {
        "estimator": estimator,
        "depth": depths,
        "leaf_count": int(leaf_count),
        "maximum_estimator": float(np.max(estimator)),
    }
