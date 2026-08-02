"""Exact kernels for BFFT-guided partition-of-unity image models.

The routines in this module deliberately operate on one *measured*
owner/runner assignment.  They do not enumerate candidate diagrams.  The
resulting block matrix is the renderer's own interaction graph: a block
``(i, j)`` exists exactly when cells ``i`` and ``j`` jointly explain a pixel.

The native extension supplies the hot accumulation and rendering loops when
available.  NumPy/Numba implementations are retained as a portable reference.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse

from ._core import (
    _check,
    _vision_assemble_normal,
    _vision_bucket_first_label,
    _vision_bucket_two_labels,
    _vision_curvature_population_f32,
    _vision_fast_march_first_label,
    _vision_fast_march_labels,
    _vision_metric_edge_costs_f32,
    _vision_prepare_continuous_metric,
    _vision_hard_affine_fit,
    _vision_hard_basis_refit,
    _vision_binary_dilation_cross_u8,
    _vision_render_affine,
    _vision_resize_bilinear_f64,
    _vision_scan_paired_offsets,
    _vision_scan_residual_ridges,
    _vision_separable_filter_f64,
    _vision_soft_support_diffuse,
    _vision_sobel_f64,
    _vision_support_forward,
    _vision_support_normal_apply,
    _vision_support_transpose,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(fn):  # pragma: no cover
    return fn


_compile = njit(cache=True, fastmath=False) if njit is not None else _identity


def _ptr(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def vision_backend():
    """Human-readable hot-kernel backend used by this installation."""
    native = (
        _vision_assemble_normal is not None and
        _vision_render_affine is not None and
        _vision_scan_residual_ridges is not None)
    return "native C++" if native else (
        "Numba" if njit is not None else "portable Python")


def separable_filter_native(
    fields, kernel_y, kernel_x, *, mirror_without_edge=False, threads=0,
):
    """Filter one contiguous CxHxW batch, or return ``None`` if unavailable."""
    if _vision_separable_filter_f64 is None:
        return None
    value = np.ascontiguousarray(fields, dtype=np.float64)
    ky = np.ascontiguousarray(kernel_y, dtype=np.float64)
    kx = np.ascontiguousarray(kernel_x, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("separable filter fields must have shape CxHxW")
    if (ky.ndim != 1 or kx.ndim != 1 or ky.size % 2 != 1 or
            kx.size % 2 != 1 or ky.size == 0 or kx.size == 0):
        raise ValueError("separable filter kernels must be nonempty and odd")
    channels, height, width = value.shape
    scratch = np.empty_like(value)
    output = np.empty_like(value)
    _check(_vision_separable_filter_f64(
        channels, height, width, ky.size, kx.size,
        int(bool(mirror_without_edge)), max(int(threads), 0),
        _ptr(value, ctypes.c_double), _ptr(ky, ctypes.c_double),
        _ptr(kx, ctypes.c_double), _ptr(scratch, ctypes.c_double),
        _ptr(output, ctypes.c_double),
    ), "bfft_vision_separable_filter_f64")
    return output


def resize_bilinear_native(fields, shape, *, threads=0):
    """Resize one contiguous CxHxW batch, or return ``None`` if unavailable."""
    if _vision_resize_bilinear_f64 is None:
        return None
    value = np.ascontiguousarray(fields, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("bilinear resize fields must have shape CxHxW")
    output_height, output_width = map(int, shape)
    if output_height < 1 or output_width < 1:
        raise ValueError("bilinear resize dimensions must be positive")
    channels, input_height, input_width = value.shape
    output = np.empty(
        (channels, output_height, output_width), dtype=np.float64)
    _check(_vision_resize_bilinear_f64(
        channels, input_height, input_width, output_height, output_width,
        max(int(threads), 0), _ptr(value, ctypes.c_double),
        _ptr(output, ctypes.c_double),
    ), "bfft_vision_resize_bilinear_f64")
    return output


def sobel_native(fields, *, threads=0):
    """Return normalized Sobel derivatives, or ``None`` if unavailable."""
    if _vision_sobel_f64 is None:
        return None
    value = np.ascontiguousarray(fields, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("Sobel fields must have shape CxHxW")
    channels, height, width = value.shape
    gradient_x = np.empty_like(value)
    gradient_y = np.empty_like(value)
    _check(_vision_sobel_f64(
        channels, height, width, max(int(threads), 0),
        _ptr(value, ctypes.c_double), _ptr(gradient_x, ctypes.c_double),
        _ptr(gradient_y, ctypes.c_double),
    ), "bfft_vision_sobel_f64")
    return gradient_x, gradient_y


def binary_dilation_cross_native(mask, iterations, *, threads=0):
    """Dilate one HxW mask, or return ``None`` for an older native library."""
    if _vision_binary_dilation_cross_u8 is None:
        return None
    value = np.ascontiguousarray(mask, dtype=np.bool_)
    if value.ndim != 2:
        raise ValueError("binary dilation mask must have shape HxW")
    height, width = value.shape
    scratch = np.empty_like(value)
    output = np.empty_like(value)
    _check(_vision_binary_dilation_cross_u8(
        height, width, max(int(iterations), 0), max(int(threads), 0),
        _ptr(value, ctypes.c_uint8), _ptr(scratch, ctypes.c_uint8),
        _ptr(output, ctypes.c_uint8),
    ), "bfft_vision_binary_dilation_cross_u8")
    return output


def bucket_two_labels_native(
    seed_pixel, reach, direction_costs, delta, span, shift,
):
    """Run the exact native two-label Dial walk, or return ``None``."""
    if _vision_bucket_two_labels is None:
        return None
    seeds = np.ascontiguousarray(seed_pixel, dtype=np.int64)
    radius = np.ascontiguousarray(reach, dtype=np.float64)
    costs = np.ascontiguousarray(direction_costs, dtype=np.float32)
    if costs.ndim != 3 or costs.shape[0] != 8:
        raise ValueError("direction costs must have shape 8xHxW")
    if seeds.ndim != 1 or radius.shape != seeds.shape:
        raise ValueError("seed pixels and reach must be equal-length vectors")
    height, width = costs.shape[1:]
    pixels = height * width
    owner = np.empty(pixels, dtype=np.int32)
    runner = np.empty(pixels, dtype=np.int32)
    distance = np.empty(pixels, dtype=np.float64)
    second_distance = np.empty(pixels, dtype=np.float64)
    parent = np.empty(pixels, dtype=np.int32)
    pushes = ctypes.c_size_t()
    _check(_vision_bucket_two_labels(
        height, width, seeds.size,
        _ptr(seeds, ctypes.c_int64), _ptr(radius, ctypes.c_double),
        _ptr(costs, ctypes.c_float), float(delta), int(span), float(shift),
        _ptr(owner, ctypes.c_int32), _ptr(runner, ctypes.c_int32),
        _ptr(distance, ctypes.c_double),
        _ptr(second_distance, ctypes.c_double),
        _ptr(parent, ctypes.c_int32), ctypes.byref(pushes),
    ), "bfft_vision_bucket_two_labels")
    return owner, runner, distance, second_distance, parent, int(pushes.value)


def curvature_population_native(
    precision_xx,
    precision_xy,
    precision_yy,
    base_measure,
    base_implied_cells,
):
    """Return the native curvature-density fields, or ``None`` if unavailable."""
    if _vision_curvature_population_f32 is None:
        return None
    qxx = np.ascontiguousarray(precision_xx, dtype=np.float32)
    qxy = np.ascontiguousarray(precision_xy, dtype=np.float32)
    qyy = np.ascontiguousarray(precision_yy, dtype=np.float32)
    measure = np.ascontiguousarray(base_measure, dtype=np.float32)
    if not (qxx.ndim == 2 and qxx.shape == qxy.shape == qyy.shape ==
            measure.shape):
        raise ValueError("curvature population fields must share one 2-D shape")
    height, width = qxx.shape
    corrected = np.empty_like(measure)
    curvature = np.empty_like(measure)
    sagitta = np.empty_like(measure)
    factor = np.empty_like(measure)
    implied = ctypes.c_double()
    _check(_vision_curvature_population_f32(
        height,
        width,
        _ptr(qxx, ctypes.c_float),
        _ptr(qxy, ctypes.c_float),
        _ptr(qyy, ctypes.c_float),
        _ptr(measure, ctypes.c_float),
        float(base_implied_cells),
        _ptr(corrected, ctypes.c_float),
        _ptr(curvature, ctypes.c_float),
        _ptr(sagitta, ctypes.c_float),
        _ptr(factor, ctypes.c_float),
        ctypes.byref(implied),
    ), "bfft_vision_curvature_population_f32")
    return {
        "measure": corrected,
        "director_curvature": curvature,
        "curvature_sagitta_ratio": sagitta,
        "curvature_population_factor": factor,
        "implied_cells": float(implied.value),
    }


def soft_support_diffuse_native(
    field, conductance, passes, coupling=0.8, threads=0,
):
    """Return the native soft field, or ``None`` for an older library."""
    if _vision_soft_support_diffuse is None:
        return None
    value = np.ascontiguousarray(field, dtype=np.float64)
    scalar = value.ndim == 2
    if scalar:
        value = np.ascontiguousarray(value[..., None])
    if value.ndim != 3:
        raise ValueError("soft-support field must have shape HxW or HxWxC")
    height, width, channels = value.shape
    expected = {
        "horizontal": (height, width - 1),
        "vertical": (height - 1, width),
        "diagonal_down_right": (height - 1, width - 1),
        "diagonal_down_left": (height - 1, width - 1),
    }
    edge = {}
    for name, shape in expected.items():
        edge[name] = np.ascontiguousarray(
            conductance[name], dtype=np.float64)
        if edge[name].shape != shape:
            raise ValueError(
                f"{name} conductance has shape {edge[name].shape}, "
                f"expected {shape}")
    output = np.empty_like(value)
    scratch = np.empty_like(value)
    _check(_vision_soft_support_diffuse(
        height,
        width,
        channels,
        max(int(passes), 0),
        max(int(threads), 0),
        float(coupling),
        _ptr(value, ctypes.c_double),
        _ptr(edge["horizontal"], ctypes.c_double),
        _ptr(edge["vertical"], ctypes.c_double),
        _ptr(edge["diagonal_down_right"], ctypes.c_double),
        _ptr(edge["diagonal_down_left"], ctypes.c_double),
        _ptr(output, ctypes.c_double),
        _ptr(scratch, ctypes.c_double),
    ), "bfft_vision_soft_support_diffuse")
    return output[..., 0] if scalar else output


def prepare_continuous_metric_native(
    superbase, mxx, mxy, myy, *, consistency_limit=1.75,
):
    """Prepare one continuous-march stencil/CSR bundle natively."""
    if _vision_prepare_continuous_metric is None:
        return None
    vectors = np.ascontiguousarray(superbase, dtype=np.int32)
    a = np.ascontiguousarray(mxx, dtype=np.float64)
    b = np.ascontiguousarray(mxy, dtype=np.float64)
    c = np.ascontiguousarray(myy, dtype=np.float64)
    if not (a.ndim == 2 and a.shape == b.shape == c.shape):
        raise ValueError("continuous metric fields must share one 2-D shape")
    height, width = a.shape
    if vectors.shape != (height, width, 3, 2):
        raise ValueError("continuous superbases must have shape HxWx3x2")
    pixels = height * width
    directions = np.empty((height, width, 6, 2), dtype=np.int32)
    direction_costs = np.empty((height, width, 6), dtype=np.float64)
    direction_valid = np.empty((height, width, 6), dtype=np.bool_)
    cardinal_costs = np.empty((height, width, 4), dtype=np.float64)
    inverse_offset = np.empty(pixels + 1, dtype=np.int64)
    inverse_receiver = np.empty(10 * pixels, dtype=np.int32)
    inverse_count = ctypes.c_size_t()
    _check(_vision_prepare_continuous_metric(
        height,
        width,
        max(float(consistency_limit), 1.0),
        _ptr(vectors, ctypes.c_int32),
        _ptr(a, ctypes.c_double),
        _ptr(b, ctypes.c_double),
        _ptr(c, ctypes.c_double),
        _ptr(directions, ctypes.c_int32),
        _ptr(direction_costs, ctypes.c_double),
        _ptr(direction_valid, ctypes.c_uint8),
        _ptr(cardinal_costs, ctypes.c_double),
        _ptr(inverse_offset, ctypes.c_int64),
        inverse_receiver.size,
        _ptr(inverse_receiver, ctypes.c_int32),
        ctypes.byref(inverse_count),
    ), "bfft_vision_prepare_continuous_metric")
    return (
        directions,
        direction_costs,
        direction_valid,
        cardinal_costs,
        inverse_offset,
        inverse_receiver[:inverse_count.value],
    )


def fast_march_first_label_native(
    seed_pixel,
    seed_value,
    seed_label,
    seed_gradient_x,
    seed_gradient_y,
    prepared,
    *,
    source_gradients=True,
):
    """Run the exact native continuous first-arrival walk when available.

    ``None`` means the installed shared library predates this kernel. The
    returned tuple has the same contract as the Numba reference, including
    acceptance order and heap diagnostics.
    """
    if _vision_fast_march_first_label is None:
        return None
    seeds = np.ascontiguousarray(seed_pixel, dtype=np.int32)
    values = np.ascontiguousarray(seed_value, dtype=np.float64)
    labels = np.ascontiguousarray(seed_label, dtype=np.int32)
    seed_gx = np.ascontiguousarray(seed_gradient_x, dtype=np.float64)
    seed_gy = np.ascontiguousarray(seed_gradient_y, dtype=np.float64)
    if not (
        seeds.ndim == values.ndim == labels.ndim ==
        seed_gx.ndim == seed_gy.ndim == 1
    ):
        raise ValueError("fast-march seed arrays must be one-dimensional")
    seed_count = seeds.size
    if not (
        values.size == labels.size == seed_gx.size ==
        seed_gy.size == seed_count
    ):
        raise ValueError("fast-march seed arrays must have equal length")

    mxx = np.ascontiguousarray(prepared["mxx"], dtype=np.float64)
    mxy = np.ascontiguousarray(prepared["mxy"], dtype=np.float64)
    myy = np.ascontiguousarray(prepared["myy"], dtype=np.float64)
    if not (mxx.ndim == 2 and mxx.shape == mxy.shape == myy.shape):
        raise ValueError("fast-march metric fields must share one 2-D shape")
    height, width = mxx.shape
    pixels = height * width
    directions = np.ascontiguousarray(
        prepared["directions"], dtype=np.int32)
    direction_costs = np.ascontiguousarray(
        prepared["direction_costs"], dtype=np.float64)
    direction_valid = np.ascontiguousarray(
        prepared["direction_valid"], dtype=np.uint8)
    cardinal_costs = np.ascontiguousarray(
        prepared["cardinal_costs"], dtype=np.float64)
    inverse_offset = np.ascontiguousarray(
        prepared["inverse_offset"], dtype=np.int64)
    inverse_receiver = np.ascontiguousarray(
        prepared["inverse_receiver"], dtype=np.int32)
    if directions.shape != (height, width, 6, 2):
        raise ValueError("fast-march directions must have shape HxWx6x2")
    if (
        direction_costs.shape != (height, width, 6) or
        direction_valid.shape != (height, width, 6) or
        cardinal_costs.shape != (height, width, 4)
    ):
        raise ValueError("fast-march stencil costs have invalid shapes")
    if inverse_offset.shape != (pixels + 1,):
        raise ValueError("fast-march inverse offset has invalid shape")

    owner = np.empty(pixels, dtype=np.int32)
    distance = np.empty(pixels, dtype=np.float64)
    gradient_x = np.empty(pixels, dtype=np.float64)
    gradient_y = np.empty(pixels, dtype=np.float64)
    source_gradient_x = (
        np.empty(pixels, dtype=np.float64) if source_gradients else None)
    source_gradient_y = (
        np.empty(pixels, dtype=np.float64) if source_gradients else None)
    parent_first = np.empty(pixels, dtype=np.int32)
    parent_second = np.empty(pixels, dtype=np.int32)
    parent_fraction = np.empty(pixels, dtype=np.float64)
    acceptance_order = np.empty(pixels, dtype=np.int32)
    accepted_count = ctypes.c_size_t()
    push_count = ctypes.c_size_t()
    maximum_heap_size = ctypes.c_size_t()
    _check(_vision_fast_march_first_label(
        height,
        width,
        seed_count,
        _ptr(seeds, ctypes.c_int32),
        _ptr(values, ctypes.c_double),
        _ptr(labels, ctypes.c_int32),
        _ptr(seed_gx, ctypes.c_double),
        _ptr(seed_gy, ctypes.c_double),
        _ptr(directions, ctypes.c_int32),
        _ptr(direction_costs, ctypes.c_double),
        _ptr(direction_valid, ctypes.c_uint8),
        _ptr(cardinal_costs, ctypes.c_double),
        _ptr(inverse_offset, ctypes.c_int64),
        inverse_receiver.size,
        _ptr(inverse_receiver, ctypes.c_int32),
        _ptr(mxx, ctypes.c_double),
        _ptr(mxy, ctypes.c_double),
        _ptr(myy, ctypes.c_double),
        _ptr(owner, ctypes.c_int32),
        _ptr(distance, ctypes.c_double),
        _ptr(gradient_x, ctypes.c_double),
        _ptr(gradient_y, ctypes.c_double),
        (
            _ptr(source_gradient_x, ctypes.c_double)
            if source_gradient_x is not None else None
        ),
        (
            _ptr(source_gradient_y, ctypes.c_double)
            if source_gradient_y is not None else None
        ),
        _ptr(parent_first, ctypes.c_int32),
        _ptr(parent_second, ctypes.c_int32),
        _ptr(parent_fraction, ctypes.c_double),
        _ptr(acceptance_order, ctypes.c_int32),
        ctypes.byref(accepted_count),
        ctypes.byref(push_count),
        ctypes.byref(maximum_heap_size),
    ), "bfft_vision_fast_march_first_label")
    return (
        owner,
        distance,
        gradient_x,
        gradient_y,
        source_gradient_x,
        source_gradient_y,
        parent_first,
        parent_second,
        parent_fraction,
        acceptance_order[:accepted_count.value],
        int(push_count.value),
        int(maximum_heap_size.value),
    )


def fast_march_labels_native(
    seed_pixel,
    seed_value,
    seed_label,
    seed_gradient_x,
    seed_gradient_y,
    prepared,
):
    """Run the exact owner/distance-only continuous first-arrival walk."""
    if _vision_fast_march_labels is None:
        full = fast_march_first_label_native(
            seed_pixel,
            seed_value,
            seed_label,
            seed_gradient_x,
            seed_gradient_y,
            prepared,
        )
        if full is None:
            return None
        return full[0], full[1], full[10], full[11]

    seeds = np.ascontiguousarray(seed_pixel, dtype=np.int32)
    values = np.ascontiguousarray(seed_value, dtype=np.float64)
    labels = np.ascontiguousarray(seed_label, dtype=np.int32)
    seed_gx = np.ascontiguousarray(seed_gradient_x, dtype=np.float64)
    seed_gy = np.ascontiguousarray(seed_gradient_y, dtype=np.float64)
    if not (
        seeds.ndim == values.ndim == labels.ndim ==
        seed_gx.ndim == seed_gy.ndim == 1
    ):
        raise ValueError("fast-march seed arrays must be one-dimensional")
    seed_count = seeds.size
    if not (
        values.size == labels.size == seed_gx.size ==
        seed_gy.size == seed_count
    ):
        raise ValueError("fast-march seed arrays must have equal length")

    mxx = np.ascontiguousarray(prepared["mxx"], dtype=np.float64)
    mxy = np.ascontiguousarray(prepared["mxy"], dtype=np.float64)
    myy = np.ascontiguousarray(prepared["myy"], dtype=np.float64)
    height, width = mxx.shape
    directions = np.ascontiguousarray(
        prepared["directions"], dtype=np.int32)
    direction_costs = np.ascontiguousarray(
        prepared["direction_costs"], dtype=np.float64)
    direction_valid = np.ascontiguousarray(
        prepared["direction_valid"], dtype=np.uint8)
    cardinal_costs = np.ascontiguousarray(
        prepared["cardinal_costs"], dtype=np.float64)
    inverse_offset = np.ascontiguousarray(
        prepared["inverse_offset"], dtype=np.int64)
    inverse_receiver = np.ascontiguousarray(
        prepared["inverse_receiver"], dtype=np.int32)
    pixels = height * width
    owner = np.empty(pixels, dtype=np.int32)
    distance = np.empty(pixels, dtype=np.float64)
    push_count = ctypes.c_size_t()
    maximum_heap_size = ctypes.c_size_t()
    _check(_vision_fast_march_labels(
        height,
        width,
        seed_count,
        _ptr(seeds, ctypes.c_int32),
        _ptr(values, ctypes.c_double),
        _ptr(labels, ctypes.c_int32),
        _ptr(seed_gx, ctypes.c_double),
        _ptr(seed_gy, ctypes.c_double),
        _ptr(directions, ctypes.c_int32),
        _ptr(direction_costs, ctypes.c_double),
        _ptr(direction_valid, ctypes.c_uint8),
        _ptr(cardinal_costs, ctypes.c_double),
        _ptr(inverse_offset, ctypes.c_int64),
        inverse_receiver.size,
        _ptr(inverse_receiver, ctypes.c_int32),
        _ptr(mxx, ctypes.c_double),
        _ptr(mxy, ctypes.c_double),
        _ptr(myy, ctypes.c_double),
        _ptr(owner, ctypes.c_int32),
        _ptr(distance, ctypes.c_double),
        ctypes.byref(push_count),
        ctypes.byref(maximum_heap_size),
    ), "bfft_vision_fast_march_labels")
    return (
        owner,
        distance,
        int(push_count.value),
        int(maximum_heap_size.value),
    )


def metric_edge_costs_native(
    precision_xx,
    precision_xy,
    precision_yy,
    *,
    precision_gain,
    boundary_xx=None,
    boundary_xy=None,
    boundary_yy=None,
    boundary_gain=0.0,
):
    """Stream a frozen tensor metric into an eight-edge float32 cost stack."""
    if _vision_metric_edge_costs_f32 is None:
        return None
    qxx = np.ascontiguousarray(precision_xx, dtype=np.float32)
    qxy = np.ascontiguousarray(precision_xy, dtype=np.float32)
    qyy = np.ascontiguousarray(precision_yy, dtype=np.float32)
    if not (qxx.ndim == 2 and qxx.shape == qxy.shape == qyy.shape):
        raise ValueError("precision tensors must share one 2-D shape")
    use_boundary = float(boundary_gain) > 0.0
    if use_boundary:
        bxx = np.ascontiguousarray(boundary_xx, dtype=np.float32)
        bxy = np.ascontiguousarray(boundary_xy, dtype=np.float32)
        byy = np.ascontiguousarray(boundary_yy, dtype=np.float32)
        if not (bxx.shape == bxy.shape == byy.shape == qxx.shape):
            raise ValueError("boundary tensors must match precision tensors")
        bxx_pointer = _ptr(bxx, ctypes.c_float)
        bxy_pointer = _ptr(bxy, ctypes.c_float)
        byy_pointer = _ptr(byy, ctypes.c_float)
    else:
        bxx_pointer = None
        bxy_pointer = None
        byy_pointer = None
    height, width = qxx.shape
    costs = np.empty((8, height, width), dtype=np.float32)
    _check(_vision_metric_edge_costs_f32(
        height,
        width,
        _ptr(qxx, ctypes.c_float),
        _ptr(qxy, ctypes.c_float),
        _ptr(qyy, ctypes.c_float),
        bxx_pointer,
        bxy_pointer,
        byy_pointer,
        float(precision_gain),
        float(boundary_gain),
        _ptr(costs, ctypes.c_float),
    ), "bfft_vision_metric_edge_costs_f32")
    return costs


def bucket_first_label_native(
    seed_pixel,
    reach,
    costs,
    delta,
    span,
    shift,
):
    """Run the exact native monotone-bucket first-owner graph walk."""
    if _vision_bucket_first_label is None:
        return None
    seeds = np.ascontiguousarray(seed_pixel, dtype=np.int64)
    source_reach = np.ascontiguousarray(reach, dtype=np.float64)
    edge_costs = np.ascontiguousarray(costs, dtype=np.float32)
    if (
        seeds.ndim != 1 or source_reach.shape != seeds.shape or
        edge_costs.ndim != 3 or edge_costs.shape[0] != 8
    ):
        raise ValueError("invalid monotone-bucket transport arrays")
    height, width = edge_costs.shape[1:]
    pixels = height * width
    owner = np.empty(pixels, dtype=np.int32)
    distance = np.empty(pixels, dtype=np.float64)
    parent = np.empty(pixels, dtype=np.int32)
    push_count = ctypes.c_size_t()
    _check(_vision_bucket_first_label(
        height,
        width,
        seeds.size,
        _ptr(seeds, ctypes.c_int64),
        _ptr(source_reach, ctypes.c_double),
        _ptr(edge_costs, ctypes.c_float),
        float(delta),
        int(span),
        float(shift),
        _ptr(owner, ctypes.c_int32),
        _ptr(distance, ctypes.c_double),
        _ptr(parent, ctypes.c_int32),
        ctypes.byref(push_count),
    ), "bfft_vision_bucket_first_label")
    return owner, distance, parent, int(push_count.value)


def hard_affine_fit_native(labels, target):
    """Fit conditioned affine hard regions natively, or return ``None``."""
    if _vision_hard_affine_fit is None:
        return None
    label_field = np.ascontiguousarray(labels, dtype=np.int32)
    target_field = np.ascontiguousarray(target, dtype=np.float64)
    if label_field.ndim != 2:
        raise ValueError("hard-affine labels must be a two-dimensional field")
    height, width = label_field.shape
    if target_field.shape != (height, width, 3):
        raise ValueError("hard-affine target must have shape HxWx3")
    if label_field.size == 0 or np.min(label_field) < 0:
        raise ValueError("hard-affine labels must be nonnegative")
    cells = int(np.max(label_field)) + 1
    basis = np.empty((height, width, 3), dtype=np.float64)
    count = np.empty(cells, dtype=np.float64)
    radius = np.empty(cells, dtype=np.float64)
    centroid = np.empty((cells, 2), dtype=np.float64)
    reconstruction = np.empty_like(target_field)
    _check(_vision_hard_affine_fit(
        height,
        width,
        cells,
        _ptr(label_field, ctypes.c_int32),
        _ptr(target_field, ctypes.c_double),
        _ptr(basis, ctypes.c_double),
        _ptr(count, ctypes.c_double),
        _ptr(radius, ctypes.c_double),
        _ptr(centroid, ctypes.c_double),
        _ptr(reconstruction, ctypes.c_double),
    ), "bfft_vision_hard_affine_fit")
    return (
        label_field.ravel(),
        basis.reshape(-1, 3),
        count,
        radius,
        centroid,
        reconstruction,
    )


def hard_basis_refit_native(labels, design, target, count, radius):
    """Refit an augmented hard-region basis natively, or return ``None``."""
    if _vision_hard_basis_refit is None:
        return None
    label_vector = np.ascontiguousarray(labels, dtype=np.int32).ravel()
    design_matrix = np.ascontiguousarray(design, dtype=np.float64)
    target_matrix = np.ascontiguousarray(target, dtype=np.float64).reshape(
        -1, 3)
    count_vector = np.ascontiguousarray(count, dtype=np.float64)
    radius_vector = np.ascontiguousarray(radius, dtype=np.float64)
    if design_matrix.ndim != 2 or design_matrix.shape[0] != label_vector.size:
        raise ValueError("hard-basis design must have one row per label")
    if target_matrix.shape[0] != label_vector.size:
        raise ValueError("hard-basis target must have one row per label")
    if count_vector.ndim != 1 or radius_vector.shape != count_vector.shape:
        raise ValueError("hard-basis count and radius must be equal vectors")
    reconstruction = np.empty_like(target_matrix)
    _check(_vision_hard_basis_refit(
        label_vector.size,
        count_vector.size,
        design_matrix.shape[1],
        _ptr(label_vector, ctypes.c_int32),
        _ptr(design_matrix, ctypes.c_double),
        _ptr(target_matrix, ctypes.c_double),
        _ptr(count_vector, ctypes.c_double),
        _ptr(radius_vector, ctypes.c_double),
        _ptr(reconstruction, ctypes.c_double),
    ), "bfft_vision_hard_basis_refit")
    return reconstruction


def compact_support_operators(rows, sites, weight, basis_x, basis_y,
                              pixels, cells):
    """Return native compact-support ``A``, ``A.T`` and ``A.T A`` calls."""
    if (_vision_support_forward is None or
            _vision_support_transpose is None or
            _vision_support_normal_apply is None):
        return None
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    sites = np.ascontiguousarray(sites, dtype=np.int32)
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    basis_x = np.ascontiguousarray(basis_x, dtype=np.float64)
    basis_y = np.ascontiguousarray(basis_y, dtype=np.float64)
    sample_count = rows.size
    pixels = int(pixels)
    cells = int(cells)
    scratch = np.empty(pixels, dtype=np.float64)

    def forward(coefficient):
        coefficient = np.ascontiguousarray(coefficient, dtype=np.float64)
        output = np.empty(pixels, dtype=np.float64)
        _check(_vision_support_forward(
            sample_count, pixels, cells,
            _ptr(rows, ctypes.c_int32), _ptr(sites, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(basis_x, ctypes.c_double),
            _ptr(basis_y, ctypes.c_double),
            _ptr(coefficient, ctypes.c_double),
            _ptr(output, ctypes.c_double)),
            "bfft_vision_support_forward")
        return output

    def transpose(pixel):
        pixel = np.ascontiguousarray(pixel, dtype=np.float64)
        output = np.empty(3 * cells, dtype=np.float64)
        _check(_vision_support_transpose(
            sample_count, pixels, cells,
            _ptr(rows, ctypes.c_int32), _ptr(sites, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(basis_x, ctypes.c_double),
            _ptr(basis_y, ctypes.c_double), _ptr(pixel, ctypes.c_double),
            _ptr(output, ctypes.c_double)),
            "bfft_vision_support_transpose")
        return output

    def normal(coefficient):
        coefficient = np.ascontiguousarray(coefficient, dtype=np.float64)
        output = np.empty(3 * cells, dtype=np.float64)
        _check(_vision_support_normal_apply(
            sample_count, pixels, cells,
            _ptr(rows, ctypes.c_int32), _ptr(sites, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(basis_x, ctypes.c_double),
            _ptr(basis_y, ctypes.c_double),
            _ptr(coefficient, ctypes.c_double),
            _ptr(scratch, ctypes.c_double), _ptr(output, ctypes.c_double)),
            "bfft_vision_support_normal_apply")
        return output

    return forward, transpose, normal


@dataclass(frozen=True)
class CoownershipGraph:
    """Fixed block-CSR topology for one actual owner/runner assignment."""

    cells: int
    width: int
    block_row: np.ndarray
    block_col: np.ndarray
    position: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    diag_of: np.ndarray
    slot_forward: np.ndarray
    slot_reverse: np.ndarray

    @property
    def block_count(self) -> int:
        return int(self.block_row.size)

    @property
    def edge_count(self) -> int:
        return (self.block_count - self.cells) // 2


class SingleStageDecompositionObjective:
    """RGB + one-stage cartoon + one-stage texture reconstruction objective.

    The target split is computed exactly once.  Candidate reconstructions are
    decomposed on demand, which removes the invariant target decomposition
    from line searches without relying on an identity/checksum cache.
    """

    def __init__(
        self,
        target_rgb,
        *,
        lam=0.05,
        mu=40.0,
        passes=24,
        threads=4,
        space="oklab_lc",
        solver=1,
        target_lab=None,
    ):
        from .effects import meyer_channels

        self.target_rgb = np.ascontiguousarray(
            np.clip(target_rgb, 0.0, 1.0), dtype=np.float64)
        self.lam = float(lam)
        self.mu = float(mu)
        self.passes = int(passes)
        self.threads = int(threads)
        self.space = str(space)
        self.solver = int(solver)
        split = meyer_channels(
            self.target_rgb, space=self.space, lam=self.lam, mu=self.mu,
            passes=self.passes, threads=self.threads, solver=self.solver,
            working_lab=target_lab)
        scale = np.maximum(split.scale[None, None, :], 1e-12)
        self.target_cartoon = split.cartoon / scale
        self.target_texture = split.texture / scale
        self.last_residual_energy = None
        self.last_reconstruction_lab = None
        self.evaluation_count = 0
        self.restore_count = 0

    def capture_state(self):
        """Return the immutable residual state of the latest evaluation.

        Candidate selection often needs to restore a previously scored
        reconstruction.  Its scalar record is already retained by the
        caller; retaining this array reference avoids decomposing the same
        reconstruction again merely to restore ``last_residual_energy``.
        """

        return (
            self.last_residual_energy,
            self.last_reconstruction_lab,
        )

    def restore_state(self, state):
        """Restore a state returned by :meth:`capture_state` exactly."""

        if state is None or state[0] is None:
            raise ValueError("cannot restore an unevaluated objective state")
        (
            self.last_residual_energy,
            self.last_reconstruction_lab,
        ) = state
        self.restore_count += 1

    def evaluate(self, reconstruction_rgb):
        """Return the three MSE terms and their equally weighted sum."""
        from .effects import meyer_channels, srgb_to_lab

        self.evaluation_count += 1
        reconstruction = np.ascontiguousarray(
            np.clip(reconstruction_rgb, 0.0, 1.0), dtype=np.float64)
        if reconstruction.shape != self.target_rgb.shape:
            raise ValueError("reconstruction shape differs from target")
        reconstruction_lab = srgb_to_lab(reconstruction)
        split = meyer_channels(
            reconstruction, space=self.space, lam=self.lam, mu=self.mu,
            passes=self.passes, threads=self.threads, solver=self.solver,
            working_lab=reconstruction_lab)
        scale = np.maximum(split.scale[None, None, :], 1e-12)
        cartoon = split.cartoon / scale
        texture = split.texture / scale
        rgb_error = np.square(self.target_rgb - reconstruction)
        cartoon_error = np.square(self.target_cartoon - cartoon)
        texture_error = np.square(self.target_texture - texture)
        rgb_mse = float(np.mean(rgb_error))
        cartoon_mse = float(np.mean(cartoon_error))
        texture_mse = float(np.mean(texture_error))
        self.last_residual_energy = (
            np.mean(rgb_error, axis=2)
            + np.mean(cartoon_error, axis=2)
            + np.mean(texture_error, axis=2)
        )
        self.last_reconstruction_lab = reconstruction_lab
        return {
            "rgb_mse": rgb_mse,
            "psnr": -10.0 * math.log10(max(rgb_mse, 1e-12)),
            "cartoon_mse": cartoon_mse,
            "texture_mse": texture_mse,
            "objective": rgb_mse + cartoon_mse + texture_mse,
        }


@_compile
def _fill_scalar_indices(block_col, row_offsets, indptr, indices,
                         cells, width):
    for cell in range(cells):
        start = row_offsets[cell]
        stop = row_offsets[cell + 1]
        for sub in range(width):
            cursor = indptr[width * cell + sub]
            for k in range(start, stop):
                column = width * block_col[k]
                for part in range(width):
                    indices[cursor + part] = column + part
                cursor += width


def coownership_graph(owner, other, valid, cells, width=3):
    """Construct the exact interaction topology of the current rendering.

    The topology may be shared between cartoon and texture fits only when
    they use this same current owner/runner assignment.  It is intentionally
    not cached across geometry updates or hypothetical candidate placements.
    """
    owner = np.ascontiguousarray(owner, dtype=np.int64)
    other = np.ascontiguousarray(other, dtype=np.int64)
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    cells = int(cells)
    width = int(width)
    visible = np.flatnonzero(valid)
    i = owner[visible]
    j = other[visible]
    low = np.minimum(i, j)
    high = np.maximum(i, j)
    keys = np.unique(low * cells + high)
    pair_a = keys // cells
    pair_b = keys % cells

    diagonal = np.arange(cells, dtype=np.int64)
    block_row = np.concatenate((diagonal, pair_a, pair_b))
    block_col = np.concatenate((diagonal, pair_b, pair_a))
    order = np.lexsort((block_col, block_row))
    block_row = np.ascontiguousarray(block_row[order])
    block_col = np.ascontiguousarray(block_col[order])
    relocated = np.empty(order.size, dtype=np.int64)
    relocated[order] = np.arange(order.size)
    edges = keys.size
    diag_of = np.ascontiguousarray(relocated[:cells])
    forward_of = relocated[cells:cells + edges]
    reverse_of = relocated[cells + edges:]

    counts = np.bincount(block_row, minlength=cells)
    row_offsets = np.zeros(cells + 1, dtype=np.int64)
    np.cumsum(counts, out=row_offsets[1:])
    position = np.ascontiguousarray(
        np.arange(block_row.size) - row_offsets[block_row])

    indptr = np.zeros(width * cells + 1, dtype=np.int64)
    np.cumsum(np.repeat(width * counts, width), out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    _fill_scalar_indices(
        block_col, row_offsets, indptr, indices, cells, width)

    slot_forward = np.full(owner.size, -1, dtype=np.int64)
    slot_reverse = np.full(owner.size, -1, dtype=np.int64)
    if visible.size:
        found = np.searchsorted(keys, low * cells + high)
        owner_is_low = i == low
        slot_forward[visible] = np.where(
            owner_is_low, forward_of[found], reverse_of[found])
        slot_reverse[visible] = np.where(
            owner_is_low, reverse_of[found], forward_of[found])
    return CoownershipGraph(
        cells, width, block_row, block_col, position, indptr, indices,
        diag_of, slot_forward, slot_reverse)


@_compile
def _accumulate_reference(owner, other, valid, w1, w2, first, second,
                          target, diag_of, slot_forward, slot_reverse,
                          blocks, rhs):
    width = first.shape[1]
    channels = target.shape[1]
    u = np.empty(width, dtype=np.float64)
    v = np.empty(width, dtype=np.float64)
    for p in range(owner.size):
        i = owner[p]
        if i < 0:
            continue
        for a in range(width):
            u[a] = w1[p] * first[p, a]
        diagonal = diag_of[i]
        for a in range(width):
            for b in range(width):
                blocks[diagonal, a, b] += u[a] * u[b]
            for channel in range(channels):
                rhs[i, a, channel] += u[a] * target[p, channel]
        if not valid[p]:
            continue
        j = other[p]
        for a in range(width):
            v[a] = w2[p] * second[p, a]
        diagonal = diag_of[j]
        for a in range(width):
            for b in range(width):
                blocks[diagonal, a, b] += v[a] * v[b]
            for channel in range(channels):
                rhs[j, a, channel] += v[a] * target[p, channel]
        forward = slot_forward[p]
        reverse = slot_reverse[p]
        for a in range(width):
            for b in range(width):
                blocks[forward, a, b] += u[a] * v[b]
                blocks[reverse, a, b] += v[a] * u[b]


@_compile
def _blocks_to_data(blocks, block_row, position, indptr, data, width):
    for k in range(blocks.shape[0]):
        cell = block_row[k]
        base = width * position[k]
        for a in range(width):
            cursor = indptr[width * cell + a] + base
            for b in range(width):
                data[cursor + b] = blocks[k, a, b]


def assemble_normal(owner, other, valid, w1, w2, first, second, target,
                    graph, regularization=None):
    """Fused exact normal assembly without materializing a design matrix."""
    w1 = np.ascontiguousarray(w1, dtype=np.float64)
    w2 = np.ascontiguousarray(w2, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("target must have shape (pixels, channels)")
    if first.shape != second.shape:
        raise ValueError("first and second basis arrays must have equal shape")
    if first.shape[1] != graph.width:
        raise ValueError("basis width does not match graph width")

    blocks = np.zeros(
        (graph.block_count, graph.width, graph.width), dtype=np.float64)
    rhs = np.zeros(
        (graph.cells, graph.width, target.shape[1]), dtype=np.float64)
    if _vision_assemble_normal is not None:
        owner32 = np.ascontiguousarray(owner, dtype=np.int32)
        other32 = np.ascontiguousarray(other, dtype=np.int32)
        valid8 = np.ascontiguousarray(valid, dtype=np.uint8)
        _check(_vision_assemble_normal(
            owner32.size, graph.cells, graph.width, graph.block_count,
            _ptr(owner32, ctypes.c_int32), _ptr(other32, ctypes.c_int32),
            _ptr(valid8, ctypes.c_uint8), _ptr(w1, ctypes.c_double),
            _ptr(w2, ctypes.c_double), _ptr(first, ctypes.c_double),
            _ptr(second, ctypes.c_double), _ptr(target, ctypes.c_double),
            _ptr(graph.diag_of, ctypes.c_int64),
            _ptr(graph.slot_forward, ctypes.c_int64),
            _ptr(graph.slot_reverse, ctypes.c_int64),
            _ptr(blocks, ctypes.c_double), _ptr(rhs, ctypes.c_double)),
            "bfft_vision_assemble_normal")
    else:
        owner64 = np.ascontiguousarray(owner, dtype=np.int64)
        other64 = np.ascontiguousarray(other, dtype=np.int64)
        valid_bool = np.ascontiguousarray(valid, dtype=np.bool_)
        _accumulate_reference(
            owner64, other64, valid_bool, w1, w2, first, second, target,
            graph.diag_of, graph.slot_forward, graph.slot_reverse, blocks, rhs)
    if regularization is not None:
        reg = np.asarray(regularization, dtype=np.float64)
        if reg.shape == (graph.width,):
            reg = np.tile(reg, graph.cells)
        if reg.shape != (graph.cells * graph.width,):
            raise ValueError("regularization must have width or cells*width values")
        for part in range(graph.width):
            blocks[graph.diag_of, part, part] += reg[part::graph.width]

    data = np.empty(graph.indices.size, dtype=np.float64)
    _blocks_to_data(
        blocks, graph.block_row, graph.position, graph.indptr, data,
        graph.width)
    matrix = sparse.csr_matrix(
        (data, graph.indices, graph.indptr),
        shape=(graph.cells * graph.width, graph.cells * graph.width))
    return matrix.tocsc(), rhs.reshape(
        graph.cells * graph.width, target.shape[1]), blocks


@_compile
def _render_reference(coeff, owner, other, valid, w1, w2, first, second,
                      pred_first, pred_second, field):
    width = coeff.shape[1]
    channels = coeff.shape[2]
    for p in range(owner.size):
        i = owner[p]
        j = other[p] if valid[p] else i
        for channel in range(channels):
            left = 0.0
            right = 0.0
            for a in range(width):
                left += coeff[i, a, channel] * first[p, a]
                right += coeff[j, a, channel] * second[p, a]
            pred_first[p, channel] = left
            pred_second[p, channel] = right
            field[p, channel] = w1[p] * left + w2[p] * right


def render_partition(coeff, owner, other, valid, w1, w2, first, second):
    """Render both cell predictions and their partition-of-unity blend."""
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    w1 = np.ascontiguousarray(w1, dtype=np.float64)
    w2 = np.ascontiguousarray(w2, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    pixel_count = np.asarray(owner).size
    shape = (pixel_count, coeff.shape[2])
    pred_first = np.empty(shape, dtype=np.float64)
    pred_second = np.empty(shape, dtype=np.float64)
    field = np.empty(shape, dtype=np.float64)
    if _vision_render_affine is not None:
        owner32 = np.ascontiguousarray(owner, dtype=np.int32)
        other32 = np.ascontiguousarray(
            np.where(valid, other, owner), dtype=np.int32)
        _check(_vision_render_affine(
            pixel_count, coeff.shape[0], coeff.shape[1],
            _ptr(owner32, ctypes.c_int32), _ptr(other32, ctypes.c_int32),
            _ptr(w1, ctypes.c_double), _ptr(w2, ctypes.c_double),
            _ptr(first, ctypes.c_double), _ptr(second, ctypes.c_double),
            _ptr(coeff, ctypes.c_double), _ptr(pred_first, ctypes.c_double),
            _ptr(pred_second, ctypes.c_double), _ptr(field, ctypes.c_double)),
            "bfft_vision_render_affine")
    else:
        owner64 = np.ascontiguousarray(owner, dtype=np.int64)
        other64 = np.ascontiguousarray(other, dtype=np.int64)
        valid_bool = np.ascontiguousarray(valid, dtype=np.bool_)
        _render_reference(
            coeff, owner64, other64, valid_bool, w1, w2, first, second,
            pred_first, pred_second, field)
    return field, pred_first, pred_second


@_compile
def _ridge_scan_reference(owner, weight, residual, dx, dy, cosines, sines,
                          spacing, cells, angles, bins, span,
                          channel_weights):
    stride = angles * bins * residual.shape[1]
    channels = residual.shape[1]
    accumulator = np.zeros(cells * stride, dtype=np.float64)
    mass = np.zeros(cells, dtype=np.float64)
    total = np.zeros((cells, channels), dtype=np.float64)
    scale = bins / (2.0 * span)
    for p in range(owner.size):
        cell = owner[p]
        if cell < 0:
            continue
        phi = weight[p]
        mass[cell] += phi
        for channel in range(channels):
            total[cell, channel] += phi * residual[p, channel]
        px = dx[p] / spacing
        py = dy[p] / spacing
        base = cell * stride
        for angle in range(angles):
            projection = px * cosines[angle] + py * sines[angle]
            index = int((projection + span) * scale)
            if index < 0:
                index = 0
            elif index >= bins:
                index = bins - 1
            slot = base + (angle * bins + index) * channels
            for channel in range(channels):
                accumulator[slot + channel] += (
                    phi * residual[p, channel])

    best_score = np.zeros(cells, dtype=np.float64)
    best_angle = np.zeros(cells, dtype=np.int64)
    best_bin = np.zeros(cells, dtype=np.int64)
    running = np.empty(channels, dtype=np.float64)
    for cell in range(cells):
        denominator = max(mass[cell], 1e-9)
        base = cell * stride
        seen = False
        for angle in range(angles):
            for channel in range(channels):
                running[channel] = 0.0
            for bin_index in range(bins):
                slot = base + (angle * bins + bin_index) * channels
                value = 0.0
                for channel in range(channels):
                    running[channel] += accumulator[slot + channel]
                    contrast = total[cell, channel] - 2.0 * running[channel]
                    value += channel_weights[channel] * contrast * contrast
                value /= denominator
                if not seen or value > best_score[cell]:
                    best_score[cell] = value
                    best_angle[cell] = angle
                    best_bin[cell] = bin_index
                    seen = True
    return best_score, best_angle, best_bin


def measure_residual_ridges(owner, weight, residual, dx, dy, spacing,
                            cells, angles=16, bins=41, span=2.5,
                            channel_weights=(1.0, 1.5, 1.5)):
    """Measure one bounded residual ridge per current cell in one image pass."""
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    residual = np.ascontiguousarray(residual, dtype=np.float64)
    dx = np.ascontiguousarray(dx, dtype=np.float64)
    dy = np.ascontiguousarray(dy, dtype=np.float64)
    channel_weights = np.ascontiguousarray(
        channel_weights, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, int(angles), endpoint=False)
    cosines = np.ascontiguousarray(np.cos(theta))
    sines = np.ascontiguousarray(np.sin(theta))
    if _vision_scan_residual_ridges is not None:
        owner32 = np.ascontiguousarray(owner, dtype=np.int32)
        score = np.empty(int(cells), dtype=np.float64)
        angle_index = np.empty(int(cells), dtype=np.int32)
        bin_index = np.empty(int(cells), dtype=np.int32)
        _check(_vision_scan_residual_ridges(
            owner32.size, int(cells), int(angles), int(bins),
            float(spacing), float(span), _ptr(owner32, ctypes.c_int32),
            _ptr(weight, ctypes.c_double), _ptr(residual, ctypes.c_double),
            _ptr(dx, ctypes.c_double), _ptr(dy, ctypes.c_double),
            _ptr(cosines, ctypes.c_double), _ptr(sines, ctypes.c_double),
            _ptr(channel_weights, ctypes.c_double),
            _ptr(score, ctypes.c_double), _ptr(angle_index, ctypes.c_int32),
            _ptr(bin_index, ctypes.c_int32)),
            "bfft_vision_scan_residual_ridges")
    else:
        owner64 = np.ascontiguousarray(owner, dtype=np.int64)
        score, angle_index, bin_index = _ridge_scan_reference(
            owner64, weight, residual, dx, dy, cosines, sines,
            float(spacing), int(cells), int(angles), int(bins), float(span),
            channel_weights)
    offset = ((bin_index + 0.5) / int(bins) * (2.0 * float(span))
              - float(span))
    return score, theta[angle_index], offset


def measure_paired_offsets(
    owner,
    weight,
    residual,
    projection,
    cells,
    bins=161,
    span=2.5,
    channel_weights=(1.0, 1.5, 1.5),
):
    """Measure one finite split offset along each cell's fixed coordinate."""

    owner = np.ascontiguousarray(owner, dtype=np.int32).ravel()
    weight = np.ascontiguousarray(weight, dtype=np.float64).ravel()
    residual = np.ascontiguousarray(residual, dtype=np.float64).reshape(-1, 3)
    projection = np.ascontiguousarray(
        projection, dtype=np.float64).ravel()
    channel_weights = np.ascontiguousarray(
        channel_weights, dtype=np.float64)
    cells = int(cells)
    bins = int(bins)
    if not (
        owner.size == weight.size == residual.shape[0] == projection.size
        and cells > 0
        and bins > 0
        and channel_weights.shape == (3,)
    ):
        raise ValueError("paired-offset arrays have inconsistent geometry")

    if _vision_scan_paired_offsets is not None:
        score = np.empty(cells, dtype=np.float64)
        bin_index = np.empty(cells, dtype=np.int32)
        _check(_vision_scan_paired_offsets(
            owner.size,
            cells,
            bins,
            float(span),
            _ptr(owner, ctypes.c_int32),
            _ptr(weight, ctypes.c_double),
            _ptr(residual, ctypes.c_double),
            _ptr(projection, ctypes.c_double),
            _ptr(channel_weights, ctypes.c_double),
            _ptr(score, ctypes.c_double),
            _ptr(bin_index, ctypes.c_int32),
        ), "bfft_vision_scan_paired_offsets")
    else:
        scale = bins / (2.0 * float(span))
        index = np.clip(
            ((projection + float(span)) * scale).astype(np.int64),
            0,
            bins - 1,
        )
        histogram = np.zeros((cells, bins, 3), dtype=np.float64)
        np.add.at(
            histogram,
            (owner, index),
            weight[:, None] * residual,
        )
        running = np.cumsum(histogram, axis=1)
        total = running[:, -1]
        mass = np.bincount(owner, weights=weight, minlength=cells)
        contrast = total[:, None, :] - 2.0 * running
        value = np.sum(
            channel_weights[None, None, :] * contrast * contrast,
            axis=2,
        ) / np.maximum(mass[:, None], 1e-9)
        bin_index = np.argmax(value, axis=1).astype(np.int32)
        score = value[np.arange(cells), bin_index]
    offset = (
        (bin_index.astype(np.float64) + 0.5)
        / bins
        * (2.0 * float(span))
        - float(span)
    )
    return score, offset
