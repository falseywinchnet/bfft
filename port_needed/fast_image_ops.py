"""Compiled image primitives used by the segmenting pipeline.

The Gaussian supports in this pipeline are at most 25 taps wide.  Batched
separable convolution is cheaper than padding every row and column into the
public power-of-two BFFT API, and it avoids Python crossings per scanline.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


def _identity(function):  # pragma: no cover
    return function


_compile = (
    njit(cache=True, parallel=True, fastmath=False)
    if njit is not None else _identity
)


@_compile
def _gaussian_batch(fields, kernel):
    channels, height, width = fields.shape
    radius = len(kernel) // 2
    temporary = np.empty_like(fields)
    output = np.empty_like(fields)
    for row in prange(channels * height):
        channel = row // height
        y = row - channel * height
        for x in range(width):
            value = 0.0
            for offset in range(-radius, radius + 1):
                source_x = x + offset
                while source_x < 0 or source_x >= width:
                    if source_x < 0:
                        source_x = -source_x - 1
                    else:
                        source_x = 2 * width - source_x - 1
                value += kernel[offset + radius] * fields[
                    channel, y, source_x]
            temporary[channel, y, x] = value
    for column in prange(channels * width):
        channel = column // width
        x = column - channel * width
        for y in range(height):
            value = 0.0
            for offset in range(-radius, radius + 1):
                source_y = y + offset
                while source_y < 0 or source_y >= height:
                    if source_y < 0:
                        source_y = -source_y - 1
                    else:
                        source_y = 2 * height - source_y - 1
                value += kernel[offset + radius] * temporary[
                    channel, source_y, x]
            output[channel, y, x] = value
    return output


@_compile
def _gaussian_batch_xy(fields, kernel_y, kernel_x):
    channels, height, width = fields.shape
    radius_y = len(kernel_y) // 2
    radius_x = len(kernel_x) // 2
    temporary = np.empty_like(fields)
    output = np.empty_like(fields)
    for row in prange(channels * height):
        channel = row // height
        y = row - channel * height
        for x in range(width):
            value = 0.0
            for offset in range(-radius_x, radius_x + 1):
                source_x = x + offset
                while source_x < 0 or source_x >= width:
                    if source_x < 0:
                        source_x = -source_x
                    else:
                        source_x = 2 * width - source_x - 2
                value += kernel_x[offset + radius_x] * fields[
                    channel, y, source_x]
            temporary[channel, y, x] = value
    for column in prange(channels * width):
        channel = column // width
        x = column - channel * width
        for y in range(height):
            value = 0.0
            for offset in range(-radius_y, radius_y + 1):
                source_y = y + offset
                while source_y < 0 or source_y >= height:
                    if source_y < 0:
                        source_y = -source_y
                    else:
                        source_y = 2 * height - source_y - 2
                value += kernel_y[offset + radius_y] * temporary[
                    channel, source_y, x]
            output[channel, y, x] = value
    return output


@_compile
def _resize_bilinear_batch(fields, output_height, output_width):
    channels, input_height, input_width = fields.shape
    output = np.empty(
        (channels, output_height, output_width), dtype=fields.dtype)
    scale_y = input_height / output_height
    scale_x = input_width / output_width
    for row in prange(channels * output_height):
        channel = row // output_height
        y = row - channel * output_height
        source_y = (y + 0.5) * scale_y - 0.5
        y0 = int(np.floor(source_y))
        fraction_y = source_y - y0
        y1 = y0 + 1
        if y0 < 0:
            y0 = -y0
        elif y0 >= input_height:
            y0 = 2 * input_height - y0 - 2
        if y1 < 0:
            y1 = -y1
        elif y1 >= input_height:
            y1 = 2 * input_height - y1 - 2
        for x in range(output_width):
            source_x = (x + 0.5) * scale_x - 0.5
            x0 = int(np.floor(source_x))
            fraction_x = source_x - x0
            x1 = x0 + 1
            if x0 < 0:
                x0 = -x0
            elif x0 >= input_width:
                x0 = 2 * input_width - x0 - 2
            if x1 < 0:
                x1 = -x1
            elif x1 >= input_width:
                x1 = 2 * input_width - x1 - 2
            top = (
                (1.0 - fraction_x) * fields[channel, y0, x0]
                + fraction_x * fields[channel, y0, x1]
            )
            bottom = (
                (1.0 - fraction_x) * fields[channel, y1, x0]
                + fraction_x * fields[channel, y1, x1]
            )
            output[channel, y, x] = (
                (1.0 - fraction_y) * top + fraction_y * bottom)
    return output


@_compile
def _sobel_batch(fields):
    channels, height, width = fields.shape
    gx = np.empty_like(fields)
    gy = np.empty_like(fields)
    for row in prange(channels * height):
        channel = row // height
        y = row - channel * height
        above = max(y - 1, 0)
        below = min(y + 1, height - 1)
        for x in range(width):
            left = max(x - 1, 0)
            right = min(x + 1, width - 1)
            gx[channel, y, x] = (
                fields[channel, above, right]
                + 2.0 * fields[channel, y, right]
                + fields[channel, below, right]
                - fields[channel, above, left]
                - 2.0 * fields[channel, y, left]
                - fields[channel, below, left]
            ) / 8.0
            gy[channel, y, x] = (
                fields[channel, below, left]
                + 2.0 * fields[channel, below, x]
                + fields[channel, below, right]
                - fields[channel, above, left]
                - 2.0 * fields[channel, above, x]
                - fields[channel, above, right]
            ) / 8.0
    return gx, gy


@_compile
def _binary_dilation_cross(mask, iterations):
    current = mask.copy()
    height, width = current.shape
    for _ in range(max(iterations, 0)):
        output = current.copy()
        for y in prange(height):
            for x in range(width):
                if current[y, x]:
                    continue
                output[y, x] = (
                    (y > 0 and current[y - 1, x])
                    or (y + 1 < height and current[y + 1, x])
                    or (x > 0 and current[y, x - 1])
                    or (x + 1 < width and current[y, x + 1])
                )
        current = output
    return current


def gaussian_filter(value: np.ndarray, sigma: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    scalar = array.ndim == 2
    fields = array[None] if scalar else array
    if fields.ndim != 3:
        raise ValueError("gaussian_filter expects HxW or CxHxW")
    smoothing = max(float(sigma), 0.0)
    if smoothing == 0.0:
        return array.copy()
    radius = int(4.0 * smoothing + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / smoothing))
    kernel /= np.sum(kernel)
    output = _gaussian_batch(
        np.ascontiguousarray(fields, dtype=np.float64),
        np.ascontiguousarray(kernel),
    )
    return output[0] if scalar else output


def _gaussian_kernel(sigma: float) -> np.ndarray:
    smoothing = max(float(sigma), 0.0)
    if smoothing == 0.0:
        return np.ones(1, dtype=np.float64)
    radius = int(4.0 * smoothing + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / smoothing))
    kernel /= np.sum(kernel)
    return np.ascontiguousarray(kernel)


def resize(
    value: np.ndarray,
    shape: tuple[int, int],
    *,
    order: int = 1,
    anti_aliasing: bool = True,
) -> np.ndarray:
    """Resize HxW or HxWxC data without SciPy.

    Pixel-centre coordinates and the prefilter match ``skimage.resize`` for
    linear interpolation.  The channel axis is never filtered or rescaled.
    """
    array = np.asarray(value, dtype=np.float64)
    scalar = array.ndim == 2
    if scalar:
        fields = array[None]
    elif array.ndim == 3:
        fields = np.moveaxis(array, -1, 0)
    else:
        raise ValueError("resize expects HxW or HxWxC")
    output_height, output_width = map(int, shape)
    if output_height < 1 or output_width < 1:
        raise ValueError("resize output dimensions must be positive")
    fields = np.ascontiguousarray(fields, dtype=np.float64)
    if order == 0:
        input_height, input_width = fields.shape[1:]
        y = np.minimum(
            ((np.arange(output_height) + 0.5)
             * input_height / output_height).astype(np.intp),
            input_height - 1,
        )
        x = np.minimum(
            ((np.arange(output_width) + 0.5)
             * input_width / output_width).astype(np.intp),
            input_width - 1,
        )
        result = fields[:, y[:, None], x[None, :]]
    elif order == 1:
        if anti_aliasing:
            input_height, input_width = fields.shape[1:]
            sigma_y = max((input_height / output_height - 1.0) / 2.0, 0.0)
            sigma_x = max((input_width / output_width - 1.0) / 2.0, 0.0)
            if sigma_y > 0.0 or sigma_x > 0.0:
                fields = _gaussian_batch_xy(
                    fields,
                    _gaussian_kernel(sigma_y),
                    _gaussian_kernel(sigma_x),
                )
        result = _resize_bilinear_batch(
            fields, output_height, output_width)
    else:
        raise ValueError("fast resize supports nearest or linear interpolation")
    return result[0] if scalar else np.moveaxis(result, 0, -1)


def sobel(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    scalar = array.ndim == 2
    fields = array[None] if scalar else array
    if fields.ndim != 3:
        raise ValueError("sobel expects HxW or CxHxW")
    gx, gy = _sobel_batch(np.ascontiguousarray(fields, dtype=np.float64))
    return (gx[0], gy[0]) if scalar else (gx, gy)


def binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    return _binary_dilation_cross(
        np.ascontiguousarray(mask, dtype=np.bool_),
        max(int(iterations), 0),
    )
