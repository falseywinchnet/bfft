"""Fast owner-masked eikonal Lanczos image resampling.

This is a display/resampling primitive, not a segmentation stage.  It uses a
fixed, scale-aware Lanczos-2 footprint in the local tensor chart:

* the dominant tensor eigenvector is computed algebraically (no atan2);
* Lanczos weights come from a linear lookup table (no per-tap trig);
* RGB accumulation, DC normalization, and range clamping are fused;
* taps never cross structural owner IDs; and
* the output rows run in one Numba parallel region.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


def _identity(function):  # pragma: no cover
    return function


_compile_parallel = (
    njit(cache=True, parallel=True, fastmath=False)
    if njit is not None else _identity
)


def _lanczos2_table(size=4097):
    distance = np.linspace(0.0, 2.0, int(size), dtype=np.float64)
    weight = np.sinc(distance) * np.sinc(distance / 2.0)
    weight[-1] = 0.0
    return np.ascontiguousarray(weight)


LANCZOS2_TABLE = _lanczos2_table()


@_compile_parallel
def _resize_kernel(
    image,
    labels,
    tensor_xx,
    tensor_xy,
    tensor_yy,
    output_height,
    output_width,
    anisotropy,
    clamp_range,
    table,
):
    input_height, input_width, _channels = image.shape
    output = np.empty(
        (output_height, output_width, 3), dtype=np.float64)
    scale_y = output_height / input_height
    scale_x = output_width / input_width
    filter_scale = min(min(scale_x, scale_y), 1.0)
    maximum_stretch = 1.0 + max(anisotropy, 0.0)
    radius = int(math.ceil(2.0 * maximum_stretch / filter_scale))
    table_scale = (len(table) - 1) / 2.0

    for output_y in prange(output_height):
        source_y = (output_y + 0.5) / scale_y - 0.5
        center_y = int(math.floor(source_y + 0.5))
        center_y = min(max(center_y, 0), input_height - 1)
        for output_x in range(output_width):
            source_x = (output_x + 0.5) / scale_x - 0.5
            center_x = int(math.floor(source_x + 0.5))
            center_x = min(max(center_x, 0), input_width - 1)
            owner = labels[center_y, center_x]

            xx = max(tensor_xx[center_y, center_x], 0.0)
            xy = tensor_xy[center_y, center_x]
            yy = max(tensor_yy[center_y, center_x], 0.0)
            difference = xx - yy
            discriminant = math.sqrt(max(
                difference * difference + 4.0 * xy * xy, 0.0))
            trace = xx + yy
            coherence = min(max(
                discriminant / max(trace, 1e-15), 0.0), 1.0)

            # Dominant symmetric-2x2 eigenvector without angle recovery.
            eigenvalue = 0.5 * (trace + discriminant)
            if abs(xy) > 1e-15:
                normal_x = xy
                normal_y = eigenvalue - xx
                inverse_norm = 1.0 / math.sqrt(max(
                    normal_x * normal_x + normal_y * normal_y, 1e-30))
                normal_x *= inverse_norm
                normal_y *= inverse_norm
            elif xx >= yy:
                normal_x = 1.0
                normal_y = 0.0
            else:
                normal_x = 0.0
                normal_y = 1.0
            tangent_x = -normal_y
            tangent_y = normal_x
            stretch = 1.0 + anisotropy * coherence
            inverse_normal_scale = stretch
            inverse_tangent_scale = 1.0 / stretch

            total_weight = 0.0
            accum0 = 0.0
            accum1 = 0.0
            accum2 = 0.0
            minimum0 = 1e300
            minimum1 = 1e300
            minimum2 = 1e300
            maximum0 = -1e300
            maximum1 = -1e300
            maximum2 = -1e300
            for sample_y_raw in range(
                int(math.floor(source_y)) - radius,
                int(math.floor(source_y)) + radius + 1,
            ):
                sample_y = min(max(
                    sample_y_raw, 0), input_height - 1)
                dy = sample_y_raw - source_y
                for sample_x_raw in range(
                    int(math.floor(source_x)) - radius,
                    int(math.floor(source_x)) + radius + 1,
                ):
                    sample_x = min(max(
                        sample_x_raw, 0), input_width - 1)
                    if labels[sample_y, sample_x] != owner:
                        continue
                    dx = sample_x_raw - source_x
                    normal_distance = abs(filter_scale * (
                        dx * normal_x + dy * normal_y
                    ) * inverse_normal_scale)
                    tangent_distance = abs(filter_scale * (
                        dx * tangent_x + dy * tangent_y
                    ) * inverse_tangent_scale)
                    if normal_distance >= 2.0 or tangent_distance >= 2.0:
                        continue

                    normal_position = normal_distance * table_scale
                    normal_index = min(
                        int(normal_position), len(table) - 2)
                    normal_fraction = normal_position - normal_index
                    normal_weight = (
                        table[normal_index]
                        + normal_fraction * (
                            table[normal_index + 1]
                            - table[normal_index])
                    )
                    tangent_position = tangent_distance * table_scale
                    tangent_index = min(
                        int(tangent_position), len(table) - 2)
                    tangent_fraction = tangent_position - tangent_index
                    tangent_weight = (
                        table[tangent_index]
                        + tangent_fraction * (
                            table[tangent_index + 1]
                            - table[tangent_index])
                    )
                    weight = normal_weight * tangent_weight
                    total_weight += weight
                    value0 = image[sample_y, sample_x, 0]
                    value1 = image[sample_y, sample_x, 1]
                    value2 = image[sample_y, sample_x, 2]
                    accum0 += weight * value0
                    accum1 += weight * value1
                    accum2 += weight * value2
                    minimum0 = min(minimum0, value0)
                    minimum1 = min(minimum1, value1)
                    minimum2 = min(minimum2, value2)
                    maximum0 = max(maximum0, value0)
                    maximum1 = max(maximum1, value1)
                    maximum2 = max(maximum2, value2)

            if abs(total_weight) < 1e-12:
                output[output_y, output_x] = image[center_y, center_x]
            else:
                value0 = accum0 / total_weight
                value1 = accum1 / total_weight
                value2 = accum2 / total_weight
                if clamp_range:
                    value0 = min(max(value0, minimum0), maximum0)
                    value1 = min(max(value1, minimum1), maximum1)
                    value2 = min(max(value2, minimum2), maximum2)
                output[output_y, output_x, 0] = value0
                output[output_y, output_x, 1] = value1
                output[output_y, output_x, 2] = value2
    return output


def eikonal_lanczos_resize(
    image,
    output_shape,
    labels,
    tensor,
    *,
    anisotropy=0.5,
    clamp_range=True,
):
    """Resize RGB data through an owner-masked local eikonal chart."""
    value = np.ascontiguousarray(image, dtype=np.float64)
    owner = np.ascontiguousarray(labels, dtype=np.int32)
    xx, xy, yy = (
        np.ascontiguousarray(component, dtype=np.float64)
        for component in tensor
    )
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("eikonal Lanczos input must have shape HxWx3")
    if owner.shape != value.shape[:2]:
        raise ValueError("owner labels must match the input image")
    if xx.shape != owner.shape or xy.shape != owner.shape or yy.shape != owner.shape:
        raise ValueError("metric tensor fields must match the input image")
    output_height, output_width = map(int, output_shape)
    if output_height < 1 or output_width < 1:
        raise ValueError("output dimensions must be positive")
    if (output_height, output_width) == value.shape[:2]:
        return value.copy()
    return _resize_kernel(
        value,
        owner,
        xx,
        xy,
        yy,
        output_height,
        output_width,
        float(anisotropy),
        bool(clamp_range),
        LANCZOS2_TABLE,
    )
