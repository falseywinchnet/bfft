import numpy as np

from bfft import _core
from port_needed import fast_image_ops as image_ops


def _reflect(index, size, whole_sample):
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index if whole_sample else -index - 1
        else:
            index = (
                2 * size - index - 2 if whole_sample
                else 2 * size - index - 1
            )
    return index


def _separable_reference(fields, kernel_y, kernel_x, whole_sample):
    channels, height, width = fields.shape
    temporary = np.empty_like(fields)
    output = np.empty_like(fields)
    radius_x = len(kernel_x) // 2
    radius_y = len(kernel_y) // 2
    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                temporary[channel, y, x] = sum(
                    kernel_x[offset + radius_x]
                    * fields[
                        channel,
                        y,
                        _reflect(x + offset, width, whole_sample),
                    ]
                    for offset in range(-radius_x, radius_x + 1)
                )
        for y in range(height):
            for x in range(width):
                output[channel, y, x] = sum(
                    kernel_y[offset + radius_y]
                    * temporary[
                        channel,
                        _reflect(y + offset, height, whole_sample),
                        x,
                    ]
                    for offset in range(-radius_y, radius_y + 1)
                )
    return output


def _bilinear_reference(fields, shape):
    channels, input_height, input_width = fields.shape
    output_height, output_width = shape
    output = np.empty((channels, output_height, output_width))
    scale_y = input_height / output_height
    scale_x = input_width / output_width
    for channel in range(channels):
        for y in range(output_height):
            source_y = (y + 0.5) * scale_y - 0.5
            raw_y0 = int(np.floor(source_y))
            fraction_y = source_y - raw_y0
            y0 = _reflect(raw_y0, input_height, True)
            y1 = _reflect(raw_y0 + 1, input_height, True)
            for x in range(output_width):
                source_x = (x + 0.5) * scale_x - 0.5
                raw_x0 = int(np.floor(source_x))
                fraction_x = source_x - raw_x0
                x0 = _reflect(raw_x0, input_width, True)
                x1 = _reflect(raw_x0 + 1, input_width, True)
                top = (
                    (1.0 - fraction_x) * fields[channel, y0, x0]
                    + fraction_x * fields[channel, y0, x1]
                )
                bottom = (
                    (1.0 - fraction_x) * fields[channel, y1, x0]
                    + fraction_x * fields[channel, y1, x1]
                )
                output[channel, y, x] = (
                    (1.0 - fraction_y) * top + fraction_y * bottom
                )
    return output


def test_native_v3_image_kernels_match_reference():
    if _core._vision_separable_filter_f64 is None:
        return
    rng = np.random.default_rng(20260802)
    fields = np.ascontiguousarray(rng.normal(size=(3, 13, 17)))
    kernel_y = image_ops._gaussian_kernel(0.85)
    kernel_x = image_ops._gaussian_kernel(1.35)

    for whole_sample in (False, True):
        result = image_ops.separable_filter_native(
            fields,
            kernel_y,
            kernel_x,
            mirror_without_edge=whole_sample,
            threads=3,
        )
        reference = _separable_reference(
            fields, kernel_y, kernel_x, whole_sample)
        np.testing.assert_allclose(result, reference, atol=4e-15, rtol=0.0)

    result = image_ops.resize_bilinear_native(fields, (19, 11), threads=3)
    reference = _bilinear_reference(fields, (19, 11))
    np.testing.assert_allclose(result, reference, atol=3e-15, rtol=0.0)

    gx, gy = image_ops.sobel_native(fields, threads=3)
    expected_x = np.empty_like(fields)
    expected_y = np.empty_like(fields)
    for channel in range(fields.shape[0]):
        for y in range(fields.shape[1]):
            above = max(y - 1, 0)
            below = min(y + 1, fields.shape[1] - 1)
            for x in range(fields.shape[2]):
                left = max(x - 1, 0)
                right = min(x + 1, fields.shape[2] - 1)
                expected_x[channel, y, x] = (
                    fields[channel, above, right]
                    + 2 * fields[channel, y, right]
                    + fields[channel, below, right]
                    - fields[channel, above, left]
                    - 2 * fields[channel, y, left]
                    - fields[channel, below, left]
                ) / 8
                expected_y[channel, y, x] = (
                    fields[channel, below, left]
                    + 2 * fields[channel, below, x]
                    + fields[channel, below, right]
                    - fields[channel, above, left]
                    - 2 * fields[channel, above, x]
                    - fields[channel, above, right]
                ) / 8
    np.testing.assert_array_equal(gx, expected_x)
    np.testing.assert_array_equal(gy, expected_y)

    mask = rng.random((23, 19)) > 0.86
    expected = mask.copy()
    for _ in range(4):
        expanded = expected.copy()
        expanded[1:] |= expected[:-1]
        expanded[:-1] |= expected[1:]
        expanded[:, 1:] |= expected[:, :-1]
        expanded[:, :-1] |= expected[:, 1:]
        expected = expanded
    result = image_ops.binary_dilation_cross_native(mask, 4, threads=3)
    np.testing.assert_array_equal(result, expected)


def test_public_v3_image_path_avoids_numba_when_native_is_present():
    if _core._vision_separable_filter_f64 is None:
        return
    fields = np.arange(3 * 12 * 14, dtype=np.float64).reshape(3, 12, 14)
    image_ops.gaussian_filter(fields, 1.0)
    image_ops.resize(np.moveaxis(fields, 0, -1), (9, 11))
    image_ops.sobel(fields)
    image_ops.binary_dilation(fields[0] > 20, 2)
    for fallback in (
        image_ops._gaussian_batch,
        image_ops._gaussian_batch_xy,
        image_ops._resize_bilinear_batch,
        image_ops._sobel_batch,
        image_ops._binary_dilation_cross,
    ):
        assert not getattr(fallback, "signatures", ())
