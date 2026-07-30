import numpy as np

from bfft.vision import measure_paired_offsets


def test_paired_offset_scan_matches_direct_cell_scan():
    rng = np.random.default_rng(0x50414952)
    pixels, cells, bins = 137, 7, 21
    span = 2.5
    owner = ((np.arange(pixels) * 13 + 4) % cells).astype(np.int32)
    weight = rng.uniform(0.1, 1.0, pixels)
    residual = rng.uniform(-1.0, 1.0, (pixels, 3))
    projection = rng.uniform(-4.0, 4.0, pixels)
    channel_weight = np.array([1.0, 1.5, 1.5])

    actual_score, actual_offset = measure_paired_offsets(
        owner,
        weight,
        residual,
        projection,
        cells,
        bins=bins,
        span=span,
        channel_weights=channel_weight,
    )

    expected_score = np.empty(cells)
    expected_bin = np.empty(cells, dtype=np.int32)
    scale = bins / (2.0 * span)
    for cell in range(cells):
        selected = np.flatnonzero(owner == cell)
        histogram = np.zeros((bins, 3))
        index = np.clip(
            ((projection[selected] + span) * scale).astype(np.int64),
            0,
            bins - 1,
        )
        np.add.at(
            histogram,
            index,
            weight[selected, None] * residual[selected],
        )
        running = np.cumsum(histogram, axis=0)
        total = running[-1]
        contrast = total[None] - 2.0 * running
        value = np.sum(channel_weight[None] * contrast * contrast, axis=1)
        value /= max(float(np.sum(weight[selected])), 1e-9)
        expected_bin[cell] = int(np.argmax(value))
        expected_score[cell] = value[expected_bin[cell]]
    expected_offset = (
        (expected_bin.astype(np.float64) + 0.5)
        / bins
        * (2.0 * span)
        - span
    )

    np.testing.assert_allclose(actual_score, expected_score, rtol=2e-15)
    np.testing.assert_array_equal(actual_offset, expected_offset)
