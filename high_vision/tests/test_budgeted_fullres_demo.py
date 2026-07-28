"""Regression checks for the bounded full-resolution low-light pipeline."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from budgeted_fullres_demo import (  # noqa: E402
    Budget,
    block_sum,
    circular_cross_wiener,
    frame_stream,
    run,
    source_image,
)


def test_block_sum_preserves_detected_electrons():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    reduced = block_sum(image, 16)
    assert reduced.shape == (16, 16)
    np.testing.assert_allclose(np.sum(reduced), np.sum(image))


def test_sparse_camera_stream_is_reproducible():
    source = source_image(64)
    config = Budget(
        size=64, frames=3, photons_at_white=0.05,
        shift_radius=3, thumbnail=32, sampler="events", seed=12)
    first = list(frame_stream(source, config))
    second = list(frame_stream(source, config))
    for a, b in zip(first, second):
        assert a[0] == b[0]
        assert a[2] == b[2]
        np.testing.assert_array_equal(a[1], b[1])


def test_circular_cross_support_reduces_independent_noise():
    rng = np.random.default_rng(31)
    source = source_image(64)
    first = source + rng.normal(0.0, 0.18, source.shape)
    second = source + rng.normal(0.0, 0.18, source.shape)
    filtered, filtered_first, filtered_second, diagnostics = (
        circular_cross_wiener(first, second, ring_width=2))
    raw = 0.5 * (first + second)
    raw_noise = np.sqrt(np.mean((0.5 * (first - second)) ** 2))
    filtered_noise = np.sqrt(np.mean(
        (0.5 * (filtered_first - filtered_second)) ** 2))
    raw_error = np.mean((raw - source) ** 2)
    filtered_error = np.mean((filtered - source) ** 2)
    assert diagnostics["rings"] > 10
    assert filtered_noise < raw_noise * 0.5
    assert filtered_error < raw_error


def test_bounded_registration_improves_full_grid_fusion():
    result, _ = run(Budget(
        size=128,
        frames=128,
        photons_at_white=0.5,
        shift_radius=6,
        motion_step=1,
        registration_group=4,
        registration_rounds=2,
        thumbnail=64,
        registration_upsample=8,
        sampler="events",
        seed=3,
    ))
    scores = result["metrics"]
    assert (
        scores["bounded_registered"]["psnr_db"]
        > scores["unregistered"]["psnr_db"] + 1.0
    )
    assert result["registration_error_pixels"]["median"] <= 1.5
