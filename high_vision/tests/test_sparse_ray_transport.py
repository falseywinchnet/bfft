"""Checks for tempered photon-ray transport."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from sparse_ray_transport import (  # noqa: E402
    RayBench,
    backproject,
    capture,
    shift_mask,
    shift_posterior,
    source_image,
)


def test_exact_ray_posterior_peaks_at_the_observed_shift():
    config = RayBench(
        grid=32,
        frames=1,
        photons_at_white=1000.0,
        shift_radius=4,
        temperature=0.05,
    )
    source = source_image(config.grid)
    shift = (3, -2)
    counts = np.rint(
        config.photons_at_white
        * np.roll(source, shift, axis=(0, 1))
    ).astype(np.uint16)[None, ...]

    posterior, _ = shift_posterior(
        counts,
        source,
        config,
        shift_mask(config.grid, config.shift_radius),
    )

    peak = np.unravel_index(np.argmax(posterior[0]), source.shape)
    signed = tuple(
        int(value if value <= config.grid // 2 else value - config.grid)
        for value in peak
    )
    assert signed == shift


def test_delta_ray_backprojection_inverts_the_shift():
    config = RayBench(
        grid=32,
        frames=1,
        photons_at_white=1.0,
        shift_radius=4,
    )
    source = source_image(config.grid)
    shift = (2, -3)
    counts = np.roll(source, shift, axis=(0, 1))[None, ...]
    posterior = np.zeros_like(counts)
    posterior[0, shift[0] % config.grid, shift[1] % config.grid] = 1.0

    recovered = backproject(counts, posterior, config)

    np.testing.assert_allclose(recovered, source, atol=1e-10)


def test_sparse_capture_is_reproducible():
    config = RayBench(grid=32, frames=8, shift_radius=3)
    source = source_image(config.grid)

    first = capture(
        source,
        config.frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed,
    )
    second = capture(
        source,
        config.frames,
        config.photons_at_white,
        config.background_photons,
        config.shift_radius,
        config.seed,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
