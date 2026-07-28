"""Checks for independent-orbit nucleation and gauge consensus."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from realistic_multisource_consensus import (  # noqa: E402
    consensus_radial_support,
    fit_phase_ramp,
    frequency_geometry,
)


def test_phase_ramp_alignment_removes_only_translation_gauge():
    grid = 32
    rng = np.random.default_rng(41)
    reference = rng.uniform(-np.pi, np.pi, size=(grid, grid))
    fy, fx, rings = frequency_geometry(grid)
    coefficient_y = 0.031
    coefficient_x = -0.024
    source = reference + coefficient_y * fy + coefficient_x * fx
    mask = rings <= 5

    aligned, shift = fit_phase_ramp(source, reference, mask)

    np.testing.assert_allclose(
        aligned[mask], reference[mask], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        shift,
        (
            -coefficient_y * grid / (2.0 * np.pi),
            -coefficient_x * grid / (2.0 * np.pi),
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_independent_weak_sources_can_publish_one_consensus_ring():
    significance = np.asarray([
        [8.0, 9.0, 8.0, 7.0, 1.44, -0.88],
        [8.0, 9.0, 8.0, 7.0, 1.83, 0.29],
        [8.0, 9.0, 8.0, 7.0, 3.51, 1.06],
        [8.0, 9.0, 8.0, 7.0, 2.59, 0.54],
    ])
    phase_consistency = np.asarray([1.0, 1.0, 1.0, 1.0, 0.86, 0.72])
    amplitude_consistency = np.asarray([1.0, 1.0, 1.0, 1.0, 0.74, 0.68])
    nucleus = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    combined, gain = consensus_radial_support(
        significance,
        phase_consistency,
        amplitude_consistency,
        nucleus,
    )

    assert combined[4] > 4.6
    assert gain[4] > 0.1
    assert gain[5] == 0.0
