"""Focused checks for the realistic pre-mean orbit bootstrap."""

from pathlib import Path
from itertools import permutations
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from realistic_moonlight_bench import MoonlightCamera  # noqa: E402
from poisson_orbit_demo import select_steps  # noqa: E402
from realistic_orbit_bootstrap import (  # noqa: E402
    calibrated_thumbnail,
    calibrated_sublattice_thumbnails,
    distinct_gatherer_product,
    poisson_factorial_bispectrum,
    solve_phase_factorized,
    solve_phase_marching,
    split_half_radial_support,
    trimmed_complex_location,
)


def test_calibrated_thumbnail_removes_known_fixed_sensor_maps():
    camera = MoonlightCamera(size=64, canvas_margin=8, frames=2)
    y, x = np.mgrid[:camera.size, :camera.size]
    maps = {
        "prnu": 0.9 + 0.2 * x / (camera.size - 1),
        "dsnu": 0.02 * np.sin(y),
        "hot_rate": 0.01 * ((x + y) % 7 == 0),
    }
    radiance = 0.37 + 0.11 * np.sin(x / 5.0) * np.cos(y / 7.0)
    frame = radiance * maps["prnu"] + maps["dsnu"] + maps["hot_rate"]

    measured = calibrated_thumbnail(frame, maps, camera, grid=16)
    expected = radiance.reshape(16, 4, 16, 4).sum(axis=(1, 3)) / 16.0
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_sublattice_gatherers_partition_the_thumbnail_exactly():
    camera = MoonlightCamera(size=64, canvas_margin=8, frames=2)
    rng = np.random.default_rng(3)
    frame = rng.normal(size=(64, 64))
    maps = {
        "prnu": 0.9 + 0.2 * rng.random((64, 64)),
        "dsnu": 0.02 * rng.normal(size=(64, 64)),
        "hot_rate": 0.01 * rng.random((64, 64)),
    }

    ordinary = calibrated_thumbnail(frame, maps, camera, grid=16)
    gatherers = calibrated_sublattice_thumbnails(
        frame, maps, camera, grid=16)

    np.testing.assert_allclose(
        np.mean(gatherers, axis=0), ordinary, rtol=1e-12, atol=1e-12)


def test_distinct_gatherer_closed_form_matches_all_permutations():
    rng = np.random.default_rng(5)
    spectra = (
        rng.normal(size=(2, 4, 8, 8))
        + 1j * rng.normal(size=(2, 4, 8, 8))
    )
    step = (1, 2)
    measured = distinct_gatherer_product(spectra, step)
    shifted = np.roll(np.roll(
        spectra, -step[0], axis=2), -step[1], axis=3)
    explicit = np.zeros((2, 8, 8), dtype=np.complex128)
    for first, second, third in permutations(range(4), 3):
        explicit += (
            spectra[:, first]
            * spectra[:, second, step[0], step[1], None, None]
            * np.conj(shifted[:, third])
        )
    explicit /= 24.0

    np.testing.assert_allclose(measured, explicit, rtol=1e-12, atol=1e-12)
    sample_y = np.asarray([0, 1, 3, 6])
    sample_x = np.asarray([2, 7, 4, 1])
    bounded = distinct_gatherer_product(
        spectra, step, (sample_y, sample_x))
    np.testing.assert_allclose(
        bounded,
        explicit[:, sample_y, sample_x],
        rtol=1e-12,
        atol=1e-12,
    )


def test_split_half_support_rejects_inconsistent_biphase():
    halves = np.empty((2, 4, 64, 64), dtype=np.complex128)
    halves[0] = 1.0
    halves[1] = 1.0j

    _, agreement, significance, gain = split_half_radial_support(
        halves, grid=64)

    np.testing.assert_allclose(agreement, 0.0, atol=1e-15)
    np.testing.assert_allclose(significance, 0.0, atol=1e-13)
    assert gain[0] == 1.0
    np.testing.assert_array_equal(gain[1:], 0.0)


def test_factorial_bispectrum_removes_poisson_self_collisions():
    grid = 16
    block_area = 25
    rng = np.random.default_rng(31)
    mean_thumbnail = 0.1 + rng.random((grid, grid))
    spectrum = np.fft.fft2(mean_thumbnail)
    photon_dc = float(spectrum[0, 0].real)
    expected_power = (
        np.abs(spectrum) ** 2 + photon_dc / block_area)
    steps = select_steps(grid, grid, 10, full_circle_support=False)
    true_bispectrum = []
    raw_bispectrum = []
    for sy, sx in steps:
        shifted = np.roll(np.roll(
            spectrum, -sy, axis=0), -sx, axis=1)
        truth = spectrum * spectrum[sy, sx] * np.conj(shifted)
        collisions = (
            np.abs(spectrum) ** 2
            + np.abs(spectrum[sy, sx]) ** 2
            + np.abs(shifted) ** 2
        ) / block_area + photon_dc / block_area ** 2
        true_bispectrum.append(truth)
        raw_bispectrum.append(truth + collisions)

    corrected = poisson_factorial_bispectrum(
        np.asarray(raw_bispectrum),
        expected_power,
        photon_dc,
        steps,
        block_area,
    )

    np.testing.assert_allclose(
        corrected, np.asarray(true_bispectrum), rtol=1e-12, atol=1e-12)


def test_trimmed_complex_location_bounds_cubic_outliers():
    lanes = np.asarray([
        1.0 + 2.0j,
        1.1 + 1.9j,
        0.9 + 2.1j,
        1.0 + 2.0j,
        1.05 + 2.05j,
        0.95 + 1.95j,
        1e6 - 1e6j,
        -1e6 + 1e6j,
    ])

    fused = trimmed_complex_location(lanes, trim=2)

    np.testing.assert_allclose(fused, 1.0 + 2.0j, atol=0.04)


def test_split_half_support_accepts_repeated_biphase():
    halves = np.ones((2, 4, 64, 64), dtype=np.complex128)

    _, agreement, significance, gain = split_half_radial_support(
        halves, grid=64)

    np.testing.assert_array_equal(agreement, 1.0)
    assert significance[1] > 6.0
    assert np.all(gain == 1.0)


def test_marching_phase_fusor_recovers_an_exact_supported_orbit():
    grid = 32
    rng = np.random.default_rng(19)
    image = rng.random((grid, grid))
    spectrum = np.fft.fft2(image)
    source_phase = np.angle(spectrum)
    steps = select_steps(grid, grid, 20, full_circle_support=False)
    bispectrum = []
    for sy, sx in steps:
        bispectrum.append(
            spectrum
            * spectrum[sy, sx]
            * np.conj(np.roll(np.roll(
                spectrum, -sy, axis=0), -sx, axis=1)))
    bispectrum = np.asarray(bispectrum)
    coherence = np.ones_like(bispectrum.real)
    frequencies = np.fft.fftfreq(grid) * grid
    radius = np.hypot(frequencies[:, None], frequencies[None, :])
    support = (radius <= 8.0).astype(np.float64)

    recovered, info = solve_phase_marching(
        bispectrum, coherence, steps, support)

    fy = frequencies[:, None]
    fx = frequencies[None, :]
    gauged_source = (
        source_phase
        - fy * source_phase[1, 0]
        - fx * source_phase[0, 1]
    )
    selected = support > 0.0
    phase_error = np.angle(np.exp(
        1j * (recovered[selected] - gauged_source[selected])))
    np.testing.assert_allclose(phase_error, 0.0, atol=1e-10)
    assert info["optimizer_iterations"] == 0
    assert info["unsupported_phase_frequencies"] == 0

    reconciled, factor_info = solve_phase_factorized(
        bispectrum, coherence, steps, support)
    reconciled_error = np.angle(np.exp(
        1j * (reconciled[selected] - gauged_source[selected])))
    np.testing.assert_allclose(reconciled_error, 0.0, atol=1e-7)
    assert factor_info["optimizer_iterations"] == 0
    assert factor_info["linear_factorizations"] == 1
