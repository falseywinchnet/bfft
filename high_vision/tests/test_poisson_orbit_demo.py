"""Regression checks for the registration-free photon-orbit experiment."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from poisson_orbit_demo import Camera, run, select_steps  # noqa: E402


def _cross_bispectrum(image, steps):
    spectrum = np.fft.fft2(image)
    result = []
    for sy, sx in steps:
        result.append(
            spectrum
            * spectrum[sy, sx]
            * np.conj(np.roll(np.roll(
                spectrum, -sy, axis=0), -sx, axis=1)))
    return np.asarray(result)


def test_cross_bispectrum_ignores_cyclic_registration():
    rng = np.random.default_rng(4)
    image = rng.random((16, 16))
    steps = [(0, 1), (1, 0), (1, 2), (3, 1)]
    expected = _cross_bispectrum(image, steps)
    shifted = np.roll(image, (5, -3), axis=(0, 1))
    measured = _cross_bispectrum(shifted, steps)
    np.testing.assert_allclose(measured, expected, rtol=1e-11, atol=1e-10)


def test_full_circle_control_retains_antipodal_witnesses():
    half = select_steps(32, 32, 16, full_circle_support=False)
    full = select_steps(32, 32, 16, full_circle_support=True)
    assert (0, 31) not in half
    assert (0, 31) in full
    assert len(half) == len(full) == 16


def test_low_light_orbit_recovery_beats_unregistered_stacking():
    result, _ = run(Camera(
        size=24,
        frames=256,
        photons_at_white=2.0,
        bispectrum_steps=10,
        optimizer_iterations=80,
        seed=2,
        batch=64,
    ))
    scores = result["metrics"]
    assert scores["single"]["psnr_db"] < scores["unregistered_mean"]["psnr_db"]
    assert (
        scores["orbit_cross_bispectrum"]["psnr_db"]
        > scores["unregistered_mean"]["psnr_db"] + 6.0
    )
