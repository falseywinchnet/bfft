"""Regression checks for circle-valued closure transport consensus."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from closure_transport_consensus import (  # noqa: E402
    ClosureTransportConfig,
    connection_phase_seed,
    solve_closure_transport_consensus,
    synchronize_translation_gauge,
)
from poisson_orbit_demo import select_steps  # noqa: E402


def _exact_problem(grid: int = 24, radius: float = 6.0):
    rng = np.random.default_rng(113)
    image = rng.random((grid, grid))
    spectrum = np.fft.fft2(image)
    steps = select_steps(
        grid, grid, 16, full_circle_support=False)
    bispectrum = np.asarray([
        spectrum
        * spectrum[sy, sx]
        * np.conj(np.roll(np.roll(
            spectrum, -sy, axis=0), -sx, axis=1))
        for sy, sx in steps
    ])
    frequency = np.fft.fftfreq(grid) * grid
    support = (
        np.hypot(frequency[:, None], frequency[None, :]) <= radius
    ).astype(np.float64)
    return spectrum, steps, bispectrum, support


def _gauged_truth(spectrum: np.ndarray) -> np.ndarray:
    grid = spectrum.shape[0]
    frequency = np.fft.fftfreq(grid) * grid
    phase = np.angle(spectrum)
    phase = (
        phase
        - frequency[:, None] * phase[1, 0]
        - frequency[None, :] * phase[0, 1]
    )
    return np.exp(1j * phase)


def test_connection_seed_recovers_exact_closure_without_radial_march():
    spectrum, steps, bispectrum, support = _exact_problem()

    phase, active, info = connection_phase_seed(
        bispectrum,
        np.ones_like(bispectrum.real),
        steps,
        support,
    )

    truth = _gauged_truth(spectrum)
    error = np.angle(phase[active] * np.conj(truth[active]))
    np.testing.assert_allclose(error, 0.0, atol=1e-7)
    assert info["nucleated_frequencies"] == int(np.count_nonzero(support))


def test_convolutional_gauge_sync_removes_an_integer_translation():
    spectrum, _, _, support = _exact_problem()
    reference = _gauged_truth(spectrum)
    grid = spectrum.shape[0]
    fy = np.fft.fftfreq(grid)[:, None]
    fx = np.fft.fftfreq(grid)[None, :]
    displaced = reference * np.exp(
        -2j * np.pi * (3 * fy - 2 * fx))

    aligned, shift = synchronize_translation_gauge(
        displaced, reference, support)

    np.testing.assert_allclose(
        aligned[support > 0.0],
        reference[support > 0.0],
        atol=1e-10,
    )
    assert shift == (3, -2)


def test_interleaved_transport_recovers_exact_multisource_orbit():
    spectrum, steps, bispectrum, support = _exact_problem()
    sources = 4
    bispectra = np.repeat(bispectrum[None, ...], sources, axis=0)
    coherence = np.ones_like(bispectra.real)
    source_support = np.repeat(
        support[None, ...], sources, axis=0)

    phase, publication, info, states = (
        solve_closure_transport_consensus(
            bispectra,
            coherence,
            steps,
            source_support,
            ClosureTransportConfig(sweeps=8),
        )
    )

    truth = _gauged_truth(spectrum)
    selected = support > 0.0
    error = np.angle(phase[selected] * np.conj(truth[selected]))
    np.testing.assert_allclose(error, 0.0, atol=2e-6)
    assert np.count_nonzero(publication) == np.count_nonzero(support)
    assert info["trace"][-1]["closure_residual"] < 1e-7
    assert info["trace"][-1]["consensus_residual"] < 1e-7
    assert len(states) == sources


def test_consensus_rejects_a_phase_incoherent_single_source_front():
    spectrum, steps, bispectrum, support = _exact_problem()
    sources = 3
    bispectra = np.repeat(bispectrum[None, ...], sources, axis=0)
    coherence = np.ones_like(bispectra.real)
    source_support = np.repeat(
        support[None, ...], sources, axis=0)
    frequency = np.fft.fftfreq(support.shape[0]) * support.shape[0]
    outer = np.hypot(
        frequency[:, None], frequency[None, :]) > 4.0
    source_support[1:, outer] = 0.0

    _, publication, _, _ = solve_closure_transport_consensus(
        bispectra,
        coherence,
        steps,
        source_support,
        ClosureTransportConfig(sweeps=4, minimum_sources=2),
    )

    assert np.all(publication[outer] == 0.0)
