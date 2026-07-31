import numpy as np

from experiments.meyer_harmonic_gball_relaxation import harmonic_grid_relax


def test_zero_diffusion_is_exact_frame_identity():
    source = np.random.default_rng(4).standard_normal((3, 73, 119))
    reconstruction, _ = harmonic_grid_relax(
        source,
        window_size=16,
        hop=8,
        diffusion=0.0,
    )
    assert np.max(np.abs(reconstruction - source)) < 2e-12


def test_measured_transport_preserves_off_bin_carrier():
    height, width = 128, 192
    y, x = np.mgrid[:height, :width]
    carrier = np.cos(
        2.0 * np.pi * (x / 7.3 + y / 19.0) + 0.37)
    source = np.stack((carrier, 0.7 * carrier, 0.2 * carrier))
    reconstruction, _ = harmonic_grid_relax(
        source,
        window_size=16,
        hop=8,
        diffusion=1.0,
        consistency_power=1.0,
    )
    core = (
        slice(None),
        slice(32, -32),
        slice(32, -32),
    )
    relative_error = (
        np.linalg.norm((reconstruction - source)[core])
        / np.linalg.norm(source[core])
    )
    assert relative_error < 0.003


def test_phase_inconsistent_carrier_is_contractively_relaxed():
    height, width = 128, 192
    y, x = np.mgrid[:height, :width]
    phase = np.random.default_rng(1).uniform(
        0.0, 2.0 * np.pi, (height // 16, width // 16))
    phase = np.repeat(np.repeat(phase, 16, axis=0), 16, axis=1)
    carrier = np.cos(
        2.0 * np.pi * (x / 7.3 + y / 19.0) + phase)
    source = np.stack((carrier, 0.7 * carrier, 0.2 * carrier))
    reconstruction, record = harmonic_grid_relax(
        source,
        window_size=16,
        hop=8,
        diffusion=1.0,
        consistency_power=1.0,
    )
    assert np.linalg.norm(reconstruction) < 0.97 * np.linalg.norm(source)
    assert record["removed_energy_fraction"] > 0.004
