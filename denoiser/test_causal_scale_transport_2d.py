"""Invariants for the continuous coarse-to-fine transport filtration."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.fft import dctn, idctn
from scipy import sparse

from .causal_scale_transport_2d import (
    _heat_state,
    _heat_tensor,
    _isotropic_selling_spectrum,
    _krylov_spectral_purity,
    _scale_phase_birth,
    _selling_jet_hadamard_pull,
    _spectral_measure_coordinates,
    _tensor_coordinates,
    causal_scale_transport_observation_2d,
)
from .continual_eikonal_noise_transport_2d import _continual_flux_laplacian


class CausalScaleTransport2DTests(unittest.TestCase):
    def test_dct_generator_is_exact_identity_selling_laplacian(self):
        rng = np.random.default_rng(311)
        field = rng.normal(size=(8, 9))
        metric = {
            "metric_xx": np.ones(field.shape),
            "metric_xy": np.zeros(field.shape),
            "metric_yy": np.ones(field.shape),
        }
        laplacian, _transport, _diagnostic = _continual_flux_laplacian(
            metric, np.ones(field.shape))
        spectrum = _isotropic_selling_spectrum(field.shape)
        coefficient = dctn(field, type=2, norm="ortho")
        spectral_action = idctn(
            spectrum * coefficient, type=2, norm="ortho")
        np.testing.assert_allclose(
            spectral_action,
            (laplacian @ field.reshape(-1)).reshape(field.shape),
            atol=2e-15,
            rtol=2e-14,
        )

    def test_heat_semigroup_preserves_constant_and_mean(self):
        spectrum = _isotropic_selling_spectrum((8, 8))
        constant = np.full((8, 8), 0.37)
        coefficient = dctn(constant, type=2, norm="ortho")
        for time in (0.0, 1.0, 17.0, 1000.0):
            state = _heat_state(coefficient, spectrum, time)
            np.testing.assert_allclose(state, constant, atol=3e-16, rtol=0.0)
            self.assertAlmostEqual(float(np.mean(state)), 0.37, places=15)

    def test_filtration_is_exactly_telescoping_and_bookkept(self):
        rng = np.random.default_rng(313)
        field = rng.normal(size=(8, 8))
        readout, residual, diagnostic = (
            causal_scale_transport_observation_2d(field))
        self.assertLess(diagnostic["decomposition_maximum_error"], 4e-15)
        self.assertLess(diagnostic["reconstruction_maximum_error"], 4e-15)
        np.testing.assert_allclose(
            readout + residual, field, atol=4e-15, rtol=0.0)
        self.assertLess(diagnostic["endpoint_distance_from_mean"], 2e-14)

    def test_nested_trace_refinement_changes_only_numerical_resolution(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.2 + 0.05 * xx + 0.1 * np.sin(0.7 * yy)
        generation_counts = []
        for refinement in (0, 1):
            readout, residual, diagnostic = (
                causal_scale_transport_observation_2d(
                    field, trace_refinement=refinement))
            generation_counts.append(len(diagnostic["generations"]))
            self.assertLess(
                diagnostic["decomposition_maximum_error"], 4e-15)
            np.testing.assert_allclose(
                readout + residual, field, atol=4e-15, rtol=0.0)
        self.assertEqual(generation_counts[1], 2 * generation_counts[0])

    def test_every_scale_coordinate_and_confidence_stays_in_its_cone(self):
        yy, xx = np.mgrid[:8, :8]
        field = 0.3 + 0.2 * np.sin(0.5 * xx) + 0.1 * (yy > 3)
        _readout, _residual, diagnostic = (
            causal_scale_transport_observation_2d(field))
        confidence = diagnostic["confidence_coarse_to_fine"]
        self.assertGreaterEqual(float(np.min(confidence)), 0.0)
        self.assertLessEqual(float(np.max(confidence)), 1.0)
        for key in (
            "hadamard_pull_coarse_to_fine",
            "spectral_phase_coarse_to_fine",
            "spectral_connection_coarse_to_fine",
            "transport_pull_coarse_to_fine",
            "pulled_phase_coarse_to_fine",
            "hadamard_pulled_coarse_to_fine",
            "phase_susceptibility_coarse_to_fine",
            "selling_jet_pull_coarse_to_fine",
            "unresolved_krylov_purity_coarse_to_fine",
        ):
            coordinate = np.asarray(diagnostic[key])
            self.assertGreaterEqual(float(np.min(coordinate)), 0.0)
            self.assertLessEqual(float(np.max(coordinate)), 1.0)
        for generation in diagnostic["generations"]:
            self.assertGreaterEqual(generation["directional_fraction"], 0.0)
            self.assertLessEqual(generation["directional_fraction"], 1.0)
            self.assertAlmostEqual(
                generation["directional_fraction"]
                + generation["isotropic_fraction"],
                1.0,
                places=14,
            )
            self.assertGreaterEqual(generation["mean_confidence"], 0.0)
            self.assertLessEqual(generation["mean_confidence"], 1.0)
            self.assertGreaterEqual(
                generation["normalized_rayleigh_eigenvalue"], -1e-14)
            self.assertLessEqual(
                generation["normalized_rayleigh_eigenvalue"], 2.0 + 1e-14)
            self.assertGreaterEqual(
                generation["spectral_refusal_distance"], 0.0)
            self.assertLessEqual(
                generation["spectral_refusal_distance"], 1.0)
            self.assertGreaterEqual(
                generation["spectral_susceptibility"], 0.0)
            self.assertLessEqual(
                generation["spectral_susceptibility"], 1.0)
            self.assertGreaterEqual(
                generation["resolvent_energy_survival"], 0.0)
            self.assertLessEqual(
                generation["resolvent_energy_survival"], 1.0 + 1e-14)

    def test_transport_tensor_distinguishes_direction_from_isotropy(self):
        yy, xx = np.mgrid[:16, :16]
        oriented = np.sin(2.0 * np.pi * xx / 8.0)
        rng = np.random.default_rng(317)
        isotropic = rng.normal(size=(16, 16))
        spectrum = _isotropic_selling_spectrum(oriented.shape)
        oriented_coordinate = _tensor_coordinates(
            _heat_tensor(oriented, spectrum, 4.0))
        isotropic_coordinate = _tensor_coordinates(
            _heat_tensor(isotropic, spectrum, 4.0))
        self.assertGreater(
            oriented_coordinate["directional_fraction"],
            isotropic_coordinate["directional_fraction"],
        )

    def test_hadamard_pull_is_spectral_rank_not_amplitude_equality(self):
        yy, xx = np.mgrid[:8, :8]
        base = 0.2 + 0.03 * xx - 0.01 * yy
        charts = np.stack((
            base, 2.0 * base,
            -3.0 * base, base,
            0.5 * base, -2.0 * base,
        ))
        zero = sparse.csr_matrix((64, 64), dtype=np.float64)
        phase, pull, spectral_phase, projection, diagnostic = _scale_phase_birth(
            charts, zero, 1.0)
        np.testing.assert_allclose(pull, 1.0, atol=3e-15, rtol=0.0)
        self.assertLess(float(np.min(phase)), 1.0)
        np.testing.assert_allclose(
            spectral_phase, phase, atol=3e-15, rtol=0.0)
        self.assertTrue(np.all(np.isfinite(projection)))
        self.assertAlmostEqual(
            diagnostic["mean_hadamard_effective_rank"], 1.0, places=14)

    def test_hadamard_pull_rejects_transport_independent_volume(self):
        yy, xx = np.mgrid[:8, :8]
        checker = np.where(((xx + yy) & 1) == 0, -1.0, 1.0)
        first = np.ones((8, 8))
        charts = np.stack((first, checker) * 3)
        pixels = 64
        averaging = np.full((pixels, pixels), 1.0 / pixels)
        mixing_laplacian = sparse.csr_matrix(np.eye(pixels) - averaging)
        _phase, pull, spectral_phase, projection, diagnostic = _scale_phase_birth(
            charts, mixing_laplacian, 1e12)
        self.assertLess(float(np.max(pull)), 2e-22)
        self.assertLess(float(np.max(spectral_phase)), 2e-22)
        self.assertLess(float(np.max(np.abs(projection))), 2e-12)
        self.assertAlmostEqual(
            diagnostic["mean_hadamard_effective_rank"], 2.0, places=12)

    def test_selling_jet_pull_is_a_positive_amplitude_invariant_law(self):
        yy, xx = np.mgrid[:8, :8]
        base = 0.2 + np.sin(0.7 * xx) + 0.3 * np.cos(0.9 * yy)
        charts = np.stack((
            base, 2.0 * base,
            -3.0 * base, base,
            0.5 * base, -2.0 * base,
        ))
        metric = {
            "metric_xx": np.ones(base.shape),
            "metric_xy": np.zeros(base.shape),
            "metric_yy": np.ones(base.shape),
        }
        laplacian, _transport, stencil = _continual_flux_laplacian(
            metric, np.ones(base.shape))
        pull, diagnostic = _selling_jet_hadamard_pull(
            charts,
            laplacian,
            1.0 / float(stencil["maximum_degree"]),
        )
        np.testing.assert_allclose(pull, 1.0, atol=3e-15, rtol=0.0)
        self.assertAlmostEqual(
            diagnostic["mean_selling_jet_surviving_volume"], 0.0, places=14)

    def test_krylov_purity_recognizes_both_transport_eigenvalue_phases(self):
        height, width = 8, 8
        metric = {
            "metric_xx": np.ones((height, width)),
            "metric_xy": np.zeros((height, width)),
            "metric_yy": np.ones((height, width)),
        }
        laplacian, _transport, stencil = _continual_flux_laplacian(
            metric, np.ones((height, width)))
        time = 1.0 / float(stencil["maximum_degree"])
        yy, xx = np.mgrid[:height, :width]
        for field in (
            np.ones((height, width)),
            np.cos(np.pi * (xx + 0.5) * (width - 1) / width)
            * np.cos(np.pi * (yy + 0.5) * (height - 1) / height),
        ):
            pull, diagnostic = _krylov_spectral_purity(
                field, laplacian, time)
            np.testing.assert_allclose(pull, 1.0, atol=2e-14, rtol=0.0)
            self.assertAlmostEqual(
                diagnostic["mean_krylov_spectral_impurity"], 0.0, places=13)

    def test_spectral_measure_dispersion_is_bounded_and_eigen_exact(self):
        height, width = 8, 8
        metric = {
            "metric_xx": np.ones((height, width)),
            "metric_xy": np.zeros((height, width)),
            "metric_yy": np.ones((height, width)),
        }
        laplacian, _transport, stencil = _continual_flux_laplacian(
            metric, np.ones((height, width)))
        yy, xx = np.mgrid[:height, :width]
        eigenfield = (
            np.cos(np.pi * (xx + 0.5) * 3.0 / width)
            * np.cos(np.pi * (yy + 0.5) * 5.0 / height)
        )
        eigen = _spectral_measure_coordinates(
            eigenfield, laplacian, float(stencil["maximum_degree"]))
        self.assertLess(eigen["normalized_spectral_dispersion"], 2e-14)
        rng = np.random.default_rng(319)
        mixture = _spectral_measure_coordinates(
            rng.normal(size=(height, width)),
            laplacian,
            float(stencil["maximum_degree"]),
        )
        self.assertGreaterEqual(
            mixture["normalized_spectral_dispersion"], 0.0)
        self.assertLessEqual(
            mixture["normalized_spectral_dispersion"], 1.0)


if __name__ == "__main__":
    unittest.main()
