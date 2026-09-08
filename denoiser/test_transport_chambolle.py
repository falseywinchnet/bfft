"""Structural tests for the transport-admissible Chambolle operator."""

from __future__ import annotations

import unittest

import numpy as np

from .transport_chambolle import (
    TransportChambolleResolution,
    _flux_readout,
    _forward_gradient,
    _project_flux_zonotope,
    denoise_transport_chambolle,
    transport_flux_body,
    transport_flux_bodies,
)


class TransportChambolleTests(unittest.TestCase):
    def test_component_batched_bodies_match_scalar_bodies(self):
        rng = np.random.default_rng(20260828)
        images = rng.normal(size=(4, 16, 19))
        expected = [transport_flux_body(image) for image in images]
        actual = transport_flux_bodies(images)
        for scalar, batched in zip(expected, actual):
            for field in scalar.__dataclass_fields__:
                np.testing.assert_allclose(
                    getattr(batched, field), getattr(scalar, field),
                    atol=2e-14, rtol=0.0, equal_nan=True)

    def test_gradient_and_flux_readout_are_adjoint(self):
        rng = np.random.default_rng(4)
        image = rng.normal(size=(12, 15))
        flux_y = rng.normal(size=image.shape)
        flux_x = rng.normal(size=image.shape)
        # Outward free-end fluxes are not represented by the gradient.
        flux_y[-1] = 0.0
        flux_x[:, -1] = 0.0
        gy, gx = _forward_gradient(image)
        left = float(np.sum(gy * flux_y + gx * flux_x))
        right = float(np.sum(image * _flux_readout(flux_y, flux_x)))
        self.assertAlmostEqual(left, right, places=12)

    def test_constant_field_is_exact_transport_identity(self):
        image = np.full((16, 19), 0.375)
        estimate, diagnostic = denoise_transport_chambolle(image)
        np.testing.assert_array_equal(estimate, image)
        self.assertEqual(diagnostic["status"], "transport identity")

    def test_flux_projection_respects_measured_zonotope(self):
        y, x = np.mgrid[:18, :20]
        image = 0.2 + 0.01 * x + 0.02 * y
        image[8, 9] += 0.7
        body = transport_flux_body(image)
        rng = np.random.default_rng(9)
        py, px = _project_flux_zonotope(
            rng.normal(size=image.shape), rng.normal(size=image.shape), body)
        normal = py * body.normal_y + px * body.normal_x
        tangent = -py * body.normal_x + px * body.normal_y
        self.assertLessEqual(
            float(np.max(np.abs(normal) - body.normal_capacity)), 1e-14)
        self.assertLessEqual(
            float(np.max(np.abs(tangent) - body.tangent_capacity)), 1e-14)

    def test_impulse_is_contracted_without_global_weight(self):
        image = np.full((24, 24), 0.3)
        image[12, 12] = 1.0
        estimate, diagnostic = denoise_transport_chambolle(image)
        self.assertLess(estimate[12, 12], image[12, 12])
        self.assertGreaterEqual(estimate[12, 12], 0.3 - 1e-12)
        self.assertGreater(diagnostic["mean_unsupported_amplitude"], 0.0)

    def test_dual_descent_is_monotone(self):
        rng = np.random.default_rng(12)
        y, x = np.mgrid[:20, :22]
        truth = (x > 10).astype(float) * 0.7 + 0.1
        observation = truth + 0.12 * rng.normal(size=truth.shape)
        _estimate, diagnostic = denoise_transport_chambolle(
            observation,
            TransportChambolleResolution(maximum_iterations=80),
        )
        trace = np.asarray(diagnostic["contraction_objective_trace"])
        self.assertLessEqual(float(np.max(np.diff(trace))), 1e-10)

    def test_output_and_residual_recompose_observation(self):
        rng = np.random.default_rng(31)
        image = rng.random((17, 21))
        estimate, diagnostic = denoise_transport_chambolle(image)
        residual = image - estimate
        np.testing.assert_allclose(estimate + residual, image, atol=0.0, rtol=0.0)
        self.assertEqual(diagnostic["recomposition_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
