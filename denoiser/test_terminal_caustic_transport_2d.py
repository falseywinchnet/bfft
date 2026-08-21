"""Invariants for the scalar terminal-caustic theorem probe."""

from __future__ import annotations

import unittest

import numpy as np

from .terminal_caustic_transport_2d import (
    phase_integrated_terminal_caustic_readout_2d,
    scalar_pushforward_collision_barycenter,
)


class TerminalCausticTransport2DTests(unittest.TestCase):
    def test_constant_is_exact(self):
        image = np.full((8, 8), 0.37)
        output, diagnostic = phase_integrated_terminal_caustic_readout_2d(
            image,
            angular_count=4,
            quantile_count=8,
            phase_count=2,
        )
        np.testing.assert_allclose(output, image, atol=2e-15, rtol=0.0)
        self.assertEqual(diagnostic["physical_parameters"], "none")

    def test_scalar_projection_is_branch_relabel_invariant(self):
        rng = np.random.default_rng(811)
        signal = rng.random((3, 4, 9))
        haar = rng.random(signal.shape)
        haar /= np.sum(haar, axis=-1, keepdims=True)
        action = rng.normal(size=signal.shape)
        order = 1.0 + 2.0 * rng.random(signal.shape[:2])
        expected, _ = scalar_pushforward_collision_barycenter(
            signal, haar, action, order)
        permutation = rng.permutation(signal.shape[-1])
        actual, _ = scalar_pushforward_collision_barycenter(
            signal[..., permutation],
            haar[..., permutation],
            action[..., permutation],
            order,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0.0)

    def test_order_one_is_terminal_path_mean(self):
        signal = np.broadcast_to(
            np.linspace(0.0, 1.0, 8), (2, 3, 8)).copy()
        haar = np.full(signal.shape, 1.0 / signal.shape[-1])
        action = np.zeros_like(signal)
        output, _ = scalar_pushforward_collision_barycenter(
            signal, haar, action, np.ones(signal.shape[:2]))
        np.testing.assert_allclose(output, 0.5, atol=2e-15, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
