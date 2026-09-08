from __future__ import annotations

import unittest

import numpy as np

from experiments.antithetic_itd_enrichment import conv_itd_baseline
from experiments.conv_itd_zonotopic_enrichment import (
    enriched_decompose,
    gaussian_radial_modes,
    rademacher_generators,
)


class ZonotopicEnrichmentTests(unittest.TestCase):
    def test_radial_rule_has_standard_normal_moments(self) -> None:
        radius, weight = gaussian_radial_modes(5)
        self.assertAlmostEqual(float(np.sum(weight)), 1.0, places=14)
        self.assertAlmostEqual(float(weight@radius**2), 1.0, places=13)
        self.assertAlmostEqual(float(weight@radius**4), 3.0, places=12)

    def test_vertices_are_zero_mean_unit_rms(self) -> None:
        profile = rademacher_generators(128, 8, 17)
        np.testing.assert_allclose(np.mean(profile, axis=0), 0.0, atol=2e-15)
        np.testing.assert_allclose(np.sqrt(np.mean(profile*profile, axis=0)), 1.0, atol=2e-15)

    def test_conv_itd_is_sign_equivariant_on_noise(self) -> None:
        q = rademacher_generators(96, 1, 19)[:, 0]
        np.testing.assert_allclose(conv_itd_baseline(-q), -conv_itd_baseline(q), atol=2e-12, rtol=2e-12)

    def test_enriched_decomposition_closes(self) -> None:
        t = np.linspace(0.0, 1.0, 96)
        signal = .1*t + np.sin(2*np.pi*7*t) + .2*np.sin(2*np.pi*19*t)
        rotations, baseline, levels = enriched_decompose(
            signal,
            alpha=.08,
            probe_count=3,
            max_levels=5,
            transported=True,
            normalize_rotation=True,
            seed=23,
        )
        self.assertTrue(levels)
        reconstruction = baseline+np.sum(rotations, axis=0)
        np.testing.assert_allclose(reconstruction, signal, atol=3e-15, rtol=3e-15)


if __name__ == "__main__":
    unittest.main()
