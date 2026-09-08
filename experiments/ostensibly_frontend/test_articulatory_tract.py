from __future__ import annotations

import unittest

import numpy as np

from .articulatory_tract import (
    initial_all_trajectory,
    interpolate_trajectory,
    synthesize_articulatory_trajectory,
    tract_diameter_from_controls,
)


class ArticulatoryTractTests(unittest.TestCase):
    def test_tract_profile_is_positive_and_constriction_is_local(self) -> None:
        open_profile = tract_diameter_from_controls(17.7, 2.05, 0.82, 38.0, 1.45)
        lateral = tract_diameter_from_controls(14.0, 2.15, 1.2, 38.0, 0.78)
        self.assertEqual(open_profile.shape, (44,))
        self.assertTrue(np.all(lateral > 0.0))
        self.assertLess(float(lateral[38]), float(open_profile[38]))

    def test_trajectory_interpolation_preserves_flat_vowel(self) -> None:
        frames = initial_all_trajectory()
        values = interpolate_trajectory(frames, np.asarray((0.10, 0.20, 0.30)))
        np.testing.assert_allclose(values["tongue_index"], 17.7, atol=1e-12)
        np.testing.assert_allclose(values["tongue_diameter"], 2.05, atol=1e-12)

    def test_continuous_all_synthesis_is_finite_and_nonempty(self) -> None:
        result = synthesize_articulatory_trajectory(
            initial_all_trajectory(0.52), sample_rate=12_000, control_hop=12
        )
        self.assertEqual(result.samples.shape, (6240,))
        self.assertTrue(np.all(np.isfinite(result.samples)))
        self.assertGreater(float(np.max(np.abs(result.samples))), 0.5)
        self.assertEqual(result.tract_diameter.shape[1], 44)


if __name__ == "__main__":
    unittest.main()
