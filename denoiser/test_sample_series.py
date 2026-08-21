"""Controls for composited 1-D series and shared corruptions."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .sample_series import COMPONENTS, PRESETS, compose_series, corrupt
except ImportError:
    from sample_series import COMPONENTS, PRESETS, compose_series, corrupt


class SampleSeriesTests(unittest.TestCase):
    def test_every_component_is_independently_selectable(self):
        for component in COMPONENTS:
            _x, signal, fields = compose_series(128, (component,))
            self.assertEqual(set(fields), {component})
            self.assertEqual(signal.shape, (128,))
            self.assertTrue(np.all((signal >= 0.0) & (signal <= 1.0)))

    def test_presets_produce_distinct_composites(self):
        smooth = compose_series(256, PRESETS["smooth geometry"])[1]
        oscillatory = compose_series(256, PRESETS["oscillatory composite"])[1]
        self.assertGreater(float(np.max(np.abs(smooth - oscillatory))), 0.1)

    def test_corruption_is_deterministic_and_none_is_exact(self):
        clean = compose_series(128, PRESETS["mixed transport stress"])[1]
        np.testing.assert_array_equal(
            corrupt(clean, "none", amount=0.2, density=0.2, seed=3), clean)
        first = corrupt(
            clean, "mixed replacement + uniform",
            amount=0.2, density=0.1, seed=11)
        second = corrupt(
            clean, "mixed replacement + uniform",
            amount=0.2, density=0.1, seed=11)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()

