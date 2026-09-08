from __future__ import annotations

import unittest

import numpy as np

from experiments.easu_reference import easu_nested_2x, easu_resize


class EasuReferenceTest(unittest.TestCase):
    def test_nested_shape_channels_and_constant(self) -> None:
        for source in (
            np.full((7, 9), 0.37),
            np.full((7, 9, 1), 0.37),
            np.full((7, 9, 3), 0.37),
        ):
            output = easu_nested_2x(source)
            self.assertEqual(output.shape[:2], (13, 17))
            self.assertEqual(output.ndim, source.ndim)
            np.testing.assert_allclose(output, 0.37, atol=2.0e-15)

    def test_mandatory_clamp_preserves_global_source_range(self) -> None:
        rng = np.random.default_rng(881)
        source = rng.uniform(size=(8, 10, 3))
        output = easu_nested_2x(source)
        self.assertGreaterEqual(float(np.min(output)), float(np.min(source)))
        self.assertLessEqual(float(np.max(output)), float(np.max(source)))

    def test_general_resize_has_requested_shape(self) -> None:
        source = np.zeros((8, 10, 3))
        self.assertEqual(easu_resize(source, 0.5).shape, (4, 5, 3))
        self.assertEqual(easu_resize(source, 2.0).shape, (16, 20, 3))


if __name__ == "__main__":
    unittest.main()
