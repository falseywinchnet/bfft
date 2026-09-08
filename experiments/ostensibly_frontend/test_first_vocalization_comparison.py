import unittest

import numpy as np

from .run_first_vocalization_comparison import _crossfade, duration_match


class FirstVocalizationComparisonTests(unittest.TestCase):
    def test_crossfade_has_expected_length_and_endpoints(self) -> None:
        left = np.ones(10)
        right = np.zeros(8)
        output = _crossfade((left, right), 4)
        self.assertEqual(output.shape, (14,))
        self.assertEqual(output[0], 1.0)
        self.assertEqual(output[-1], 0.0)

    def test_duration_match_preserves_endpoints_and_target_length(self) -> None:
        source = np.asarray([-0.5, 0.25, 1.0])
        output = duration_match(source, 9)
        self.assertEqual(output.shape, (9,))
        self.assertEqual(output[0], source[0])
        self.assertEqual(output[-1], source[-1])


if __name__ == "__main__":
    unittest.main()
