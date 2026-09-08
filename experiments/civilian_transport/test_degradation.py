import unittest
import numpy as np
from .data import make_case, FAMILIES
from .degradation import fixture, acquisition_indices, COUNTS, A, FAMILY_A, TIMES, pair_region


class DegradationTests(unittest.TestCase):
    def test_nested_acquisition_fixed_endpoints(self):
        previous = set(range(65))
        for count in COUNTS:
            selected = acquisition_indices(count)
            self.assertEqual(len(selected), count)
            self.assertEqual((selected[0], selected[-1]), (0, 64))
            self.assertTrue(set(selected).issubset(previous))
            previous = set(selected)

    def test_matched_noise_and_truth(self):
        a = fixture('helix', 40001, .35, 65, True)
        b = fixture('helix', 40001, 1.4, 9, True)
        np.testing.assert_array_equal(a['truth'], b['truth'])
        np.testing.assert_allclose((a['observations']-a['truth']-a['bias_vector'])/.35,
                                   (b['observations']-b['truth']-b['bias_vector'])/1.4, atol=1e-13)
        self.assertAlmostEqual(np.linalg.norm(a['bias_vector']), .75)

    def test_fixed_time_contract(self):
        for family in FAMILIES:
            case = make_case(family, 45678, samples=len(TIMES), observation_times=TIMES)
            np.testing.assert_array_equal(case['times'], TIMES)
        with self.assertRaises(ValueError):
            make_case('line', 0, samples=3, observation_times=[0, 0, 1])

    def test_bounds_are_global_and_pair_radius_monotone(self):
        self.assertGreater(A, 5)
        self.assertLess(A, 5.1)
        self.assertEqual(A, max(FAMILY_A.values()))
        previous = 0
        for sigma in (.1, .35, .7, 1.4, 2.8):
            case = fixture('line', 40000, sigma, 65, False)
            _, r, best, _ = pair_region(case, sigma, False)
            self.assertGreater(r[best], previous)
            previous = r[best]
        previous = 0
        for count in COUNTS:
            case = fixture('line', 40000, .35, count, False)
            _, r, best, _ = pair_region(case, .35, False)
            self.assertGreaterEqual(r[best]+1e-12, previous)
            previous = r[best]


if __name__ == '__main__':
    unittest.main()
