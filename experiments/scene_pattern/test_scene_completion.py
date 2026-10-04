import unittest
from .scene_completion import consensus


class CompletionTests(unittest.TestCase):
    def test_two_paths_agree_on_actual_coordinates(self):
        q = consensus([(0, [10, 20]), (1, [10.4, 19.8])])
        self.assertEqual(q['sources'], [0, 1])
        self.assertAlmostEqual(q['xy'][0], 10.2)

    def test_conflicting_hypotheses_are_not_averaged_into_a_new_point(self):
        self.assertIsNone(consensus([(0, [0, 0]), (1, [1, 0]), (2, [9, 0]), (3, [10, 0])]))
        self.assertIsNone(consensus([(0, [0, 0]), (1, [3, 0])]))

    def test_single_path_and_duplicate_source_are_not_corroboration(self):
        self.assertIsNone(consensus([(0, [0, 0])]))
        with self.assertRaises(ValueError):
            consensus([(0, [0, 0]), (0, [0, 0])])

    def test_majority_rejects_outlying_path_without_changing_its_members(self):
        q = consensus([(0, [10, 20]), (1, [10.4, 19.8]), (2, [40, 50])])
        self.assertEqual(q['sources'], [0, 1])
        self.assertEqual(q['rejected_paths'], 1)


if __name__ == '__main__':
    unittest.main()
