from __future__ import annotations

import unittest

import torch

from ML_experiment.run_spiral_evidence_horizon import make_task, spiral_truth
from ML_experiment.run_spiral_capacity_sweep import CONFIGURATIONS, PROBE_CONFIGURATIONS


class SpiralEvidenceHorizonTests(unittest.TestCase):
    def test_capacity_sweep_separates_resource_axes(self):
        self.assertEqual(len(CONFIGURATIONS), len(set(CONFIGURATIONS)))
        self.assertIn((38, 1000), CONFIGURATIONS)
        self.assertIn((38, 2000), CONFIGURATIONS)
        self.assertIn((54, 500), CONFIGURATIONS)
        self.assertIn((76, 500), CONFIGURATIONS)
        self.assertTrue(PROBE_CONFIGURATIONS.issubset(set(CONFIGURATIONS)))

    def test_horizons_are_balanced_and_equally_withheld(self):
        for turns in (2, 4, 8):
            task = make_task(turns, seed=0, points_per_turn=30)
            self.assertEqual(len(task.tail_x), turns)
            self.assertEqual(len(task.tail_y), turns)
            observed_labels = torch.cat((task.y_train, task.y_val))
            self.assertEqual(float(observed_labels.float().mean()), 0.5)
            self.assertEqual(float(task.y_test.float().mean()), 0.5)
            self.assertEqual(len(task.x_test), 900 * turns)

    def test_analytic_truth_labels_both_noise_free_arms(self):
        for turns in (2, 4, 8):
            turn_coordinate = torch.linspace(0.0, 2.0 * turns, 257)
            radius = 0.1 + 0.45 * turn_coordinate
            theta = 0.55 + 2.0 * torch.pi * turn_coordinate
            arm0 = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), 1)
            arm1 = -arm0
            self.assertTrue(torch.equal(spiral_truth(arm0), torch.zeros(257, dtype=torch.long)))
            self.assertTrue(torch.equal(spiral_truth(arm1), torch.ones(257, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
