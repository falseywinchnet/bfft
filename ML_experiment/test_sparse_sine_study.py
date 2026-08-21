from __future__ import annotations

import math
import unittest

import torch

from ML_experiment.sparse_sine_study import (
    AcquisitionData, SupportWhitened, curve_metrics, curve_probe, make_model,
)
from ML_experiment.tasks import sparse_sine_1d
from ML_experiment.sparse_sine_witness_descent import InterleavedWitnessAtlas


class SparseSineStudyTests(unittest.TestCase):
    def test_sampling_measure_thins_by_ninety_six_times(self):
        task = sparse_sine_1d(0)
        self.assertEqual(task.segment_counts[:2], (768, 768))
        self.assertEqual(task.segment_counts[-1], 8)
        self.assertEqual(task.segment_counts[0] / task.segment_counts[-1], 96)
        self.assertEqual(len(task.tail_x), 5)
        self.assertAlmostEqual(float(task.x_test.min()), 0.0)
        self.assertAlmostEqual(float(task.x_test.max()), 1.0)

    def test_every_observed_period_retains_training_pairs(self):
        task = sparse_sine_1d(2)
        data = AcquisitionData(task)
        self.assertEqual(len(data.by_segment), 10)
        self.assertTrue(all(len(indices) >= 5 for indices in data.by_segment))
        self.assertLessEqual(float(data.sorted_x.max()), 1.0)

    def test_local_hermite_transport_recovers_the_observed_geometry(self):
        task = sparse_sine_1d(1)
        data = AcquisitionData(task)
        x = torch.linspace(float(data.sorted_x[0]), float(data.sorted_x[-1]), 5001)[:, None]
        index = torch.searchsorted(
            data.sorted_x[:, 0], x[:, 0], right=True
        ).clamp(1, len(data.sorted_x) - 1) - 1
        x0, x1 = data.sorted_x[index], data.sorted_x[index + 1]
        y0, y1 = data.sorted_y[index], data.sorted_y[index + 1]
        m0, m1 = data.sorted_first[index], data.sorted_first[index + 1]
        h = x1 - x0
        alpha = (x - x0) / h
        a2, a3 = alpha.square(), alpha.pow(3)
        prediction = (
            (2 * a3 - 3 * a2 + 1) * y0
            + (a3 - 2 * a2 + alpha) * h * m0
            + (-2 * a3 + 3 * a2) * y1
            + (a3 - a2) * h * m1
        )
        truth = (
            torch.sin(2 * math.pi * task.observed_periods * x + task.phase_offset)
            - task.target_mean
        ) / task.target_std
        r2 = 1.0 - float((prediction - truth).square().mean() / truth.square().mean())
        self.assertGreater(r2, 0.999)

    def test_segment_metrics_use_uniform_coordinate_evaluation(self):
        task = sparse_sine_1d(0)
        model = make_model("self_context", 16)
        metrics = curve_metrics(model, task)
        self.assertEqual(len(metrics["segment_metrics"]), 10)
        self.assertEqual(
            [row["count"] for row in metrics["segment_metrics"]],
            list(task.segment_counts),
        )

    def test_support_sampling_tracks_coordinate_measure(self):
        task = sparse_sine_1d(0)
        data = AcquisitionData(task)
        generator = torch.Generator().manual_seed(77)
        index = data.support(50000, generator)
        segment = data.segment[index]
        fractions = torch.bincount(segment, minlength=10).float() / len(index)
        # Each complete period owns approximately one tenth of the coordinate
        # interval despite the 96x variation in observation count.
        self.assertTrue(torch.all((fractions > 0.075) & (fractions < 0.125)))

    def test_stratified_support_covers_every_period_per_batch(self):
        task = sparse_sine_1d(0)
        data = AcquisitionData(task)
        index = data.stratified_support(
            1000, torch.Generator().manual_seed(91),
        )
        fractions = (
            torch.bincount(data.segment[index], minlength=10).float() / len(index)
        )
        self.assertTrue(torch.all((fractions > 0.085) & (fractions < 0.115)))

    def test_support_whitening_uses_coordinate_not_observation_moments(self):
        task = sparse_sine_1d(0)
        data = AcquisitionData(task)
        wrapped = SupportWhitened(torch.nn.Identity(), data)
        self.assertAlmostEqual(float(wrapped.input_mean), 0.5, delta=0.015)
        self.assertAlmostEqual(
            float(wrapped.input_scale), 1 / math.sqrt(12), delta=0.015,
        )

    def test_witness_atlas_is_disjoint_and_spans_support(self):
        data = AcquisitionData(sparse_sine_1d(0))
        atlas = InterleavedWitnessAtlas(data, cells=16, folds=4)
        seen = torch.zeros(len(data.x), dtype=torch.long)
        for fold in range(4):
            witness = atlas.witness(fold)
            seen[witness] += 1
            self.assertLess(float(data.x[witness].min()), 0.03)
            self.assertGreater(float(data.x[witness].max()), 0.94)
            draw = atlas.stratified_train(
                512, fold, torch.Generator().manual_seed(800 + fold)
            )
            self.assertFalse(torch.isin(draw, witness).any())
        self.assertTrue(torch.equal(seen, torch.ones_like(seen)))

    def test_probe_extends_diagnostic_not_scored_horizon(self):
        task = sparse_sine_1d(0)
        probe = curve_probe(make_model("self_context", 8), task, points=31)
        self.assertEqual(probe["observed_limit"], 1.0)
        self.assertEqual(probe["scored_extrapolation_limit"], 1.5)
        self.assertEqual(probe["diagnostic_limit"], 3.0)
        self.assertEqual(probe["x"][-1], 3.0)


if __name__ == "__main__":
    unittest.main()
