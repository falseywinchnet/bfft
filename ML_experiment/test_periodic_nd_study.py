from __future__ import annotations

import unittest

import torch

from ML_experiment.models import parameter_count
from ML_experiment.periodic_nd_study import ScalarChartBank, make_model


class PeriodicNDStudyTests(unittest.TestCase):
    def test_scalar_chart_variants_have_expected_shape(self):
        x = torch.randn(7, 8)
        for name in (
            "axis_chart_16",
            "orthogonal_chart_24",
            "orthogonal_identity_24",
            "qr_chart_24",
            "free_chart_8x24",
            "self_commuting_chart",
        ):
            with self.subTest(name=name):
                self.assertEqual(make_model(name, 8, 1, 38)(x).shape, (7, 1))

    def test_axis_chart_preserves_observed_coordinate_frame(self):
        model = ScalarChartBank(8, 1, rays=8, units=12, frame_mode="axes")
        torch.testing.assert_close(model.rays_matrix(), torch.eye(8))

    def test_learned_frames_are_orthogonal(self):
        for mode in ("orthogonal", "orthogonal_identity", "qr"):
            with self.subTest(mode=mode):
                model = ScalarChartBank(8, 1, rays=8, units=12, frame_mode=mode)
                frame = model.rays_matrix()
                torch.testing.assert_close(
                    frame @ frame.T, torch.eye(8), atol=2e-5, rtol=2e-5
                )

    def test_axis_chart_has_no_mixed_input_curvature(self):
        model = ScalarChartBank(4, 1, rays=4, units=8, frame_mode="axes")
        point = torch.randn(4, requires_grad=True)
        hessian = torch.autograd.functional.hessian(
            lambda value: model(value[None]).sum(), point
        )
        off_diagonal = hessian - torch.diag_embed(torch.diagonal(hessian))
        torch.testing.assert_close(off_diagonal, torch.zeros_like(off_diagonal))

    def test_small_chart_bank_is_below_existing_self_context_budget(self):
        chart = make_model("axis_chart_16", 8, 1, 38)
        self_context = make_model("self_context", 8, 1, 38)
        self.assertLess(parameter_count(chart), parameter_count(self_context))


if __name__ == "__main__":
    unittest.main()
