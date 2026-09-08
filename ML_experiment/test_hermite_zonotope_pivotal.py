import unittest

import torch

from ML_experiment.hermite_zonotope_pivotal import ObservedHermiteAtlas
from ML_experiment.models import parameter_count
from ML_experiment.odd_context_hybrids import make_hybrid
from ML_experiment.run_gradient_zonotope_battery import ProjectiveWitnessAtlas
from ML_experiment.tasks import complex_spiral_3d, radial_stripes, spiral


class ObservedHermiteAtlasTests(unittest.TestCase):
    def test_scalar_hermite_stays_inside_observed_support(self):
        task = complex_spiral_3d(0)
        atlas = ObservedHermiteAtlas(task, pool_size=512, seed=1)
        self.assertGreaterEqual(float(atlas.hermite_x.min()), float(task.x_train.min()))
        self.assertLessEqual(float(atlas.hermite_x.max()), float(task.x_train.max()))
        self.assertEqual(atlas.hermite_y.shape, (512, 3))
        self.assertTrue(torch.isfinite(atlas.hermite_y).all())

    def test_classification_transport_uses_only_observed_labels(self):
        task = spiral(0)
        atlas = ObservedHermiteAtlas(task, pool_size=512, seed=2)
        self.assertTrue(set(atlas.hermite_y.tolist()) <= set(task.y_train.tolist()))
        self.assertEqual(atlas.hermite_x.shape, (512, 2))
        self.assertGreaterEqual(atlas.geometry_diagnostics["hermite_acceptance"], 0.0)
        self.assertLessEqual(atlas.geometry_diagnostics["hermite_acceptance"], 1.0)

    def test_structural_probability_is_normalized_and_nonuniform(self):
        task = radial_stripes(0)
        atlas = ObservedHermiteAtlas(task, pool_size=256, seed=3)
        self.assertAlmostEqual(float(atlas.real_probability.sum()), 1.0, places=5)
        self.assertGreater(float(atlas.real_probability.std()), 0.0)

    def test_curvature_response_backport_has_finite_gradients(self):
        for name in (
            "self_context_curvature_response",
            "self_context_curvature_response2",
            "self_context_curvature_response_bounded",
            "self_context_curvature_response_geometric",
            "self_context_curvature_response_signed",
            "self_context_curvature_response_signed_geometric",
        ):
            model = make_hybrid(name, 2, 2, 16)
            x = torch.randn(12, 2)
            loss = model(x).square().mean()
            loss.backward()
            self.assertEqual(model(x).shape, (12, 2))
            self.assertTrue(all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ))

    def test_curvature_response_adds_only_one_parameter(self):
        baseline = make_hybrid("self_context", 2, 2, 16)
        candidate = make_hybrid("self_context_curvature_response", 2, 2, 16)
        self.assertEqual(parameter_count(candidate), parameter_count(baseline) + 1)

    def test_signed_curvature_response_begins_as_exact_self_context(self):
        torch.manual_seed(9)
        baseline = make_hybrid("self_context", 2, 2, 16)
        torch.manual_seed(9)
        candidate = make_hybrid(
            "self_context_curvature_response_signed", 2, 2, 16
        )
        x = torch.randn(19, 2)
        self.assertTrue(torch.equal(candidate(x), baseline(x)))
        self.assertEqual(float(candidate.curvature_scale), 0.0)

    def test_selection_curvature_is_a_finite_bounded_chart_transport(self):
        model = make_hybrid("self_context_selection_curvature", 2, 2, 16)
        x = torch.randn(19, 2)
        loss = model(x).square().mean()
        loss.backward()
        for layer in (model.up, model.down):
            self.assertGreater(float(layer.last_selection_blend), 0.0)
            self.assertLess(float(layer.last_selection_blend), 1.0)
            self.assertTrue(torch.isfinite(layer.last_selection_shift).all())
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_low_rank_witness_uses_intrinsic_radius(self):
        from ML_experiment.tasks import nd_spiral

        atlas = ProjectiveWitnessAtlas(nd_spiral(1, seed=0))
        self.assertEqual(atlas.intrinsic_rank, 2)
        self.assertEqual(atlas.geometry_mode, "intrinsic_radius")
        generator = torch.Generator().manual_seed(3)
        sample = atlas.sample_band(64, 0, generator)
        self.assertTrue((atlas.band[sample] == 0).all())

    def test_high_dimensional_spiral_builds_observed_hermite_chart(self):
        from ML_experiment.tasks import nd_spiral

        low = nd_spiral(1, seed=0)
        low_atlas = ObservedHermiteAtlas(low, pool_size=128, seed=4)
        self.assertEqual(low_atlas.hermite_x.shape, (128, low.input_dim))
        self.assertTrue(torch.isfinite(low_atlas.hermite_x).all())
        self.assertTrue(set(low_atlas.hermite_y.tolist()) <= {0, 1})
        self.assertEqual(low_atlas.geometry_diagnostics["observed_intrinsic_rank"], 2)

        high = nd_spiral(8, seed=0)
        high_atlas = ObservedHermiteAtlas(high, pool_size=128, seed=4)
        self.assertEqual(
            high_atlas.geometry_diagnostics["observed_intrinsic_rank"],
            high.input_dim,
        )


if __name__ == "__main__":
    unittest.main()
