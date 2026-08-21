import unittest

import torch

from ML_experiment.odd_context_hybrids import (
    CONTROLLED_VARIANTS, FACTOR_VARIANTS, ContextualCubicBridge,
    CubicResidual, VARIANTS, make_hybrid)


class OddContextHybridTests(unittest.TestCase):
    def test_all_variants_have_safe_shapes_and_gradients(self):
        x = torch.randn(6, 16)
        for name in dict.fromkeys((*VARIANTS, *CONTROLLED_VARIANTS, *FACTOR_VARIANTS)):
            with self.subTest(name=name):
                model = make_hybrid(name, 16, 2, 16)
                output = model(x)
                self.assertEqual(output.shape, (6, 2))
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_capacity_controls_match_their_bridge_targets(self):
        for control, bridge in (
            ("self_capacity_match_rank8", "self_contextual_angular_rank8"),
            ("self_capacity_match_full", "self_contextual_angular"),
        ):
            control_count = sum(p.numel() for p in make_hybrid(control, 16, 2, 24).parameters())
            bridge_count = sum(p.numel() for p in make_hybrid(bridge, 16, 2, 24).parameters())
            self.assertLess(abs(control_count - bridge_count), 200)

    def test_tied_bridge_rank_is_a_real_capacity_axis(self):
        small = make_hybrid("self_contextual_tied_rank8", 16, 2, 24)
        large = make_hybrid("self_contextual_tied_rank24", 16, 2, 24)
        self.assertLess(
            sum(p.numel() for p in small.parameters()),
            sum(p.numel() for p in large.parameters()),
        )

    def test_tight_projection_is_semi_orthogonal(self):
        for shape in ((8, 24), (48, 24)):
            weight = torch.randn(*shape)
            frame = ContextualCubicBridge._semi_orthogonal(weight)
            gram = frame @ frame.T if shape[0] <= shape[1] else frame.T @ frame
            self.assertTrue(torch.allclose(gram, torch.eye(len(gram)), atol=2e-5))

    def test_learned_cone_starts_between_raw_and_unit_rays(self):
        model = make_hybrid("self_contextual_full_learned_cone", 16, 2, 24)
        exponent = torch.sigmoid(model.bridge.cone_logits)
        self.assertTrue(torch.allclose(exponent, torch.full((3,), 0.5)))

    def test_homogeneity_atlas_starts_as_degree_one(self):
        model = make_hybrid(
            "self_contextual_full_learned_cone_ray_degrees", 16, 2, 24
        )
        degree = 3.0 * torch.sigmoid(model.bridge.degree_logits)
        self.assertEqual(tuple(degree.shape), (48,))
        self.assertTrue(torch.allclose(degree, torch.ones_like(degree)))
        self.assertEqual(tuple(model(torch.randn(7, 16)).shape), (7, 2))

    def test_global_homogeneity_has_one_degree(self):
        model = make_hybrid(
            "self_contextual_full_learned_cone_global_degree", 16, 2, 24
        )
        self.assertEqual(tuple(model.bridge.degree_logits.shape), (1,))

    def test_weight_norm_is_exactly_raw_at_initialization(self):
        bridge = ContextualCubicBridge(7, 11, 5, 13, conditioning="weight_norm")
        source, context = torch.randn(9, 7), torch.randn(9, 11)
        for index, (layer, value) in enumerate(
            ((bridge.a, source), (bridge.b, source), (bridge.c, context))
        ):
            self.assertTrue(torch.allclose(
                bridge._project(layer, value, index), layer(value),
                atol=2e-6, rtol=2e-6,
            ))

    def test_angular_cubic_is_odd_and_degree_one(self):
        layer = CubicResidual(7, 5, 24, "angular")
        x = torch.randn(11, 7)
        with torch.no_grad():
            value = layer(x)
            self.assertTrue(torch.allclose(layer(-x), -value, atol=2e-5, rtol=2e-5))
            self.assertTrue(torch.allclose(layer(3 * x), 3 * value, atol=2e-4, rtol=2e-4))

    def test_degree2_cubic_has_odd_parity_and_quadratic_positive_scale(self):
        layer = CubicResidual(7, 5, 24, "degree2")
        x = torch.randn(11, 7)
        with torch.no_grad():
            value = layer(x)
            self.assertTrue(torch.allclose(layer(-x), -value, atol=2e-5, rtol=2e-5))
            self.assertTrue(torch.allclose(layer(3 * x), 9 * value, atol=2e-4, rtol=2e-4))


if __name__ == "__main__":
    unittest.main()
