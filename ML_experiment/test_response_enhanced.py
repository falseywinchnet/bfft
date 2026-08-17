"""Tests for relational response self-context and frame flow."""
from __future__ import annotations

import unittest

import torch

from ML_experiment.models import parameter_count
from ML_experiment.response_enhanced import (
    RELATIONAL_CFF_DEEP,
    RELATIONAL_SCL,
    RelationalCFFLinear,
    RelationalSelfContextLinear,
    allocation_summary,
    make_response_variant,
)


class ResponseEnhancedTests(unittest.TestCase):
    def test_layers_forward_backward_and_preserve_requested_shape(self):
        for layer_type in (RelationalSelfContextLinear, RelationalCFFLinear):
            layer = layer_type(12, 18)
            x = torch.randn(7, 12, requires_grad=True)
            output = layer(x)
            self.assertEqual(output.shape, (7, 18))
            output.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))

    def test_response_receives_relational_features(self):
        layer = RelationalSelfContextLinear(12, 12, rank=4)
        self.assertEqual(layer.response[0].in_features, 7 + 5 * 4)
        _, projected, _ = layer._allocate(torch.randn(5, 12))
        state = torch.randn(5, 12)
        _, _, matched = layer._allocate(state, projected)
        _, _, changed = layer._allocate(state, torch.roll(projected, 1, 0))
        self.assertGreater(float((matched - changed).abs().max()), 1e-7)

    def test_deep_cff_has_extra_response_depth(self):
        scl = RelationalSelfContextLinear(12, 12)
        cff = RelationalCFFLinear(12, 12)
        self.assertEqual(sum(isinstance(item, torch.nn.Linear) for item in scl.response), 2)
        self.assertEqual(sum(isinstance(item, torch.nn.Linear) for item in cff.response), 3)
        self.assertGreater(parameter_count(cff), parameter_count(scl))

    def test_network_variants_and_allocation_diagnostics(self):
        for name in (RELATIONAL_SCL, RELATIONAL_CFF_DEEP):
            model = make_response_variant(name, 2, 3, 12)
            output = model(torch.randn(16, 2))
            self.assertEqual(output.shape, (16, 3))
            diagnostics = allocation_summary(model)
            self.assertTrue(0 <= diagnostics["allocation_entropy"] <= 1)
            self.assertTrue(0 < diagnostics["allocation_max_weight"] <= 1)


if __name__ == "__main__":
    unittest.main()
