import unittest

import torch
import torch.nn.functional as F

from ML_experiment.context_backprop import (
    backward_with_context_split,
    clip_context_gradient_channels_,
)
from ML_experiment.models import SoftEikonalNet


class ContextBackpropSplitTest(unittest.TestCase):
    def _model(self):
        torch.manual_seed(7)
        return SoftEikonalNet(
            4, 2, width=8, directions=8, rank=2,
            self_context_strength=.35, context_steps=1,
        )

    def test_forward_value_is_preserved_and_channels_recombine(self):
        model = self._model()
        x = torch.randn(12, 4)
        y = torch.randn(12, 2)
        reference = F.mse_loss(model(x), y).detach()
        loss, diagnostics = backward_with_context_split(
            model, lambda: F.mse_loss(model(x), y)
        )
        self.assertTrue(torch.equal(loss.detach(), reference))
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            self.assertTrue(torch.equal(
                parameter.grad,
                parameter._fixed_chart_grad + parameter._context_chart_grad,
            ))
        self.assertGreaterEqual(diagnostics["fixed_chart_alignment"], -1e-10)

    def test_global_context_small_gain_certificate(self):
        model = self._model()
        x = torch.randn(16, 4)
        y = torch.randn(16, 2)
        _, diagnostics = backward_with_context_split(
            model, lambda: F.mse_loss(model(x), y)
        )
        self.assertLessEqual(
            diagnostics["context_chart_applied_norm"],
            diagnostics["fixed_chart_gradient_norm"] * (1 + 1e-6),
        )
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            fixed_norm = float(parameter._fixed_chart_grad.norm())
            context_norm = float(parameter._context_chart_grad.norm())
            if fixed_norm > 1e-20:
                self.assertLessEqual(context_norm, fixed_norm * (1 + 1e-6))

    def test_zero_gain_removes_only_chart_feedback(self):
        model = self._model()
        x = torch.randn(10, 4)
        y = torch.randn(10, 2)
        _, diagnostics = backward_with_context_split(
            model, lambda: F.mse_loss(model(x), y), context_gain=0.0
        )
        self.assertEqual(diagnostics["context_chart_scale"], 0.0)
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.equal(
                    parameter.grad, parameter._fixed_chart_grad
                ))

    def test_clipping_preserves_source_channel_identity(self):
        model = self._model()
        x = torch.randn(10, 4)
        y = 1000 * torch.randn(10, 2)
        backward_with_context_split(model, lambda: F.mse_loss(model(x), y))
        total_norm = clip_context_gradient_channels_(model.parameters(), .1)
        self.assertGreater(float(total_norm), .1)
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.allclose(
                    parameter.grad,
                    parameter._fixed_chart_grad + parameter._context_chart_grad,
                    atol=1e-7,
                ))


if __name__ == "__main__":
    unittest.main()
