from __future__ import annotations

import unittest

import torch

from ML_experiment.models import SoftEikonalLinear, SoftEikonalNet


class ContextBackwardRestraintTests(unittest.TestCase):
    def test_forward_value_is_identical_in_all_backward_modes(self):
        torch.manual_seed(17)
        model = SoftEikonalNet(3, 2, 8, self_context_strength=.25)
        value = torch.randn(11, 3)
        outputs = []
        for mode in ("exact", "detached", "nonexpansive"):
            model.set_context_backward_mode(mode)
            outputs.append(model(value).detach())
        for output in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], output))

    def test_nonexpansive_mode_bounds_normalization_jacobian(self):
        layer = SoftEikonalLinear(
            4, 4, self_context_strength=.25,
            context_backward_mode="nonexpansive",
        ).double()
        reference = torch.tensor([[2.0, -1.0, .5, 1.5]], dtype=torch.float64)

        def normalize(state):
            normalized, state_rms, reference_rms = layer._normalize_like(
                state.reshape(1, 4), reference
            )
            return layer._temper_context_backward(
                normalized, state_rms, reference_rms
            ).reshape(4)

        state = torch.tensor([.02, -.01, .03, .01], dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(normalize, state)
        self.assertLessEqual(float(torch.linalg.matrix_norm(jacobian, ord=2)), 1.0 + 1e-9)

    def test_detached_mode_zeros_only_context_normalization_gradient(self):
        layer = SoftEikonalLinear(
            3, 3, self_context_strength=.25,
            context_backward_mode="detached",
        ).double()
        state = torch.tensor([[.2, -.3, .4]], dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
        normalized, state_rms, reference_rms = layer._normalize_like(state, reference)
        restrained = layer._temper_context_backward(
            normalized, state_rms, reference_rms
        )
        restrained.square().sum().backward()
        self.assertTrue(torch.equal(state.grad, torch.zeros_like(state)))


if __name__ == "__main__":
    unittest.main()
