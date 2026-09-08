from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from ML_experiment.optimizers import (
    Lepton,
    Muon,
    MuonWithAuxAdamW,
    RidgeAdamW,
    TurnAdamW,
    TransportLepton,
    TransportMuon,
    _transport_matrix_request,
    bregman_soft_polar,
    zeropower_newton_schulz5,
)


def _reference_newton_schulz(matrix, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = matrix.shape[0] > matrix.shape[1]
    value = matrix.float().T if transposed else matrix.float()
    value = value / value.norm().clamp_min(eps)
    for _ in range(steps):
        gram = value @ value.T
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    return (value.T if transposed else value).to(matrix.dtype)


class _TinyHiddenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 5)
        self.body = nn.Linear(5, 5)
        self.output = nn.Linear(5, 2)

    def forward(self, value):
        return self.output(torch.tanh(self.body(torch.tanh(self.embed(value)))))


class MuonTests(unittest.TestCase):
    def test_newton_schulz_matches_reference_polynomial(self):
        matrix = torch.tensor(
            [[1.0, -2.0, 0.5], [0.25, 1.5, -0.75]], dtype=torch.float64
        )
        actual = zeropower_newton_schulz5(matrix)
        expected = _reference_newton_schulz(matrix)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(
            actual,
            zeropower_newton_schulz5(37.0 * matrix),
            atol=2e-5,
            rtol=2e-5,
        ))

    def test_newton_schulz_handles_wide_tall_and_zero_matrices(self):
        generator = torch.Generator().manual_seed(917)
        for shape in ((3, 7), (7, 3), (4, 4)):
            matrix = torch.randn(shape, generator=generator)
            update = zeropower_newton_schulz5(matrix)
            self.assertEqual(update.shape, matrix.shape)
            self.assertTrue(torch.isfinite(update).all())
            singular = torch.linalg.svdvals(update)
            self.assertGreater(float(singular.min()), 0.45)
            self.assertLess(float(singular.max()), 1.55)
        self.assertTrue(torch.equal(
            zeropower_newton_schulz5(torch.zeros(3, 5)),
            torch.zeros(3, 5),
        ))

    def test_learning_rate_adjustments_match_reference_definitions(self):
        self.assertTrue(math.isclose(
            Muon._adjusted_lr(0.02, (8, 32), "original"), 0.02
        ))
        self.assertTrue(math.isclose(
            Muon._adjusted_lr(0.02, (32, 8), "original"), 0.04
        ))
        self.assertTrue(math.isclose(
            Muon._adjusted_lr(0.003, (8, 32), "match_rms_adamw"),
            0.003 * 0.2 * math.sqrt(32),
        ))

    def test_full_matrix_transport_preserves_norm_and_frame_component(self):
        value = torch.tensor([[2.0, 3.0], [-4.0, 1.0]], dtype=torch.float64)
        frame_from = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        frame_to = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        transported = _transport_matrix_request(value, frame_from, frame_to)
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(transported),
            torch.linalg.vector_norm(value),
        ))
        self.assertTrue(torch.allclose(
            torch.sum(value * frame_from),
            torch.sum(transported * frame_to),
        ))

    def test_transport_muon_rotates_carried_momentum_before_reuse(self):
        parameter = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        optimizer = TransportMuon(
            [parameter], lr=0.0, weight_decay=0.0, momentum=0.5,
            nesterov=False,
        )
        parameter.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        optimizer.step()
        first = optimizer.state[parameter]["momentum_buffer"].clone()
        parameter.grad = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        optimizer.step()
        second = optimizer.state[parameter]["momentum_buffer"]
        untransported = 0.5 * first + 0.5 * parameter.grad
        self.assertFalse(torch.allclose(second, untransported))
        self.assertAlmostEqual(optimizer.last_transport_ratio, 1.0, places=12)
        self.assertGreater(optimizer.last_turn, 0.0)

    def test_lepton_is_the_smoothed_nuclear_dual_map(self):
        matrix = torch.diag(torch.tensor([1.0, 0.1, 0.001], dtype=torch.float64))
        epsilon = 0.05
        update = bregman_soft_polar(matrix, epsilon)
        expected = torch.diag(
            torch.diagonal(matrix)
            / torch.sqrt(torch.diagonal(matrix).square() + epsilon ** 2)
        )
        self.assertTrue(torch.allclose(update, expected, atol=1e-6, rtol=1e-6))
        singular = torch.linalg.svdvals(update)
        self.assertGreater(float(singular[0]), 0.99)
        self.assertLess(float(singular[-1]), 0.03)

    def test_lepton_approaches_exact_polar_as_epsilon_vanishes(self):
        matrix = torch.tensor(
            [[1.0, 0.2, -0.1], [0.3, -0.7, 0.4]], dtype=torch.float64
        )
        left, _, right = torch.linalg.svd(matrix, full_matrices=False)
        polar = left @ right
        update = bregman_soft_polar(matrix, 1e-6)
        self.assertTrue(torch.allclose(update, polar, atol=2e-5, rtol=2e-5))

    def test_lepton_maintains_a_bounded_dual_request(self):
        parameter = nn.Parameter(torch.zeros(3, 2))
        optimizer = Lepton(
            [parameter], lr=0.1, weight_decay=0.0,
            bregman_epsilon=0.05,
        )
        parameter.grad = torch.tensor(
            [[1000.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        )
        optimizer.step()
        adjusted = Muon._adjusted_lr(0.1, parameter.shape, "match_rms_adamw")
        singular = torch.linalg.svdvals(-parameter.detach() / adjusted)
        self.assertTrue(torch.isfinite(singular).all())
        self.assertLessEqual(float(singular.max()), 1.0 + 1e-6)
        self.assertGreater(float(singular[0]), float(singular[-1]))

    def test_transport_lepton_moves_the_dual_state_before_mirror_readout(self):
        parameter = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        optimizer = TransportLepton(
            [parameter], lr=0.0, weight_decay=0.0, momentum=0.5,
            nesterov=False, bregman_epsilon=0.05,
        )
        parameter.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        optimizer.step()
        first = optimizer.state[parameter]["dual_buffer"].clone()
        parameter.grad = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        optimizer.step()
        second = optimizer.state[parameter]["dual_buffer"]
        untransported = 0.5 * first + 0.5 * parameter.grad
        self.assertFalse(torch.allclose(second, untransported))
        self.assertAlmostEqual(optimizer.last_transport_ratio, 1.0, places=12)
        self.assertGreater(optimizer.last_turn, 0.0)

    def test_current_frame_transport_finishes_in_the_observed_frame(self):
        parameter = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        optimizer = TransportLepton(
            [parameter], lr=0.0, weight_decay=0.0, momentum=0.5,
            nesterov=False, bregman_epsilon=0.05, frame_target="current",
        )
        first = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        second = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        parameter.grad = first
        optimizer.step()
        parameter.grad = second
        optimizer.step()
        signature = optimizer.state[parameter]["dual_signature"]
        self.assertTrue(torch.equal(signature, second))
        self.assertAlmostEqual(optimizer.last_frame_residual, 0.0, places=12)

    def test_gradient_certificate_bounds_the_applied_matrix_step(self):
        parameter = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        optimizer = TransportLepton(
            [parameter], lr=0.1, weight_decay=0.0, momentum=0.0,
            nesterov=False, bregman_epsilon=0.05, gradient_trust=1.0,
        )
        gradient = torch.tensor(
            [[1e-6, 0.0], [0.0, -2e-6]], dtype=torch.float64
        )
        parameter.grad = gradient
        before = parameter.detach().clone()
        optimizer.step()
        applied = torch.linalg.vector_norm(parameter.detach() - before)
        self.assertLessEqual(
            float(applied), float(torch.linalg.vector_norm(gradient)) + 1e-15
        )
        self.assertLess(optimizer.last_certificate_scale, 1.0)

    def test_zero_ridge_matches_adamw(self):
        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
        reference = nn.Parameter(initial.clone())
        candidate = nn.Parameter(initial.clone())
        adamw = torch.optim.AdamW(
            [reference], lr=0.03, betas=(.9, .999), eps=1e-8,
            weight_decay=0.02,
        )
        ridge = RidgeAdamW(
            [candidate], lr=0.03, betas=(.9, .999), eps=1e-8,
            weight_decay=0.02, ridge=0.0,
        )
        gradients = (
            torch.tensor([[2.0, -0.2], [0.03, 4.0]], dtype=torch.float64),
            torch.tensor([[-1.0, 0.4], [0.08, 2.0]], dtype=torch.float64),
            torch.tensor([[0.2, 0.7], [-0.1, -3.0]], dtype=torch.float64),
        )
        for gradient in gradients:
            reference.grad = gradient.clone()
            candidate.grad = gradient.clone()
            adamw.step()
            ridge.step()
        self.assertTrue(torch.allclose(reference, candidate, atol=1e-12, rtol=1e-12))

    def test_turn_adam_brakes_on_gradient_reversal(self):
        parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        optimizer = TurnAdamW(
            [parameter], lr=0.1, weight_decay=0.0, turn_brake=1.5,
            recovery=.02,
        )
        parameter.grad = torch.tensor([1.0, 0.0], dtype=torch.float64)
        optimizer.step()
        parameter.grad = torch.tensor([-1.0, 0.0], dtype=torch.float64)
        optimizer.step()
        self.assertAlmostEqual(optimizer.last_turn, 1.0, places=12)
        self.assertLess(optimizer.last_lr_scale, 0.25)

    def test_hybrid_routes_hidden_matrices_and_steps_every_parameter(self):
        torch.manual_seed(81)
        model = _TinyHiddenModel()
        optimizer = MuonWithAuxAdamW(model, lr=3e-3, weight_decay=0.0)

        muon_ids = {
            id(parameter)
            for group in optimizer.muon.param_groups
            for parameter in group["params"]
        }
        auxiliary_ids = {
            id(parameter)
            for group in optimizer.adamw.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(muon_ids, {id(model.body.weight)})
        self.assertTrue(muon_ids.isdisjoint(auxiliary_ids))
        self.assertEqual(
            muon_ids | auxiliary_ids, {id(p) for p in model.parameters()}
        )

        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        value = torch.randn(11, 3)
        target = torch.randn(11, 2)
        loss = (model(value) - target).square().mean()
        loss.backward()
        optimizer.step()

        self.assertIn("momentum_buffer", optimizer.muon.state[model.body.weight])
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertFalse(torch.equal(parameter.detach(), before[name]))


if __name__ == "__main__":
    unittest.main()
