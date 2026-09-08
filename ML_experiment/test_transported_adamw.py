from __future__ import annotations

import unittest

import torch

from ML_experiment.optimizers import TransportedAdamW, zeropower_newton_schulz5


def _one_step(parameter, gradient, **kwargs):
    optimizer = TransportedAdamW([parameter], weight_decay=0.0, **kwargs)
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    optimizer.step()
    return before - parameter.detach(), optimizer


class TransportedAdamWTests(unittest.TestCase):
    def test_first_step_is_row_rms_normalized_not_muon(self):
        torch.manual_seed(11)
        gradient = torch.randn(5, 7, dtype=torch.float64)
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        update, _ = _one_step(parameter, gradient, lr=1.0, eps=1e-12)
        row_norms = torch.linalg.vector_norm(update, dim=1)
        self.assertTrue(torch.allclose(
            row_norms, torch.full_like(row_norms, gradient.shape[1] ** .5),
            atol=1e-8,
        ))
        polar = zeropower_newton_schulz5(gradient, steps=20, eps=1e-12)
        self.assertGreater(float(torch.linalg.vector_norm(update - polar)), .1)

    def test_shared_right_rotation_is_equivariant(self):
        torch.manual_seed(13)
        source = torch.randn(4, 6, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(
            torch.randn(6, 6, dtype=torch.float64)
        )
        left = torch.nn.Parameter(torch.zeros_like(source))
        right = torch.nn.Parameter(torch.zeros_like(source))
        left_optimizer = TransportedAdamW(
            [left], lr=.01, eps=1e-12, weight_decay=0.0
        )
        right_optimizer = TransportedAdamW(
            [right], lr=.01, eps=1e-12, weight_decay=0.0
        )
        for _ in range(4):
            gradient = torch.randn_like(source)
            left.grad = gradient
            right.grad = gradient @ rotation
            left_optimizer.step()
            right_optimizer.step()
        self.assertTrue(torch.allclose(
            right, left @ rotation, atol=2e-7, rtol=2e-7
        ))

    def test_second_moment_retains_off_diagonal_covariance(self):
        gradient = torch.tensor([[1.0, 2.0, -3.0]])
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        _, optimizer = _one_step(parameter, gradient, lr=.01)
        covariance = optimizer.state[parameter]["second"][0]
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        self.assertGreater(float(torch.linalg.vector_norm(off_diagonal)), 0.0)

    def test_anchor_style_transport_preserves_moment_norms(self):
        previous = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        current = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        transform = TransportedAdamW._transport_matrices(
            previous, current, 1e-12
        )
        moment = torch.tensor([[.2, -.4, .7]], dtype=torch.float64)
        moved = torch.einsum("bi,bij->bj", moment, transform)
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(moment, dim=1),
            torch.linalg.vector_norm(moved, dim=1), atol=1e-12,
        ))

    def test_zero_row_initializes_when_it_first_becomes_live(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
        optimizer = TransportedAdamW(
            [parameter], lr=.01, eps=1e-12, transport_eps=1e-10,
            weight_decay=0.0,
        )
        parameter.grad = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
        )
        optimizer.step()
        parameter.grad = torch.tensor(
            [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
        )
        optimizer.step()
        signature = optimizer.state[parameter]["signature"]
        self.assertTrue(torch.allclose(
            signature[0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        ))
        self.assertTrue(torch.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()
