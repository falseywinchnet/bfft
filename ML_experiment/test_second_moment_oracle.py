from __future__ import annotations

import unittest

import torch

from ML_experiment.muon_primacy import make_operator_problem, support_polar
from ML_experiment.second_moment_oracle import MomentState, first_step_certificate


class SecondMomentOracleTests(unittest.TestCase):
    def test_right_matrix_first_step_is_muon_direction(self):
        problem = make_operator_problem(6, 1e3, seed=19)
        gradient = problem.gradient(torch.zeros_like(problem.target))
        update = MomentState(
            gradient.shape, "right_matrix", dtype=gradient.dtype
        ).update(gradient)
        self.assertTrue(
            torch.allclose(update, support_polar(gradient), atol=2e-8, rtol=2e-8)
        )

    def test_full_second_moment_is_rotation_equivariant(self):
        generator = torch.Generator().manual_seed(23)
        gradients = [
            torch.randn(2, 3, generator=generator, dtype=torch.float64)
            for _ in range(4)
        ]
        transform, _ = torch.linalg.qr(
            torch.randn(6, 6, generator=generator, dtype=torch.float64)
        )
        original = MomentState(torch.Size((2, 3)), "full")
        rotated = MomentState(torch.Size((2, 3)), "full")
        for gradient in gradients:
            expected = original.update(gradient).reshape(-1)
            observed = rotated.update(
                (transform @ gradient.reshape(-1)).reshape(2, 3)
            ).reshape(-1)
            self.assertTrue(
                torch.allclose(observed, transform @ expected, atol=3e-7, rtol=3e-7)
            )

    def test_scalar_and_full_keep_first_gradient_direction(self):
        problem = make_operator_problem(5, 1e2, seed=29)
        gradient = problem.gradient(torch.zeros_like(problem.target))
        for mode in ("scalar", "full"):
            update = MomentState(
                gradient.shape, mode, dtype=gradient.dtype
            ).update(gradient)
            cosine = torch.nn.functional.cosine_similarity(
                update.reshape(1, -1), gradient.reshape(1, -1)
            )
            self.assertGreater(float(cosine), 1.0 - 1e-8)

    def test_certificate_marks_matrix_boundary(self):
        problem = make_operator_problem(5, 1e3, seed=31)
        result = first_step_certificate(problem)
        self.assertLess(result["right_matrix"]["distance_to_muon_direction"], 1e-7)
        self.assertLess(
            result["right_matrix"]["relative_loss_after_one_step"], 1e-12
        )


if __name__ == "__main__":
    unittest.main()
