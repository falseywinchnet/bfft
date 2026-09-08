from __future__ import annotations

import unittest

import torch

from ML_experiment.muon_primacy import (
    best_scalar_step,
    make_block_sweep_covariances,
    make_operator_problem,
    one_step_certificate,
    run_streaming,
    support_polar,
)


class MuonPrimacyTests(unittest.TestCase):
    def test_anisotropic_operator_gradient_has_target_polar_factor(self):
        for condition in (1.0, 1e2, 1e6):
            problem = make_operator_problem(12, condition, seed=93)
            gradient = problem.gradient(torch.zeros_like(problem.target))
            self.assertTrue(torch.allclose(
                support_polar(gradient),
                -problem.target,
                atol=2e-9,
                rtol=2e-9,
            ))

    def test_unit_exact_muon_step_reaches_the_optimum(self):
        problem = make_operator_problem(16, 1e5, seed=17)
        gradient = problem.gradient(torch.zeros_like(problem.target))
        weight = -support_polar(gradient)
        self.assertLess(float(problem.loss(weight)), 1e-20)

    def test_best_scalar_adam_sign_step_cannot_match_dense_operator(self):
        problem = make_operator_problem(16, 1e4, seed=17)
        gradient = problem.gradient(torch.zeros_like(problem.target))
        _, residual = best_scalar_step(
            -torch.sign(gradient), problem.target, problem.covariance
        )
        self.assertGreater(residual, 1e-2)

    def test_certificate_separates_operator_and_coordinate_geometry(self):
        problem = make_operator_problem(16, 1e4, seed=17)
        certificate = one_step_certificate(problem)
        self.assertLess(
            certificate["exact_muon"]["relative_loss_after_one_step"], 1e-20
        )
        self.assertGreater(
            certificate["adam_sign_limit"]["relative_loss_after_one_step"],
            1e-2,
        )

    def test_block_sweep_recovers_target_in_one_pass(self):
        problem = make_operator_problem(16, 1e6, seed=17)
        blocks = make_block_sweep_covariances(problem, batch_size=4)
        result = run_streaming(problem, blocks, "exact_muon", lr=1.0)
        self.assertEqual(result["steps_to_1e-8"], 4)
        self.assertLess(result["final_operator_error"], 1e-8)

    def test_block_sweep_is_an_exact_covariance_partition(self):
        problem = make_operator_problem(16, 1e6, seed=17)
        blocks = make_block_sweep_covariances(problem, batch_size=4)
        self.assertTrue(torch.allclose(
            sum(blocks, torch.zeros_like(problem.covariance)),
            problem.covariance,
            atol=1e-12,
            rtol=1e-12,
        ))


if __name__ == "__main__":
    unittest.main()
