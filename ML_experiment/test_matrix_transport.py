import unittest

import torch

from ML_experiment.optimizers import MatrixTransport


class MatrixTransportTest(unittest.TestCase):
    def test_first_full_rank_matrix_step_has_adam_scale_without_sign_map(self):
        torch.manual_seed(3)
        parameter = torch.nn.Parameter(torch.zeros(5, 5))
        gradient = torch.randn(5, 5)
        optimizer = MatrixTransport(
            [parameter], lr=.04, weight_decay=0, matrix_ridge=1e-8
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        request = -parameter.detach() / .04
        self.assertAlmostEqual(float(request.square().mean()), 1.0, places=3)
        self.assertGreater(float(torch.sum(request * gradient)), 0.0)
        # A polar-shaped request is not Adam's elementwise sign request.
        self.assertGreater(float((request - gradient.sign()).abs().mean()), .05)

    def test_biorthogonal_coordinate_equivariance(self):
        torch.manual_seed(9)
        left_rotation, _ = torch.linalg.qr(torch.randn(4, 4))
        right_rotation, _ = torch.linalg.qr(torch.randn(6, 6))
        initial = torch.randn(4, 6)
        left = torch.nn.Parameter(initial.clone())
        right = torch.nn.Parameter(
            left_rotation.transpose(0, 1) @ initial @ right_rotation
        )
        left_optimizer = MatrixTransport([left], lr=.01, weight_decay=.02)
        right_optimizer = MatrixTransport([right], lr=.01, weight_decay=.02)
        for _ in range(5):
            gradient = torch.randn(4, 6)
            left.grad = gradient.clone()
            right.grad = (
                left_rotation.transpose(0, 1) @ gradient @ right_rotation
            )
            left_optimizer.step()
            right_optimizer.step()
        expected = left_rotation.transpose(0, 1) @ left @ right_rotation
        self.assertTrue(torch.allclose(right, expected, atol=2e-4, rtol=2e-4))

    def test_request_cannot_reverse_raw_gradient(self):
        parameter = torch.nn.Parameter(torch.zeros(3, 4))
        optimizer = MatrixTransport(
            [parameter], lr=.02, weight_decay=0,
            restrain_transverse=False,
        )
        for seed in range(5):
            gradient = torch.randn(3, 4, generator=torch.Generator().manual_seed(seed))
            before = parameter.detach().clone()
            parameter.grad = gradient
            optimizer.step()
            request = (before - parameter.detach()) / .02
            self.assertGreaterEqual(float(torch.sum(request * gradient)), -2e-5)

    def test_cached_roots_obey_maximum_staleness(self):
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        optimizer = MatrixTransport(
            [parameter], lr=.01, weight_decay=0,
            root_drift_threshold=1e30, max_root_staleness=3,
        )
        generator = torch.Generator().manual_seed(81)
        refresh_steps = []
        for step in range(1, 9):
            parameter.grad = torch.randn(4, 4, generator=generator)
            optimizer.step()
            if optimizer.last_root_refresh_fraction == 1.0:
                refresh_steps.append(step)
            self.assertLessEqual(optimizer.last_root_staleness, 2.0)
        self.assertEqual(refresh_steps, [1, 4, 7])

    def test_gauge_transport_reexpresses_the_same_raw_covector(self):
        torch.manual_seed(91)
        factors = []
        for size in (4, 4, 5, 5):
            basis = torch.randn(size, size)
            factors.append(
                basis @ basis.transpose(0, 1) + .3 * torch.eye(size)
            )
        old_left, new_left, old_right, new_right = factors
        raw = torch.randn(4, 5)
        old_coordinates = old_left @ raw @ old_right
        moved = MatrixTransport._gauge_transport(
            old_coordinates, old_left, old_right, new_left, new_right
        )
        expected = new_left @ raw @ new_right
        self.assertTrue(
            torch.allclose(moved, expected, atol=2e-4, rtol=2e-4)
        )

    def test_nesterov_read_preserves_the_first_step(self):
        torch.manual_seed(101)
        gradient = torch.randn(4, 5)
        ordinary = torch.nn.Parameter(torch.zeros(4, 5))
        nesterov = torch.nn.Parameter(torch.zeros(4, 5))
        ordinary_optimizer = MatrixTransport(
            [ordinary], lr=.02, weight_decay=0, nesterov=False
        )
        nesterov_optimizer = MatrixTransport(
            [nesterov], lr=.02, weight_decay=0, nesterov=True
        )
        ordinary.grad = gradient.clone()
        nesterov.grad = gradient.clone()
        ordinary_optimizer.step()
        nesterov_optimizer.step()
        self.assertTrue(
            torch.allclose(ordinary, nesterov, atol=1e-6, rtol=1e-6)
        )

    def test_residual_polar_governor_is_reversible(self):
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        optimizer = MatrixTransport(
            [parameter], residual_polar=True, residual_polar_beta=.5
        )
        optimizer.set_residual_entropy_rank(.1)
        anisotropic_weight = optimizer.last_polar_weight
        optimizer.set_residual_entropy_rank(1.0)
        isotropic_weight = optimizer.last_polar_weight

        self.assertAlmostEqual(anisotropic_weight, .45)
        self.assertAlmostEqual(isotropic_weight, .225)
        self.assertLess(isotropic_weight, anisotropic_weight)

    def test_longitudinal_fusion_closes_live_axis_and_keeps_transverse_memory(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        optimizer = MatrixTransport(
            [parameter], lr=.02, betas=(.8, .9), weight_decay=0,
            gain_rate=0, longitudinal_fusion=True, closure_certificate=False,
        )
        parameter.grad = torch.tensor([2.0, 0.0, 0.0])
        optimizer.step()
        before = parameter.detach().clone()
        gradient = torch.tensor([0.0, 1.0, 0.0])
        parameter.grad = gradient
        optimizer.step()

        state = optimizer.state[parameter]
        beta1, beta2 = optimizer.param_groups[0]["betas"]
        bias1 = 1.0 - beta1 ** state["step"]
        bias2 = 1.0 - beta2 ** state["step"]
        live = gradient / (state["mean_square"] / bias2).sqrt().add(
            optimizer.param_groups[0]["eps"]
        )
        signature = live / torch.linalg.vector_norm(live)
        averaged = state["momentum"] / bias1
        transverse = averaged - torch.dot(averaged, signature) * signature
        expected = live + beta1 * transverse
        observed = (before - parameter.detach()) / .02

        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(
            torch.dot(observed, signature), torch.linalg.vector_norm(live),
            atol=1e-6, rtol=1e-6,
        ))
        self.assertGreater(float(torch.linalg.vector_norm(transverse)), 0.0)

    def test_tangent_gauge_reexpresses_the_same_physical_vector(self):
        torch.manual_seed(111)
        factors = []
        for size in (4, 4, 5, 5):
            basis = torch.randn(size, size)
            factors.append(
                basis @ basis.transpose(0, 1) + .4 * torch.eye(size)
            )
        old_left, new_left, old_right, new_right = factors
        old_coordinates = torch.randn(4, 5)
        moved = MatrixTransport._tangent_gauge_transport(
            old_coordinates, old_left, old_right, new_left, new_right
        )
        old_physical = old_left @ old_coordinates @ old_right
        new_physical = new_left @ moved @ new_right
        self.assertTrue(
            torch.allclose(old_physical, new_physical, atol=3e-4, rtol=3e-4)
        )


if __name__ == "__main__":
    unittest.main()
