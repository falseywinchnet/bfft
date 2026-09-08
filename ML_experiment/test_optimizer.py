import unittest

import torch

from ML_experiment.optimizer import ResidualPolarTransport
from ML_experiment.optimizers import MatrixTransport


class ResidualPolarTransportTest(unittest.TestCase):
    def test_explicit_observer_detects_operator_anisotropy(self):
        generator = torch.Generator().manual_seed(12)
        x = torch.randn(256, 4, generator=generator)
        isotropic = x @ torch.eye(4)
        anisotropic = x @ torch.diag(torch.tensor([1.0, .03, .003, .0003]))
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        optimizer = ResidualPolarTransport(
            [parameter], lr=.01, weight_decay=0, observer_center=False
        )

        isotropic_rank = optimizer.observe_residual(x, isotropic)
        optimizer._pending_global_ranks.clear()
        anisotropic_rank = optimizer.observe_residual(x, anisotropic)

        self.assertGreater(isotropic_rank, .99)
        self.assertLess(anisotropic_rank, .3)

    def test_polar_governor_releases_when_residual_becomes_isotropic(self):
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        optimizer = ResidualPolarTransport(
            [parameter], lr=.01, weight_decay=0, gain_rate=0,
            polar_beta=.5,
        )
        gradient = torch.randn(4, 4, generator=torch.Generator().manual_seed(13))

        optimizer.set_residual_entropy_rank(.1)
        parameter.grad = gradient.clone()
        optimizer.step()
        anisotropic_weight = optimizer.last_polar_weight

        optimizer.set_residual_entropy_rank(1.0)
        parameter.grad = gradient.clone()
        optimizer.step()
        isotropic_weight = optimizer.last_polar_weight

        self.assertAlmostEqual(anisotropic_weight, .45)
        self.assertAlmostEqual(isotropic_weight, .225)
        self.assertLess(isotropic_weight, anisotropic_weight)

    def test_attached_linear_observer_updates_only_from_training_backward(self):
        torch.manual_seed(14)
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 3),
        )
        optimizer = ResidualPolarTransport(
            model.parameters(), lr=.003, weight_decay=0
        ).attach_model(model)
        x = torch.randn(64, 5)
        target = torch.randn(64, 3)
        with torch.no_grad():
            model(x)
        self.assertTrue(all(not stack for stack in optimizer._activation_stacks.values()))
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

        matrix_states = [optimizer.state[p] for p in model.parameters()
                         if p.ndim == 2]
        self.assertTrue(matrix_states)
        self.assertTrue(all(0 <= s["residual_entropy_rank"] <= 1
                            for s in matrix_states))
        self.assertTrue(any(s["polar_weight"] > 0 for s in matrix_states))
        optimizer.remove_observers()
        self.assertFalse(optimizer._observer_handles)

    def test_module_filter_leaves_unobserved_matrix_on_transport(self):
        torch.manual_seed(141)
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 4),
        )
        observed = {model[2]}
        optimizer = ResidualPolarTransport(
            model.parameters(), lr=.003, weight_decay=0
        ).attach_model(model, module_filter=lambda module: module in observed)

        x = torch.randn(64, 5)
        target = torch.randn(64, 4)
        torch.nn.functional.mse_loss(model(x), target).backward()
        optimizer.step()

        self.assertEqual(optimizer.state[model[0].weight]["polar_weight"], 0)
        self.assertGreater(optimizer.state[model[2].weight]["polar_weight"], 0)
        optimizer.remove_observers()

    def test_standalone_update_matches_evaluated_experimental_arm(self):
        generator = torch.Generator().manual_seed(15)
        standalone_parameter = torch.nn.Parameter(torch.zeros(4, 5))
        experimental_parameter = torch.nn.Parameter(torch.zeros(4, 5))
        standalone = ResidualPolarTransport(
            [standalone_parameter], lr=.01, weight_decay=0, gain_rate=0
        )
        experimental = MatrixTransport(
            [experimental_parameter], lr=.01, weight_decay=0, gain_rate=0,
            longitudinal_fusion=True, residual_polar=True,
        )

        for rank in (.8, .3, .95, .2):
            gradient = torch.randn(4, 5, generator=generator)
            standalone.set_residual_entropy_rank(rank)
            experimental.set_residual_entropy_rank(rank)
            standalone_parameter.grad = gradient.clone()
            experimental_parameter.grad = gradient.clone()
            standalone.step()
            experimental.step()

        self.assertTrue(torch.allclose(
            standalone_parameter, experimental_parameter,
            atol=2e-6, rtol=2e-6,
        ))

    def test_applied_request_cannot_reverse_the_gradient(self):
        parameter = torch.nn.Parameter(torch.zeros(3, 4))
        optimizer = ResidualPolarTransport(
            [parameter], lr=.02, weight_decay=0, gain_rate=0
        )
        for seed in range(5):
            gradient = torch.randn(
                3, 4, generator=torch.Generator().manual_seed(seed)
            )
            before = parameter.detach().clone()
            parameter.grad = gradient.clone()
            optimizer.set_residual_entropy_rank(.2)
            optimizer.step()
            request = (before - parameter.detach()) / .02
            self.assertGreaterEqual(float(torch.sum(request * gradient)), -2e-5)


if __name__ == "__main__":
    unittest.main()
