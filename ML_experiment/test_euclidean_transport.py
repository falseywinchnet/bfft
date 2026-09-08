import unittest

import torch

from ML_experiment.optimizers import EuclideanTransport


class EuclideanTransportTest(unittest.TestCase):
    def test_first_step_has_unit_rms_per_row_without_changing_shape(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 4))
        gradient = torch.tensor([[1.0, 2.0, -3.0, 4.0], [.1, -.2, .3, -.4]])
        optimizer = EuclideanTransport([parameter], lr=.05, weight_decay=0)
        parameter.grad = gradient.clone()
        optimizer.step()
        request = -parameter.detach() / .05
        self.assertTrue(torch.allclose(
            request.square().mean(1), torch.ones(2), atol=2e-6
        ))
        cosine = torch.nn.functional.cosine_similarity(request, gradient, dim=1)
        self.assertTrue(torch.allclose(cosine, torch.ones(2), atol=2e-6))

    def test_row_coordinate_rotation_equivariance(self):
        torch.manual_seed(4)
        transform, _ = torch.linalg.qr(torch.randn(5, 5))
        initial = torch.randn(3, 5)
        left = torch.nn.Parameter(initial.clone())
        right = torch.nn.Parameter(initial @ transform)
        left_optimizer = EuclideanTransport([left], lr=.02, weight_decay=.01)
        right_optimizer = EuclideanTransport([right], lr=.02, weight_decay=.01)
        for _ in range(6):
            gradient = torch.randn(3, 5)
            left.grad = gradient.clone()
            right.grad = gradient @ transform
            left_optimizer.step()
            right_optimizer.step()
        self.assertTrue(torch.allclose(
            right, left @ transform, atol=3e-6, rtol=3e-6
        ))

    def test_update_never_reverses_live_gradient(self):
        parameter = torch.nn.Parameter(torch.zeros(4))
        optimizer = EuclideanTransport(
            [parameter], lr=.1, weight_decay=0, restrain_transverse=False
        )
        for gradient in (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([-3.0, 1.0, -2.0, .5]),
            torch.tensor([2.0, -4.0, 1.0, -3.0]),
        ):
            before = parameter.detach().clone()
            parameter.grad = gradient.clone()
            optimizer.step()
            request = (before - parameter.detach()) / .1
            self.assertGreaterEqual(float(torch.dot(request, gradient)), -1e-6)

    def test_transport_state_is_finite_at_antipodal_turn(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        optimizer = EuclideanTransport([parameter], lr=.01, weight_decay=0)
        parameter.grad = torch.tensor([1.0, 0.0, 0.0])
        optimizer.step()
        parameter.grad = torch.tensor([-1.0, 0.0, 0.0])
        optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertTrue(torch.isfinite(
            optimizer.state[parameter]["momentum"]
        ).all())


if __name__ == "__main__":
    unittest.main()
