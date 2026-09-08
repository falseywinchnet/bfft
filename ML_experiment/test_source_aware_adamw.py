import unittest

import torch

from ML_experiment.optimizers import SourceAwareAdamW


def attach_sources(parameter, fixed, chart):
    parameter.grad = fixed + chart
    parameter._fixed_chart_grad = fixed
    parameter._context_chart_grad = chart


class SourceAwareAdamWTest(unittest.TestCase):
    def test_zero_chart_matches_adamw(self):
        left = torch.nn.Parameter(torch.tensor([1.0, -2.0, .5]))
        right = torch.nn.Parameter(left.detach().clone())
        baseline = torch.optim.AdamW([left], lr=.03, weight_decay=.02)
        source = SourceAwareAdamW([right], lr=.03, weight_decay=.02)
        for step in range(1, 6):
            gradient = torch.tensor([.3, -.1 * step, .2])
            left.grad = gradient.clone()
            attach_sources(right, gradient.clone(), torch.zeros_like(gradient))
            baseline.step()
            source.step()
        self.assertTrue(torch.allclose(left, right, atol=2e-7, rtol=2e-7))

    def test_chart_does_not_write_adam_state(self):
        left = torch.nn.Parameter(torch.zeros(3))
        right = torch.nn.Parameter(torch.zeros(3))
        left_optimizer = SourceAwareAdamW([left], lr=.01, weight_decay=0)
        right_optimizer = SourceAwareAdamW([right], lr=.01, weight_decay=0)
        fixed = torch.tensor([1.0, 2.0, -3.0])
        attach_sources(left, fixed.clone(), torch.tensor([2.0, -1.0, .5]))
        attach_sources(right, fixed.clone(), torch.tensor([-7.0, 4.0, 2.0]))
        left_optimizer.step()
        right_optimizer.step()
        for name in ("exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(
                left_optimizer.state[left][name], right_optimizer.state[right][name]
            ))

    def test_applied_request_cannot_reverse_fixed_decision(self):
        parameter = torch.nn.Parameter(torch.tensor([.2, -.4, .8]))
        optimizer = SourceAwareAdamW([parameter], lr=.1, weight_decay=0)
        fixed = torch.tensor([1.0, 2.0, -1.0])
        chart = torch.tensor([-20.0, -30.0, 20.0])
        before = parameter.detach().clone()
        attach_sources(parameter, fixed, chart)
        optimizer.step()
        request = (before - parameter.detach()) / .1
        self.assertGreaterEqual(float(torch.dot(request, fixed)), -1e-6)
        self.assertLessEqual(optimizer.last_context_request_scale, 1.0)
        self.assertGreaterEqual(optimizer.last_fixed_alignment, -1e-6)

    def test_missing_split_is_an_error(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([1.0])
        optimizer = SourceAwareAdamW([parameter])
        with self.assertRaisesRegex(RuntimeError, "context_split"):
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
