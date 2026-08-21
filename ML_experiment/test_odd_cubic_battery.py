import unittest

import torch

from ML_experiment.nd_spiral_wall import ShallowOddCubicNet
from ML_experiment.tasks import TASK_BUILDERS


class OddCubicBatteryTests(unittest.TestCase):
    def test_every_task_shape_and_gradient(self):
        for name, builder in TASK_BUILDERS.items():
            with self.subTest(task=name):
                task = builder(0)
                model = ShallowOddCubicNet(task.input_dim, task.output_dim, 16)
                output = model(task.x_train[:4])
                self.assertEqual(output.shape, (4, task.output_dim))
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
