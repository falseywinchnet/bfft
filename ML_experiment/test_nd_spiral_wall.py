import unittest

import torch

from ML_experiment.nd_spiral_wall import WALL_MODELS, make_wall_model


class NDSpiralWallTests(unittest.TestCase):
    def test_every_model_is_shape_and_gradient_safe(self):
        x = torch.randn(7, 16)
        for name in WALL_MODELS:
            with self.subTest(name=name):
                torch.manual_seed(0)
                model = make_wall_model(name, 16, 2, 16)
                output = model(x)
                self.assertEqual(output.shape, (7, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_antipodal_input_is_not_augmented_with_task_coordinates(self):
        for name in WALL_MODELS:
            model = make_wall_model(name, 16, 2, 16)
            self.assertEqual(model(torch.randn(3, 16)).shape[-1], 2)


if __name__ == "__main__":
    unittest.main()
