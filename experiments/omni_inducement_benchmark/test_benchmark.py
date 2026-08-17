from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import torch

import models
from metrics import shape_metrics, tail_metrics
from render import decision_atlas, surface_atlas
from tasks import TASK_BUILDERS


class BenchmarkTests(unittest.TestCase):
    def test_every_model_forwards_and_backpropagates(self):
        task = TASK_BUILDERS["spiral_2d"](0)
        for name in models.MODEL_BUILDERS:
            with self.subTest(model=name):
                torch.manual_seed(3); model = models.make_model(name, task.input_dim, 16)
                context_x, context_y = task.x_train[:96], task.y_train[:96]
                output = model(task.x_train[96:112], context_x, context_y)
                self.assertEqual(tuple(output.shape), (16, 2))
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))

    def test_no_forbidden_activation_module_or_function(self):
        source = inspect.getsource(models)
        forbidden = "nn." + "GELU"
        forbidden_function = "F." + "gelu"
        self.assertNotIn(forbidden, source)
        self.assertNotIn(forbidden_function, source)
        x = torch.linspace(-4, 4, 31)
        expected = x * torch.sigmoid((torch.pi / torch.sqrt(torch.tensor(3.0))) * x)
        self.assertTrue(torch.allclose(models.LELU()(x), expected))

    def test_tasks_are_balanced_and_ranked(self):
        for name, builder in TASK_BUILDERS.items():
            with self.subTest(task=name):
                task = builder(0)
                self.assertGreaterEqual(task.intrinsic_rank, 2)
                self.assertLess(abs(float(task.y_train.float().mean()) - .5), .04)
                self.assertEqual(len(task.tail_x), 10)
                for labels in task.tail_y:
                    self.assertLess(abs(float(labels.float().mean()) - .5), .04)

    def test_shape_and_tail_metrics(self):
        task = TASK_BUILDERS["checkerboard_2d"](0)
        model = models.make_model("linear", 2, 16)
        shape = shape_metrics(model, task, resolution=28); tail = tail_metrics(model, task)
        self.assertIn("boundary_f1", shape); self.assertIn("tail_min_class_recall", tail)
        self.assertGreater(shape["true_components"], 2)

    def test_renderers_mark_cartesian_orientation(self):
        task = TASK_BUILDERS["spiral_2d"](0); model = models.make_model("linear", 2, 16)
        with tempfile.TemporaryDirectory() as directory:
            decision = Path(directory) / "decision.svg"; surface = Path(directory) / "surface.svg"
            decision_atlas(decision, task, {"linear": model}, resolution=12)
            surface_atlas(surface, task, {"linear": model}, resolution=11)
            self.assertIn("not mirrored", decision.read_text())
            self.assertIn("front: y min", surface.read_text())


if __name__ == "__main__": unittest.main()
