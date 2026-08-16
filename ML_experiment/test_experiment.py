from __future__ import annotations

import unittest
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ML_experiment.models import parameter_count
from ML_experiment.run_benchmark import chart_loss, secant_loss
from ML_experiment.tasks import TASK_BUILDERS
from ML_experiment.variants import VARIANTS, make_variant


class SupersetTests(unittest.TestCase):
    def test_catalog_is_the_union_superset(self):
        required = {"spiral", "checkerboard", "two_moons", "pinwheel", "xor_quads", "sinusoid_bounds",
                    "radial_stripes", "swiss_cheese", "lorenz_lobes", "periodic_wells", "ripple", "ring_sdf",
                    "complex_spiral_3d", "periodic_nd", "hyperchecker", "multiscale_1d", "chirp_1d",
                    "localized_steps_1d", "fourier_mix_1d", "nd_spiral_low_rank", "nd_spiral_high_rank",
                    "hypercube_checker"}
        self.assertEqual(required, set(TASK_BUILDERS))

    def test_exact_parameter_budget(self):
        for dimensions in ((1, 1), (2, 2), (16, 2)):
            counts = {parameter_count(make_variant(name, *dimensions, 16)) for name in VARIANTS}
            self.assertEqual(len(counts), 1)

    def test_context_variants_change_function_without_parameters(self):
        x = torch.randn(11, 3); torch.manual_seed(7); base = make_variant("self_context", 3, 2, 16)
        for name in ("self_context_hard", "self_context_iterated", "self_context_uncertainty"):
            torch.manual_seed(7); other = make_variant(name, 3, 2, 16)
            self.assertEqual(parameter_count(base), parameter_count(other)); self.assertFalse(torch.allclose(base(x), other(x)))

    def test_relational_objectives_backpropagate(self):
        for name, objective in (("self_context_secant", secant_loss), ("self_context_chart", chart_loss)):
            model = make_variant(name, 2, 2, 16); x = torch.randn(32, 2); y = torch.randint(2, (32,)); g = torch.Generator().manual_seed(3)
            loss = objective(model, x, y, "classification", g, x.std(0, keepdim=True) * .055) if name.endswith("chart") else objective(model, x, y, "classification", g)
            loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_nd_tasks_have_inner_and_outer_support(self):
        for name in ("nd_spiral_low_rank", "nd_spiral_high_rank", "hypercube_checker"):
            task = TASK_BUILDERS[name](0); self.assertEqual(task.input_dim, 16); self.assertEqual(len(task.tail_x), 10)
            self.assertEqual(task.output_dim, 2); self.assertGreater(len(task.x_test), 1000)


if __name__ == "__main__": unittest.main()
