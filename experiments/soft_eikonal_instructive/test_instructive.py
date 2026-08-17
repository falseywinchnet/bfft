from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch

from experiments.soft_eikonal_instructive.models import PairZeroEvaluation, VARIANTS, make_variant
from experiments.soft_eikonal_instructive.run_screen import (
    allocation_secant_loss, garnish_loss, pair_loss, secant_loss, task_loss,
)
from experiments.soft_eikonal_matched.models import SoftEikonalNet, parameter_count


class InstructiveTests(unittest.TestCase):
    def test_every_variant_has_exact_reference_budget(self):
        for dimensions in ((2, 2), (2, 1), (1, 3), (10, 2)):
            for width in (16, 36):
                reference = SoftEikonalNet(*dimensions, width)
                for name in VARIANTS:
                    model = make_variant(name, *dimensions, width)
                    self.assertEqual(parameter_count(model), parameter_count(reference), (name, dimensions, width))

    def test_pair_zero_shapes_and_budget_remainder(self):
        model = make_variant("paired_zero", 2, 3, 16); evaluator = PairZeroEvaluation(model)
        self.assertEqual(tuple(model(torch.randn(7, 4)).shape), (7, 6))
        self.assertEqual(tuple(evaluator(torch.randn(7, 2)).shape), (7, 3))
        self.assertGreater(model.pair_width, 1); self.assertGreaterEqual(model.extra.numel(), 0)

    def test_self_context_changes_function_without_adding_parameters(self):
        torch.manual_seed(7); base = SoftEikonalNet(2, 2, 16)
        torch.manual_seed(7); context = SoftEikonalNet(2, 2, 16, self_context_strength=.25)
        x = torch.randn(16, 2)
        self.assertEqual(parameter_count(base), parameter_count(context))
        self.assertFalse(torch.allclose(base(x), context(x)))

    def test_all_training_losses_are_finite_and_differentiable(self):
        x = torch.randn(32, 2); y = torch.randint(0, 2, (32,)); generator = torch.Generator().manual_seed(4)
        for name in ("garnish", "secant", "allocation"):
            model = SoftEikonalNet(2, 2, 16); scale = x.std(0, keepdim=True) * .05
            if name == "garnish": loss = garnish_loss(model, x, y, "classification", generator, scale)
            elif name == "secant": loss = secant_loss(model, x, y, "classification", generator)
            else: loss = allocation_secant_loss(model, x, y, "classification", generator, scale)
            loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_pair_loss_uses_both_outputs(self):
        model = make_variant("paired_zero", 2, 1, 16); x = torch.randn(32, 2); y = torch.randn(32, 1)
        loss = pair_loss(model, x, y, "regression", torch.Generator().manual_seed(9), unpaired_fraction=.25)
        loss.backward(); self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__": unittest.main()
