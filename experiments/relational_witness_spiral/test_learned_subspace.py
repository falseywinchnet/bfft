import importlib.util
from pathlib import Path
import unittest

import torch

HERE=Path(__file__).parent
SPEC=importlib.util.spec_from_file_location('learned',HERE/'run_learned_subspace.py')
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class LearnedSubspaceTests(unittest.TestCase):
    def test_dynamic_layer_starts_dense_but_has_live_mixture_gradient(self):
        torch.manual_seed(3)
        layer=M.LearnedSubspaceLinear(5,7,'gram')
        x=torch.randn(11,5)
        expected=layer.base(x).detach()
        actual=layer(x)
        torch.testing.assert_close(actual,expected)
        actual.square().mean().backward()
        final=layer.response[-1]
        self.assertGreater(float(final.weight.grad.abs().sum()+final.bias.grad.abs().sum()),0.)

    def test_static_factorization_is_linear(self):
        torch.manual_seed(4)
        layer=M.LearnedSubspaceLinear(4,6,'static')
        with torch.no_grad(): layer.mix.fill_(.4)
        a=torch.randn(3,4); b=torch.randn(3,4)
        # Bias is the only affine offset.
        lhs=layer(a+b)-layer(torch.zeros_like(a))
        rhs=(layer(a)-layer(torch.zeros_like(a)))+(layer(b)-layer(torch.zeros_like(b)))
        torch.testing.assert_close(lhs,rhs)


if __name__=='__main__': unittest.main()
