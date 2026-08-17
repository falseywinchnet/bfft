import unittest
import torch
from run_associative_shells import AssociativeShellNet,sample_episode


class AssociativeShellTests(unittest.TestCase):
    def test_memory_is_permutation_invariant(self):
        torch.manual_seed(2); model=AssociativeShellNet(8,2); x=torch.randn(20,2); y=torch.randint(0,2,(20,)); p=torch.randperm(20)
        self.assertTrue(torch.allclose(model.memory(x,y),model.memory(x[p],y[p])))

    def test_context_and_query_are_disjoint(self):
        indices=torch.arange(100); c,q=sample_episode(indices,60,30,torch.Generator().manual_seed(4))
        self.assertFalse(bool(torch.isin(c,q).any()))

    def test_memory_changes_output_and_receives_gradient(self):
        torch.manual_seed(5); model=AssociativeShellNet(8,3); cx=torch.randn(24,2); cy=torch.randint(0,2,(24,)); q=torch.randn(7,2)
        memory=model.memory(cx,cy); output=model(q,memory); zero=model(q,torch.zeros_like(memory))
        self.assertFalse(torch.allclose(output,zero)); output.square().mean().backward()
        self.assertGreater(float(model.key[0].weight.grad.norm()),0)
        self.assertGreater(float(model.value[0].weight.grad.norm()),0)


if __name__=='__main__': unittest.main()
