import unittest
import torch
from run_projective_quotient import ProjectiveQuotient,sample_episode


class ProjectiveQuotientTests(unittest.TestCase):
    def test_scene_is_permutation_invariant(self):
        torch.manual_seed(2); model=ProjectiveQuotient(8,6,4,2)
        x=torch.randn(30,2); y=torch.randint(0,2,(30,)); q=torch.randn(9,2); p=torch.randperm(30)
        self.assertTrue(torch.allclose(model.forward_episode(x,y,q),model.forward_episode(x[p],y[p],q),atol=1e-8,rtol=1e-7))

    def test_no_direct_binary_head(self):
        model=ProjectiveQuotient(8,6,4,2); x=torch.randn(24,2); y=torch.randint(0,2,(24,)); q=torch.randn(7,2)
        self.assertEqual(model.forward_episode(x,y,q).shape,(7,2))
        self.assertFalse(any(m.out_features==2 for m in model.modules() if isinstance(m,torch.nn.Linear)))

    def test_projective_placement_receives_gradient(self):
        torch.manual_seed(4); model=ProjectiveQuotient(8,6,4,2)
        x=torch.randn(32,2); y=torch.randint(0,2,(32,)); q=torch.randn(8,2); target=torch.randint(0,2,(8,))
        torch.nn.functional.cross_entropy(model.forward_episode(x,y,q),target).backward()
        self.assertGreater(float(model.generators.grad.norm()),0); self.assertGreater(float(model.feature[0].weight.grad.norm()),0)

    def test_context_and_query_are_disjoint(self):
        c,q=sample_episode(torch.arange(100),60,30,torch.Generator().manual_seed(5))
        self.assertFalse(bool(torch.isin(c,q).any()))


if __name__=="__main__": unittest.main()
