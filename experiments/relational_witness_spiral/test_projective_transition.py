import unittest
import torch
from run_projective_transition import ProjectiveTransition,occlusion_episode


class ProjectiveTransitionTests(unittest.TestCase):
    def test_transition_is_stochastic(self):
        torch.manual_seed(2); model=ProjectiveTransition(8,6,4,2,3)
        x=torch.randn(32,2); y=torch.randint(0,2,(32,)); state=model.context_state(x,y)
        self.assertTrue(torch.allclose(state["transition"].sum(1),torch.ones(4)))
        self.assertTrue(bool((state["transition"]>=0).all()))

    def test_context_permutation_does_not_change_prediction(self):
        torch.manual_seed(3); model=ProjectiveTransition(8,6,4,2,3)
        x=torch.randn(28,2); y=torch.randint(0,2,(28,)); q=torch.randn(7,2); p=torch.randperm(28)
        a=model.predict_episode(x,y,q)[0]; b=model.predict_episode(x[p],y[p],q)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=1e-6))

    def test_transition_and_selector_receive_gradient(self):
        torch.manual_seed(4); model=ProjectiveTransition(8,6,4,2,3)
        x=torch.randn(30,2); y=torch.randint(0,2,(30,)); q=torch.randn(8,2); target=torch.randint(0,2,(8,))
        torch.nn.functional.cross_entropy(model.predict_episode(x,y,q)[0],target).backward()
        self.assertGreater(float(model.clock.weight.grad.norm()),0)
        self.assertGreater(float(model.step_selector[0].weight.grad.norm()),0)

    def test_occlusion_uses_disjoint_inner_points(self):
        x=torch.randn(100,2); indices=torch.arange(100)
        context,query=occlusion_episode(x,indices,20,torch.Generator().manual_seed(5))
        self.assertFalse(bool(torch.isin(context,query).any())); self.assertGreater(len(query),0)


if __name__=="__main__": unittest.main()
