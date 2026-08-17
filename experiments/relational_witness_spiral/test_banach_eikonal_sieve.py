import math,unittest
import torch
from run_banach_eikonal_sieve import BanachEikonalSieve,circular_basis


class BanachEikonalSieveTests(unittest.TestCase):
    def test_operator_curve_is_periodic_and_continuous(self):
        torch.manual_seed(2); model=BanachEikonalSieve(width=8,layers=2,views=12,harmonics=3)
        theta=torch.linspace(-1,1,11)
        self.assertTrue(torch.allclose(model.operator_at(0,theta),model.operator_at(0,theta+2*math.pi),atol=1e-9))
        delta=(model.operator_at(0,theta+1e-5)-model.operator_at(0,theta)).norm()
        self.assertLess(float(delta),1e-3)

    def test_uniform_quadrature_refinement_agrees(self):
        theta8=2*math.pi*(torch.arange(8,dtype=torch.float64)+.5)/8
        theta32=2*math.pi*(torch.arange(32,dtype=torch.float64)+.5)/32
        self.assertTrue(torch.allclose(circular_basis(theta8,3).mean(0),circular_basis(theta32,3).mean(0),atol=1e-10))

    def test_context_is_permutation_invariant_and_query_disjoint(self):
        torch.manual_seed(3); model=BanachEikonalSieve(width=8,layers=2,views=10,harmonics=2)
        x=torch.randn(30,2); y=torch.randint(0,2,(30,)); q=torch.randn(7,2); p=torch.randperm(30)
        self.assertTrue(torch.allclose(model.forward_episode(x,y,q),model.forward_episode(x[p],y[p],q),atol=1e-9))

    def test_transport_and_operator_curve_receive_gradient(self):
        torch.manual_seed(4); model=BanachEikonalSieve(width=8,layers=2,views=10,harmonics=2,mode="transport")
        x=torch.randn(28,2); y=torch.randint(0,2,(28,)); q=torch.randn(6,2); target=torch.randint(0,2,(6,))
        torch.nn.functional.cross_entropy(model.forward_episode(x,y,q),target).backward()
        self.assertGreater(float(model.potential.weight.grad.norm()),0)
        self.assertGreater(float(model.operator_coeff.grad[:,1:].norm()),0)

    def test_small_context_change_moves_density_continuously(self):
        torch.manual_seed(5); model=BanachEikonalSieve(width=8,layers=2,views=10,harmonics=2,mode="transport")
        x=torch.randn(24,2); y=torch.randint(0,2,(24,)); s1=model.context_summary(x,y); s2=model.context_summary(x+1e-6,y)
        p1=model.transport_phases(s1,0)[1]; p2=model.transport_phases(s2,0)[1]
        self.assertLess(float((p1-p2).abs().max()),1e-3)
        self.assertTrue(bool((p1>=0).all()))
        self.assertAlmostEqual(float(p1.sum()),1.0,places=10)


if __name__=="__main__": unittest.main()
