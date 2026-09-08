import unittest
import numpy as np
from .dynamic_factors import disk_direction,gap_and_direction,choose_factor,first_branch_event
from .model import ReducedMeyerMap,State,project_disk
from .certificate import certificate
from experiments.benchmark_meyer_flow_jump import benchmark_scene

class DynamicTests(unittest.TestCase):
    def test_boundary_derivative_both_directions(self):
        x=np.array([1.]);y=np.array([0.])
        for h,expected in [(1.,0.),(-1.,-1.)]:
            dx,dy=disk_direction(x,y,np.array([h]),y,1.)
            self.assertAlmostEqual(float(dx[0]),expected)
            eps=1e-7;p,_=project_disk(x+eps*h,y,1.)
            np.testing.assert_allclose((p-x)/eps,dx,atol=1e-8)

    def test_gap_derivative_and_value(self):
        rng=np.random.default_rng(312);m=ReducedMeyerMap(benchmark_scene(16),.05,40)
        z=State(*(rng.normal(size=m.shape)*8 for _ in range(6)))
        h=State(*(rng.normal(size=m.shape) for _ in range(6)))
        value,slope=gap_and_direction(m,z,h)
        self.assertAlmostEqual(value,certificate(m,z)['gap'],places=7)
        eps=1e-5
        fd=(certificate(m,m.add_scaled(z,h,eps))['gap']-certificate(m,m.add_scaled(z,h,-eps))['gap'])/(2*eps)
        self.assertAlmostEqual(slope,fd,delta=1e-3)

    def test_event_location(self):
        m=ReducedMeyerMap(np.zeros((8,8)),.05,40)
        zero=np.zeros(m.shape);z=State(*(zero.copy() for _ in range(6)))
        h=State(zero,zero,np.ones(m.shape)*2,zero,zero,zero)
        self.assertAlmostEqual(first_branch_event(z,h,m),m.ru/2)

    def test_acceptance_contract_across_trajectory(self):
        for lam,mu in [(.02,20),(.05,40),(.1,80)]:
            m=ReducedMeyerMap(benchmark_scene(32),lam,mu);z=m.initial()
            for _ in range(24):
                base=m.step(z);z,d=choose_factor(m,z,base)
                self.assertGreaterEqual(d['alpha'],1)
                self.assertLessEqual(d['alpha'],2)
                self.assertLessEqual(certificate(m,z)['gap'],d['base_gap']+d['roundoff_allowance']+1e-9)

if __name__=='__main__':unittest.main()
