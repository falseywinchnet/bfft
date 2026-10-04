import unittest
import numpy as np
from .bpdn_adapter import SparseCoding,make_solver
from .certified_transport import discover_certified,run_certified
from .engine import run

class Tests(unittest.TestCase):
    def model(self,passes=30):
        rng=np.random.default_rng(7);D=rng.normal(size=(16,32));D/=np.linalg.norm(D,axis=0)
        S=rng.normal(size=(16,3));return SparseCoding(make_solver(D,S,.1,passes))
    def test_library_equivalence(self):
        for passes in [1,2,30]:
            m=self.model(passes);got,_=run(m,passes,False);ref=m.solver.solve()
            np.testing.assert_allclose(got,ref,atol=1e-14,rtol=1e-13)
    def test_tangent(self):
        m=self.model();z=m.initial
        for _ in range(17):z=m.step(z)
        h=np.random.default_rng(3).normal(size=z.size);h/=np.linalg.norm(h);eps=1e-6
        np.testing.assert_allclose(m.linearize(z).action(h),(m.step(z+eps*h)-m.step(z-eps*h))/(2*eps),atol=1e-9,rtol=1e-6)
    def test_bound(self):
        m=self.model();got,info=run_certified(m,300,tolerance=.1,depths=(4,8))
        ref,_=run(m,300,False)
        self.assertLessEqual(np.linalg.norm(got-ref),info['output_bound']+1e-10)

if __name__=='__main__':unittest.main()
