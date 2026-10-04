import unittest
import numpy as np
from .sporco_adapter import make_solver,TVDeconv
from .certified_transport import discover_certified,run_certified
from .engine import run


class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.image=rng.random((16,19));self.psf=rng.uniform(.1,1,(3,5));self.psf/=self.psf.sum()
    def model(self,passes=30):return TVDeconv(make_solver(self.image,self.psf,.08,passes))
    def test_library_equivalence(self):
        for passes in [1,2,9,30]:
            model=self.model(passes);got,_=run(model,passes-1,False)
            ref=model.solver.solve()
            np.testing.assert_allclose(got,ref,atol=2e-14,rtol=2e-13)
    def test_tangent_and_nonexpansive(self):
        model=self.model();rng=np.random.default_rng(45);z=model.initial
        for _ in range(9):z=model.step(z)
        h=rng.normal(size=z.size);h/=np.linalg.norm(h);eps=1e-6
        lin=model.linearize(z)
        np.testing.assert_allclose(lin.action(h),(model.step(z+eps*h)-model.step(z-eps*h))/(2*eps),atol=2e-9,rtol=1e-5)
        for _ in range(12):
            a=rng.normal(size=z.size);b=rng.normal(size=z.size)
            self.assertLessEqual(np.linalg.norm(model.step(a)-model.step(b)),np.linalg.norm(a-b)*(1+1e-14))
    def test_certificate(self):
        model=self.model();z=model.initial
        for _ in range(40):z=model.step(z)
        for tol in [.01,.1,1.]:
            got,info=discover_certified(model,z,32,tolerance=tol)
            truth=z.copy()
            for _ in range(info['horizon']):truth=model.step(truth)
            self.assertLessEqual(np.linalg.norm(got-truth),info['bound']+1e-12)
        got,info=run_certified(model,200,tolerance=.1)
        ref,_=run(model,200,False)
        self.assertLessEqual(np.linalg.norm(got-ref),info['output_bound']+1e-11)


if __name__=='__main__':unittest.main()
