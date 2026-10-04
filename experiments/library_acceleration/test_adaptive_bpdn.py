import unittest
import numpy as np
from sporco.admm.bpdn import BPDN
from .adaptive_bpdn import solve
from .bpdn_adapter import make_solver,SparseCoding
from .engine import run
from .certified_transport import discover_certified

class Tests(unittest.TestCase):
    def problem(self):
        rng=np.random.default_rng(18);D=rng.normal(size=(16,32));D/=np.linalg.norm(D,axis=0)
        return D,rng.normal(size=(16,2))
    def test_native_schedule(self):
        D,S=self.problem();opt=BPDN.Options(dict(FastSolve=True,MaxMainIter=160,RelStopTol=0,AbsStopTol=0))
        library=BPDN(D,S,.1,opt);reference=library.solve()
        got,stats=solve(D,S,.1,maxiter=160,accelerated=False,finite=True)
        np.testing.assert_allclose(got,reference,atol=1e-14,rtol=1e-13)
        self.assertTrue(stats['rho_events'])
    def test_relaxed_quotient(self):
        D,S=self.problem();solver=make_solver(D,S,.1,100,relaxation=1.8)
        model=SparseCoding(solver);got,_=run(model,100,False)
        np.testing.assert_allclose(got,solver.solve(),atol=1e-13,rtol=1e-12)
        z=model.initial
        for _ in range(20):z=model.step(z)
        got,e=discover_certified(model,z,32,tolerance=.1)
        truth=z.copy()
        for _ in range(e['horizon']):truth=model.step(truth)
        self.assertLessEqual(np.linalg.norm(got-truth),e['bound']+1e-11)
    def test_no_skipped_events(self):
        D,S=self.problem();_,stats=solve(D,S,.1,maxiter=300,finite=True)
        for e in stats['events']:
            end=e['at']+e['horizon']
            self.assertEqual(e['at']//10,end//10)
            self.assertEqual(e['at']//64,end//64)
        self.assertEqual(stats['passes'],300)

if __name__=='__main__':unittest.main()
