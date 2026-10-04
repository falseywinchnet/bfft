import unittest
import numpy as np
from .frame_transport import GuardedCarrier,SpectralCarrier,AtomicCarrier,run
from .study_energy_theory import source_problem

class FrameTests(unittest.TestCase):
    def test_frame_uncertainty_contains_measured_relation_error(self):
        rng=np.random.default_rng(777);K=np.exp(rng.normal(size=(19,17)))
        c=GuardedCarrier(K,12,audit=True)
        for i in range(50):
            x=np.exp(4*np.sin(np.linspace(0,5,17)+.08*i)+rng.normal(size=17)*.001)
            c.apply(x,1e-8)
        for e in c.events:self.assertLessEqual(e['relation_error'],e['retained_uncertainty']+1e-12)
        self.assertLessEqual(c.audit_max,1e-12)

    def test_previously_failing_strong_frame(self):
        C,a,b=source_problem(128,0);K=np.exp(-C/.001)
        ordinary=run(K,a,b,method='ordinary')
        for method in ['guarded','spectral','atomic']:
            guarded=run(K,a,b,method=method,audit=True)
            error=np.ptp(guarded['y']-ordinary['y'])
            self.assertLess(error,1e-7)
            self.assertLessEqual(error,guarded['bound']+1e-12)
            self.assertLessEqual(guarded['audit_violation'],1e-12)
            self.assertGreater(guarded['dropped'],0)


if __name__=='__main__':unittest.main()
