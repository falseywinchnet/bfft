import unittest
import numpy as np
from .model import ReducedMeyerMap
from .certificate import certificate
from .defect_relaxation import DefectRelaxation
from .diagnose_ramp_refresh import refresh_endpoint,terms


class RampDiagnosisTests(unittest.TestCase):
    def test_branch_ablation_and_gap_components(self):
        m=ReducedMeyerMap(255.*np.arange(128)[None,:]/128,.02,20)
        z=m.initial();op=DefectRelaxation(m,'refresh')
        for _ in range(80):
            expected=op.step(z)
            actual=refresh_endpoint(m,m.step(z),'uw')
            np.testing.assert_allclose(m.pack(actual),m.pack(expected),atol=1e-11,rtol=0)
            # A refresh moves the ordinary endpoint's projected witness into
            # the retained incoming slot without changing either primal field.
            old_witness=terms(m,m.step(z))
            moved_witness=terms(m,actual,True)
            for key in old_witness:
                self.assertAlmostEqual(old_witness[key],moved_witness[key],places=10)
            for branch in ['', 'u','w','uw']:
                current=refresh_endpoint(m,m.step(z),branch)
                t=terms(m,current)
                self.assertAlmostEqual(t['gap'],certificate(m,current)['gap_per_pixel'],places=11)
                for incoming in [False,True]:
                    component=terms(m,current,incoming)
                    self.assertGreaterEqual(min(component[k] for k in ['cartoon','texture','balance']),-1e-11)
            z=actual

    def test_repeated_row_invariance(self):
        f=255.*np.arange(128)[None,:]/128
        a,b=ReducedMeyerMap(f,.02,20),ReducedMeyerMap(np.repeat(f,8,axis=0),.02,20)
        for branches in ['', 'u','w','uw']:
            za,zb=a.initial(),b.initial()
            for _ in range(100):
                za=refresh_endpoint(a,a.step(za),branches)
                zb=refresh_endpoint(b,b.step(zb),branches)
            for x,y in zip(za.fields(),zb.fields()):
                np.testing.assert_allclose(y,np.repeat(x,8,axis=0),atol=1e-10,rtol=0)
            self.assertAlmostEqual(certificate(a,za)['gap_per_pixel'],certificate(b,zb)['gap_per_pixel'],places=10)


if __name__=='__main__':unittest.main()
