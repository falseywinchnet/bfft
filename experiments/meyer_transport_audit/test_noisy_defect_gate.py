import unittest
import numpy as np
from .model import ReducedMeyerMap,grad,project_disk,State
from .defect_relaxation import DefectRelaxation
from .noisy_defect_gate import NoisyDefectGate,alignment


class NoisyDefectGateTests(unittest.TestCase):
    def test_deterministic_implementation_is_unchanged(self):
        m=ReducedMeyerMap(np.random.default_rng(8).normal(100,20,(12,16)),.05,40)
        original=DefectRelaxation(m,'aligned_refresh');new=NoisyDefectGate(m,'deterministic')
        z=m.initial()
        for _ in range(32):
            zn=new.step(z);old=original.step(z)
            np.testing.assert_array_equal(m.pack(zn),m.pack(old));z=old

    def test_probability_and_collinear_decisions(self):
        m=ReducedMeyerMap(np.zeros((1,20000)),.05,40)
        op=NoisyDefectGate(m,'local',11)
        one=np.ones(m.shape);zero=np.zeros(m.shape)
        for c in [-1.,-.9,-.5,0.,.5,.9,1.]:
            previous=(one,zero);extra=(c*one,np.sqrt(1-c*c)*one)
            np.testing.assert_allclose(alignment(previous,extra),c,atol=1e-14)
            accepted=op.admit_defect(previous,extra)
            self.assertAlmostEqual(float(np.mean(accepted)),(1+c)/2,delta=.02)
            if abs(c)==1:self.assertTrue(np.all(accepted==(c>0)))

    def test_only_selection_changes_and_fixed_point(self):
        m=ReducedMeyerMap(np.random.default_rng(3).normal(100,30,(12,16)),.05,40)
        for mode in ['shared','local']:
            op=NoisyDefectGate(m,mode,11);repeat=NoisyDefectGate(m,mode,11);z=m.initial()
            for _ in range(16):
                baseline=DefectRelaxation(m,'ordinary').step(z);zn=op.step(z)
                np.testing.assert_array_equal(m.pack(zn),m.pack(repeat.step(z)))
                np.testing.assert_array_equal(zn.u,baseline.u);np.testing.assert_array_equal(zn.w,baseline.w)
                for x,tx,ty,t0x,t0y,r in [(zn.u,zn.tux,zn.tuy,baseline.tux,baseline.tuy,m.ru),(zn.w,zn.twx,zn.twy,baseline.twx,baseline.twy,m.rw)]:
                    gx,gy=grad(x);bx,by=tx-gx,ty-gy;px,py=t0x-gx,t0y-gy;qx,qy=project_disk(t0x,t0y,r)
                    self.assertTrue(np.all(((np.abs(bx-px)<1e-12)&(np.abs(by-py)<1e-12))|((np.abs(bx-qx)<1e-12)&(np.abs(by-qy)<1e-12))))
                    self.assertLessEqual(float(np.max(np.hypot(bx,by))),r+1e-12)
                z=zn
        f=100+.1*np.sin(2*np.pi*np.arange(32)[None,:]/32);m=ReducedMeyerMap(f,.05,40)
        zero=np.zeros_like(f);u=np.full_like(f,100);tw=-np.cumsum((m.cw/m.etaw)*(f-u),axis=1)
        fixed=State(u,zero.copy(),zero.copy(),zero.copy(),tw,zero.copy())
        for mode in ['shared','local']:
            op=NoisyDefectGate(m,mode,29);z=fixed
            for _ in range(16):z=op.step(z)
            np.testing.assert_allclose(m.pack(z),m.pack(fixed),atol=1e-11,rtol=0)


if __name__=='__main__':unittest.main()
