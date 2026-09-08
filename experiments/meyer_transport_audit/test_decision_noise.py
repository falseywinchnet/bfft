import unittest
from unittest.mock import patch
import numpy as np
from .model import ReducedMeyerMap,State,grad,project_disk
from .defect_relaxation import DefectRelaxation
from .decision_noise import DecisionNoise


class DecisionNoiseTests(unittest.TestCase):
    def test_candidates_feasibility_reproducibility_and_fft_budget(self):
        m=ReducedMeyerMap(np.random.default_rng(3).normal(100,30,(16,16)),.05,40)
        for mode in ['random_global','random_branch','random_site','random_independent','random_gated']:
            a,b=DecisionNoise(m,mode,11),DecisionNoise(m,mode,11)
            z=m.initial()
            for _ in range(8):
                base=DefectRelaxation(m,'ordinary').step(z)
                fft,ifft=np.fft.fft2,np.fft.ifft2
                with patch('numpy.fft.fft2',wraps=fft) as ff,patch('numpy.fft.ifft2',wraps=ifft) as inv:
                    nextz=a.step(z)
                    self.assertEqual(ff.call_count,2);self.assertEqual(inv.call_count,2)
                np.testing.assert_array_equal(m.pack(nextz),m.pack(b.step(z)))
                np.testing.assert_array_equal(nextz.u,base.u);np.testing.assert_array_equal(nextz.w,base.w)
                for x,tx,ty,bt,bty,r in [(nextz.u,nextz.tux,nextz.tuy,base.tux,base.tuy,m.ru),(nextz.w,nextz.twx,nextz.twy,base.twx,base.twy,m.rw)]:
                    gx,gy=grad(x);px,py=bt-gx,bty-gy;qx,qy=project_disk(bt,bty,r)
                    bx,by=tx-gx,ty-gy
                    self.assertLessEqual(float(np.max(np.hypot(bx,by))),r+1e-12)
                    self.assertTrue(np.all(((np.abs(bx-px)<1e-12)&(np.abs(by-py)<1e-12))|((np.abs(bx-qx)<1e-12)&(np.abs(by-qy)<1e-12))))
                z=nextz

    def test_shared_nonconstant_fixed_point(self):
        f=100+.1*np.sin(2*np.pi*np.arange(32)[None,:]/32);m=ReducedMeyerMap(f,.05,40)
        zero=np.zeros_like(f);u=np.full_like(f,100);tw=-np.cumsum((m.cw/m.etaw)*(f-u),axis=1)
        z=State(u,zero.copy(),zero.copy(),zero.copy(),tw,zero.copy())
        for seed in [11,29,47]:
            for mode in ['random_global','random_branch','random_site','random_independent','random_gated']:
                op=DecisionNoise(m,mode,seed);x=z
                for _ in range(8):x=op.step(x)
                np.testing.assert_allclose(m.pack(x),m.pack(z),atol=1e-11,rtol=0)


if __name__=='__main__':unittest.main()
