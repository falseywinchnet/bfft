import unittest
import numpy as np
from skimage.restoration import denoise_tv_chambolle, richardson_lucy
from .adapters import Chambolle, RichardsonLucy
from .engine import run


class Tests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(29)
        self.image = self.rng.uniform(.1, 1, (13, 17))
        self.psf = self.rng.uniform(.1, 1, (3, 5)); self.psf /= self.psf.sum()

    def test_tv_library_equivalence(self):
        for shape in [(13,17), (5,7,3)]:
            im=self.rng.random(shape)
            for count in [1,2,17]:
                got,_=run(Chambolle(im),count-1,False)
                ref=denoise_tv_chambolle(im,weight=.1,eps=0,max_num_iter=count)
                np.testing.assert_allclose(got,ref,atol=3e-15,rtol=3e-14)

    def test_rl_library_equivalence(self):
        for count in [1,2,17]:
            got,_=run(RichardsonLucy(self.image,self.psf),count,False)
            ref=richardson_lucy(self.image,self.psf,num_iter=count,clip=False)
            np.testing.assert_allclose(got,ref,atol=1e-14,rtol=1e-14)

    def test_tangents(self):
        for m in [Chambolle(self.image),RichardsonLucy(self.image,self.psf)]:
            x=m.initial.copy()
            for _ in range(8):x=m.step(x)
            h=self.rng.normal(size=x.size);h/=np.linalg.norm(h)
            if isinstance(m,Chambolle):
                hh=h.reshape((m.ndim,)+m.shape)
                hh[0,-1,:]=0;hh[1,:,-1]=0
            lin=m.linearize(x);eps=1e-6
            numeric=(m.step(x+eps*h)-m.step(x-eps*h))/(2*eps)
            np.testing.assert_allclose(lin.action(h),numeric,atol=1e-9,rtol=1e-6)
            np.testing.assert_allclose(lin.next_state,m.step(x),atol=1e-14)

    def test_budget_and_zero_residual(self):
        for m in [Chambolle(np.ones((8,9))),RichardsonLucy(self.image,self.psf)]:
            for count in [0,1,18,70]:
                out,stats=run(m,count)
                self.assertEqual(stats['passes'],count)
                self.assertTrue(np.isfinite(out).all())


if __name__=='__main__':unittest.main()
