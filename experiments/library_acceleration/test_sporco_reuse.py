import unittest
import numpy as np
from sporco.admm.tvl2 import TVL2Deconv
from .sporco_reuse import TVL2DeconvReuse

class Tests(unittest.TestCase):
    def test_native_controls(self):
        rng=np.random.default_rng(40)
        for complex_data in [False,True]:
          for relaxation in [1.,1.8]:
            im=rng.random((16,19));psf=np.ones((3,3))/9
            if complex_data:im=im+1j*rng.random(im.shape)
            opts=dict(MaxMainIter=60,RelStopTol=0,AbsStopTol=0,RelaxParam=relaxation,LinSolveCheck=True)
            a=TVL2Deconv(psf,im,.03,TVL2Deconv.Options(opts))
            b=TVL2DeconvReuse(psf,im,.03,TVL2Deconv.Options(opts))
            out=a.solve();got=b.solve()
            np.testing.assert_allclose(got,out,atol=3e-12,rtol=1e-11)
            np.testing.assert_allclose(b.getitstat().Rho,a.getitstat().Rho,atol=1e-10,rtol=1e-10)
            self.assertEqual(a.k,b.k)
    def test_one_less_forward_transform(self):
        im=np.random.default_rng(4).random((13,17));psf=np.ones((3,3))/9
        counts=[]
        for cls in [TVL2Deconv,TVL2DeconvReuse]:
            solver=cls(psf,im,.1,TVL2Deconv.Options(dict(MaxMainIter=1,FastSolve=True,AutoRho=dict(Enabled=False))))
            original=solver.fftn;calls=[]
            def counted(*args,**kwargs):
                calls.append(1);return original(*args,**kwargs)
            solver.fftn=counted;solver.solve();counts.append(len(calls))
        self.assertEqual(counts[0]-counts[1],1)

if __name__=='__main__':unittest.main()
