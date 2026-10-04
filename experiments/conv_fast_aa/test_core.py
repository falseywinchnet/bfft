import unittest
import numpy as np
from .core import tangent_integral, halfplane_coverage, conv_box, jet


class Invariants(unittest.TestCase):
    def test_affine_and_constant(self):
        y,x=np.indices((19,21)); a=.3+.01*x+.02*y
        for profile in ('box','quintic'):
            np.testing.assert_allclose(tangent_integral(a,profile=profile)[3:-3,3:-3],a[3:-3,3:-3],atol=1e-15)
        np.testing.assert_array_equal(tangent_integral(np.ones((9,9))),1)

    def test_range_complement_transpose(self):
        a=np.random.default_rng(23).random((23,25))
        for profile in ('box','quintic'):
            b=tangent_integral(a,profile=profile)
            self.assertGreaterEqual(b.min(),a.min()); self.assertLessEqual(b.max(),a.max())
            np.testing.assert_allclose(b,1-tangent_integral(1-a,profile=profile),atol=1e-14)
            np.testing.assert_allclose(b,tangent_integral(a.T,profile=profile).T,atol=1e-14)

    def test_line_integral_independent_quadrature(self):
        # Four-point Gauss on each side: exact also with quartic weight.
        a=np.random.default_rng(9).random((11,13)); gx,gy=jet(a)
        q,w=np.polynomial.legendre.leggauss(4)
        for profile in ('box','quintic'):
            result=tangent_integral(a,profile=profile)
            for y,x in ((4,4),(5,7),(7,8)):
                den=max(abs(gx[y,x]),abs(gy[y,x])); tx,ty=-gy[y,x]/den,gx[y,x]/den
                v=0
                for sign in (-1,1):
                    for t,wt in zip((q+1)/2,w/2):
                        px,py=x+sign*t*tx,y+sign*t*ty
                        ix,iy=int(np.floor(px)),int(np.floor(py)); u,z=px-ix,py-iy
                        sample=(1-u)*(1-z)*a[iy,ix]+u*(1-z)*a[iy,ix+1]+(1-u)*z*a[iy+1,ix]+u*z*a[iy+1,ix+1]
                        density=.5 if profile=='box' else 15/16*(1-t*t)**2
                        v+=wt*density*sample
                self.assertAlmostEqual(result[y,x],v,places=13)

    def test_coverage_independent_midpoint(self):
        q=(np.arange(768)+.5)/768-.5
        for n in ((1,0),(1,1),(.3,.9),(-.7,.2)):
            for d in (-.6,-.2,0,.13,.55):
                measured=np.mean(d+n[0]*q[:,None]+n[1]*q[None,:]>=0)
                self.assertLess(abs(float(halfplane_coverage(d,*n))-measured),.0015)

    def test_conv_affine(self):
        y,x=np.indices((13,15)); a=.02*x+.03*y
        np.testing.assert_allclose(conv_box(a)[3:-3,3:-3],a[3:-3,3:-3],atol=1e-13)

    def test_tensor_rgb_invariants(self):
        a=np.random.default_rng(43).random((17,19,3))
        b=tangent_integral(a,tensor=True)
        self.assertGreaterEqual(b.min(),a.min());self.assertLessEqual(b.max(),a.max())
        np.testing.assert_allclose(b,np.swapaxes(tangent_integral(np.swapaxes(a,0,1),tensor=True),0,1),atol=1e-13)
        np.testing.assert_allclose(b,tangent_integral(a[...,::-1],tensor=True)[...,::-1],atol=1e-13)
        np.testing.assert_allclose(b,np.flip(tangent_integral(np.flip(a,axis=0),tensor=True),axis=0),atol=1e-13)

    def test_zero_jet_exact_cancellation(self):
        a=np.tile([.2,.7,.2,.7,.2,.7,.2,.7,.2],(9,1))
        gx,gy=jet(a)
        np.testing.assert_array_equal(gx[:,2:-2],0)
        np.testing.assert_array_equal(gy,0)


if __name__=='__main__': unittest.main()
