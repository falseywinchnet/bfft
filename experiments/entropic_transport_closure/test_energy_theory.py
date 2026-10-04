import unittest
import numpy as np
from .energy_theory import (geometry, osc, phi, linear_factor, quadratic_factor,
                            second_jet_matrix, power_and_gramian, uniform_bounds,
                            observable_rows)


class EnergyTheory(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(207)

    def random_geometry(self, m=7, n=6):
        a=self.rng.random(m)+.1; a/=a.sum()
        b=self.rng.random(n)+.1; b/=b.sum()
        return geometry(self.rng.normal(size=(m,n))*2, a, b, self.rng.normal(size=n))

    def test_energy_and_mixed_dirichlet(self):
        for _ in range(20):
            g=self.random_geometry()
            u,v=self.rng.normal(size=(2,len(g.y)))
            E=g.energy(u)
            self.assertAlmostEqual(E,float(u@g.energy_matrix()@u),places=12)
            self.assertLessEqual(g.weighted_l1(g.bilinear(u,v)),
                                 np.sqrt(g.energy(u)*g.energy(v))+1e-13)
            self.assertLessEqual(g.weighted_l1(g.third(u)),5*osc(u)*E+1e-12)

    def test_third_derivative_independently(self):
        # Differentiate the second derivative at shifted anchors.
        g=self.random_geometry(); h=self.rng.normal(size=len(g.y)); t=1e-5
        gp=geometry(g.log_kernel,g.a,g.b,g.y+t*h)
        gm=geometry(g.log_kernel,g.a,g.b,g.y-t*h)
        finite=(gp.bilinear(h,h)-gm.bilinear(h,h))/t
        np.testing.assert_allclose(finite,g.third(h),rtol=2e-7,atol=2e-9)

    def test_finite_remainders_and_tilt_comparison(self):
        for _ in range(12):
            g=self.random_geometry()
            direction=self.rng.normal(size=len(g.y)); direction/=osc(direction)
            for radius in [.02,.2,1.,2.]:
                h=direction*radius; E=g.energy(h)
                shifted=geometry(g.log_kernel,g.a,g.b,g.y+h)
                ratio=shifted.P/g.P
                self.assertGreaterEqual(ratio.min(),np.exp(-radius)-1e-12)
                self.assertLessEqual(ratio.max(),np.exp(radius)+1e-12)
                self.assertLessEqual(shifted.energy(h),np.exp(radius)*E+1e-12)
                first=shifted.f-g.f-g.J@h
                second=first-g.bilinear(h,h)
                self.assertLessEqual(g.weighted_l1(first),linear_factor(radius)*E+1e-12)
                self.assertLessEqual(g.weighted_l1(second),quadratic_factor(radius)*E+1e-12)

    def test_phi_limits(self):
        self.assertEqual(phi(2,0),.5)
        self.assertEqual(phi(3,0),1/6)
        self.assertAlmostEqual(quadratic_factor(1e-8)/1e-8,5/6,places=7)

    def test_ledger_with_resonant_jordan_and_unit_mode(self):
        L=np.array([[1.,0.,0.],[0.,.9,1.],[0.,0.,.9]])
        W=np.array([[1.,.1,0.],[.1,2.,.3],[0.,.3,1.]])
        for H in [0,1,2,7,31,64]:
            P,S=power_and_gramian(L,W,H)
            p=np.eye(3); direct=np.zeros((3,3))
            for _ in range(H):
                direct+=p.T@W@p; p=L@p
            np.testing.assert_allclose(P,p,atol=3e-13)
            np.testing.assert_allclose(S,direct,atol=2e-11)

    def test_fixed_point_second_jet_and_uniform_bounds(self):
        n=3; a=np.ones(n)/n
        A=np.array([[.50,.35,.15],[.20,.45,.35],[.30,.20,.50]])
        for eps in [.3,.03]:
            K=(1-eps)*np.eye(n)+eps*A
            g=geometry(np.log(K),a,a,np.zeros(n))
            np.testing.assert_allclose(g.f,0,atol=1e-14)
            h=np.array([.1,-.06,-.04]); bounds=uniform_bounds(g,h)
            L=second_jet_matrix(g); lifted=np.r_[h,np.outer(h,h).ravel()]
            ell=h.copy(); q=np.zeros(n); actual=h.copy(); energy_sum=0.
            for _ in range(150):
                self.assertLessEqual(osc(actual-ell),bounds['linear']+1e-11)
                self.assertLessEqual(osc(actual-ell-q),bounds['second_jet']+1e-11)
                np.testing.assert_allclose(lifted[:n],ell+q,atol=1e-12)
                energy_sum+=g.energy(ell)
                q=g.J@q+g.bilinear(ell,ell); ell=g.J@ell
                lifted=L@lifted
                actual=geometry(np.log(K),a,a,actual).f
                actual-=actual.mean()
            self.assertLessEqual(energy_sum,float(a@(h*h))+1e-12)

    def test_observability_and_two_sample_ambiguity(self):
        J=np.diag([.9,.8,.7]); output=np.ones(3); hidden=np.array([1.,-2.,1.])
        self.assertAlmostEqual(output@hidden,0)
        self.assertAlmostEqual(output@J@hidden,0)
        self.assertAlmostEqual(output@J@J@hidden,.02)
        rows=observable_rows(J,output)
        self.assertEqual(len(rows),3)
        np.testing.assert_allclose(rows@J,rows@J@rows.T@rows,atol=1e-12)
        self.assertEqual(len(observable_rows(J,np.array([1.,0.,0.]))),1)


if __name__=='__main__':
    unittest.main()
