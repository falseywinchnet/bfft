import unittest
import numpy as np
from experiments.conv_functional_current import (
    admit_cell, basis, currents, minimum, SLOPE_GRAM,
)
from experiments.convstar import synthesize_polyphase, project_signed_fibres


class FunctionalCurrentTests(unittest.TestCase):
    def test_positive_shoulder_with_negative_bernstein_coefficient(self):
        # p(u)=(u-1/2)^2+eps; c are p's degree-4 coefficients / 5.
        eps = .001
        a = np.array([.25+eps, eps, -1/12+eps, eps, .25+eps])/5
        self.assertLess(a.min(), 0)
        self.assertAlmostEqual(minimum(a)[0], eps/5)
        admitted = admit_cell(a, a.sum(), 1, SLOPE_GRAM, True)
        np.testing.assert_allclose(admitted, a, atol=1e-14)
        coefficient = admit_cell(a,a.sum(),1,SLOPE_GRAM,False)
        self.assertGreater(np.linalg.norm(coefficient-a), .01)
        paper=project_signed_fibres(a[None,:,None],np.ones((1,5,1)),
                                    np.array([[a.sum()]]))[0,:,0]
        self.assertAlmostEqual(5*basis(.5)@paper,253/9600,places=13)

    def test_gram_is_continuous_slope_metric(self):
        z,w=np.polynomial.legendre.leggauss(8)
        b=5*basis((z+1)/2)
        np.testing.assert_allclose(b.T@(w[:,None]*b)/2,SLOPE_GRAM,atol=1e-14)

    def test_random_cones_and_nested_objective(self):
        rng=np.random.default_rng(729)
        for _ in range(40):
            a=rng.normal(size=5)
            mass=float(rng.uniform(.05,2))
            a+=(mass-a.sum())/5
            c=admit_cell(a,mass,1,SLOPE_GRAM,True)
            b=admit_cell(a,mass,1,SLOPE_GRAM,False)
            self.assertGreaterEqual(minimum(c)[0],-1e-9)
            self.assertAlmostEqual(c.sum(),mass,places=9)
            self.assertLessEqual((c-a)@SLOPE_GRAM@(c-a),
                                 (b-a)@SLOPE_GRAM@(b-a)+1e-8)
            reflected=admit_cell(a[::-1],mass,1,SLOPE_GRAM,True)
            np.testing.assert_allclose(reflected,c[::-1],atol=2e-5)

    def test_affine_cardinality_mass_and_mixed_cells(self):
        rng=np.random.default_rng(43)
        for y in (np.ones(13), 2*np.arange(13)+1.,rng.normal(size=13)):
            c,p,a,s=currents(y)
            np.testing.assert_allclose(c.sum(axis=1)[:,0],np.diff(y),atol=1e-10)
            mixed=np.any(s!=s[:,0:1,:],axis=1)[:,0]
            np.testing.assert_array_equal(c[mixed],p[mixed])
            result=synthesize_polyphase(y[:,None],c,8)[:,0]
            np.testing.assert_array_equal(result[::8],y)
            if np.allclose(np.diff(y),np.diff(y)[0]):
                np.testing.assert_allclose(result,np.linspace(y[0],y[-1],97),atol=1e-12)


if __name__=='__main__':
    unittest.main()
