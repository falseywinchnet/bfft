import unittest
import numpy as np
from .probe_gaussian_closure import step, jump, fixed_precision, coupling


class GaussianClosureTests(unittest.TestCase):
    def test_matrix_jump_and_schur_marginals(self):
        A=np.array([[1.,.2],[.2,2.]])
        B=np.array([[2.,-.3],[-.3,.7]])
        for t in [1.,.1,.01]:
            pstar=fixed_precision(A,B,t); _,got=coupling(A,pstar,t)
            np.testing.assert_allclose(got,B,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(step(A,B,t,pstar),pstar,rtol=1e-12)
            p=np.eye(2)/t
            for H in range(1,65):
                p=step(A,B,t,p)
                if H in [1,2,7,16,64]:
                    got,_,_=jump(A,B,t,np.eye(2)/t,H)
                    np.testing.assert_allclose(got,p,rtol=2e-10,atol=2e-10)

    def test_isotropic_rotation_and_scalar_identity(self):
        a,b,t=1.,2.,.01; A=a*np.eye(2); B=b*np.eye(2); p=np.eye(2)/t
        scalar=1/t
        for _ in range(100):
            p=step(A,B,t,p)
            scalar=((1+t*t/(a*b))*scalar+1/b)/(t*t/a*scalar+1)
        np.testing.assert_allclose(p,np.eye(2)*scalar,rtol=1e-13)
        root=fixed_precision(A,B,t)[0,0]
        cross=(np.sqrt(t*t+4*a*b)-t)/2
        self.assertAlmostEqual(root,a/(t*cross),places=11)


if __name__=='__main__': unittest.main()
