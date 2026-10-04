import unittest
import numpy as np
from .multiplicative_algebra import algebra_basis,AlgebraCarrier,run
from .frame_transport import run as ordinary

class AlgebraTests(unittest.TestCase):
    def test_binary_joint_algebra_closes(self):
        W=np.array([[-1,-1],[-1,1],[1,-1],[1,1]]*3,dtype=float)
        Q,event=algebra_basis(W,32)
        self.assertEqual(event['rank'],4);self.assertTrue(event['closed'])
        np.testing.assert_allclose(Q.T@Q,np.eye(4),atol=1e-12)
        rng=np.random.default_rng(76);K=rng.random((9,12))+.01
        c=AlgebraCarrier(K,32,supplied_generators=W)
        for t in np.linspace(-3,3,40):
            x=np.exp(W@np.array([np.sin(t),np.cos(t)]))
            out,bound=c.apply(x)
            self.assertLessEqual(np.ptp(np.log(out/(K@x))),bound+1e-12)
        self.assertEqual(c.products,4);self.assertEqual(c.reuses,39)

    def test_mixed_products_are_needed(self):
        W=np.array([[-1,-1],[-1,1],[1,-1],[1,1]]*3,dtype=float)
        Q,_=algebra_basis(W,32,mixed=False)
        self.assertEqual(Q.shape[1],3)
        x=np.exp(W@np.array([.7,.9]))
        self.assertGreater(np.linalg.norm(x-Q@(Q.T@x)),.1)

    def test_axis_control_does_not_gain_mixed_terms_from_orthogonalization(self):
        rng=np.random.default_rng(99)
        W=rng.choice([-1.,1.],size=(128,4));W-=W.mean(axis=0)
        W/=np.sqrt(np.mean(W*W,axis=0))
        Q,event=algebra_basis(W,64,mixed=False)
        self.assertEqual(Q.shape[1],5)
        mixed,event=algebra_basis(W,64,mixed=True)
        self.assertEqual(mixed.shape[1],16)
        self.assertTrue(event['closed'])

    def test_live_full_trajectory(self):
        rng=np.random.default_rng(178);K=np.exp(rng.normal(size=(17,13)))
        a=np.ones(17)/17;b=np.ones(13)/13
        exact=ordinary(K,a,b,passes=64,method='ordinary')
        out=run(K,a,b,passes=64,budget=12)
        self.assertLessEqual(np.ptp(out['y']-exact['y']),out['bound']+1e-12)

if __name__=='__main__':unittest.main()
