import unittest
import numpy as np
from .primitive_transport import ResponseCarrier,finite_run

class PrimitiveTransportTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(543)
        self.K=np.exp(self.rng.normal(size=(17,13)))

    def test_carried_action_survives_frame_change(self):
        c=ResponseCarrier(self.K,8,transport=True)
        for _ in range(5):c.apply(np.exp(self.rng.normal(size=13)),1e-10)
        x=np.exp(self.rng.normal(size=13)*2);out=self.K@x
        count=c.products;c.rebase(x,out)
        self.assertEqual(count,c.products)
        P=self.K*x[None,:]/out[:,None]
        np.testing.assert_allclose(c.Y,P@c.Q,atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(c.Q.T@c.Q,np.eye(c.Q.shape[1]),atol=1e-12)

    def test_reused_product_and_enclosure(self):
        c=ResponseCarrier(self.K,8)
        x=np.ones(13);c.apply(x,1e-8)
        request=x*(1+1e-10*self.rng.normal(size=13))
        out,bound,measured=c.apply(request,1e-8)
        self.assertFalse(measured)
        error=np.ptp(np.log(out/(self.K@request)))
        self.assertLessEqual(error,bound+1e-14)

    def test_complete_finite_trajectory(self):
        a=self.rng.random(17)+.1;a/=a.sum()
        b=self.rng.random(13)+.1;b/=b.sum()
        exact=finite_run(self.K,a,b,passes=80,carried=False)
        for transport in [False,True]:
            out=finite_run(self.K,a,b,passes=80,rank=8,transport=transport)
            error=np.ptp(out['y']-exact['y'])
            self.assertLessEqual(error,out['trajectory_bound']+1e-12)
            self.assertGreater(out['reuses'],0)
            self.assertEqual(out['products']+out['reuses'],160)

if __name__=='__main__':unittest.main()
