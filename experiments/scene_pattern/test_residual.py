import unittest
import numpy as np
from scipy.ndimage import gaussian_filter,shift
from .residual import local_delays,field,shifted_points

class ResidualTests(unittest.TestCase):
    def test_known_fractional_shift(self):
        rng=np.random.default_rng(22)
        a=gaussian_filter(rng.random((96,120,3)),(.8,.8,0)).astype(np.float32)
        b=shift(a,(.65,-.8,0),order=3,mode='reflect')
        centers=np.array([[.3,.3],[.5,.5],[.7,.7]])
        records=local_delays(a,b,np.ones((96,120),bool),centers)
        self.assertEqual(len(records),3)
        for r in records:np.testing.assert_allclose(r['delay'],[-.8,.65],atol=.18)
    def test_no_evidence_zero_field(self):
        f=field([],(30,40));self.assertTrue(np.all(f==0))
        q=np.array([[-.1,.5],[1.1,.5],[.5,.5]])
        np.testing.assert_allclose(shifted_points(q,f),q)
    def test_uncovered_regions_rejected(self):
        rng=np.random.default_rng(3);a=rng.random((80,80,3)).astype(np.float32)
        self.assertEqual(local_delays(a,a,np.zeros((80,80),bool),np.array([[.5,.5]])),[])
    def test_field_direction_and_boundary(self):
        f=field([dict(center=[.5,.5],delay=[1.,-.5],before=1.,after=.1)],(100,100))
        self.assertTrue(np.all(f[0]==0))
        q=np.array([[.5,.5],[1.2,.5]])
        p=shifted_points(q,f)
        self.assertGreater(p[0,0],q[0,0]);self.assertLess(p[0,1],q[0,1]);np.testing.assert_allclose(p[1],q[1])
