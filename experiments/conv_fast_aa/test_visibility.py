import unittest
import numpy as np
from .visibility import scenes,visible_moments,primitive


class Visibility(unittest.TestCase):
    def test_independent_partition(self):
        rng=np.random.default_rng(56)
        for name,triangles in scenes().items():
            for p in rng.integers(0,31,(18,2)):
                a=visible_moments(triangles,*p,oracle=True);b=visible_moments(triangles,*p,oracle=False)
                np.testing.assert_allclose(a,b,atol=2e-11,err_msg=name)
                self.assertGreaterEqual(a[:,0].min(),-1e-12)
                self.assertLessEqual(a[:,0].sum(),1+1e-12)

    def test_depth_crossing_inside_pixel(self):
        v=[[-3,-3],[9,-3],[-3,9]]
        a=primitive(v,z=(1,0,0));b=primitive(v,z=(-1,0,1))
        m=visible_moments([a,b],0,0)
        np.testing.assert_allclose(m[:,0],[.5,.5],atol=1e-12)
        np.testing.assert_allclose(m[:,1],[.125,.375],atol=1e-12)

    def test_duplicate_and_coplanar(self):
        a=primitive([[-3,-3],[9,-3],[-3,9]])
        m=visible_moments([a,a.copy()],0,0)
        np.testing.assert_allclose(m[:,0],[1,0],atol=1e-12)

    def test_overflow_is_error(self):
        a=next(iter(scenes().values()))[0]
        with self.assertRaises(ValueError):visible_moments([a]*5,0,0)


if __name__=='__main__':unittest.main()
