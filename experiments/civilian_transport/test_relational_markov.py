import unittest
import numpy as np
from .relational_markov import RelationalKalman as Fast, transitions
from .relational_reference import RelationalKalman as Dense, prior

class MarkovTests(unittest.TestCase):
    def test_midpoint_kernel_realization(self):
        for cells in (8,32,128):
            f,q=transitions((.3,1.5,6.),14.,cells)
            for i,l in enumerate((.3,1.5,6.)):
                for n in (0,1,cells-1):
                    d=np.sqrt(3)*(14./cells)*n/l
                    self.assertAlmostEqual(4*f[n,i,2,2],4*(1+d)*np.exp(-d),places=12)
                    np.testing.assert_allclose(f[n,i,2:,2:] @ (4*np.eye(2)) @ f[n,i,2:,2:].T+q[n,i,2:,2:],4*np.eye(2),atol=2e-13)

    def test_streams_and_full_exports(self):
        rng=np.random.default_rng(77)
        for cells in (8,32,128):
            for sever in (None,-1.,.1,3.37,14.,20.):
                for coords in ('current','zak'):
                    y=rng.normal(size=(13,3));ts=np.r_[0,np.sort(rng.uniform(0,14,11)),14.]
                    a=Fast(y[0],.35,14.,cells,coordinates=coords,sever_at=sever)
                    b=Dense(y[0],.35,14.,cells,coordinates=coords,sever_at=sever)
                    for j,(t,v) in enumerate(zip(ts[1:],y[1:])):
                        a.update(t,v,beta_steps=3 if j%3==0 else 1);b.update(t,v,beta_steps=3 if j%3==0 else 1)
                        aa=a.forecast([t,(t+14)/2,14.]);bb=b.forecast([t,(t+14)/2,14.])
                        for key in aa:np.testing.assert_allclose(aa[key],bb[key],atol=3e-8,rtol=2e-8)
                        np.testing.assert_allclose(a.position(t),bb['mean'][0],atol=3e-8)
                    for key,v in a.forecast([0,3,14]).items():
                        np.testing.assert_allclose(v,b.forecast([0,3,14])[key],atol=3e-8)
                    np.testing.assert_allclose(a.means,b.means,atol=3e-8)
                    np.testing.assert_allclose(a.covs,b.covs,atol=3e-8)

    def test_dense_cache_invalidated(self):
        a=Fast([0,0,0]);a.update(.2,[1,2,3]);r=a._dense()
        a.update(.3,[2,3,4]);self.assertIsNone(a._reference)
        self.assertIsNot(a._dense(),r)

class NativeMarkovTests(MarkovTests):
    def setUp(self):
        global Fast
        from .relational_native import RelationalKalman
        self.previous=Fast;Fast=RelationalKalman
    def tearDown(self):
        global Fast
        Fast=self.previous

if __name__=='__main__':unittest.main()
