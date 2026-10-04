import unittest
import numpy as np
from .adapters import Chambolle
from .chambolle_floor import objective

class Tests(unittest.TestCase):
    def test_dual_gap_identity(self):
        rng=np.random.default_rng(83);model=Chambolle(rng.normal(size=(11,13)),.2)
        p=rng.normal(size=(2,11,13));p*=.19/np.maximum(1,np.linalg.norm(p,axis=0))
        p[0,-1,:]=0;p[1,:,-1]=0
        u,g,P,D,E=objective(model,p.ravel())
        expected=float(np.sum(model.weight*np.linalg.norm(g,axis=0)+np.sum(g*p,axis=0)))
        self.assertAlmostEqual(P-D,expected,places=11)
        self.assertGreaterEqual(P-D,0.)
        self.assertNotAlmostEqual(P,E*u.size)
    def test_constant_floor(self):
        model=Chambolle(np.full((11,13),.4))
        _,_,P,D,E=objective(model,model.initial)
        self.assertEqual(P,0.);self.assertEqual(D,0.);self.assertEqual(E,0.)

if __name__=='__main__':unittest.main()
