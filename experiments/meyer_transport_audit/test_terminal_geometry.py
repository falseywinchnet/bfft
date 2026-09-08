import unittest
import numpy as np
from .terminal_geometry import constant_bottom,measure,fields,direct
from .model import ReducedMeyerMap
from .certificate import certificate

class TerminalTests(unittest.TestCase):
    def test_gap_matches_existing_certificate(self):
        f=np.random.default_rng(8).normal(size=(12,16))*20
        m=ReducedMeyerMap(f,.05,40);z=m.initial()
        p=measure(f,z.u,*fields(m,z),m.lam,m.mu)
        self.assertAlmostEqual(p['gap_per_pixel'],certificate(m,z)['gap_per_pixel'],places=11)

    def test_direct_closes_two_conditions_only(self):
        f=np.random.default_rng(9).normal(size=(12,16))*20
        m=ReducedMeyerMap(f,.05,40);r=direct(m,m.initial())
        self.assertLess(abs(r['texture_complementarity']),1e-11)
        self.assertLess(r['balance'],1e-23)
        self.assertGreater(r['cartoon_complementarity'],1.)

    def test_poisson_certificate_known_sinusoid(self):
        y,x=np.mgrid[:32,:32];f=100+2*np.sin(2*np.pi*(x+2*y)/32)
        r=constant_bottom(f,.05,20)
        self.assertTrue(r['poisson_flux_fits'])
        self.assertLess(r['gap_per_pixel'],1e-24)
        failed=constant_bottom(f,.05,.01)
        self.assertFalse(failed['poisson_flux_fits'])
        self.assertGreater(failed['gap_per_pixel'],.01)

    def test_constant_bottom(self):
        r=constant_bottom(np.full((12,16),93.),.05,20)
        self.assertTrue(r['poisson_flux_fits'])
        self.assertEqual(r['gap_per_pixel'],0.)

if __name__=='__main__':unittest.main()
