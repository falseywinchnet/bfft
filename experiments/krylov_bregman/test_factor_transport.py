import unittest
import numpy as np
from .curved_meyer import MeyerQuotient
from .factor_transport import FactorChart,response,response_adjoint_basis,enrich_from_defect

class FactorTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(44)
        self.m=MeyerQuotient(self.rng.normal(size=(8,8))*40+100)
        self.z=self.m.initial.copy()
        for _ in range(4):self.z=self.m.step(self.z)

    def test_stationary_chart(self):
        m=MeyerQuotient(np.zeros((8,8)))
        chart=FactorChart.acquire(m,m.initial,4)
        self.assertEqual(chart.basis.actions,0)
        np.testing.assert_array_equal(chart.candidate(64),m.initial)

    def test_spatial_response_adjoint(self):
        m=self.m;q=self.rng.normal(size=4*m.n);Q=self.rng.normal(size=(5*m.n,3))
        np.testing.assert_allclose(Q.T@response(m,q),response_adjoint_basis(m,Q).T@q,atol=2e-13)

    def test_exact_factor_identity_across_boundaries(self):
        m=self.m;chart=FactorChart.acquire(m,self.z,4);lin=m.linearize(self.z)
        for scale in (.01,1,100):
            c=scale*self.rng.normal(size=chart.basis.actions);d=chart.basis.Q@c
            predicted=lin.next_state+lin.action(d)+response(m,chart.remainder(c))
            np.testing.assert_allclose(predicted,m.step(self.z+d),atol=3e-13)

    def test_reduced_update_equals_actual_projected_map(self):
        chart=FactorChart.acquire(self.m,self.z,4);b=chart.basis
        c=np.zeros(b.actions)
        for _ in range(16):
            source=np.zeros(b.actions);source[0]=b.beta
            nextc=source+b.H@c+chart.reduced_response.T@chart.remainder(c)
            np.testing.assert_allclose(nextc,b.Q.T@(self.m.step(self.z+b.Q@c)-self.z),atol=2e-12)
            c=nextc
        np.testing.assert_allclose(chart.coordinates(16),c,atol=2e-12)

    def test_defect_enrichment_retains_galerkin_identity(self):
        chart=enrich_from_defect(FactorChart.acquire(self.m,self.z,4))
        self.assertGreater(chart.basis.actions,4)
        b=chart.basis;c=self.rng.normal(size=b.actions)
        source=np.zeros(b.actions);source[0]=b.beta
        predicted=source+b.H@c+chart.reduced_response.T@chart.remainder(c)
        np.testing.assert_allclose(predicted,b.Q.T@(self.m.step(self.z+b.Q@c)-self.z),atol=2e-12)
        np.testing.assert_allclose(chart.reduced_response,response_adjoint_basis(self.m,b.Q),atol=2e-13)

    def test_no_full_map_in_reduced_recurrence(self):
        chart=FactorChart.acquire(self.m,self.z,4)
        self.m.step=lambda _: (_ for _ in ()).throw(AssertionError('expensive map replay'))
        self.assertTrue(np.all(np.isfinite(chart.candidate(16))))

if __name__=='__main__':unittest.main()
