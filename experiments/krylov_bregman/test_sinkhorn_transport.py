import unittest
import numpy as np
from .sinkhorn_transport import Sinkhorn,gauge,solve
from .sinkhorn_study import problem
from .core import finite_flow

class SinkhornTests(unittest.TestCase):
    def model(self):
        return Sinkhorn(*problem(16,0,'mixture'),.03)

    def test_matches_independent_log_update(self):
        m=self.model();y=np.linspace(-3,2,16)
        np.testing.assert_allclose(m.step(y),m.log_step(y),atol=2e-14)
        np.testing.assert_allclose(m.step(y+137),m.step(y),atol=3e-14)

    def test_tangent_and_gauge(self):
        m=self.model();rng=np.random.default_rng(3);y=rng.normal(size=16);v=gauge(rng.normal(size=16))
        lin=m.linearize(y);h=1e-5
        np.testing.assert_allclose(lin.action(v),(m.step(y+h*v)-m.step(y-h*v))/(2*h),atol=2e-9)
        self.assertLess(np.linalg.norm(lin.action(np.ones(16))),1e-14)

    def test_residual_is_actual_plan_marginal_error(self):
        m=self.model();y=np.linspace(-4,3,16);plan=m.plan(y)
        np.testing.assert_allclose(plan.sum(axis=1),m.p,atol=1e-14)
        self.assertAlmostEqual(m.error(y),np.sum(np.abs(plan.sum(axis=0)-m.q)),places=13)

    def test_extreme_state_uses_stable_fallback(self):
        m=self.model();y=np.linspace(-1000,1000,16)
        np.testing.assert_allclose(m.step(y),m.log_step(y),atol=1e-12)
        self.assertGreater(m.fallbacks,0)
        self.assertTrue(np.isfinite(m.error(y)))
        self.assertTrue(np.all(np.isfinite(m.linearize(y).action(np.arange(16)))))

    def test_all_methods_reach_same_plan(self):
        m=self.model();ref,record=solve(m,'ordinary',target=1e-11,budget=20000)
        self.assertEqual(record['status'],'target');p=m.plan(ref)
        for method in ('ordinary','fixed','discovered','anderson'):
            z,r=solve(m,method,target=1e-8,budget=20000)
            self.assertEqual(r['status'],'target',method)
            self.assertLess(np.sum(np.abs(m.plan(z)-p)),1e-7)
            self.assertLess(r['error'],1e-8)

    def test_action_accounting(self):
        m=self.model();_,b,_=finite_flow(m,m.initial,4,16,False)
        self.assertEqual(m.calls,1);self.assertEqual(m.actions,b.actions)
        m.error(m.initial);self.assertEqual(m.checks,1)

if __name__=='__main__':unittest.main()
