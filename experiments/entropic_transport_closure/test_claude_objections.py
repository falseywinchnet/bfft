import unittest
import numpy as np
from .energy_theory import geometry, power_and_gramian
from .probe_claude_objections import observability_gramian, rule_outputs, log_second_bound, nested_problem, history_prediction, duplication_probe
from .study_energy_theory import spectrum


class ObjectionTests(unittest.TestCase):
    def test_gramian_and_tail(self):
        rng=np.random.default_rng(8); lam=np.array([1.,.99,.7,.2]); O=rng.normal(size=(3,4)); H=37
        direct=sum((O*lam[None,:]**j).T@(O*lam[None,:]**j) for j in range(H))
        W=observability_gramian(lam,O,H)
        np.testing.assert_allclose(W,direct,atol=1e-12)
        _,powered=power_and_gramian(np.diag(lam),O.T@O,H)
        np.testing.assert_allclose(W,powered,atol=1e-12)
        values,U=np.linalg.eigh(W); x=U[:,:2]@rng.normal(size=2)
        self.assertLessEqual(x@W@x,values[1]*(x@x)+1e-12)

    def test_rule_rows_match_independent_bilinear(self):
        rng=np.random.default_rng(22); n=7; a=np.ones(n)/n
        g=geometry(rng.normal(size=(n,n)),a,a,rng.normal(size=n)); _,V=spectrum(g)
        rows=rule_outputs(g,V,3)
        expected=np.array([[2*V[:,i]@(g.c*g.bilinear(V[:,k],V[:,j])) for k in range(n-1)] for j in range(3) for i in range(3)])
        np.testing.assert_allclose(rows,expected,atol=1e-12)

    def test_nested_density_cost_geometry(self):
        small=nested_problem(64); large=nested_problem(256)
        np.testing.assert_array_equal(small[0],large[0][:64,:64])
        for index in [1,2]: np.testing.assert_allclose(small[index],large[index][:64]/large[index][:64].sum())

    def test_duplicate_atoms_preserve_dynamics_but_worsen_cmin_bound(self):
        rows=duplication_probe(); first=rows[0]
        for row in rows:
            self.assertAlmostEqual(row['energy'],first['energy'],places=12)
            self.assertAlmostEqual(row['second_jet_error_64'],first['second_jet_error_64'],places=11)
            self.assertAlmostEqual(row['one_step_centered_c2_to_infinity'],first['one_step_centered_c2_to_infinity'],places=12)
        self.assertGreater(rows[-1]['log10_uniform_bound'],first['log10_uniform_bound']+3)

    def test_large_radius_log_bound_and_exact_history(self):
        rng=np.random.default_rng(25); a=np.ones(5)/5
        g=geometry(rng.normal(size=(5,5)),a,a,np.zeros(5))
        bound=log_second_bound(g,np.arange(5)*100.)
        self.assertTrue(np.isfinite(bound['log10_uniform_bound']))
        seq=np.array([.9**j+.3*.8**j for j in range(64)])
        fit=history_prediction(seq,2,0.,rng)
        self.assertLess(fit['relative_future_rms'],1e-8)


if __name__=='__main__': unittest.main()
