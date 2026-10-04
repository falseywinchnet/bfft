import unittest
import numpy as np
from .probe_directional_limit import spectral_summary, activity_summary, oracle_replay, affine_minimax_three


class DirectionalLimitTests(unittest.TestCase):
    def test_isotropic_tight_bound_and_aligned_counterexample(self):
        d=12
        isotropic=activity_summary(np.eye(d))
        self.assertAlmostEqual(isotropic['effective_dimension'],d)
        self.assertAlmostEqual(isotropic['best_scalar_capture'],1/d)
        self.assertEqual(isotropic['ranks']['0.99'],d)
        aligned=activity_summary(np.tile(np.arange(d),(40,1)))
        self.assertAlmostEqual(aligned['effective_dimension'],1)
        self.assertEqual(aligned['ranks']['0.99'],1)

    def test_capture_bound_rotation_and_cancellation(self):
        rng=np.random.default_rng(9); X=rng.normal(size=(25,9)); Q=np.linalg.qr(rng.normal(size=(9,9)))[0]
        r=activity_summary(X); rotated=activity_summary(X@Q)
        np.testing.assert_allclose(r['capture'],rotated['capture'],atol=1e-13)
        self.assertLess(r['cancellation_identity_error'],1e-12)
        for k,capture in enumerate(r['capture'],start=1):
            self.assertLessEqual(capture,k/r['effective_dimension']+1e-12)
        M=X.T@X/len(X)
        values,U=np.linalg.eigh(M); P=U[:,-3:]@U[:,-3:].T
        loss=np.linalg.norm(X-X@P)**2/np.linalg.norm(X)**2
        self.assertAlmostEqual(loss,1-r['capture'][2])

    def test_affine_minimax_obstruction_is_attained(self):
        r=affine_minimax_three([0.,1.,2.],[0.,1.,0.])
        self.assertAlmostEqual(r['absolute_minimax_error'],.5)
        self.assertAlmostEqual(max(abs(x) for x in r['attaining_errors']),.5)
        rng=np.random.default_rng(801)
        for _ in range(20):
            x=np.sort(rng.random(3)); y=rng.normal(size=3)
            row=affine_minimax_three(x,y)
            self.assertAlmostEqual(max(abs(v) for v in row['attaining_errors']),row['absolute_minimax_error'],places=11)

    def test_excitation_weighting_matches_output_ensemble(self):
        from .probe_claude_objections import observability_gramian
        rng=np.random.default_rng(510); lam=np.array([.99,.7,.2]); O=rng.normal(size=(2,3)); X=rng.normal(size=(8,3)); H=10
        Sigma=X.T@X/len(X); W=observability_gramian(lam,O,H)
        w,U=np.linalg.eigh(Sigma); root=(U*np.sqrt(w))@U.T
        direct=np.mean([sum(np.linalg.norm(O@(lam**j*x))**2 for j in range(H)) for x in X])
        self.assertAlmostEqual(float(np.trace(root@W@root)),direct)

    def test_scalar_output_does_not_mean_scalar_temporal_state(self):
        lam=np.array([.99,.7,.2]); O=np.ones((1,3)); alpha=np.array([1.,-.8,.3])
        row=oracle_replay(lam,O,alpha,20)
        self.assertGreater(row['ranks']['0.99'],1)
        full=row['trials'][2]
        self.assertLess(full['actual_orbit_closed_relative_rms'],1e-12)
        self.assertGreater(row['trials'][0]['invariance_defect'],1e-3)


if __name__=='__main__': unittest.main()
