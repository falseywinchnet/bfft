import unittest
import numpy as np
from .relational_kalman import design, prior, zak, condition, forecast, btb_proximal_check, RelationalKalman


class RelationalKalmanTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0, 6, 17)
        self.y = np.c_[np.sin(self.t), self.t, .2*self.t**2]
        self.y += np.random.default_rng(2).normal(size=self.y.shape)*.35

    def test_exact_current_integrals(self):
        b = design([0, .17, 6.13, 8], cells=64)
        np.testing.assert_allclose(b[:, 1:] @ np.full(64, np.sqrt(8/64)), [0, .17, 6.13, 8])

    def test_zak_unitarity_realness(self):
        u = zak()
        np.testing.assert_allclose(u.conj().T @ u, np.eye(65), atol=2e-15)
        x = np.random.default_rng(3).normal(size=(65, 3))
        np.testing.assert_allclose(u.conj().T @ (u @ x), x, atol=5e-15)

    def test_coordinate_equivalence(self):
        a = forecast(self.t, self.y, .35, [6., 7., 8.], coordinates='current')
        b = forecast(self.t, self.y, .35, [6., 7., 8.])
        for key in ('mean', 'covariance', 'weights'):
            np.testing.assert_allclose(a[key], b[key], atol=2e-9)

    def test_likelihood_counted_once(self):
        a = forecast(self.t, self.y, .35, [8.])
        b = forecast(self.t, self.y, .35, [8.], beta_steps=7)
        for key in ('mean', 'covariance', 'weights'):
            np.testing.assert_allclose(a[key], b[key], atol=3e-9)

    def test_spatial_equivariance(self):
        r, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
        shift = np.array([30., -20., 40.])
        a = forecast(self.t, self.y, .35, [8.])
        b = forecast(self.t, self.y @ r+shift, .35, [8.])
        np.testing.assert_allclose(b['mean'], a['mean'] @ r+shift, atol=1e-9)
        np.testing.assert_allclose(b['covariance'][0], r.T @ a['covariance'][0] @ r, atol=1e-9)
        np.testing.assert_allclose(b['weights'], a['weights'], atol=1e-9)

    def test_covariance_psd_and_information(self):
        p = prior(1.5, .35)
        b = design(self.t[1:])
        _, p1, _ = condition(np.zeros((65, 3)), p, b, self.y[1:]-self.y[0], .35**2)
        self.assertGreater(np.linalg.eigvalsh(p1).min(), -1e-10)
        self.assertGreater(np.linalg.eigvalsh(p-p1).min(), -1e-10)
        _, p2, _ = condition(np.zeros((65, 3)), p, b, self.y[1:]-self.y[0], 4*.35**2)
        self.assertGreater(np.linalg.eigvalsh(p2-p1).min(), -1e-10)

    def test_future_nullspace(self):
        b = design(self.t)
        p = prior(1.5, .35, sever_at=6.)
        np.testing.assert_allclose(b[:, 49:], 0, atol=0)
        np.testing.assert_allclose(p[49:, :49], 0, atol=0)

    def test_mixture_variance_includes_branches(self):
        a = forecast(self.t, self.y, .35, [8.])
        within = np.dot(a['weights'], a['component_variance'][:, 0])*np.eye(3)
        self.assertGreater(np.linalg.eigvalsh(a['covariance'][0]-within).min(), -1e-10)

    def test_actual_proximal_iteration(self):
        # Small, noisy case admits a quick meaningful convergence test.
        t = np.linspace(0, 1, 6)
        y = np.c_[t, t*t, t*0]
        a = btb_proximal_check(t, y, 2., cells=16, steps=100)
        b = btb_proximal_check(t, y, 2., cells=16, steps=2000)
        self.assertLess(b['relative_mean_error'], a['relative_mean_error'])
        self.assertLess(b['relative_mean_error'], 1e-9)

    def test_streaming_equals_joint_conditioning(self):
        model = RelationalKalman(self.y[0], .35)
        for time, observation in zip(self.t[1:], self.y[1:]):
            model.update(time, observation, beta_steps=4)
        actual = model.forecast([6., 7., 8.])
        expected = forecast(self.t, self.y, .35, [6., 7., 8.])
        for key in ('mean', 'covariance', 'weights', 'log_evidence'):
            np.testing.assert_allclose(actual[key], expected[key], atol=3e-9)
        with self.assertRaises(ValueError):
            model.update(self.t[-1], self.y[-1])
        with self.assertRaises(ValueError):
            model.forecast([8.1])

    def test_prior_predictive_coverage(self):
        # A calibration check under the declared model, distinct from motion fixtures.
        rng = np.random.default_rng(12)
        cells = 16
        p = prior(1.5, .35, cells=cells)
        b = design(self.t[1:], cells=cells)
        f = design([8.], cells=cells)[0]
        m, post, _ = condition(np.zeros((cells+1, 3)), p, b, np.zeros((16, 3)), .35**2)
        s = b @ p @ b.T+.35**2*np.eye(16)
        gain = p @ b.T @ np.linalg.inv(s)
        q = np.linalg.cholesky(p) @ rng.normal(size=(cells+1, 1200))
        y = b @ q+.35*rng.normal(size=(16, 1200))
        errors = f @ (q-gain @ y)/np.sqrt(f @ post @ f)
        self.assertLess(abs(float(np.mean(errors**2))-1), .12)
        self.assertTrue(.93 < np.mean(np.abs(errors)<1.959963985) < .97)


if __name__ == '__main__':
    unittest.main()
