import unittest
import numpy as np
from .geometry import rotate_and_integrate, action, unit_rows
from .geometric_filter import GeometricFilter
from .baselines import ConventionalFilter, linear_model, measurement, H
from .data import make_case


class GeometryTests(unittest.TestCase):
    def test_finite_flow_composition_and_speed(self):
        rng = np.random.default_rng(9)
        u = unit_rows(rng.normal(size=(30, 3)))
        w = rng.normal(size=(30, 3))
        u1, d1 = rotate_and_integrate(u, w, .37)
        u2, d2 = rotate_and_integrate(u1, w, .61)
        u3, d3 = rotate_and_integrate(u, w, .98)
        np.testing.assert_allclose(u2, u3, atol=1e-13)
        np.testing.assert_allclose(d1+d2, d3, atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(u3, axis=1), 1., atol=1e-13)

    def test_zero_turn_and_full_circle(self):
        u = np.array([1., 0., 0.])
        turned, moved = rotate_and_integrate(u, np.zeros(3), 3.)
        np.testing.assert_allclose(turned, u)
        np.testing.assert_allclose(moved, 3*u)
        turned, moved = rotate_and_integrate(u, [0., 0., 1.], 2*np.pi)
        np.testing.assert_allclose(turned, u, atol=1e-13)
        np.testing.assert_allclose(moved, 0., atol=1e-13)

    def test_rotation_equivariance(self):
        rng = np.random.default_rng(3)
        r, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(r) < 0:
            r[:, 0] *= -1
        u, w = rng.normal(size=(2, 3))
        a, b = rotate_and_integrate(u, w, .3)
        c, d = rotate_and_integrate(r @ u, r @ w, .3)
        np.testing.assert_allclose(c, r @ a, atol=1e-13)
        np.testing.assert_allclose(d, r @ b, atol=1e-13)
        cost = action(.1, .2, w, w+u, u, .3, .2, .3, .1)
        rotated = action(.1, .2, r @ w, r @ (w+u), r @ u, .3, .2, .3, .1)
        np.testing.assert_allclose(cost, rotated, rtol=1e-13, atol=1e-12)

    def test_nonnegative_action_and_hard_constraint(self):
        self.assertEqual(action(0., 0., np.zeros(3), np.zeros(3), np.zeros(3), .1, .2, .3, .4), 0.)
        self.assertTrue(np.isinf(action(0., 1., np.zeros(3), np.zeros(3), np.zeros(3), .1, 0., .3, .4)))

    def test_average_generator_invents_boundary_crossing(self):
        t = np.linspace(0., 2., 101)
        supported = np.array([rotate_and_integrate([1., 0., 0.], [0., 0., sign], h)[1]
                              for sign in (-1, 1) for h in t]).reshape(2, len(t), 3)
        straight = np.array([rotate_and_integrate([1., 0., 0.], [0., 0., 0.], h)[1] for h in t])
        self.assertLess(supported[:, :, 0].max(), 1.5)
        self.assertGreater(straight[:, 0].max(), 1.5)
        self.assertGreater(np.linalg.norm(supported[:, -1].mean(axis=0)-straight[-1]), 1.)

    def test_covariance_and_noise_mean_identity(self):
        f = GeometricFilter(np.zeros(3), particles=128)
        f.predict(.1)
        y = np.array([.1, -.2, .3])
        f.update(y)
        mean = f.weights @ f.mean.sum(axis=1)
        np.testing.assert_allclose(mean+f.last_noise, y, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(f.cov).min(), -1e-12)
        self.assertAlmostEqual(f.weights.sum(), 1., places=12)
        self.assertEqual(f.n, 256)  # neither contamination branch is collapsed

    def test_forecasts_do_not_mutate_filter_or_random_stream(self):
        f = GeometricFilter(np.zeros(3), particles=64, seed=4)
        f.predict(.1)
        f.update(np.array([.2, .1, .05]))
        arrays = [getattr(f, name).copy() for name in ('u', 'omega', 'weights', 'mean', 'cov')]
        rng = str(f.rng.bit_generator.state)
        a = f.forecast_paths([.1]*10, 64, 3)
        b = f.forecast_paths([.1]*10, 64, 3)
        np.testing.assert_array_equal(a, b)
        for before, name in zip(arrays, ('u', 'omega', 'weights', 'mean', 'cov')):
            np.testing.assert_array_equal(before, getattr(f, name))
        self.assertEqual(rng, str(f.rng.bit_generator.state))

    def test_missing_observation_does_not_reweight(self):
        f = GeometricFilter(np.zeros(3), particles=64)
        f.predict(.1)
        weights = f.weights.copy()
        self.assertIsNone(f.update(None))
        np.testing.assert_array_equal(weights, f.weights)

    def test_standard_kalman_update_independent_formula(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(12, 12))
        p = a @ a.T+np.eye(12)
        m, y = rng.normal(size=12), rng.normal(size=3)
        updated, covariance, _, _ = measurement(m, p, y, .4, False)
        precision = np.linalg.inv(p)+H.T @ H/.16
        expected_cov = np.linalg.inv(precision)
        expected = expected_cov @ (np.linalg.solve(p, m)+H.T @ y/.16)
        np.testing.assert_allclose(updated, expected, atol=1e-12)
        np.testing.assert_allclose(covariance, expected_cov, atol=1e-12)

    def test_linear_covariances_psd(self):
        for kind in ['cv', 'ca', (.3, -.2, .1)]:
            for dt in [.001, .1, 1.]:
                _, q = linear_model(kind, dt, 2.)
                self.assertGreater(np.linalg.eigvalsh(q).min(), -1e-12)

    def test_all_methods_position_only_smoke_with_gap_outlier(self):
        case = make_case('helix', 12, True, samples=31)
        for name in ['cv', 'ca', 'imm', 'robust_imm', 'turn_ukf', 'geometric']:
            f = GeometricFilter(case['observations'][0], particles=64) if name == 'geometric' else ConventionalFilter(case['observations'][0], name)
            for k in range(1, 31):
                f.predict(case['times'][k]-case['times'][k-1])
                y = case['observations'][k]
                f.update(y if np.isfinite(y).all() else None)
                mean, cov = f.state()
                self.assertTrue(np.isfinite(mean).all())
                self.assertGreater(np.linalg.eigvalsh(cov).min(), -1e-10)
            path = f.forecast_paths([.1, .15], samples=32)
            self.assertEqual(path.shape, (32, 3, 3))
            self.assertTrue(np.isfinite(path).all())


if __name__ == '__main__':
    unittest.main()
