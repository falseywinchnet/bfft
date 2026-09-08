import unittest

import numpy as np

from .core import DIM, GRAVITY, H, JointFilter, Mode, flow, make_filter, moments, process_covariance
from .study import simulate


class TransportTests(unittest.TestCase):
    def test_ballistic_closed_form(self):
        f = JointFilter([Mode(0., .4, 0.)], augmented=False)
        initial, initial_cov = f.state()
        for _ in range(20):
            f.predict()
        mean, cov = f.state()
        expected = initial.copy()
        expected[:2] += initial[2:4]+GRAVITY*.5
        expected[2:4] += GRAVITY
        transition = np.eye(DIM)
        transition[:2, 2:4] = np.eye(2)
        np.testing.assert_allclose(mean, expected, atol=1e-12)
        np.testing.assert_allclose(cov, transition @ initial_cov @ transition.T, atol=1e-12)

    def test_kalman_equals_independent_batch_conditioning(self):
        f = JointFilter([Mode(0., .4, 0.)], augmented=False)
        mean, cov = f.state()
        mean, cov = mean[:4], cov[:4, :4]
        times = np.arange(6)*f.dt
        design = np.vstack([np.c_[np.eye(2), t*np.eye(2)] for t in times])
        offset = np.concatenate([.5*GRAVITY*t*t for t in times])
        observations = (design @ mean+offset+np.random.default_rng(8).normal(size=12)*.4)
        gram = design @ cov @ design.T+.16*np.eye(12)
        gain = np.linalg.solve(gram, design @ cov).T
        conditional = mean+gain @ (observations-design @ mean-offset)
        conditional_cov = cov-gain @ design @ cov
        for k, obs in enumerate(observations.reshape(-1, 2)):
            if k:
                f.predict()
            f.update(obs)
        transition = np.eye(4)
        transition[:2, 2:4] = times[-1]*np.eye(2)
        final = transition @ conditional+np.r_[.5*GRAVITY*times[-1]**2, GRAVITY*times[-1]]
        np.testing.assert_allclose(f.state()[0][:4], final, atol=1e-11)
        np.testing.assert_allclose(f.state()[1][:4, :4], transition @ conditional_cov @ transition.T, atol=1e-11)

    def test_nonlinear_transport_jacobian(self):
        mean = np.random.default_rng(1).normal(size=DIM)
        mode = Mode(.017, .4, .5)
        _, jac = flow(mean, mode, .05)
        numeric = np.column_stack([(flow(mean+np.eye(DIM)[k]*1e-5, mode, .05)[0]
                                    - flow(mean-np.eye(DIM)[k]*1e-5, mode, .05)[0])/2e-5
                                   for k in range(DIM)])
        np.testing.assert_allclose(jac, numeric, atol=1e-10)

    def test_observation_decomposition_with_cross_covariance(self):
        f = make_filter('joint_transport')
        f.predict()
        observation = np.array([1., 31.])
        f.update(observation)
        mean, cov = f.last_noise
        observation_map = np.c_[H, np.eye(2)]
        np.testing.assert_allclose(observation_map @ mean, observation, atol=1e-11)
        np.testing.assert_allclose(observation_map @ cov @ observation_map.T, 0., atol=1e-11)
        self.assertLess(np.linalg.norm(cov[:DIM, DIM:]), 10.)
        self.assertGreater(np.linalg.norm(cov[:DIM, DIM:]), 0.)
        self.assertGreater(np.trace(cov[-2:, -2:]), 0.)

    def test_mixture_total_covariance(self):
        mean, cov = moments(np.array([.5, .5]), np.array([[-2., 0.], [2., 0.]]),
                            np.array([np.eye(2), np.eye(2)]))
        np.testing.assert_allclose(mean, 0.)
        np.testing.assert_allclose(cov, np.diag([5., 1.]))

    def test_psd_and_normalization_through_missing_and_outliers(self):
        f = make_filter('joint_transport')
        for record in simulate(101, 'combined'):
            f.predict()
            f.update(record['observation'])
            self.assertAlmostEqual(f.weights.sum(), 1., places=12)
            self.assertGreaterEqual(f.weights.min(), 0.)
            self.assertGreater(np.linalg.eigvalsh(f.covs).min(), -1e-10)
            self.assertGreater(np.linalg.eigvalsh(f.state()[1]).min(), -1e-10)
        for mode in f.modes:
            self.assertGreater(np.linalg.eigvalsh(process_covariance(mode, .05)).min(), -1e-12)

    def test_forecast_is_pure_and_composes_without_observations(self):
        f = make_filter('joint_transport')
        f.update(np.array([.2, 30.1]))
        means, covs, weights = f.means.copy(), f.covs.copy(), f.weights.copy()
        a = f.forecast(12)
        b = f.forecast(5).forecast(7)
        np.testing.assert_array_equal(f.means, means)
        np.testing.assert_array_equal(f.covs, covs)
        np.testing.assert_array_equal(f.weights, weights)
        np.testing.assert_allclose(a.means, b.means)
        np.testing.assert_allclose(a.covs, b.covs)
        self.assertIsNone(a.last_noise)

    def test_position_bias_has_instantaneous_null_direction(self):
        # A measurement alone cannot distinguish position from camera offset.
        gauge = np.zeros(DIM)
        gauge[0], gauge[8] = 1., -1.
        np.testing.assert_array_equal(H @ gauge, np.zeros(2))

    def test_simulator_ballistic_matches_analytic_trajectory(self):
        rows = simulate(4, 'ballistic')
        initial = rows[0]['state']
        for row in rows:
            t = row['t']
            expected = np.r_[initial[:2]+initial[2:4]*t+.5*GRAVITY*t*t,
                             initial[2:4]+GRAVITY*t]
            np.testing.assert_allclose(row['state'], expected, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
