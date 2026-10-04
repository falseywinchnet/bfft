import unittest
from unittest.mock import patch
import numpy as np

from experiments.entropic_transport_closure.sinkhorn import (
    Kernel, evaluate, prepare, rollout, qnorm, solve, solve_blocks,
    projective_power, mobius_matrix,
)


class AppliedClosureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(84)
        self.K = np.exp(rng.normal(size=(13, 11)))
        self.a = rng.uniform(.2, 1, 13); self.a /= self.a.sum()
        self.b = rng.uniform(.2, 1, 11); self.b /= self.b.sum()
        self.y = rng.normal(size=11)

    def test_weighted_basis_and_true_second_derivative(self):
        kernel = Kernel(self.K)
        state = evaluate(kernel, self.a, self.b, self.y)
        model = prepare(kernel, state, depth=4, quadratic=True)
        weights = state.c/state.c.sum()
        np.testing.assert_allclose(model.U.T@(weights[:, None]*model.U), np.eye(4), atol=2e-14)
        np.testing.assert_allclose(weights@model.U, 0, atol=2e-14)
        np.testing.assert_allclose(model.H, model.H.T, atol=2e-14)
        coords = np.array([.2, -.4, .7, .3])
        h = model.U@coords
        full, _ = model.full_and_reduced(coords)
        second = 2*(full-state.d-model.Jcols@coords)
        eps = 1e-4
        plus = evaluate(kernel, self.a, self.b, state.y+eps*h).f
        minus = evaluate(kernel, self.a, self.b, state.y-eps*h).f
        np.testing.assert_allclose((plus+minus-2*state.f)/eps**2, second, atol=4e-7)

    def test_cubic_remainder_and_evolving_reduced_derivative(self):
        kernel = Kernel(self.K)
        state = evaluate(kernel, self.a, self.b, self.y)
        model = prepare(kernel, state, 4, True)
        rng = np.random.default_rng(93)
        for scale in [.01, .1, .5]:
            coords = scale*rng.normal(size=4)
            h = model.U@coords
            full, _ = model.full_and_reduced(coords)
            actual = evaluate(kernel, self.a, self.b, state.y+h).f-state.y
            # evaluate centers the input gauge; compare in quotient norm.
            self.assertLessEqual(qnorm(actual-full), 11/96*np.ptp(h)**3+2e-14)
            direction = rng.normal(size=4)
            eps = 1e-6
            _, plus = model.full_and_reduced(coords+eps*direction)
            _, minus = model.full_and_reduced(coords-eps*direction)
            np.testing.assert_allclose((plus-minus)/(2*eps),
                model.reduced_derivative(coords)@direction, atol=2e-10)
        self.assertGreater(np.linalg.norm(model.reduced_derivative(coords)-model.H), .001)

    def test_whole_trajectory_certificate_includes_escape_no_kernel_rollout(self):
        for quadratic in [False, True]:
            kernel = Kernel(self.K)
            state = evaluate(kernel, self.a, self.b, self.y)
            for _ in range(3):
                state = evaluate(kernel, self.a, self.b, state.f)
            model = prepare(kernel, state, 3, quadratic)
            with patch.object(kernel, 'apply', side_effect=AssertionError('kernel in rollout')):
                proposal = rollout(model, 12, relative_budget=1e8)
            actual = state
            for _ in range(proposal['horizon']):
                actual = evaluate(kernel, self.a, self.b, actual.f)
            error = qnorm(actual.y-state.y-proposal['displacement'])
            self.assertLessEqual(error, proposal['bound']+2e-13)
            self.assertGreater(proposal['projection_bound'], 0)

    def test_all_methods_have_actual_marginal_certificate(self):
        for method in ['ordinary', 'linear', 'quadratic', 'quadratic2']:
            result = solve(self.K, self.a, self.b, method, cooldown=0, tolerance=1e-10)
            self.assertTrue(result['converged'])
            pi = result['u'][:, None]*self.K*result['v'][None, :]
            np.testing.assert_allclose(pi.sum(axis=1), self.a, rtol=1e-12)
            np.testing.assert_allclose(pi.sum(axis=0), self.b, rtol=1e-10)

    def test_general_kernel_accepts_real_certified_advances(self):
        x = np.linspace(0, 1, 64)
        K = np.exp(-(x[:, None]-x[None, :])**2/.003)
        a = .1+np.exp(-((x-.28)/.18)**2); a /= a.sum()
        b = .1+np.exp(-((x-.68)/.2)**2); b /= b.sum()
        for method in ['linear', 'quadratic', 'quadratic2']:
            result = solve(K, a, b, method=method, tolerance=1e-9)
            self.assertTrue(result['converged'])
            self.assertGreater(result['accepted'], 0)
            self.assertLessEqual(max(result['relative_bounds']), .05)
            pi = result['u'][:, None]*K*result['v'][None, :]
            np.testing.assert_allclose(pi.sum(axis=0), b, rtol=1e-9)

    def test_true_marginal_gate_rejects_a_nonprogressing_proposal(self):
        fake = dict(displacement=np.zeros(len(self.b)), horizon=128, relative_bound=0.)
        with patch('experiments.entropic_transport_closure.sinkhorn.rollout', return_value=fake):
            result = solve(self.K, self.a, self.b, method='linear', cooldown=0)
        self.assertTrue(result['converged'])
        self.assertGreater(result['rejected'], 0)
        self.assertEqual(result['accepted'], 0)

    def test_block_jump_and_returned_full_coupling(self):
        rng = np.random.default_rng(2)
        g, h = np.arange(40)%2, np.arange(38)%2
        r, s = rng.uniform(.3, 2, len(g)), rng.uniform(.3, 2, len(h))
        a, b = np.ones(len(g))/len(g), np.ones(len(h))/len(h)
        C = np.array([[1., .001], [.001, 1.]])
        s[h == 0] *= 3
        K = r[:, None]*C[g[:, None], h[None, :]]*s[None, :]
        results = [solve_blocks(C, g, h, r, s, a, b, method=method, tolerance=1e-10)
                   for method in ['ordinary', 'power']]
        for result in results:
            self.assertTrue(result['converged'])
            pi = result['u'][:, None]*K*result['v'][None, :]
            np.testing.assert_allclose(pi.sum(axis=1), a, rtol=1e-12)
            np.testing.assert_allclose(pi.sum(axis=0), b, rtol=1e-10)
        self.assertGreater(results[0]['checks'], 1000)
        self.assertLess(results[1]['checks'], 15)
        M = mobius_matrix(C, np.array([.5, .5]), np.array([.5, .5]))
        w = np.array([.7, .3])
        for _ in range(129):
            w = M@w; w /= w.sum()
        np.testing.assert_allclose(w, projective_power(M, np.array([.7, .3]), 129), atol=2e-14)


if __name__ == '__main__':
    unittest.main()
