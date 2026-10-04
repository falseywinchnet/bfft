import unittest
import numpy as np
from .rl import Blur, RL, remainder_parts, osc0, solve
from .study import scene


class RLTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0); t = scene(32, 0); t *= 100/t.mean()
        bl = Blur(t.shape, 2.0); self.m = RL(rng.poisson(bl(t.ravel())+5).astype(float), bl, 5.)
        self.z = self.m.initial+0.3*rng.standard_normal(t.size); self.rng = rng

    def test_tangent_and_metric(self):
        lin = self.m.linearize(self.z); h = 1e-6*self.rng.standard_normal(self.z.size)
        err = self.m.step(self.z+h)-lin.next_state-lin.action(h)
        self.assertLess(np.linalg.norm(err), 1e-5*np.linalg.norm(lin.action(h)))
        u, v = self.rng.standard_normal((2, self.z.size))
        a, b = u@lin.metric(lin.action(v)), v@lin.metric(lin.action(u))
        self.assertLess(abs(a-b), 1e-10*abs(a))

    def test_exact_remainder_and_bound(self):
        lin = self.m.linearize(self.z)
        for scale in (0.01, 0.3, 2.0):
            d = scale*self.rng.standard_normal(self.z.size)
            rem = self.m.step(self.z+d)-lin.next_state-lin.action(d)
            a, b = remainder_parts(self.m, self.z, d)
            self.assertLess(np.abs(rem-a-b).max(), 1e-9*max(1, np.abs(rem).max()))
            bound = osc0(d)**2/8
            self.assertTrue(np.all(-a >= -1e-12) and np.all(b >= -1e-12))
            self.assertLessEqual(np.abs(-a).max(), bound*(1+1e-12))
            self.assertLessEqual(np.abs(b).max(), bound*(1+1e-12))
            self.assertLessEqual(np.abs(rem).max(), bound*(1+1e-12))

    def test_certified_solve_monotone(self):
        m = self.m; _, r = solve(m, 'ordinary', -np.inf, budget=200)
        _, c = solve(m, 'certified', r['objective'], budget=400, depth=8, tau=10.)
        self.assertEqual(c['status'], 'target')
        f = [t[2] for t in c['trace']]
        self.assertTrue(all(b <= a+1e-9*abs(a) for a, b in zip(f, f[1:])))


if __name__ == '__main__':
    unittest.main()
