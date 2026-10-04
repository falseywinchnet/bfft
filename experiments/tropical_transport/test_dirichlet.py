"""Checks of the Dirichlet-curvature lemma at arbitrary (non-fixed) states."""
import unittest
import numpy as np
from .probe import Pair, problem
from .normal_form import conditionals, bilinear


class Dirichlet(unittest.TestCase):
    def test_lemma_random_states(self):
        rng = np.random.default_rng(1); worst = 0.
        for eps in (0.1, 0.01, 0.001):
            C, p, q = problem(48, 3); S = Pair(C, p, q, eps)
            for _ in range(20):
                y = rng.standard_normal(48)*rng.choice([0.1, 3, 30])
                P, Q = conditionals(S, y)
                a = np.exp(S.lp); c = a@P            # column sums of diag(a)P
                J = Q@P
                u = rng.standard_normal(48)*rng.choice([1e-3, 1, 10])
                w = rng.standard_normal(48)
                # total-variance identities
                E1 = a@(P@(u*u)-(P@u)**2); E2 = c@(Q@((P@u)**2)-(Q@(P@u))**2)
                self.assertAlmostEqual(E1, u@(c*u)-(P@u)@(a*(P@u)), delta=1e-9*max(1, abs(E1)))
                self.assertAlmostEqual(E1, u@(c*(u-J@u)), delta=1e-9*max(1, abs(E1)))
                self.assertAlmostEqual(E2, u@(c*(J@u-J@(J@u))), delta=1e-9*max(1, abs(E2)))
                lhs = abs(w@(c*bilinear(P, Q, u, u))); rhs = np.abs(w).max()*E1
                self.assertLessEqual(lhs, rhs*(1+1e-9)+1e-15); worst = max(worst, lhs/rhs)
                # B is half the second derivative of F
                t = 1e-3*max(1., 1/np.abs(u).max())
                fd = (S.F(y+t*u)+S.F(y-t*u)-2*S.F(y))/(2*t*t)
                B = bilinear(P, Q, u, u)
                # compare modulo gauge? F is not gauge-fixed here; F(y+c1)=F(y)+c1 so no drift in 2nd diff
                self.assertLess(np.abs(fd-B).max(), 1e-4*max(1e-12, np.abs(B).max())+1e-7)
        print('max lhs/rhs', worst)


if __name__ == '__main__':
    unittest.main()
