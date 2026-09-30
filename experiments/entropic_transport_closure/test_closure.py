"""Independent checks of the identities in FORMAL_ANALYSIS.md.

Run on the Mini: python3 -m unittest discover -s
experiments/entropic_transport_closure -p 'test_*.py' -v
"""
from fractions import Fraction as F
import unittest
import numpy as np


def geometry(K, a, b, y):
    v = np.exp(y)
    u = a / (K @ v)
    P = K * v[None, :] / (K @ v)[:, None]
    Q = K.T * u[None, :] / (K.T @ u)[:, None]
    return np.log(b / (K.T @ u)), P, Q


def tilt(P, d):
    weighted = P * np.exp(d)[None, :]
    normalizer = weighted.sum(axis=1)
    return weighted / normalizer[:, None], np.log(normalizer)


def lifted(P, Q, d):
    Pnext, logp = tilt(P, d)
    Qnext, logq = tilt(Q, -logp)
    return Pnext, Qnext, -logq


def variation(P, d):
    return P * (d[None, :] - (P @ d)[:, None])


def mobius(K, a, b):
    p, q = K[0]
    r, s = K[1]
    c = b[0] / b[1]
    return [[c * (q*a[0]*r + s*a[1]*p), c*q*s*sum(a)],
            [p*r*sum(a), p*a[0]*s + r*a[1]*q]]


def mm(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(2))
             for j in range(2)] for i in range(2)]


def mpow(M, n):
    R = [[F(1), F(0)], [F(0), F(1)]]
    while n:
        if n & 1:
            R = mm(R, M)
        M = mm(M, M)
        n //= 2
    return R


def action(M, t):
    return (M[0][0]*t + M[0][1]) / (M[1][0]*t + M[1][1])


def determinant(M):
    return M[0][0]*M[1][1] - M[0][1]*M[1][0]


class ClosureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.K = np.exp(rng.normal(size=(5, 4)))
        self.a = rng.uniform(.2, 1, 5); self.a /= self.a.sum()
        self.b = rng.uniform(.2, 1, 4); self.b /= self.b.sum()
        self.y = rng.normal(size=4)

    def test_exact_finite_geometry_and_increment_trajectory(self):
        y = self.y.copy()
        f, P, Q = geometry(self.K, self.a, self.b, y)
        d = f-y
        for _ in range(25):
            P, Q, dn = lifted(P, Q, d)
            y += d
            f, Pr, Qr = geometry(self.K, self.a, self.b, y)
            np.testing.assert_allclose(P, Pr, atol=2e-14)
            np.testing.assert_allclose(Q, Qr, atol=2e-14)
            np.testing.assert_allclose(dn, f-y, atol=2e-14)
            d = dn

    def test_jacobian_and_its_evolution(self):
        y = self.y
        f, P, Q = geometry(self.K, self.a, self.b, y)
        direction = np.array([.3, -.7, 1.2, -.5])
        eps = 1e-5
        fp, Pp, Qp = geometry(self.K, self.a, self.b, y+eps*direction)
        fm, Pm, Qm = geometry(self.K, self.a, self.b, y-eps*direction)
        np.testing.assert_allclose((fp-fm)/(2*eps), Q@P@direction, atol=2e-10)
        dj = variation(Q, -P@direction)@P + Q@variation(P, direction)
        np.testing.assert_allclose((Qp@Pp-Qm@Pm)/(2*eps), dj, atol=2e-10)
        self.assertGreater(np.linalg.norm(dj), .001)

    def test_gauge_and_fixed_cross_ratios(self):
        f, P, Q = geometry(self.K, self.a, self.b, self.y)
        fg, Pg, Qg = geometry(self.K, self.a, self.b, self.y+2.3)
        np.testing.assert_allclose(fg, f+2.3)
        np.testing.assert_allclose(Pg, P)
        np.testing.assert_allclose(Qg, Q)
        Pn, Qn, _ = lifted(P, Q, f-self.y)
        for V, Vn in [(P, Pn), (Q, Qn)]:
            ratio = lambda X: X[0, 0]*X[1, 1]/(X[0, 1]*X[1, 0])
            self.assertAlmostEqual(ratio(V), ratio(Vn), places=12)

    def test_moving_metric_and_joint_coupling(self):
        f, P, Q = geometry(self.K, self.a, self.b, self.y)
        d = f-self.y
        pi = self.a[:, None]*P
        c = pi.sum(axis=0)
        np.testing.assert_allclose(c, self.b*np.exp(-d))
        np.testing.assert_allclose(c[:, None]*Q, pi.T)
        J = Q@P
        S = np.sqrt(c)[:, None]*J/np.sqrt(c)[None, :]
        np.testing.assert_allclose(S, S.T, atol=2e-15)
        eig = np.linalg.eigvalsh(S)
        self.assertGreaterEqual(eig.min(), -2e-15)
        self.assertLessEqual(eig.max(), 1+2e-15)
        Pn, Qn, dn = lifted(P, Q, d)
        pin = np.exp(-np.log(P@np.exp(d)))[:, None]*pi*np.exp(d)[None, :]
        np.testing.assert_allclose(pin, self.a[:, None]*Pn)
        np.testing.assert_allclose(pin.sum(axis=0), self.b*np.exp(-dn))

    def test_variance_source_is_nonzero(self):
        _, P, _ = geometry(self.K, self.a, self.b, self.y)
        d = np.array([-.5, .1, .4, 1.3])
        variance = P@(d*d) - (P@d)**2
        eps = 1e-4
        plus = np.log(P@np.exp(eps*d))
        minus = np.log(P@np.exp(-eps*d))
        np.testing.assert_allclose((plus+minus)/eps**2, variance, atol=5e-8)
        self.assertTrue(np.all(variance > 0))

    def test_exact_nonlinear_sources_and_global_remainder_bound(self):
        f, P, Q = geometry(self.K, self.a, self.b, self.y)
        rng = np.random.default_rng(105)
        for scale in [.01, .2, 1, 4]:
            for _ in range(8):
                d = scale*rng.normal(size=4)
                rp = np.log(P@np.exp(d)) - P@d
                e = -np.log(P@np.exp(d))
                rq = np.log(Q@np.exp(e)) - Q@e
                fn, _, _ = geometry(self.K, self.a, self.b, self.y+d)
                remainder = fn-f-Q@P@d
                np.testing.assert_allclose(remainder, Q@rp-rq, atol=3e-14)
                self.assertLessEqual(np.max(np.abs(remainder)), np.ptp(d)**2/8 + 1e-14)

    def test_frozen_polynomial_rollout_error_certificate(self):
        y = self.y.copy()
        f, P, Q = geometry(self.K, self.a, self.b, y)
        J, d = Q@P, f-y
        s = np.zeros_like(y)
        actual, bound = y.copy(), 0.
        for _ in range(10):
            bound += np.ptp(s)**2/8
            s = d+J@s
            actual, _, _ = geometry(self.K, self.a, self.b, actual)
            self.assertLessEqual(np.max(np.abs(y+s-actual)), bound+3e-14)

    def test_exact_rational_state_and_changing_tangent_jump(self):
        K = [[F(2), F(1)], [F(1), F(3)]]
        a, b = [F(2, 5), F(3, 5)], [F(3, 7), F(4, 7)]
        M = mobius(K, a, b)
        t0 = F(7, 3)
        t, tangent = t0, F(1)
        derivatives = []
        for n in range(1, 13):
            derivative = determinant(M)/(M[1][0]*t+M[1][1])**2
            tangent *= derivative
            derivatives.append(derivative)
            # Independent ordinary alternating row/column scaling.
            u = [a[i]/(K[i][0]*t+K[i][1]) for i in range(2)]
            v = [b[j]/sum(K[i][j]*u[i] for i in range(2)) for j in range(2)]
            t = v[0]/v[1]
            Mn = mpow(M, n)
            self.assertEqual(t, action(Mn, t0))
            self.assertEqual(tangent, determinant(Mn)/(Mn[1][0]*t0+Mn[1][1])**2)
        self.assertNotEqual(derivatives[0], derivatives[1])
        # Cayley-Hamilton reduces every matrix power to alpha M + beta I.
        trace, det = M[0][0]+M[1][1], determinant(M)
        alpha, beta = F(0), F(1)
        for n in range(13):
            polynomial = [[alpha*M[i][j] + (beta if i == j else 0)
                           for j in range(2)] for i in range(2)]
            self.assertEqual(polynomial, mpow(M, n))
            alpha, beta = trace*alpha+beta, -det*alpha

    def test_large_block_family_exact_reduction(self):
        rng = np.random.default_rng(7)
        g = np.array([0, 1, 0, 0, 1, 1, 0])
        h = np.array([1, 0, 1, 0, 1, 0])
        C = np.array([[2., 1.], [1., 3.]])
        r, s = rng.uniform(.5, 2, len(g)), rng.uniform(.5, 2, len(h))
        K = r[:, None]*C[g[:, None], h[None, :]]*s[None, :]
        a, b = rng.uniform(.1, 1, len(g)), rng.uniform(.1, 1, len(h))
        a /= a.sum(); b /= b.sum()
        A, B = np.bincount(g, weights=a), np.bincount(h, weights=b)
        # Arbitrary initial v enters the block manifold after one full pass.
        v = rng.uniform(.3, 2, len(h))
        v = b/(K.T@(a/(K@v)))
        w = np.bincount(h, weights=s*v)
        M = np.asarray(mobius(C, A, B))
        initial_ratio = w[0]/w[1]
        for n in range(1, 16):
            v = b/(K.T@(a/(K@v)))
            w = B/(C.T@(A/(C@w)))
            np.testing.assert_allclose(v, (b/s)*(w/B)[h], atol=2e-14)
            Mn = np.linalg.matrix_power(M, n)
            self.assertAlmostEqual(w[0]/w[1], action(Mn, initial_ratio), places=12)


if __name__ == '__main__':
    unittest.main()
