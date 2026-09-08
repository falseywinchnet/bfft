import math
import unittest
from fractions import Fraction as F
from .certificates import pair_bound, certificates, intrinsic_acceleration_bound


class CertificateTests(unittest.TestCase):
    def test_sharp_bound_exact_rational_identity(self):
        # Constant acceleration and aligned opposite observation errors attain
        # the bound. Exact rational arithmetic verifies the algebraic identity.
        for gap in (F(1, 3), F(2), F(7)):
            for h in (F(0), F(1, 7), F(3)):
                A, ea, eb = F(5, 3), F(2, 9), F(4, 7)
                xa, xb, future = A*gap*gap/2, F(0), A*h*h/2
                ya, yb = xa + ea, xb - eb
                center = yb + h/gap*(yb-ya)
                radius = (1+h/gap)*eb + h/gap*ea + A*h*(gap+h)/2
                self.assertEqual(future-center, radius)
                ball = pair_bound(-float(gap), (float(ya), 0, 0), float(ea),
                                  0, (float(yb), 0, 0), float(eb), float(h), float(A))
                self.assertTrue(ball.contains((float(future), 0, 0), 1e-12))

    def test_exact_finite_turn(self):
        speed, omega = 2., .7
        path = lambda t: (speed/omega*math.sin(omega*t),
                          speed/omega*(1-math.cos(omega*t)), 0.)
        A = intrinsic_acceleration_bound(speed, 0, omega)
        for gap in (.1, .4, 2):
            for h in (0, .2, 1, 3):
                ball = pair_bound(-gap, path(-gap), 0, 0, path(0), 0, h, A)
                self.assertTrue(ball.contains(path(h), 1e-12))

    def test_append_retains_old_constraints(self):
        old = certificates([-2, -1], [(0,), (1,)], [.2, .3], 2, 1)
        new = certificates([-2, -1, 0], [(0,), (1,), (2,)], [.2, .3, .1], 2, 1)
        self.assertTrue(all(ball in new for ball in old))

    def test_error_monotonicity_and_recentering(self):
        a = pair_bound(-1, (0, 0), .1, 0, (1, 0), .2, 2, 1)
        b = pair_bound(-1, (0, 0), .4, 0, (1, 0), .5, 2, 1)
        self.assertEqual(a.center, b.center)
        self.assertGreater(b.radius, a.radius)
        self.assertAlmostEqual(a.radius_about((3, 4)), a.radius + 4)

    def test_fail_closed(self):
        for A in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                pair_bound(-1, (0,), .1, 0, (1,), .1, 1, A)
        with self.assertRaises(ValueError):
            pair_bound(0, (0,), .1, 0, (1,), .1, 1, 1)
        with self.assertRaises(ValueError):
            certificates([0], [(0,)], [.1], 1, 1)


if __name__ == '__main__':
    unittest.main()
