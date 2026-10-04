"""Independent exact-arithmetic checks for the warped-pixel CONV* addendum."""
from fractions import Fraction as F
from math import comb
import unittest


def bernstein(n, i):
    p = [F(0)] * (n + 1)
    for j in range(n - i + 1):
        p[i + j] = F(comb(n, i) * comb(n - i, j) * (-1)**j)
    return p


def evaluate(p, t):
    return sum(a * t**k for k, a in enumerate(p))


class StarWarpCompilation(unittest.TestCase):
    def test_primitive_identity_as_exact_polynomial_coefficients(self):
        for i in range(6):
            primitive = [sum(bernstein(6, j)[k] for j in range(i + 1, 7)) / 6
                         for k in range(7)]
            self.assertEqual([k * primitive[k] for k in range(1, 7)], bernstein(5, i))
            self.assertEqual(evaluate(primitive, F(0)), 0)
            self.assertEqual(evaluate(primitive, F(1)), F(1, 6))

    def test_rectangle_contraction_against_independent_monomial_integrals(self):
        a, b, c, d = F(2, 7), F(5, 6), F(1, 9), F(4, 5)
        weights = []
        for left, right in [(a, b), (c, d)]:
            axis = []
            for i in range(6):
                primitive = [sum(bernstein(6, j)[k] for j in range(i + 1, 7)) / 6
                             for k in range(7)]
                axis.append(evaluate(primitive, right) - evaluate(primitive, left))
            self.assertEqual(sum(axis), right - left)
            self.assertTrue(all(w > 0 for w in axis))
            weights.append(axis)
        direct = compiled = F(0)
        for j in range(6):
            for i in range(6):
                control = F((17*i + 23*j) % 37, 37)
                compiled += control * weights[0][i] * weights[1][j]
                for x, vx in enumerate(bernstein(5, i)):
                    for y, vy in enumerate(bernstein(5, j)):
                        direct += control*vx*vy*(b**(x+1)-a**(x+1))/(x+1)*(d**(y+1)-c**(y+1))/(y+1)
        self.assertEqual(compiled, direct)

    def test_tail_identity_and_enclosure_in_exact_rationals(self):
        for degree in range(7):
            n = degree + 1
            for r in [F(1, 100), F(1, 8), F(1, 2), F(9, 10)]:
                tail = r**n/2*(r*(1+r)/(1-r)**3+(2*n+3)*r/(1-r)**2+(n+1)*(n+2)/(1-r))
                positive_series = sum(F(comb(k+2, 2))*r**k for k in range(degree+1))
                self.assertEqual(tail, 1/(1-r)**3-positive_series)
                for z in [-r, -r/3, F(0), r/3, r]:
                    approximation = sum(F((-1)**k*comb(k+2, 2))*z**k for k in range(degree+1))
                    self.assertLessEqual(abs(1/(1+z)**3-approximation), tail)

    def test_canonical_geometry_table_budget_in_exact_rationals(self):
        from pathlib import Path
        import re
        area = Path(__file__).parent / 'area_acceleration'
        source = (area / 'geometry-area-kernel.c').read_text()
        entries = re.search(r'geometry_radius\[7\]=\{([^}]+)\}', source).group(1)
        radii = [F(v) for v in entries.split(',')]
        self.assertEqual(len(radii), 7)
        self.assertEqual(radii[0], 0)
        self.assertEqual(radii, sorted(set(radii)))
        for degree, r in enumerate(radii):
            n = degree + 1
            tail = r**n / 2 * (r*(1+r)/(1-r)**3
                    + (2*n+3)*r/(1-r)**2 + (n+1)*(n+2)/(1-r))
            self.assertLessEqual(2*(1+r)**3*tail, F(1, 10**9))


if __name__ == '__main__':
    unittest.main()
