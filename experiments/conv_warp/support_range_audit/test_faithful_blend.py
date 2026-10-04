"""Exact coefficient identities, independent of source images or fitted data."""

import unittest
from fractions import Fraction as F
from math import comb

from .faithful_blend import compile_blend


def monomial(px, py, degree):
    """Independent power-to-Bernstein identity for u**px * v**py."""
    return [[F(comb(i, px), comb(degree, px))*F(comb(j, py), comb(degree, py))
             for i in range(degree+1)] for j in range(degree+1)]


class FaithfulBlendTests(unittest.TestCase):
    def test_complete_polynomial_basis_exactly(self):
        zero = [[F(0)]*6 for _ in range(6)]
        for px in range(6):
            for py in range(6):
                field = monomial(px, py, 5)
                elevated = monomial(px, py, 6)
                for bx in range(2):
                    for by in range(2):
                        beta = monomial(bx, by, 1)
                        product = monomial(px+bx, py+by, 6)
                        self.assertEqual(compile_blend(zero, field, beta), product)
                        expected = [[elevated[j][i]-product[j][i]
                                     for i in range(7)] for j in range(7)]
                        self.assertEqual(compile_blend(field, zero, beta), expected)

    def test_identical_orders_are_degree_elevated_unchanged(self):
        field = monomial(3, 4, 5)
        beta = [[F(1, 7), F(4, 5)], [F(2, 3), F(1, 9)]]
        self.assertEqual(compile_blend(field, field, beta), monomial(3, 4, 6))

    def test_degree_six_term_and_its_integral_are_retained(self):
        zero = [[F(0)]*6 for _ in range(6)]
        result = compile_blend(zero, monomial(5, 5, 5), monomial(1, 1, 1))
        self.assertEqual(result, monomial(6, 6, 6))
        self.assertEqual(sum(map(sum, result))/49, F(1, 49))


if __name__ == "__main__":
    unittest.main()
