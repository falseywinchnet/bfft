from fractions import Fraction as F
from math import comb
import unittest
from experiments.conv_warp.star_banks import compile_xy, compile_blend, contract_2d
from experiments.conv_exact_fusion.polynomial_proof import first_cell, evaluate, scalar_profile, tails


def original_order(rows, i, j, u, v):
    line = [evaluate(p, u) for p in first_cell(rows, i)]
    return line[j]+sum(evaluate(t, v)*p[0] for t, p in zip(tails(), scalar_profile(line)[j]))


class TwoDimensionalBanks(unittest.TestCase):
    def test_both_nonlinear_orders_and_blend_compile_to_one_exact_2d_stencil(self):
        scenes = [[[F((x*x+3*y*x+7*y*y+seed*x*y*y) % 19) for x in range(7)] for y in range(7)] for seed in [0, 1, 7]]
        nonseparable = 0
        for rows in scenes:
            for u, v in [(F(0), F(1, 3)), (F(1, 5), F(4, 7)), (F(1, 2), F(1, 2)), (F(1), F(2, 3))]:
                for i, j in [(0, 0), (2, 2), (5, 5)]:
                    a = original_order(rows, i, j, u, v)
                    b = original_order(list(zip(*rows)), j, i, v, u)
                    bank = compile_xy(rows, i, j, u, v)
                    self.assertEqual(contract_2d(bank, rows), a)
                    self.assertEqual(sum(map(sum, bank)), 1)
                    beta = F(3, 7)
                    blended = compile_blend(rows, i, j, u, v, beta)
                    self.assertEqual(contract_2d(blended, rows), (1-beta)*a+beta*b)
                    nonseparable += any(blended[y][x]*blended[y+1][x+1] != blended[y][x+1]*blended[y+1][x] for y in range(6) for x in range(6))
        self.assertGreater(nonseparable, 0)

    def test_global_inverse_square_blend_is_an_exact_finite_2d_phase_filter_pair(self):
        rows = [[F((3*x+7*y) % 11, 11) for x in range(5)] for y in range(4)]
        for u, v in [(F(1, 3), F(2, 7)), (F(3, 4), F(1, 5))]:
            # Translation-invariant constant metric; all finite-image sources.
            kernel = {(dy, dx): 1/((dx+u)**2+(dy+v)**2) for dy in range(-3, 4) for dx in range(-4, 5)}
            for j in range(3):
                for i in range(4):
                    numerator = sum(kernel[j-y, i-x]*rows[y][x] for y in range(4) for x in range(5))
                    denominator = sum(kernel[j-y, i-x] for y in range(4) for x in range(5))
                    direct_weights = [[1/((F(i)+u-x)**2+(F(j)+v-y)**2) for x in range(5)] for y in range(4)]
                    direct = sum(rows[y][x]*direct_weights[y][x] for y in range(4) for x in range(5))/sum(map(sum, direct_weights))
                    self.assertEqual(numerator/denominator, direct)

    def test_integrated_bank_equals_repeated_surface_contractions_exactly(self):
        # Any common geometry rule (including projective weights) commutes
        # with the channel contraction. No quadrature accuracy assumption.
        nodes = [(F(1, 7), F(2, 5), F(3, 11)), (F(4, 9), F(3, 4), F(5, 13)), (F(7, 8), F(1, 3), F(7, 17))]
        bern = lambda i, t: comb(5, i)*t**i*(1-t)**(5-i)
        bank = [[sum(w*bern(i, u)*bern(j, v) for u, v, w in nodes) for i in range(6)] for j in range(6)]
        for channel in range(4):
            controls = [[F((i*i+7*j+channel*i*j) % 23, 23) for i in range(6)] for j in range(6)]
            direct = sum(w*sum(controls[j][i]*bern(i, u)*bern(j, v) for i in range(6) for j in range(6)) for u, v, w in nodes)
            self.assertEqual(contract_2d(bank, controls), direct)


if __name__ == '__main__':
    unittest.main()
