import unittest

import mpmath as mp

from elliptic_pi import (
    cubic_small_root,
    descend_lambda,
    descend_lambda_degree3,
    descend_lambda_degree3_normalized,
    descend_lambda_degree3_normalized_data,
    pi_cubic_implicit,
    pi_lambda3_direct,
    pi_lambda3_period_transport,
    pi_lambda3_composite_jet,
    pi_lambda3_incremental_nome,
    pi_lambda3_terminal_nome,
    pi_lemniscatic_series,
    pi_quartic_implicit,
    pi_quartic_radical,
    positive_root_s0,
    positive_root_y0,
    quartic_small_root,
    lambda3_period_data,
    lambda2_branch_invariants,
    lambda2_branch_pair,
    terminal_period_series,
)


class EllipticPiTests(unittest.TestCase):
    def test_four_point_geometry(self):
        mp.mp.dps = 80
        lam = mp.mpf(1) / 2
        j = 256 * (1 - lam + lam * lam) ** 3 / (lam * lam * (1 - lam) ** 2)
        self.assertEqual(j, 1728)

    def test_initial_polynomial_roots(self):
        mp.mp.dps = 100
        self.assertLess(abs(positive_root_y0(90) - (mp.sqrt(2) - 1)), mp.mpf("1e-85"))
        self.assertLess(
            abs(positive_root_s0(90) - (mp.sqrt(3) - 1) / 2), mp.mpf("1e-85")
        )

    def test_lambda_descent(self):
        mp.mp.dps = 100
        u = mp.mpf(1) / 2
        v = descend_lambda(u, 90)
        explicit = ((1 - mp.sqrt(1 - u)) / (1 + mp.sqrt(1 - u))) ** 2
        self.assertLess(abs(v - explicit), mp.mpf("1e-85"))

    def test_lambda2_two_sheet_bifurcation(self):
        mp.mp.dps = 100
        u = mp.mpf(1) / 2
        small, large, multiplier_small, multiplier_large = (
            lambda2_branch_pair(u)
        )
        branch_trace, branch_norm, multiplier_invariant = (
            lambda2_branch_invariants(u)
        )
        self.assertLess(abs(small * large - branch_norm), mp.mpf("1e-90"))
        self.assertLess(abs(small + large - branch_trace), mp.mpf("1e-90"))
        self.assertLess(
            abs(multiplier_small + multiplier_large - multiplier_invariant),
            mp.mpf("1e-90"),
        )
        self.assertLess(
            abs(multiplier_small * multiplier_large - multiplier_invariant),
            mp.mpf("1e-90"),
        )

        period = lambda x: mp.hyp2f1(mp.mpf(1) / 2, mp.mpf(1) / 2, 1, x)
        self.assertLess(
            abs(period(u) - multiplier_small * period(small)),
            mp.mpf("1e-90"),
        )
        # The reciprocal sheet lies beyond the hypergeometric branch cut.  It
        # carries the other period-basis component rather than a second real,
        # independently contracting copy of the small branch.
        self.assertGreater(abs(mp.im(period(large))), mp.mpf("0.1"))
        continued = mp.sqrt(small) * (
            period(small) - 1j * period(1 - small)
        )
        self.assertLess(abs(period(large) - continued), mp.mpf("1e-90"))

    def test_degree3_lambda_descent(self):
        mp.mp.dps = 100
        u = mp.mpf(1) / 2
        v = descend_lambda_degree3(u, 90)
        q = mp.exp(-3 * mp.pi)
        explicit = (mp.jtheta(2, 0, q) / mp.jtheta(3, 0, q)) ** 4
        self.assertLess(abs(v - explicit), mp.mpf("1e-85"))
        tiny = mp.mpf("1e-20")
        descended = descend_lambda_degree3(tiny, 90)
        self.assertLess(abs(descended / tiny**3 - mp.mpf(1) / 256), mp.mpf("1e-20"))

        normalized = descend_lambda_degree3_normalized(u, 90)
        self.assertLess(abs(normalized - v), mp.mpf("1e-85"))

    def test_degree3_period_alpha_correction(self):
        mp.mp.dps = 100
        u = mp.mpf(1) / 2
        v = descend_lambda_degree3(u, 90)
        first, _, multiplier, multiplier_derivative, correction = lambda3_period_data(u, v)
        step = mp.mpf("1e-35")
        v_plus = descend_lambda_degree3(u + step, 90)
        v_minus = descend_lambda_degree3(u - step, 90)
        numerical_first = (v_plus - v_minus) / (2 * step)
        multiplier_plus = lambda3_period_data(u + step, v_plus)[2]
        multiplier_minus = lambda3_period_data(u - step, v_minus)[2]
        numerical_multiplier_derivative = (multiplier_plus - multiplier_minus) / (2 * step)
        self.assertLess(abs(first - numerical_first), mp.mpf("1e-55"))
        self.assertLess(
            abs(multiplier_derivative - numerical_multiplier_derivative), mp.mpf("1e-53")
        )

        predicted = multiplier / 2 + correction
        elliptic_k = mp.ellipk(v)
        elliptic_e = mp.ellipe(v)
        direct = mp.pi / (4 * elliptic_k**2) - 3 * (elliptic_e / elliptic_k - 1)
        self.assertLess(abs(predicted - direct), mp.mpf("1e-80"))

    def test_implicit_roots_match_radicals(self):
        mp.mp.dps = 100
        y = mp.sqrt(2) - 1
        quartic = quartic_small_root(y, 90)
        root4 = mp.root(1 - y**4, 4)
        self.assertLess(abs(quartic - (1 - root4) / (1 + root4)), mp.mpf("1e-85"))

        s = (mp.sqrt(3) - 1) / 2
        cubic = cubic_small_root(s, 90)
        r = 3 / (1 + 2 * mp.root(1 - s**3, 3))
        self.assertLess(abs(cubic - (r - 1) / 2), mp.mpf("1e-85"))

    def test_pi_iterations(self):
        mp.mp.dps = 130
        implicit4 = pi_quartic_implicit(110)
        radical4 = pi_quartic_radical(110)
        implicit3 = pi_cubic_implicit(110)
        direct3 = pi_lambda3_direct(110)
        period3 = pi_lambda3_period_transport(110)
        jet3 = pi_lambda3_composite_jet(110)
        nome3 = pi_lambda3_terminal_nome(110)
        incremental3 = pi_lambda3_incremental_nome(110)
        lemniscatic = pi_lemniscatic_series(110)
        self.assertLess(abs(implicit4 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(radical4 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(implicit3 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(direct3 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(period3 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(jet3 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(nome3 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(incremental3 - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(lemniscatic - mp.pi), mp.mpf("1e-100"))
        self.assertLess(abs(implicit4 - radical4), mp.mpf("1e-100"))

    def test_period_factors_telescope_to_composite_jet(self):
        mp.mp.dps = 120
        initial = mp.mpf("0.37")
        u = initial
        ratio_product = mp.mpf(1)
        derivative_product = mp.mpf(1)
        weighted_log_derivative = mp.mpf(0)
        composite_first = mp.mpf(1)
        composite_second = mp.mpf(0)
        degree_power = mp.mpf(1)

        for _ in range(3):
            v = descend_lambda_degree3_normalized(u, 105)
            first, second, _, _, _ = lambda3_period_data(u, v)
            ratio = first * u * (1 - u) / (3 * v * (1 - v))
            log_ratio_derivative = (
                second / first
                + (1 - 2 * u) / (u * (1 - u))
                - first * (1 - 2 * v) / (v * (1 - v))
            )
            ratio_product *= ratio
            weighted_log_derivative -= derivative_product * log_ratio_derivative
            derivative_product *= first
            composite_second = (
                second * composite_first * composite_first
                + first * composite_second
            )
            composite_first *= first
            degree_power *= 3
            u = v

        endpoint_ratio = (
            composite_first * initial * (1 - initial)
            / (degree_power * u * (1 - u))
        )
        endpoint_log_derivative = (
            composite_second / composite_first
            + (1 - 2 * initial) / (initial * (1 - initial))
            - composite_first * (1 - 2 * u) / (u * (1 - u))
        )
        self.assertLess(abs(ratio_product - endpoint_ratio), mp.mpf("1e-95"))
        self.assertLess(
            abs(weighted_log_derivative + endpoint_log_derivative),
            mp.mpf("1e-95"),
        )

    def test_incremental_nome_telescope(self):
        mp.mp.dps = 110
        u = mp.mpf(1) / 2
        correction_sum = mp.mpf(0)
        inverse_weight = mp.mpf(1) / 3
        count = 4
        for _ in range(count):
            u, ratio = descend_lambda_degree3_normalized_data(u, 100)
            correction_sum += mp.log(ratio) * inverse_weight
            inverse_weight /= 3
        initial_l = mp.log(32)
        terminal_l = -mp.log(u / 16)
        reconstructed = terminal_l / (mp.mpf(3) ** count) + correction_sum
        self.assertLess(abs(initial_l - reconstructed), mp.mpf("1e-95"))

    def test_terminal_period_series(self):
        mp.mp.dps = 100
        u = mp.mpf("1e-8")
        function, square, logarithmic_derivative = terminal_period_series(u, 80)
        direct = mp.hyp2f1(mp.mpf(1) / 2, mp.mpf(1) / 2, 1, u)
        direct_h = 2 * mp.diff(
            lambda x: mp.hyp2f1(mp.mpf(1) / 2, mp.mpf(1) / 2, 1, x), u
        ) / direct
        self.assertLess(abs(function - direct), mp.mpf("1e-75"))
        self.assertLess(abs(square - direct * direct), mp.mpf("1e-75"))
        self.assertLess(abs(logarithmic_derivative - direct_h), mp.mpf("1e-72"))


if __name__ == "__main__":
    unittest.main()
