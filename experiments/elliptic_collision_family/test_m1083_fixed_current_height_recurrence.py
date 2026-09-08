from fractions import Fraction
import unittest

from experiments.elliptic_collision_family.run_m1083_fixed_current_height_recurrence import (
    FIRST_DIFFERENCE,
    FOURTH_DIFFERENCE,
    FOURTH_HEIGHT_PAIR_CONDUCTOR,
    FOURTH_HEIGHT_PAIR_MINIMAL_MODEL,
    GENERATOR_P,
    GENERATOR_Q,
    HEIGHT_QUARTIC,
    ORIGIN,
    PARAMETER_JACOBIAN_CONDUCTOR,
    PARAMETER_JACOBIAN_MINIMAL_MODEL,
    coefficient_box_shell,
    denominator_conductor_exponent,
    denominator_minimal_delta_exponent,
    fiber_from_point,
    fourth_height_pair_raw_jacobian,
    recurrence_group,
)


Q = Fraction


class FixedCurrentHeightRecurrenceTests(unittest.TestCase):
    def test_distinguished_points_and_fibers(self):
        group = recurrence_group()
        for point in (ORIGIN, GENERATOR_P, GENERATOR_Q):
            self.assertTrue(group.verify(point))
            self.assertTrue(fiber_from_point(point).verify())
        self.assertEqual(fiber_from_point(ORIGIN).base_height, Q(90))
        self.assertEqual(fiber_from_point(GENERATOR_P).base_height, Q(390))
        self.assertEqual(fiber_from_point(GENERATOR_Q).base_height, Q(-90))

    def test_first_shell_contains_the_new_exact_rank_five_height(self):
        heights = {fiber.base_height for _a, _b, fiber in coefficient_box_shell(1)}
        self.assertIn(Q(-27_610, 1_071), heights)
        self.assertIn(Q(3_380_310, 177_289), heights)
        self.assertIn(Q(-1_056_013_590, 10_100_519), heights)

    def test_discriminant_current_factorization(self):
        for point in (ORIGIN, GENERATOR_P, GENERATOR_Q):
            fiber = fiber_from_point(point)
            e, d = fiber.primitive_height
            self.assertEqual(
                fiber.integral_discriminant,
                -432 * e * e * (e * e + FOURTH_DIFFERENCE * d * d) * d**8,
            )
            a1, a2, a3, a4, a6 = fiber.integral_model
            self.assertEqual((a1, a2, a3), (0, 0, 0))
            self.assertEqual(-16 * (4 * a4**3 + 27 * a6**2), fiber.integral_discriminant)

    def test_cube_denominator_valuations_disappear(self):
        expected_delta = (0, 8, 4, 0, 8, 4, 0, 8, 4, 0)
        expected_conductor = (0, 2, 2, 0, 2, 2, 0, 2, 2, 0)
        self.assertEqual(
            tuple(denominator_minimal_delta_exponent(k) for k in range(10)),
            expected_delta,
        )
        self.assertEqual(
            tuple(denominator_conductor_exponent(k) for k in range(10)),
            expected_conductor,
        )

    def test_fourth_height_rank_zero_quotient_models(self):
        self.assertEqual(
            fourth_height_pair_raw_jacobian(),
            (0, 0, 0, -53_112_205_872, -3_570_486_866_315_136),
        )
        self.assertEqual(
            FOURTH_HEIGHT_PAIR_MINIMAL_MODEL,
            (0, -1, 0, -40_981_640, -76_514_264_400),
        )
        self.assertEqual(FOURTH_HEIGHT_PAIR_CONDUCTOR, 31_920)
        self.assertEqual(
            PARAMETER_JACOBIAN_MINIMAL_MODEL,
            (0, 1, 0, -10_919_160, 13_009_804_308),
        )
        self.assertEqual(PARAMETER_JACOBIAN_CONDUCTOR, 8_888_880)


if __name__ == "__main__":
    unittest.main()
