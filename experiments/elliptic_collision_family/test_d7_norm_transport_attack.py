from fractions import Fraction
import unittest

from experiments.elliptic_collision_family.run_d7_norm_transport_attack import (
    FIRST_HEIGHT,
    FIRST_TARGET_CONDUCTOR,
    FIRST_TARGET_MODEL,
    FOURTH_DIFFERENCE,
    HEIGHT_QUARTIC,
    INVISIBLE_POINT,
    MISSING_HEIGHT_QUOTIENT_CONDUCTOR,
    MISSING_HEIGHT_QUOTIENT_MODEL,
    PARAMETER_CONDUCTOR,
    PARAMETER_GENERATOR,
    PARAMETER_MODEL,
    QUARTIC_GENERATOR,
    SECOND_HEIGHT,
    SECOND_TARGET_CONDUCTOR,
    SECOND_TARGET_FACTORISATION,
    SECOND_TARGET_MODEL,
    VISIBLE_POINTS,
    height_from_parameter_point,
    local_bad_orbit,
    missing_height_raw_jacobian,
    recurrence_group,
    target_discriminant_currents,
    verify_target_point,
)


Q = Fraction


class D7NormTransportAttackTests(unittest.TestCase):
    def test_parameter_generator_maps_to_first_height(self):
        self.assertEqual(
            height_from_parameter_point(tuple(map(Q, PARAMETER_GENERATOR))),
            FIRST_HEIGHT,
        )
        self.assertEqual(PARAMETER_MODEL, (0, 1, 0, -14_916, 205_884))
        self.assertEqual(PARAMETER_CONDUCTOR, 34_320)

    def test_quartic_generator_and_double_height(self):
        group = recurrence_group()
        self.assertTrue(group.verify(QUARTIC_GENERATOR))
        doubled = group.add(QUARTIC_GENERATOR, QUARTIC_GENERATOR)
        p = doubled[0]
        height = (p - Q(400, p)) / 2
        self.assertEqual(abs(height), SECOND_HEIGHT)

    def test_target_models_and_factorisation(self):
        self.assertEqual(FIRST_TARGET_MODEL, (0, 0, 1, -147, 706))
        self.assertEqual(FIRST_TARGET_CONDUCTOR, 50_121)
        self.assertEqual(
            SECOND_TARGET_MODEL,
            (0, 0, 0, -15_957_501_882_672, 184_268_088_879_537_135_620),
        )
        self.assertEqual(SECOND_TARGET_CONDUCTOR, 1_750_328_941_213_617_099_804)
        self.assertEqual(
            SECOND_TARGET_CONDUCTOR,
            __import__("math").prod(prime**power for prime, power in SECOND_TARGET_FACTORISATION),
        )

    def test_rank_six_visible_and_invisible_points(self):
        self.assertTrue(all(verify_target_point(point) for point in VISIBLE_POINTS))
        self.assertTrue(verify_target_point(INVISIBLE_POINT))

    def test_discriminant_norm_current(self):
        e, d, norm = target_discriminant_currents(SECOND_HEIGHT)
        self.assertEqual((e, d), (76_719, 1_148))
        self.assertEqual(norm, 7_693_969_249)
        self.assertEqual(norm, e * e + FOURTH_DIFFERENCE * d * d)

    def test_missing_height_is_rank_zero_quotient(self):
        self.assertEqual(
            missing_height_raw_jacobian(),
            (0, 0, 0, -262_590_768, -1_590_029_889_408),
        )
        self.assertEqual(
            MISSING_HEIGHT_QUOTIENT_MODEL,
            (0, -1, 0, -202_616, -34_012_320),
        )
        self.assertEqual(MISSING_HEIGHT_QUOTIENT_CONDUCTOR, 1_680)

    def test_even_parameter_orbit_hits_bad_divisor(self):
        expected = {
            7: (4, (2, 4)),
            41: (4, (2, 4)),
            107: (8, (2, 4, 6, 8)),
            239: (8, (2, 4, 6, 8)),
            157: (10, (5, 10)),
            503: (130, (65, 130)),
        }
        for prime, (order, indices) in expected.items():
            actual_order, actual_indices, _types = local_bad_orbit(prime)
            self.assertEqual((actual_order, actual_indices), (order, indices))


if __name__ == "__main__":
    unittest.main()
