from fractions import Fraction
import unittest

from experiments.elliptic_collision_family.run_d37_rank_two_transport_attack import (
    BASE_HEIGHT,
    BASE_SHORT_POINTS,
    CHORD_THIRD_POINT,
    FIRST_SHELL,
    INVISIBLE_CURRENT_4,
    INVISIBLE_CURRENT_70,
    JOINT_BASE_POINT,
    JOINT_CONDUCTOR,
    JOINT_MODEL,
    LOCAL_SIEVE_PRIMES,
    PARAMETER_CONDUCTOR,
    PARAMETER_MODEL,
    chord_factor,
    chord_residual,
    is_full_joint_lift,
    joint_height,
    local_lift_status,
    parameter_height,
    parameter_point,
    passes_local_sieve,
    target_rhs,
    verify_target_point,
)


Q = Fraction


class D37RankTwoTransportAttackTests(unittest.TestCase):
    def test_parameter_first_shell(self):
        self.assertEqual(PARAMETER_MODEL, (0, 1, 0, -711_167_436, 7_193_953_224_864))
        self.assertEqual(PARAMETER_CONDUCTOR, 412_131_720)
        actual = set()
        for fiber in FIRST_SHELL:
            point = parameter_point(fiber.coefficients)
            self.assertIsNotNone(point)
            actual.add(abs(parameter_height(point)))
        self.assertEqual(actual, {fiber.height for fiber in FIRST_SHELL})
        self.assertEqual(sorted(fiber.rank for fiber in FIRST_SHELL), [4, 5, 5, 6])

    def test_base_points_and_currents(self):
        self.assertTrue(all(verify_target_point(point) for point in BASE_SHORT_POINTS))
        self.assertEqual(target_rhs(Q(4)) - BASE_HEIGHT**2, INVISIBLE_CURRENT_4)
        self.assertEqual(target_rhs(Q(70)) - BASE_HEIGHT**2, INVISIBLE_CURRENT_70)

    def test_chord_factor_and_third_intersection(self):
        for index in (Q(-2), Q(0), Q(1), Q(2), Q(3), Q(75, 88), Q(7, 3)):
            self.assertEqual(chord_residual(index), chord_factor(index))
        self.assertTrue(verify_target_point(CHORD_THIRD_POINT))

    def test_joint_rank_two_quotient_map(self):
        self.assertEqual(
            JOINT_MODEL,
            (0, 457_380, 0, -28_860_573_456, -13_200_249_087_305_280),
        )
        self.assertEqual(JOINT_CONDUCTOR, 82_368)
        self.assertEqual(joint_height(JOINT_BASE_POINT), BASE_HEIGHT)
        self.assertTrue(is_full_joint_lift((0, 1)))

    def test_local_forbidden_moat(self):
        # The first survivor for the original seven-prime sieve is killed by
        # the exact q=31 support-square character.
        original = (37, 41, 73, 107, 157, 239, 503)
        self.assertTrue(passes_local_sieve((18, 31), original))
        self.assertEqual(local_lift_status((18, 31), 31), "SN")
        self.assertFalse(passes_local_sieve((18, 31), LOCAL_SIEVE_PRIMES))
        self.assertTrue(passes_local_sieve((0, 1), LOCAL_SIEVE_PRIMES))


if __name__ == "__main__":
    unittest.main()
