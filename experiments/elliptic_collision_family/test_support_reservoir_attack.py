from fractions import Fraction
import unittest

from experiments.elliptic_collision_family.run_m481_norm_unit_attack import (
    H,
    height_roots,
    norm_unit_dilation,
    prym_shell,
    quotient_multiply,
)
from experiments.elliptic_collision_family.run_m481_split_lift_attack import (
    SPLIT_SUPPORTS,
    adjacent_shell as m481_adjacent_shell,
    k13_transverse_candidates,
    k37_transverse_candidates,
    mixed_transverse_candidates,
)
from experiments.elliptic_collision_family.run_rank_five_support_reservoir_audit import (
    reservoir_raw_model,
)
from experiments.elliptic_collision_family.run_support_reservoir_attack import (
    A6,
    M,
    NEW_SPLIT_HEIGHT,
    NEW_SPLIT_TRIPLE,
    ORIGIN,
    QuarticGroup,
    adjacent_shell,
    reservoir_polynomial,
)
from experiments.elliptic_collision_family.run_split_discriminant_surface_audit import (
    split_discriminant_candidates,
)
from experiments.elliptic_collision_family.run_three_current_height_conic import (
    support_current,
    support_norm,
)


Q = Fraction


class SupportReservoirAttackTests(unittest.TestCase):
    def test_generic_quartic_group_preserves_identity(self):
        group = QuarticGroup(tuple(reservoir_polynomial()), ORIGIN)
        point = (Q(31), Q(5_510))
        self.assertTrue(group.verify(point))
        self.assertEqual(group.add(point, ORIGIN), point)
        self.assertEqual(group.add(point, group.negate(point)), ORIGIN)

    def test_m217_adjacent_shell_exposes_one_new_split_support(self):
        self.assertEqual(support_norm(NEW_SPLIT_TRIPLE), M)
        current = support_current(NEW_SPLIT_TRIPLE)
        self.assertEqual(current, Q(5_169_048_744, 47_045_881))
        self.assertEqual(NEW_SPLIT_HEIGHT**2, A6 + current)
        shell_currents = {candidate.current for candidate in adjacent_shell()}
        self.assertIn(current, shell_currents)

    def test_m481_shell_exposes_13_and_37_adic_supports(self):
        expected = {
            Q(-18_989_417_760, 4_826_809),
            Q(9_108_845_036_160, 2_565_726_409),
        }
        self.assertTrue(expected.issubset({candidate.current for candidate in m481_adjacent_shell()}))
        for triple, height, label in SPLIT_SUPPORTS:
            self.assertEqual(support_norm(triple), 481, label)
            self.assertEqual(height * height, 3_961 + support_current(triple), label)

    def test_m481_transverse_shells_are_finite(self):
        self.assertEqual(len(k13_transverse_candidates()), 6)
        self.assertEqual(len(k37_transverse_candidates()), 6)
        self.assertEqual(len(mixed_transverse_candidates()), 6)
        self.assertTrue(all(candidate.fiber.verify() for candidate in k13_transverse_candidates()))

    def test_reservoir_jacobian_models(self):
        self.assertEqual(
            reservoir_raw_model(217, 1_585),
            (0, 0, 0, -433_610_786_304, -87_244_003_488_692_160),
        )
        self.assertEqual(
            reservoir_raw_model(481, 3_961),
            (0, 0, 0, -1_046_997_311_616, 770_051_568_388_576_320),
        )

    def test_norm_unit_adjacent_shell_has_only_wall_height(self):
        self.assertEqual(norm_unit_dilation(0), 1)
        self.assertEqual(height_roots(Q(1)), (Q(19), Q(41)))
        for index in (-4, -3, -2, -1, 1, 2, 3, 4):
            self.assertEqual(height_roots(norm_unit_dilation(index)), (None, None))

    def test_quotient_recurrence_leaves_one_gate_on_odd_multiples(self):
        for index in (3, 5, 7):
            point = quotient_multiply(index)
            self.assertIsNotNone(point)
            first, second = height_roots(point[0])
            self.assertIsNotNone(first)
            self.assertIsNone(second)

    def test_prym_first_two_shells_are_wall_deck_returns(self):
        for level, expected_count in ((1, 2), (2, 2)):
            hits = prym_shell(level)
            self.assertEqual(len(hits), expected_count)
            self.assertTrue(all(hit[3:] == (Q(1), Q(41), Q(19)) for hit in hits))

    def test_structural_split_discriminant_shell_is_finite(self):
        candidates = split_discriminant_candidates()
        self.assertEqual(
            tuple((curve.m, curve.a6) for curve in candidates),
            ((1_083, 21_818), (507, 28_730), (1_083, 165_818)),
        )


if __name__ == "__main__":
    unittest.main()
