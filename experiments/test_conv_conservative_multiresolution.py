from __future__ import annotations

import unittest
from itertools import product

import numpy as np

from experiments.conv_conservative_multiresolution import (
    BlockMoments,
    block_moments_2d,
    conservative_analysis_1d,
    conservative_analysis_2d,
    conservative_multilevel_analysis_1d,
    conservative_multilevel_synthesis_1d,
    conservative_restrict_1d,
    conservative_restrict_2d,
    conservative_synthesis_1d,
    conservative_synthesis_2d,
    conv_moment_1d,
    conv_moments_2d,
    face_sign_certificate_2d,
    fine_currents_1d,
    sign_changes,
    synthesize_block_moments_2d,
)


class ConservativeConvTest(unittest.TestCase):
    def test_one_dimensional_perfect_reconstruction(self) -> None:
        rng = np.random.default_rng(20260826)
        for cells in (10, 18, 34, 66):
            fine = rng.normal(size=(cells, 3))
            state = conservative_analysis_1d(fine)
            returned = conservative_synthesis_1d(state.coarse, state.detail)
            np.testing.assert_allclose(returned, fine, atol=8e-15, rtol=0)

    def test_one_dimensional_mass_is_exactly_restricted(self) -> None:
        rng = np.random.default_rng(7)
        fine = rng.normal(size=(34, 2))
        state = conservative_analysis_1d(fine)
        np.testing.assert_array_equal(
            state.coarse, conservative_restrict_1d(fine)
        )
        zero = conservative_synthesis_1d(state.coarse)
        np.testing.assert_allclose(
            conservative_restrict_1d(zero), state.coarse, atol=2e-16, rtol=0
        )

    def test_nyquist_is_annihilated(self) -> None:
        fine = (-1.0) ** np.arange(66)
        np.testing.assert_array_equal(
            conservative_restrict_1d(fine), np.zeros(33)
        )
        checker = (-1.0) ** np.indices((34, 34)).sum(axis=0)
        np.testing.assert_array_equal(
            conservative_restrict_2d(checker), np.zeros((17, 17))
        )

    def test_zero_detail_does_not_add_ordered_variation(self) -> None:
        rng = np.random.default_rng(3)
        cases = [
            np.linspace(-1.0, 2.0, 33),
            -np.linspace(-1.0, 2.0, 33),
            np.r_[np.linspace(0.0, 1.0, 17), np.linspace(0.9, -0.5, 16)],
            np.r_[np.zeros(7), np.ones(9), -np.ones(8), np.zeros(9)],
        ]
        cases.extend(rng.normal(size=33) for _ in range(32))
        for coarse in cases:
            moment = conv_moment_1d(coarse)
            current = fine_currents_1d(coarse, moment)
            self.assertLessEqual(
                sign_changes(current), sign_changes(np.diff(coarse))
            )

    def test_minimal_integer_lines_exhaust_every_transition_pattern(self) -> None:
        for values in product((-1.0, 0.0, 1.0), repeat=5):
            coarse = np.asarray(values)
            moment = conv_moment_1d(coarse)
            self.assertLessEqual(
                sign_changes(fine_currents_1d(coarse, moment)),
                sign_changes(np.diff(coarse)),
            )

    def test_affine_child_averages_are_exact(self) -> None:
        coarse = 1.25 + 0.3 * np.arange(17)
        fine = conservative_synthesis_1d(coarse)
        coordinate = np.repeat(np.arange(17, dtype=float), 2)
        coordinate[0::2] -= 0.25
        coordinate[1::2] += 0.25
        np.testing.assert_allclose(fine, 1.25 + 0.3 * coordinate, atol=2e-15)

    def test_multilevel_perfect_reconstruction(self) -> None:
        rng = np.random.default_rng(19)
        fine = rng.normal(size=(80, 2))
        pyramid = conservative_multilevel_analysis_1d(fine, levels=3)
        returned = conservative_multilevel_synthesis_1d(pyramid)
        np.testing.assert_allclose(returned, fine, atol=2e-14, rtol=0)

    def test_block_moment_round_trip_and_mass(self) -> None:
        rng = np.random.default_rng(11)
        fine = rng.normal(size=(18, 22, 3))
        moment = block_moments_2d(fine)
        returned = synthesize_block_moments_2d(moment)
        np.testing.assert_allclose(returned, fine, atol=9e-16, rtol=0)
        np.testing.assert_allclose(
            moment.mean, conservative_restrict_2d(fine), atol=3e-16, rtol=0
        )

    def test_block_moment_formula(self) -> None:
        shape = (5, 7)
        mean = np.full(shape, 2.0)
        qx = np.full(shape, 0.3)
        qy = np.full(shape, -0.2)
        qxy = np.full(shape, 0.1)
        fine = synthesize_block_moments_2d(BlockMoments(mean, qx, qy, qxy))
        returned = block_moments_2d(fine)
        for actual, expected in zip(
            (returned.mean, returned.horizontal, returned.vertical, returned.mixed),
            (mean, qx, qy, qxy),
        ):
            np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=0)

    def test_two_dimensional_conservative_conv_round_trip(self) -> None:
        rng = np.random.default_rng(23)
        fine = rng.normal(size=(18, 22, 2))
        state = conservative_analysis_2d(fine)
        returned = conservative_synthesis_2d(
            state.coarse,
            state.horizontal_detail,
            state.vertical_detail,
            state.mixed_detail,
        )
        np.testing.assert_allclose(returned, fine, atol=3e-15, rtol=0)

    def test_two_dimensional_face_current_admission(self) -> None:
        rng = np.random.default_rng(29)
        for _ in range(8):
            coarse = rng.normal(size=(9, 11))
            admitted = conv_moments_2d(coarse)
            certificate = face_sign_certificate_2d(coarse, admitted)
            self.assertLessEqual(certificate["horizontal_sign_surplus"], 0)
            self.assertLessEqual(certificate["vertical_sign_surplus"], 0)

    def test_two_dimensional_affine_child_averages_are_exact(self) -> None:
        y, x = np.indices((9, 11), dtype=float)
        coarse = 0.4 + 0.2 * x - 0.15 * y
        fine = conservative_synthesis_2d(coarse)
        fy, fx = np.indices(fine.shape, dtype=float)
        expected_x = fx / 2.0 - 0.25
        expected_y = fy / 2.0 - 0.25
        expected = 0.4 + 0.2 * expected_x - 0.15 * expected_y
        np.testing.assert_allclose(fine, expected, atol=3e-15, rtol=0)


if __name__ == "__main__":
    unittest.main()
