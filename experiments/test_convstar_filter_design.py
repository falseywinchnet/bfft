from __future__ import annotations

import unittest

import numpy as np

from experiments.convstar_filter_design import (
    design_from_bins,
    design_minimax_from_bins,
    frequency_response,
    linear_phase_target,
)


class ConvStarFilterDesignTest(unittest.TestCase):
    def test_arbitrary_complex_bins_recover_a_real_fir(self) -> None:
        rng = np.random.default_rng(20260831)
        expected = rng.normal(size=16)
        frequencies = np.linspace(0.0, 0.5, 97)
        desired = frequency_response(expected, frequencies)
        design = design_from_bins(
            desired, length=16, frequencies=frequencies
        )
        np.testing.assert_allclose(design.taps, expected, atol=2e-11, rtol=0.0)
        self.assertLess(design.certificate.stationarity_residual, 2e-12)

    def test_nonuniform_weighted_result_matches_direct_least_squares(self) -> None:
        rng = np.random.default_rng(19)
        length = 18
        frequencies = np.sort(rng.uniform(0.0, 0.5, size=83))
        desired = rng.normal(size=83) + 1j * rng.normal(size=83)
        weights = np.exp(rng.uniform(-2.0, 2.0, size=83))
        design = design_from_bins(
            desired,
            length=length,
            frequencies=frequencies,
            weights=weights,
        )
        basis = np.exp(
            -2j * np.pi
            * frequencies[:, None]
            * np.arange(length, dtype=float)[None, :]
        )
        matrix = np.concatenate((
            np.sqrt(weights)[:, None] * basis.real,
            np.sqrt(weights)[:, None] * basis.imag,
        ), axis=0)
        rhs = np.concatenate((
            np.sqrt(weights) * desired.real,
            np.sqrt(weights) * desired.imag,
        ))
        expected, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        np.testing.assert_allclose(design.taps, expected, atol=3e-11, rtol=0.0)

    def test_even_linear_phase_and_exact_dc_are_certified(self) -> None:
        length = 63
        frequencies = np.concatenate((
            np.linspace(0.0, 0.18, 300, endpoint=True),
            np.linspace(0.24, 0.5, 400, endpoint=True),
        ))
        amplitude = np.concatenate((np.ones(300), np.zeros(400)))
        desired = linear_phase_target(amplitude, frequencies, length)
        design = design_from_bins(
            desired,
            length=length,
            frequencies=frequencies,
            symmetry="even",
            exact_frequencies=[0.0],
            exact_values=[1.0],
        )
        self.assertLess(abs(np.sum(design.taps) - 1.0), 2e-11)
        self.assertLess(design.certificate.exact_constraint_residual, 2e-11)
        self.assertLess(design.certificate.symmetry_residual, 1e-15)
        self.assertLess(design.certificate.stationarity_residual, 2e-10)

    def test_odd_linear_phase_has_zero_dc(self) -> None:
        frequencies = np.linspace(0.01, 0.49, 401)
        amplitude = np.ones(frequencies.size)
        desired = linear_phase_target(
            amplitude, frequencies, 64, symmetry="odd"
        )
        design = design_from_bins(
            desired,
            length=64,
            frequencies=frequencies,
            symmetry="odd",
        )
        self.assertLess(abs(np.sum(design.taps)), 2e-13)
        self.assertLess(design.certificate.symmetry_residual, 1e-15)

    def test_weights_select_the_declared_tradeoff(self) -> None:
        length = 16
        frequencies = np.concatenate((
            np.linspace(0.0, 0.18, 21),
            np.linspace(0.25, 0.5, 31),
        ))
        amplitude = np.concatenate((np.ones(21), np.zeros(31)))
        desired = linear_phase_target(amplitude, frequencies, length)
        ordinary = design_from_bins(
            desired, length=length, frequencies=frequencies, symmetry="even"
        )
        emphasized = design_from_bins(
            desired,
            length=length,
            frequencies=frequencies,
            weights=np.concatenate((np.ones(21), np.full(31, 100.0))),
            symmetry="even",
        )
        ordinary_stop = np.linalg.norm(ordinary.response[21:])
        emphasized_stop = np.linalg.norm(emphasized.response[21:])
        ordinary_pass = np.linalg.norm(ordinary.response[:21] - desired[:21])
        emphasized_pass = np.linalg.norm(emphasized.response[:21] - desired[:21])
        self.assertLess(emphasized_stop, ordinary_stop)
        self.assertGreater(emphasized_pass, ordinary_pass)

    def test_incompatible_exact_bin_is_rejected(self) -> None:
        frequencies = np.linspace(0.0, 0.5, 129)
        with self.assertRaises(ValueError):
            design_from_bins(
                np.zeros(frequencies.size),
                length=32,
                frequencies=frequencies,
                symmetry="even",
                exact_frequencies=[0.5],
                exact_values=[1.0],
            )

    def test_filter_length_contract(self) -> None:
        for length in (16, 63, 128, 512):
            frequencies = np.linspace(0.0, 0.5, max(65, length + 1))
            design = design_from_bins(
                np.zeros(frequencies.size),
                length=length,
                frequencies=frequencies,
                regularization=1e-12,
            )
            self.assertEqual(design.taps.size, length)
        for length in (15, 513):
            with self.assertRaises(ValueError):
                design_from_bins([0.0, 0.0], length=length)

    def test_discrete_minimax_is_no_worse_than_l2_in_max_error(self) -> None:
        length = 31
        passband = np.linspace(0.0, 0.18, 180)
        stopband = np.linspace(0.25, 0.5, 220)
        frequencies = np.concatenate((passband, stopband))
        amplitude = np.concatenate((
            np.ones(passband.size), np.zeros(stopband.size)
        ))
        desired = linear_phase_target(amplitude, frequencies, length)
        least_squares = design_from_bins(
            desired,
            length=length,
            frequencies=frequencies,
            symmetry="even",
        )
        minimax = design_minimax_from_bins(
            amplitude,
            length=length,
            frequencies=frequencies,
            symmetry="even",
        )
        self.assertLessEqual(
            minimax.certificate.maximum_weighted_error,
            least_squares.certificate.maximum_weighted_error + 2e-9,
        )
        self.assertLess(minimax.certificate.stationarity_residual, 2e-9)


if __name__ == "__main__":
    unittest.main()
