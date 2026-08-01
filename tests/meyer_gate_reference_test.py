#!/usr/bin/env python3
"""Equivalence checks for the standalone Meyer gate formulations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.meyer_gate_reference import (  # noqa: E402
    ORIENTATIONS,
    box_kernel,
    compose_fir,
    cross_kernel,
    difference_kernel,
    directional_fir,
    gate_collapsed,
    gate_ring,
    gate_staged,
    response_collapsed,
    response_ring,
    response_staged,
)


class MeyerGateReferenceTest(unittest.TestCase):
    def images(self):
        rng = np.random.default_rng(20260731)
        for shape in ((8, 8), (9, 14), (12, 15), (17, 24)):
            y, x = np.mgrid[: shape[0], : shape[1]]
            structured = (
                80.0
                + 31.0 * (x >= shape[1] // 2)
                + 17.0 * np.sin(2.0 * np.pi * (x / 5.0 + y / 9.0))
                + rng.normal(0.0, 0.5, shape)
            )
            yield structured

    def test_composite_kernels_retain_origins(self):
        box3 = box_kernel(3)
        cross = cross_kernel(0.125)
        difference = difference_kernel()
        self.assertEqual(compose_fir(box3, difference)[1], 3)
        self.assertEqual(compose_fir(cross, difference)[1], 1)

    def test_box_difference_telescopes_to_two_endpoints(self):
        image = next(self.images())
        direction = (1, -1)
        box = box_kernel(3)
        box2 = compose_fir(box, box)
        box3 = compose_fir(box2, box)
        difference = difference_kernel()
        chord = compose_fir(box, difference)

        nonzero = np.flatnonzero(np.abs(chord[0]) > 0.0)
        np.testing.assert_array_equal(nonzero, (0, chord[0].size - 1))
        self.assertEqual(chord[0][0], -1.0 / 7.0)
        self.assertEqual(chord[0][-1], 1.0 / 7.0)

        direct = directional_fir(
            image, direction, *compose_fir(box3, difference)
        )
        telescoped = directional_fir(image, direction, *chord)
        telescoped = directional_fir(telescoped, direction, *box2)
        np.testing.assert_allclose(
            telescoped, direct, rtol=2e-14, atol=2e-13
        )

    def test_each_collapsed_response_matches_staged_definition(self):
        for image in self.images():
            for orientation in ORIENTATIONS:
                expected = response_staged(image, orientation)
                actual = response_collapsed(image, orientation)
                np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)

    def test_each_ring_response_matches_staged_definition(self):
        for image in self.images():
            for orientation in ORIENTATIONS:
                expected = response_staged(image, orientation)
                actual = response_ring(image, orientation)
                np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-12)

    def test_complete_gates_are_equivalent(self):
        for image in self.images():
            staged_gate, staged_raw = gate_staged(image)
            collapsed_gate, collapsed_raw = gate_collapsed(image)
            ring_gate, ring_raw = gate_ring(image)
            np.testing.assert_allclose(
                collapsed_raw, staged_raw, rtol=2e-14, atol=5e-13
            )
            np.testing.assert_allclose(
                collapsed_gate, staged_gate, rtol=3e-13, atol=3e-13
            )
            np.testing.assert_allclose(
                ring_raw, staged_raw, rtol=4e-13, atol=5e-12
            )
            np.testing.assert_allclose(
                ring_gate, staged_gate, rtol=2e-12, atol=2e-12
            )

    def test_constant_has_zero_raw_gate(self):
        image = np.full((11, 18), 137.25)
        for implementation in (gate_staged, gate_collapsed, gate_ring):
            gate, raw = implementation(image)
            np.testing.assert_allclose(raw, 0.0, atol=2e-12)
            np.testing.assert_allclose(gate, 0.0, atol=1e-24)

    def test_translation_covariance(self):
        image = next(self.images())
        shift = (3, 5)
        moved = np.roll(image, shift, axis=(0, 1))
        for implementation in (gate_staged, gate_collapsed, gate_ring):
            gate, raw = implementation(image)
            moved_gate, moved_raw = implementation(moved)
            np.testing.assert_allclose(
                moved_raw, np.roll(raw, shift, axis=(0, 1)), atol=4e-12
            )
            np.testing.assert_allclose(
                moved_gate, np.roll(gate, shift, axis=(0, 1)), atol=2e-12
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
