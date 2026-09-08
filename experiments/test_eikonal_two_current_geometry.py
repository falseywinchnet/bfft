from __future__ import annotations

import unittest

import numpy as np

from experiments.eikonal_two_current_geometry import (
    discrete_closure_defect,
    integrable_two_current,
    symbolic_certificate,
)


class EikonalTwoCurrentGeometryTests(unittest.TestCase):
    def test_symbolic_fermi_and_current_identities_are_exact(self):
        certificate = symbolic_certificate()
        self.assertTrue(certificate["all_exact"])
        self.assertEqual(certificate["density_transport_residual"], "0")
        self.assertEqual(certificate["current_closure_residual"], "0")
        self.assertEqual(certificate["pullback_closure_residual"], "0")

    def test_discrete_companion_is_exactly_closed(self):
        boundary = np.array((1, 4, 9, 16), dtype=np.int64)
        normal = np.array((
            (2, 3, 5, 7),
            (-1, 0, 1, 2),
            (8, 5, 3, 1),
        ), dtype=np.int64)
        potential, recovered, tangent = integrable_two_current(
            boundary, normal
        )
        np.testing.assert_array_equal(recovered, normal)
        np.testing.assert_array_equal(potential[0], boundary)
        np.testing.assert_array_equal(
            discrete_closure_defect(recovered, tangent),
            np.zeros((normal.shape[0], normal.shape[1] - 1), dtype=np.int64),
        )

    def test_independent_companion_edit_breaks_closure(self):
        boundary = np.array((0, 1, 0), dtype=np.int64)
        normal = np.array(((1, 2, 3), (4, 5, 6)), dtype=np.int64)
        _, recovered, tangent = integrable_two_current(boundary, normal)
        tangent[1, 1] += 1
        defect = discrete_closure_defect(recovered, tangent)
        self.assertEqual(int(np.max(np.abs(defect))), 1)

    def test_constant_trace_and_zero_current_reproduce_constant(self):
        boundary = np.full(7, 13, dtype=np.int64)
        normal = np.zeros((5, 7), dtype=np.int64)
        potential, recovered, tangent = integrable_two_current(
            boundary, normal
        )
        np.testing.assert_array_equal(potential, np.full((6, 7), 13))
        self.assertFalse(np.any(recovered))
        self.assertFalse(np.any(tangent))


if __name__ == "__main__":
    unittest.main()
