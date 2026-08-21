"""Invariants for matched 1-D scale-law readouts."""

from __future__ import annotations

import unittest

import numpy as np

from .cross_predictive_transport import (
    relation_scale_readout_forms,
    relation_scale_transport,
)


class RelationScaleReadoutFormsTests(unittest.TestCase):
    def test_constant_is_exact_for_every_readout(self):
        line = np.full(64, 0.37)
        forms, diagnostic = relation_scale_readout_forms(line)
        for value in forms.values():
            np.testing.assert_array_equal(value, line)
        self.assertEqual(diagnostic["characteristic_count"], 96)

    def test_mean_form_is_the_canonical_relation_transport(self):
        x = np.linspace(0.0, 1.0, 64, endpoint=False)
        line = 0.3 + 0.1 * x + 0.04 * np.sin(12.0 * np.pi * x)
        forms, _diagnostic = relation_scale_readout_forms(line)
        canonical, _canonical_diagnostic = relation_scale_transport(line)
        np.testing.assert_array_equal(forms["mean"], canonical)


if __name__ == "__main__":
    unittest.main()
