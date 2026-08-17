from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.spatial_dct_transport import (
    SpatialDCTTransportConfig,
    transport_spatial_dct,
)


class SpatialDCTTransportTests(unittest.TestCase):
    def test_ownership_moves_spatially_and_through_frequency_together(self):
        source = np.zeros((4, 64, 3), dtype=np.float64)
        source[0, 1, 0] = 20.0
        labels = np.zeros((2, 2), dtype=np.int32)
        result = transport_spatial_dct(
            source,
            labels,
            70,
            SpatialDCTTransportConfig(
                transport_lambda=0.5, frequency_weight=0.5
            ),
        )
        self.assertGreater(result.positive_mass[1:, 0, 0].sum(), 0.0)
        self.assertGreater(result.positive_mass[0, 1:, 0].sum(), 0.0)
        self.assertEqual(result.negative_mass.sum(), 0.0)

    def test_zero_transport_is_identity(self):
        rng = np.random.default_rng(90)
        source = rng.normal(0, 20, size=(9, 64, 3))
        labels = np.zeros((3, 3), dtype=np.int32)
        result = transport_spatial_dct(
            source, labels, 70, SpatialDCTTransportConfig(transport_lambda=0.0)
        )
        self.assertLess(float(np.max(np.abs(result.coefficients - source))), 1e-12)

    def test_signed_mass_and_edge_ownership_are_exact(self):
        rng = np.random.default_rng(91)
        source = rng.normal(0, 20, size=(16, 64, 3))
        labels = np.zeros((4, 4), dtype=np.int32)
        result = transport_spatial_dct(
            source, labels, 70, SpatialDCTTransportConfig(transport_lambda=0.3)
        )
        self.assertLess(result.positive_mass_error, 1e-11)
        self.assertLess(result.negative_mass_error, 1e-11)
        self.assertGreater(result.minimum_transported_mass, -1e-12)
        self.assertLess(result.kkt_residual, 1e-10)
        self.assertLess(result.flow_divergence_residual, 1e-10)
        self.assertEqual(result.positive_flow.shape[1:], (3,))
        self.assertGreater(result.spatial_edges, 0)
        self.assertGreater(result.frequency_edges, 0)


if __name__ == "__main__":
    unittest.main()
