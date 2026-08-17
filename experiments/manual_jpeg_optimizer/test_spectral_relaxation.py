from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.spectral_relaxation import (
    build_connections,
    connection_energy,
    connection_laplacian,
    solve_spectral_relaxation,
)


class SpectralRelaxationTests(unittest.TestCase):
    def test_connection_laplacian_is_symmetric_positive_semidefinite(self):
        rng = np.random.default_rng(20)
        block = rng.normal(size=(64, 3))
        coefficients = np.repeat(block[None], 6, axis=0)
        labels = np.zeros((2, 3), dtype=np.int32)
        connections = build_connections(coefficients, labels)
        laplacian, _ = connection_laplacian(6, connections)
        dense = laplacian.toarray()
        self.assertLess(float(np.max(np.abs(dense - dense.T))), 1e-12)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(dense)[0]), -1e-10)

    def test_identical_block_charts_attain_global_zero_bound(self):
        rng = np.random.default_rng(21)
        block = rng.normal(size=(64, 3))
        coefficients = np.repeat(block[None], 9, axis=0)
        labels = np.zeros((3, 3), dtype=np.int32)
        result = solve_spectral_relaxation(coefficients, labels)
        self.assertLess(abs(result.lower_bound), 1e-9)
        self.assertLess(result.eigen_residual, 1e-8)
        self.assertLess(result.rounded_connection_energy, 1e-8)

    def test_relaxed_value_is_a_lower_bound_on_polar_unrelaxation(self):
        rng = np.random.default_rng(22)
        coefficients = rng.normal(size=(12, 64, 3))
        labels = np.arange(12, dtype=np.int32).reshape(3, 4) % 3
        result = solve_spectral_relaxation(coefficients, labels)
        self.assertLessEqual(
            result.lower_bound,
            result.relaxed_connection_energy + 1e-8,
        )
        self.assertLessEqual(
            result.lower_bound,
            result.rounded_connection_energy + 1e-8,
        )
        gram = np.matmul(np.swapaxes(result.frames, -1, -2), result.frames)
        self.assertLess(float(np.max(np.abs(gram - np.eye(3)[None]))), 1e-10)


if __name__ == "__main__":
    unittest.main()
