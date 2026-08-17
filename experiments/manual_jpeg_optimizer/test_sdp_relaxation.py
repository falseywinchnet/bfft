from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.sdp_relaxation import (
    _coarsen_bloom_regions,
    solve_region_sdp,
)
from experiments.manual_jpeg_optimizer.spectral_relaxation import OrthogonalConnections


def random_orthogonal(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q


class SDPRelaxationTests(unittest.TestCase):
    def test_consistent_connection_field_is_exact_rank_three(self):
        rng = np.random.default_rng(30)
        frames = np.stack([random_orthogonal(rng) for _ in range(5)])
        left = np.arange(4, dtype=np.int32)
        right = left + 1
        weights = np.linspace(0.5, 1.0, 4)
        rotations = np.stack([frames[i] @ frames[j].T for i, j in zip(left, right)])
        connections = OrthogonalConnections(left, right, rotations, weights)
        rounded, gram, report = solve_region_sdp(5, connections, tolerance=1e-8)
        exact_value = 3.0 * float(np.sum(weights))
        self.assertAlmostEqual(report["relaxation_value"], exact_value, places=5)
        self.assertAlmostEqual(report["rounded_value"], exact_value, places=5)
        self.assertLess(report["certified_gap"], 2e-5)
        self.assertLess(report["maximum_diagonal_error"], 2e-7)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram)[0]), -2e-7)
        local_gram = np.matmul(rounded, np.swapaxes(rounded, -1, -2))
        self.assertLess(float(np.max(np.abs(local_gram - np.eye(3)[None]))), 1e-10)

    def test_bloom_coarsening_is_connected_and_hits_region_budget(self):
        height, width = 4, 5
        grid = np.arange(height * width).reshape(height, width)
        left = np.concatenate((grid[:, :-1].ravel(), grid[:-1].ravel()))
        right = np.concatenate((grid[:, 1:].ravel(), grid[1:].ravel()))
        edges = len(left)
        connections = OrthogonalConnections(
            left.astype(np.int32), right.astype(np.int32),
            np.repeat(np.eye(3)[None], edges, axis=0),
            np.linspace(1.0, 0.1, edges),
        )
        labels = _coarsen_bloom_regions(connections, height * width, 6)
        self.assertEqual(len(np.unique(labels)), 6)


if __name__ == "__main__":
    unittest.main()
