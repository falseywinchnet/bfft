from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.balanced_regions import (
    BalancedRegionConfig, balanced_bifurcation_regions,
)


class BalancedRegionTests(unittest.TestCase):
    def test_balanced_tree_preserves_one_owner_per_block(self):
        yy, xx = np.mgrid[:64, :96]
        ycc = np.stack((xx + yy, 20 * np.sin(xx / 8), 20 * np.cos(yy / 7)), axis=-1)
        result = balanced_bifurcation_regions(
            ycc,
            80,
            BalancedRegionConfig(target_regions=8, minimum_blocks=4),
        )
        self.assertEqual(len(np.unique(result.labels)), 8)
        self.assertEqual(result.labels.shape, (8, 12))
        self.assertTrue(np.all(np.bincount(result.labels.ravel()) >= 4))
        owned = np.concatenate([result.nodes[i].blocks for i in result.leaves])
        self.assertEqual(len(np.unique(owned)), result.labels.size)


if __name__ == "__main__":
    unittest.main()
