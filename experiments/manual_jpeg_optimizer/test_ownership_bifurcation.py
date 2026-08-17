from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manual_jpeg_optimizer.ownership_bifurcation import (
    BifurcationConfig,
    _balanced_variance_frame,
    _constituent_quantize,
    bifurcate_coefficients,
    build_ownership_tree,
    optimize_tree_pruning,
    phase_predecessor_forest,
)


class OwnershipBifurcationTests(unittest.TestCase):
    def test_constituent_route_conserves_composition_before_projection(self):
        vectors = np.array(((0.6, 0.6, 0.0), (1.2, -0.7, 0.4)))
        angle = np.pi / 4.0
        frame = np.array((
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ))
        routed, _, error = _constituent_quantize(vectors, frame)
        self.assertLess(error, 1e-14)
        self.assertTrue(np.issubdtype(routed.dtype, np.floating))

    def test_schur_horn_frame_globally_equalizes_channel_variance(self):
        values = np.array((9.0, 2.0, 0.5))
        basis = np.eye(3)
        frame = _balanced_variance_frame(values, basis)
        diagonal = np.diag(frame.T @ np.diag(values) @ frame)
        self.assertLess(float(np.max(np.abs(diagonal - np.mean(values)))), 1e-12)
        self.assertLess(
            float(np.max(np.abs(frame.T @ frame - np.eye(3)))), 1e-12
        )

    def test_phase_forest_unwraps_alternating_signs(self):
        coefficients = np.zeros((4, 64, 3), dtype=np.float64)
        base = np.array((2.0, -1.0, 0.5))
        coefficients[:, 1] = np.array((1.0, -1.0, -1.0, 1.0))[:, None] * base
        labels = np.zeros((2, 2), dtype=np.int32)
        gauge, parent = phase_predecessor_forest(coefficients, labels)
        aligned = coefficients[:, 1] * gauge[:, 1, None]
        for index in range(1, 4):
            self.assertGreater(float(np.dot(aligned[0], aligned[index])), 0.0)
            self.assertGreaterEqual(parent[index, 1], 0)

    def test_bellman_value_matches_recursive_enumeration(self):
        rng = np.random.default_rng(40)
        vectors = np.concatenate((
            rng.normal((-3, 0, 0), 0.3, size=(32, 3)),
            rng.normal((3, 0, 0), 0.3, size=(32, 3)),
        ))
        weights = np.ones(len(vectors))
        config = BifurcationConfig(
            rate_lambda=0.4,
            branch_penalty=0.1,
            maximum_depth=3,
            minimum_atoms=8,
            maximum_condition=1.01,
        )
        nodes = build_ownership_tree(vectors, weights, config)
        optimum, _ = optimize_tree_pruning(vectors, weights, nodes, config)

        def enumerate_value(index):
            node = nodes[index]
            values = [node.stop_cost]
            if node.children is not None:
                values.append(
                    config.branch_penalty
                    + enumerate_value(node.children[0])
                    + enumerate_value(node.children[1])
                )
            return min(values)

        self.assertAlmostEqual(optimum, enumerate_value(0), places=12)

    def test_every_atom_retains_exactly_one_leaf_owner(self):
        rng = np.random.default_rng(41)
        coefficients = rng.normal(0.0, 20.0, size=(9, 64, 3))
        labels = np.zeros((3, 3), dtype=np.int32)
        result = bifurcate_coefficients(
            coefficients,
            labels,
            70,
            BifurcationConfig(
                rate_lambda=0.0,
                branch_penalty=0.0,
                maximum_depth=3,
                minimum_atoms=16,
                maximum_condition=1.0,
            ),
        )
        self.assertEqual(len(result.leaf_of_atom), 9 * 63)
        self.assertTrue(np.all(result.leaf_of_atom >= 0))
        self.assertLess(result.prequantization_max_composition_error, 1e-10)
        self.assertAlmostEqual(float(np.sum(result.channel_energy_fraction)), 1.0)
        # Nearest-lattice identity is the exact zero-pressure optimum.
        self.assertEqual(result.changed_quantized_coefficients, 0)


if __name__ == "__main__":
    unittest.main()
