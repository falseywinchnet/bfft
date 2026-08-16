from __future__ import annotations

import gzip
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from tlvector_v2.affine import affine_gradient_svg, fit_rank1_affine_cells
from tlvector_v2.core import V2Config, vectorize_array_v2
from tlvector_v2.lattice import compact_lattice_loop, deterministic_svgz
from tlvector_v2.merge import merge_error_per_byte


class ConverterV2Tests(unittest.TestCase):
    def test_compact_lattice_encoder_uses_axis_commands(self):
        square = np.array([[2, 3], [7, 3], [7, 8], [2, 8]], dtype=np.float64)
        path = compact_lattice_loop(square)
        self.assertTrue(path.startswith("M2 3"))
        self.assertTrue(path.endswith("Z"))
        self.assertNotIn("L", path)
        self.assertLess(len(path), len("M2 3L7 3L7 8L2 8L2 3Z"))

    def test_svgz_is_deterministic_and_round_trips(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0H1V1Z"/></svg>\n'
        first = deterministic_svgz(svg)
        second = deterministic_svgz(svg)
        self.assertEqual(first, second)
        self.assertEqual(gzip.decompress(first).decode(), svg)

    def test_error_per_byte_merge_reduces_components_inside_budget(self):
        source = np.full((16, 16, 4), (10, 10, 10, 255), dtype=np.uint8)
        labels = np.zeros((16, 16), dtype=np.int32)
        points = [(y, x) for y in range(1, 15, 3) for x in range(1, 15, 3)]
        for y, x in points:
            source[y, x, :3] = 12
            labels[y, x] = 1
        palette = np.array([[10, 10, 10, 255], [12, 12, 12, 255]], dtype=np.uint8)
        merged, _palette, report = merge_error_per_byte(
            source, labels, palette,
            target_mse=1.0, maximum_area=1, rounds=1,
        )
        self.assertGreater(report.merges, 0)
        self.assertLess(report.components_after, report.components_before)
        self.assertLessEqual(report.mse_after, 1.0)
        self.assertEqual(np.unique(merged).tolist(), [0])

    def test_rank1_affine_gradient_beats_flat_cell(self):
        yy, xx = np.indices((20, 30), dtype=np.float64)
        source = np.empty((20, 30, 3), dtype=np.float64)
        source[..., 0] = 0.1 + 0.7 * xx / 29.0
        source[..., 1] = 0.2 + 0.4 * xx / 29.0
        source[..., 2] = 0.7 - 0.5 * xx / 29.0
        labels = np.zeros((20, 30), dtype=np.int32)
        reconstruction, _model, diagnostics = fit_rank1_affine_cells(source, labels)
        self.assertLess(diagnostics["affine_rgb_mse_255"], 1e-8)
        self.assertLess(
            diagnostics["affine_rgb_mse_255"],
            diagnostics["flat_rgb_mse_255"],
        )
        svg, rendered, svg_diagnostics = affine_gradient_svg(
            source, labels, title="linear control"
        )
        self.assertIn("<linearGradient", svg)
        self.assertIn('shape-rendering="crispEdges"', svg)
        self.assertEqual(svg_diagnostics["paths"], 1)
        np.testing.assert_allclose(rendered, reconstruction)

    def test_end_to_end_v2_is_valid_and_meets_target(self):
        colors = np.array([
            [20, 30, 40], [80, 40, 120], [150, 90, 30], [220, 180, 90],
        ], dtype=np.uint8)
        yy, xx = np.indices((24, 32))
        source = np.empty((24, 32, 4), dtype=np.uint8)
        source[..., :3] = colors[((xx // 4) + (yy // 4)) % len(colors)]
        source[..., 3] = 255
        result = vectorize_array_v2(source, V2Config(
            colors=2,
            split_budget=4,
            split_target_mse=0.1,
            final_target_mse=1.0,
            coarse_side=24,
            minimum_region=1,
            merge_maximum_area=1,
            merge_rounds=1,
        ))
        ET.fromstring(result.svg)
        self.assertEqual(gzip.decompress(result.svgz).decode(), result.svg)
        self.assertLessEqual(result.diagnostics["final_mse"], 1.0)
        self.assertEqual(result.diagnostics["target_met"], 1)


if __name__ == "__main__":
    unittest.main()
