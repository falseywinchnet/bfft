from __future__ import annotations

import gzip
import base64
from io import BytesIO
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from posterizer.core import PosterizerConfig, posterize_array
from posterizer.oklch import (
    bifurcate_palette,
    gamut_map_oklch,
    oklab_to_srgb,
    oklch_distance2,
    separate_nodes,
)
from posterizer.web_gui import convert_request
from tlvector.core import _srgb_to_oklab


class PosterizerTests(unittest.TestCase):
    def test_oklab_srgb_round_trip(self):
        rgb = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.9, 0.1, 0.2],
            [0.15, 0.65, 0.35],
        ])
        restored = oklab_to_srgb(_srgb_to_oklab(rgb))
        np.testing.assert_allclose(restored, rgb, atol=2e-7)

    def test_gamut_mapping_preserves_lightness_and_hue(self):
        lab = np.array([[0.7, 0.8, -0.4]])
        mapped = gamut_map_oklch(lab)
        self.assertAlmostEqual(mapped[0, 0], lab[0, 0])
        self.assertAlmostEqual(
            np.arctan2(mapped[0, 2], mapped[0, 1]),
            np.arctan2(lab[0, 2], lab[0, 1]),
        )
        rgb = oklab_to_srgb(mapped)
        self.assertTrue(np.all(rgb >= -1e-6))
        self.assertTrue(np.all(rgb <= 1.0 + 1e-6))

    def test_hue_distance_wraps_at_pi(self):
        chroma = 0.2
        hues = np.deg2rad(np.array([179.0, -179.0, 0.0]))
        samples = np.stack((
            np.full(3, 0.6),
            chroma * np.cos(hues),
            chroma * np.sin(hues),
            np.ones(3),
        ), axis=1)
        center_hue = np.pi
        center = np.array([[0.6, chroma * np.cos(center_hue), 0.0, 1.0]])
        distance = oklch_distance2(samples, center)[:, 0]
        self.assertLess(distance[0], 0.0001)
        self.assertLess(distance[1], 0.0001)
        self.assertGreater(distance[2], 0.1)

    def test_bifurcation_and_node_separation(self):
        first = np.tile(np.array([0.35, 0.04, 0.02, 1.0]), (40, 1))
        second = np.tile(np.array([0.78, -0.08, 0.11, 1.0]), (40, 1))
        samples = np.vstack((first, second))
        tree = bifurcate_palette(samples, 2, minimum_leaf=4)
        ordinary = separate_nodes(tree, 1.0)
        expanded = separate_nodes(tree, 1.25)
        ordinary_span = np.linalg.norm(ordinary[0, :3] - ordinary[1, :3])
        expanded_span = np.linalg.norm(expanded[0, :3] - expanded[1, :3])
        self.assertEqual(tree.splits, 1)
        self.assertGreater(tree.total_gain, 0.0)
        self.assertGreater(expanded_span, ordinary_span)

    def test_end_to_end_posterizer_is_deterministic_and_transparent(self):
        source = np.zeros((24, 32, 4), dtype=np.uint8)
        source[2:22, 3:29, 3] = 255
        yy, xx = np.indices((20, 26))
        source[2:22, 3:29, 0] = 30 + 7 * xx
        source[2:22, 3:29, 1] = 40 + 8 * yy
        source[2:22, 3:29, 2] = 160
        config = PosterizerConfig(
            colors=6,
            node_separation=1.05,
            minimum_leaf=4,
            sample_limit=2048,
            minimum_island=2,
            cleanup_rounds=1,
            trim_transparent=False,
        )
        first = posterize_array(source, config, title="control")
        second = posterize_array(source, config, title="control")
        ET.fromstring(first.svg)
        self.assertEqual(gzip.decompress(first.svgz).decode(), first.svg)
        self.assertEqual(first.svg, second.svg)
        self.assertEqual(first.svgz, second.svgz)
        self.assertEqual(first.posterized_rgba[0, 0, 3], 0)
        self.assertLessEqual(first.diagnostics["palette_colors"], 7)
        self.assertIn("H", first.svg)

    def test_inherited_mode_remains_available(self):
        source = np.full((12, 12, 4), (70, 90, 130, 255), dtype=np.uint8)
        source[:, 6:, :3] = (180, 120, 40)
        result = posterize_array(source, PosterizerConfig(
            colors=2, method="inherited", minimum_island=0, cleanup_rounds=0,
        ))
        self.assertEqual(result.diagnostics["method"], "posterizer_inherited")
        self.assertEqual(len(np.unique(result.labels)), 2)

    def test_web_endpoint_returns_all_three_formats(self):
        source = np.full((16, 20, 4), (45, 70, 120, 255), dtype=np.uint8)
        source[4:12, 5:15, :3] = (220, 110, 60)
        stream = BytesIO()
        Image.fromarray(source, "RGBA").save(stream, format="PNG")
        data = stream.getvalue()
        payload = convert_request(
            data,
            "colors=4&method=oklch&separation=1.05&lightness=1&chroma=1"
            "&hue=1&island=1&rounds=1&name=web.png",
        )
        ET.fromstring(payload["svg"])
        self.assertEqual(
            gzip.decompress(base64.b64decode(payload["svgz"])).decode(),
            payload["svg"],
        )
        with Image.open(BytesIO(base64.b64decode(payload["png"]))) as image:
            self.assertEqual(image.size, (20, 16))
        self.assertEqual(payload["diagnostics"]["source_bytes"], len(data))


if __name__ == "__main__":
    unittest.main()
