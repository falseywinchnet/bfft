from __future__ import annotations

import xml.etree.ElementTree as ET
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from tlvector.core import (
    VectorizerConfig,
    _boundary_loops,
    _loop_path,
    _normalize_alpha,
    _oklab_distance2,
    _regularized_assign,
    vectorize_array,
)
from tlvector.web_gui import convert_request


def _fixture() -> np.ndarray:
    image = np.zeros((72, 96, 4), dtype=np.uint8)
    image[:] = (246, 241, 226, 255)
    image[10:60, 8:54] = (25, 83, 114, 255)
    image[18:50, 16:46] = (236, 178, 44, 255)
    image[27:41, 25:37] = (246, 241, 226, 255)
    yy, xx = np.indices(image.shape[:2])
    circle = (xx - 72) ** 2 + (yy - 36) ** 2 <= 20 ** 2
    image[circle] = (190, 44, 67, 220)
    highlight = (xx - 67) ** 2 + (yy - 29) ** 2 <= 7 ** 2
    image[highlight] = (253, 211, 120, 235)
    return image


class VectorizerTests(unittest.TestCase):
    def test_chunked_regularization_matches_dense_reference(self):
        rng = np.random.default_rng(731)
        features = rng.normal(size=(13, 17, 4))
        palette = rng.normal(size=(9, 4))
        labels = rng.integers(0, len(palette), size=features.shape[:2], dtype=np.int32)
        active = rng.random(features.shape[:2]) > 0.27

        expected = labels.copy()
        yy, xx = np.indices(expected.shape)
        unary = _oklab_distance2(features, palette)
        for _ in range(3):
            before = expected.copy()
            for parity in (0, 1):
                padded = np.pad(expected, 1, mode="edge")
                neighbors = (
                    padded[:-2, 1:-1], padded[2:, 1:-1],
                    padded[1:-1, :-2], padded[1:-1, 2:],
                )
                costs = unary.copy()
                for candidate in range(len(palette)):
                    costs[..., candidate] += 0.19 * sum(
                        neighbor != candidate for neighbor in neighbors
                    )
                update = active & (((yy + xx) & 1) == parity)
                expected[update] = np.argmin(costs, axis=2)[update]
            if np.array_equal(expected, before):
                break

        actual = _regularized_assign(
            features, palette, labels, active, smoothness=0.19, sweeps=3
        )
        np.testing.assert_array_equal(actual, expected)

    def test_boundary_trace_keeps_hole_as_second_loop(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:11, 1:11] = True
        mask[4:8, 4:8] = False
        loops = _boundary_loops(mask)
        self.assertEqual(len(loops), 2)
        self.assertEqual(sorted(len(loop) for loop in loops), [16, 40])

    def test_vectorization_is_deterministic_and_valid_svg(self):
        config = VectorizerConfig(
            colors=5, detail_colors=2, coarse_side=48,
            minimum_region=4, simplify=0.4,
        )
        first = vectorize_array(_fixture(), config, title="fixture & check")
        second = vectorize_array(_fixture(), config, title="fixture & check")
        self.assertEqual(first.svg, second.svg)
        root = ET.fromstring(first.svg)
        self.assertEqual(root.attrib["viewBox"], "0 0 96 72")
        self.assertGreaterEqual(first.diagnostics["paths"], 3)
        self.assertGreaterEqual(first.diagnostics["loops"], first.diagnostics["paths"])

    def test_detail_children_never_escape_structural_parent(self):
        config = VectorizerConfig(
            colors=4, detail_colors=3, coarse_side=40,
            residual_quantile=0.65, minimum_region=4,
        )
        result = vectorize_array(_fixture(), config)
        structural_count = result.diagnostics["structural_colors"]
        for child in range(structural_count, len(result.parent_of)):
            pixels = result.labels == child
            self.assertTrue(np.any(pixels))
            self.assertTrue(np.all(
                result.structural_labels[pixels] == result.parent_of[child]
            ))

    def test_detail_basis_does_not_worsen_palette_reconstruction(self):
        source = _fixture()
        base = vectorize_array(source, VectorizerConfig(
            colors=4, detail_colors=0, coarse_side=40, minimum_region=4,
        ))
        detail = vectorize_array(source, VectorizerConfig(
            colors=4, detail_colors=3, coarse_side=40,
            residual_quantile=0.65, minimum_region=4,
        ))
        self.assertLessEqual(
            detail.diagnostics["rgba_mse"], base.diagnostics["rgba_mse"] + 1e-9
        )

    def test_smooth_contour_emits_curves_but_keeps_square_corners(self):
        yy, xx = np.indices((64, 64))
        circle = (xx - 32) ** 2 + (yy - 32) ** 2 <= 22 ** 2
        circle_path = _loop_path(
            _boundary_loops(circle)[0], VectorizerConfig(simplify=0.6)
        )
        self.assertIn("Q", circle_path)
        square = np.zeros((64, 64), dtype=bool)
        square[10:54, 10:54] = True
        square_path = _loop_path(
            _boundary_loops(square)[0], VectorizerConfig(simplify=0.6)
        )
        self.assertNotIn("Q", square_path)

    def test_svg_seam_guard_and_transparent_trim(self):
        source = np.zeros((20, 24, 4), dtype=np.uint8)
        source[4:16, 5:19] = (20, 80, 160, 255)
        result = vectorize_array(source, VectorizerConfig(
            colors=2, detail_colors=0, minimum_region=2, seam_overlap=0.7,
        ))
        self.assertEqual((result.width, result.height), (14, 12))
        self.assertEqual(result.diagnostics["crop_x"], 5)
        self.assertEqual(result.diagnostics["crop_y"], 4)
        self.assertIn('stroke-width="0.7"', result.svg)

    def test_browser_gui_conversion_endpoint_logic(self):
        stream = BytesIO()
        Image.fromarray(_fixture(), "RGBA").save(stream, format="PNG")
        payload = convert_request(
            stream.getvalue(),
            "colors=5&details=2&coarse=48&minimum=4&seam=0.6&name=gui.png",
        )
        self.assertIn("<svg", payload["svg"])
        self.assertIn('stroke-width="0.6"', payload["svg"])
        self.assertGreater(payload["diagnostics"]["paths"], 0)

    def test_auto_alpha_separates_cutout_edges_from_translucent_regions(self):
        edge = np.zeros((24, 24, 4), dtype=np.uint8)
        edge[5:19, 5:19, :3] = (30, 90, 190)
        edge[6:18, 6:18, 3] = 255
        edge[5, 5:19, 3] = edge[18, 5:19, 3] = 100
        edge[5:19, 5, 3] = edge[5:19, 18, 3] = 100
        normalized, mode = _normalize_alpha(edge, VectorizerConfig(alpha_mode="auto"))
        self.assertEqual(mode, "cutout")
        self.assertEqual(set(np.unique(normalized[..., 3])), {0, 255})

        translucent = np.zeros((32, 32, 4), dtype=np.uint8)
        translucent[3:29, 3:29] = (40, 120, 200, 128)
        normalized, mode = _normalize_alpha(
            translucent, VectorizerConfig(alpha_mode="auto")
        )
        self.assertEqual(mode, "preserve")
        self.assertIn(128, np.unique(normalized[..., 3]))


if __name__ == "__main__":
    unittest.main()
