from __future__ import annotations

from io import BytesIO
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.manual_jpeg_optimizer.core as core_module
from experiments.manual_jpeg_optimizer.core import (
    JPEGConfig,
    analyze_five_stages,
    block_dct,
    image_metrics,
    infer_source_quality,
    inverse_block_dct,
    optimize_jpeg,
    preprocess,
    rgb_to_ycc,
    ycc_to_rgb,
)


class ManualJPEGOptimizerTests(unittest.TestCase):
    def test_color_transform_roundtrip(self):
        rng = np.random.default_rng(4)
        rgb = rng.uniform(0, 255, size=(17, 19, 3))
        recovered = ycc_to_rgb(rgb_to_ycc(rgb))
        self.assertLess(float(np.max(np.abs(rgb - recovered))), 3e-4)

    def test_dct_roundtrip_with_padding(self):
        rng = np.random.default_rng(7)
        plane = rng.uniform(0, 255, size=(19, 23))
        recovered = inverse_block_dct(block_dct(plane), plane.shape)
        self.assertLess(float(np.max(np.abs(plane - recovered))), 1e-10)

    def test_five_named_stages_and_overlays(self):
        y, x = np.mgrid[:32, :40]
        rgb = np.stack(((x * 7) % 256, (y * 9) % 256, ((x + y) * 5) % 256), axis=-1)
        stages = analyze_five_stages(rgb, JPEGConfig(quality=75, subsampling=2))
        self.assertEqual(list(stages)[:5], [
            "1_ycbcr", "2_chroma_sampling", "3_dct_cascade",
            "4_quantization", "5_zigzag_entropy",
        ])
        self.assertIn("aligned_regions", stages)
        self.assertTrue(all(value.dtype == np.uint8 for value in stages.values()))

    def test_zero_projection_is_near_identity(self):
        rng = np.random.default_rng(8)
        rgb = rng.integers(0, 256, size=(24, 24, 3)).astype(np.float64)
        result = preprocess(rgb, JPEGConfig())
        self.assertLess(float(np.max(np.abs(rgb - result.rgb))), 3e-4)

    def test_metrics_identity(self):
        rgb = np.full((20, 20, 3), 90.0)
        ssim, psnr, edge = image_metrics(rgb, rgb)
        self.assertAlmostEqual(ssim, 1.0, places=12)
        self.assertEqual(psnr, 99.0)
        self.assertEqual(edge, 99.0)

    def test_native_fused_metrics_match_portable_reference(self):
        if core_module._native_image_metrics is None:
            self.skipTest("native BFFT vision library is unavailable")
        rng = np.random.default_rng(19)
        reference = rng.uniform(0, 255, size=(31, 37, 3))
        candidate = np.clip(reference + rng.normal(0, 7, reference.shape), 0, 255)
        prepared = core_module._prepare_metric_reference(reference)
        native = core_module._image_metrics_prepared(prepared, candidate)
        backend = core_module._native_image_metrics
        try:
            core_module._native_image_metrics = None
            portable = core_module._image_metrics_prepared(prepared, candidate)
        finally:
            core_module._native_image_metrics = backend
        np.testing.assert_allclose(native, portable, rtol=0.0, atol=1e-11)

    def test_infer_standard_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q70.jpg"
            Image.new("RGB", (16, 16), (80, 120, 160)).save(path, quality=70)
            self.assertEqual(infer_source_quality(path), 70)

    def test_frontier_trace_resolves_target_without_cartesian_sweep(self):
        y, x = np.mgrid[:48, :64]
        rgb = np.stack((
            (x * 11 + y * 3) % 256,
            (x * 5 + y * 13) % 256,
            (x * 7 + y * 17) % 256,
        ), axis=-1).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            output = Path(directory) / "output.jpg"
            Image.fromarray(rgb).save(source)
            result = optimize_jpeg(source, output, target_bytes=3_000)
            self.assertEqual(result.search_strategy, "frontier_trace")
            self.assertLessEqual(result.best.size_bytes, 3_000)
            self.assertLess(result.evaluations, 126)
            self.assertTrue(output.exists())

    def test_frontier_trace_can_leave_high_quality_source_neighborhood(self):
        y, x = np.mgrid[:128, :128]
        rgb = np.stack((
            (x * 7 + y * 3) % 256,
            (x * 2 + y * 9) % 256,
            (x * 11 + y) % 256,
        ), axis=-1).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source-q95.jpg"
            output = Path(directory) / "output.jpg"
            Image.fromarray(rgb).save(source, quality=95, subsampling=0)
            result = optimize_jpeg(source, output, target_bytes=1_800)
            self.assertEqual(result.source_quality, 95)
            self.assertLess(result.best.config.quality, 87)
            self.assertLessEqual(result.best.size_bytes, 1_800)

    def test_frontier_trace_is_cooperatively_cancellable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (16, 16), (80, 120, 160)).save(source)
            with self.assertRaises(InterruptedError):
                optimize_jpeg(
                    source,
                    Path(directory) / "output.jpg",
                    cancelled=lambda: True,
                )


if __name__ == "__main__":
    unittest.main()
