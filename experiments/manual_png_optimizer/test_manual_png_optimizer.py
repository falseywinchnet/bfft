from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from experiments.manual_png_optimizer.core import (
    PNGConfig,
    PNGCandidate,
    _apply_order,
    _best_idat,
    _better,
    _compress,
    _filter_stream,
    _filtered_rows,
    _encode_indexed,
    _ownership_flow,
    _quantize,
    _selective_ordered_diffusion,
    _smooth_transition_coverage,
    compare_pngs,
    optimize_png,
)


class PNGOptimizerTest(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:37, :53]
        self.rgb = np.stack((
            (x * 9 + y * 3) % 256,
            (x * 2 + y * 11) % 256,
            (x * 7 - y * 5) % 256,
        ), axis=-1).astype(np.uint8)

    def test_lossless_round_trip_and_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            report = root / "output.json"
            Image.fromarray(self.rgb, "RGB").save(source)
            result = optimize_png(
                source, output, PNGConfig(lossless=True), report=report
            )
            decoded = np.asarray(Image.open(output).convert("RGB"))
            self.assertTrue(np.array_equal(decoded, self.rgb))
            self.assertEqual(result.winner.ssim, 1.0)
            self.assertEqual(json.loads(report.read_text())["winner"]["size"], output.stat().st_size)

    def test_near_ssim_tie_prefers_structural_fidelity(self):
        common = dict(
            colors=64, quantizer="edge-lloyd", palette_edge_weight=1.0,
            dither="none", diffusion_strength=0.0, ownership_strength=0.0,
            palette_order="identity", filter_policy="fixed-0",
            zlib_strategy="default-m5", smooth_transition_coverage=0.5,
            elapsed_seconds=0.0, data=b"", rgba=np.zeros((1, 1, 4), np.uint8),
        )
        incumbent = PNGCandidate(
            **common, size=990, ssim=0.999950, psnr_db=50.0, edge_psnr_db=40.0,
        )
        cleaner_edge = PNGCandidate(
            **common, size=995, ssim=0.999947, psnr_db=50.0, edge_psnr_db=40.5,
        )
        material_loss = PNGCandidate(
            **common, size=995, ssim=0.999500, psnr_db=55.0, edge_psnr_db=50.0,
        )
        config = PNGConfig(target_bytes=1_000)
        self.assertTrue(_better(cleaner_edge, incumbent, config))
        self.assertFalse(_better(material_loss, incumbent, config))

    def test_indexed_bit_depths_and_alpha_round_trip(self):
        for colors in (2, 4, 16, 17):
            labels = (np.arange(35 * 41).reshape(35, 41) % colors).astype(np.int32)
            palette = np.zeros((colors, 4), dtype=np.uint8)
            palette[:, 0] = np.arange(colors) * (255 // max(colors - 1, 1))
            palette[:, 1] = 255 - palette[:, 0]
            palette[:, 2] = (np.arange(colors) * 71) % 256
            palette[:, 3] = 255
            palette[0, 3] = 0
            data, _, _ = _encode_indexed(labels, palette, {}, PNGConfig(colors=colors))
            with Image.open(__import__("io").BytesIO(data)) as decoded:
                rgba = np.asarray(decoded.convert("RGBA"))
            self.assertTrue(np.array_equal(rgba, palette[labels]))

    def test_terminal_memory_search_never_loses_to_old_memlevel(self):
        rows = np.tile(np.arange(53, dtype=np.uint8), (64, 5))[:, :257]
        compressed, _, _ = _best_idat(
            rows, 1, 9, "fast", indexed=True
        )
        filters, _ = _filtered_rows(rows, 1)
        stream = _filter_stream(
            filters, np.zeros(rows.shape[0], dtype=np.uint8)
        )
        old = _compress(stream, 9, "default", 9)
        self.assertLessEqual(len(compressed), len(old))

    def test_palette_transport_conserves_displayed_constituents(self):
        labels = (np.arange(12 * 15).reshape(12, 15) % 6).astype(np.int32)
        palette = np.arange(24, dtype=np.uint8).reshape(6, 4)
        palette[:, 3] = 255
        order = np.array((3, 0, 5, 2, 1, 4), dtype=np.int32)
        moved_labels, moved_palette = _apply_order(labels, palette, order)
        self.assertTrue(np.array_equal(palette[labels], moved_palette[moved_labels]))

    def test_zero_flow_is_identity_and_positive_flow_is_palette_valued(self):
        rgba = np.concatenate((self.rgb, np.full((*self.rgb.shape[:2], 1), 255, np.uint8)), axis=2)
        palette = np.array(((20, 30, 40, 255), (120, 130, 140, 255), (220, 225, 230, 255)), np.uint8)
        labels = np.zeros(self.rgb.shape[:2], dtype=np.int32)
        labels[:, self.rgb.shape[1] // 3: 2 * self.rgb.shape[1] // 3] = 1
        labels[:, 2 * self.rgb.shape[1] // 3:] = 2
        identity = _ownership_flow(rgba, labels, palette, 0.0, 3, 8.0)
        flowed = _ownership_flow(rgba, labels, palette, 0.001, 2, 8.0)
        self.assertTrue(np.array_equal(identity, labels))
        self.assertTrue(np.all((flowed >= 0) & (flowed < len(palette))))

    def test_fixed_palette_dithering_is_real_and_palette_bounded(self):
        gradient = np.repeat(
            np.arange(96, dtype=np.uint8)[None, :, None], 32, axis=0
        )
        rgb = np.repeat(gradient, 3, axis=2)
        rgba = np.concatenate(
            (rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)), axis=2
        )
        plain, plain_palette = _quantize(rgba, 4, "none", "median-cut")
        dithered, dithered_palette = _quantize(rgba, 4, "floyd", "median-cut")
        self.assertFalse(np.array_equal(plain_palette[plain], dithered_palette[dithered]))
        self.assertLessEqual(len(dithered_palette), 4)

    def test_selective_diffusion_is_deterministic_and_edge_gated(self):
        gradient = np.repeat(
            np.arange(64, dtype=np.uint8)[None, :, None] * 4, 24, axis=0
        )
        rgb = np.repeat(gradient, 3, axis=2)
        rgba = np.concatenate(
            (rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)), axis=2
        )
        labels, palette = _quantize(rgba, 8, "none", "median-cut")
        first = _selective_ordered_diffusion(rgba, labels, palette, 1.5, 0.0)
        second = _selective_ordered_diffusion(rgba, labels, palette, 1.5, 0.0)
        gated = _selective_ordered_diffusion(rgba, labels, palette, 1.5, 8.0)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.any(first != labels))
        self.assertLessEqual(np.count_nonzero(gated != labels), np.count_nonzero(first != labels))
        self.assertTrue(np.all((first >= 0) & (first < len(palette))))
        base_rgba = palette[labels]
        diffused_rgba = palette[first]
        self.assertGreaterEqual(
            _smooth_transition_coverage(rgba, diffused_rgba),
            _smooth_transition_coverage(rgba, base_rgba),
        )

    def test_edge_lloyd_is_deterministic(self):
        rgba = np.concatenate(
            (self.rgb, np.full((*self.rgb.shape[:2], 1), 255, np.uint8)), axis=2
        )
        first_labels, first_palette = _quantize(
            rgba, 12, "none", "edge-lloyd", 2, 1.5, 1024, 12345
        )
        second_labels, second_palette = _quantize(
            rgba, 12, "none", "edge-lloyd", 2, 1.5, 1024, 12345
        )
        self.assertTrue(np.array_equal(first_labels, second_labels))
        self.assertTrue(np.array_equal(first_palette, second_palette))

    def test_rgba_lloyd_ignores_hidden_rgb_and_preserves_display(self):
        y, x = np.mgrid[:32, :48]
        alpha = np.clip(x * 6, 0, 255).astype(np.uint8)
        rgba = np.stack((
            (x * 37 + y * 11) % 256,
            (x * 13 + y * 29) % 256,
            (x * 7 + y * 43) % 256,
            alpha,
        ), axis=-1).astype(np.uint8)
        labels, palette = _quantize(
            rgba, 32, "none", "edge-lloyd", 4, 1.0, 4096, 17
        )
        displayed = palette[labels]
        source_alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        output_alpha = displayed[..., 3:4].astype(np.float32) / 255.0
        source_white = rgba[..., :3] * source_alpha + 255.0 * (1.0 - source_alpha)
        output_white = displayed[..., :3] * output_alpha + 255.0 * (1.0 - output_alpha)
        # Endpoint ownership deliberately spends a small amount of aggregate
        # white-composite error to keep alpha-zero and alpha-255 topologically
        # exact instead of averaging them into fractional palette entries.
        self.assertLess(float(np.sqrt(np.mean((source_white-output_white) ** 2))), 25.5)
        self.assertLess(float(np.mean(np.abs(rgba[..., 3].astype(float)-displayed[..., 3]))), 18.0)
        self.assertLessEqual(len(palette), 32)

    def test_rgba_quantization_preserves_alpha_topology(self):
        y, x = np.mgrid[:48, :80]
        rgba = np.zeros((48, 80, 4), dtype=np.uint8)
        interior = (x >= 8) & (x < 72) & (y >= 8) & (y < 40)
        fringe = (
            (x >= 7) & (x < 73) & (y >= 7) & (y < 41) & ~interior
        )
        rgba[interior, 0] = np.clip(70 + x[interior] * 2, 0, 255)
        rgba[interior, 1] = np.clip(180 + x[interior] // 2, 0, 255)
        rgba[interior, 2] = 40
        rgba[interior, 3] = 255
        rgba[fringe] = np.array((220, 190, 60, 128), np.uint8)

        labels, palette = _quantize(
            rgba, 24, "none", "edge-lloyd", 3, 1.0, 4096, 23
        )
        displayed = palette[labels]
        self.assertTrue(np.all(displayed[rgba[..., 3] == 0, 3] == 0))
        self.assertTrue(np.all(displayed[rgba[..., 3] == 255, 3] == 255))
        self.assertTrue(np.all((displayed[fringe, 3] > 0) & (displayed[fringe, 3] < 255)))

        flowed = _ownership_flow(rgba, labels, palette, 0.001, 2, 8.0)
        flowed_alpha = palette[flowed, 3]
        self.assertTrue(np.all(flowed_alpha[rgba[..., 3] == 0] == 0))
        self.assertTrue(np.all(flowed_alpha[rgba[..., 3] == 255] == 255))

    def test_selective_diffusion_stays_inside_opaque_rgba_region(self):
        rgba = np.zeros((40, 96, 4), dtype=np.uint8)
        gradient = np.arange(80, dtype=np.uint8) * 3
        rgba[8:32, 8:88, :3] = gradient[None, :, None]
        rgba[8:32, 8:88, 3] = 255
        labels, palette = _quantize(rgba, 8, "none", "median-cut")
        diffused = _selective_ordered_diffusion(rgba, labels, palette, 1.5, 0.0)
        changed = diffused != labels
        self.assertTrue(np.any(changed))
        self.assertTrue(np.all(rgba[..., 3][changed] == 255))
        self.assertTrue(np.all(palette[diffused][rgba[..., 3] == 0, 3] == 0))
        self.assertTrue(np.all(palette[diffused][rgba[..., 3] == 255, 3] == 255))

    def test_lossy_output_is_valid_and_compare_agrees(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            Image.fromarray(self.rgb, "RGB").save(source)
            result = optimize_png(
                source,
                output,
                PNGConfig(colors=16, ownership_strength=0.001),
            )
            metrics = compare_pngs(source, output)
            self.assertEqual(metrics["candidate_bytes"], result.winner.size)
            self.assertAlmostEqual(metrics["ssim"], result.winner.ssim, places=12)

    def test_target_trace_refines_final_lloyd_palette_rate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.fromarray(self.rgb, "RGB").save(source, compress_level=0)
            result = optimize_png(
                source,
                root / "output.png",
                PNGConfig(target_bytes=1_400, quantizer="auto"),
            )
            lloyd_counts = {
                candidate.colors for candidate in result.candidates
                if candidate.quantizer == "edge-lloyd"
            }
            self.assertGreater(len(lloyd_counts), 1)
            self.assertLessEqual(result.winner.size, 1_400)


if __name__ == "__main__":
    unittest.main()
