"""Tests for deterministic generation and reference-grounded evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from PIL import Image

from .evaluate import candidate_metrics, evaluate_suite
from .generate import generate_suite


class CodecGroundTruthTests(unittest.TestCase):
    def test_generate_is_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            manifest = generate_suite(first, size=128, supersample=2)
            generate_suite(second, size=128, supersample=2)
            self.assertEqual(len(manifest["cases"]), 8)
            self.assertEqual(len(list((first/"upload").glob("*"))), 16)
            self.assertEqual(
                (first/"references"/"mixed.png").read_bytes(),
                (second/"references"/"mixed.png").read_bytes(),
            )

    def test_identity_scores_perfectly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate_suite(root, size=128, supersample=2)
            case = manifest["cases"][0]
            reference = root/case["reference"]
            metrics = candidate_metrics(reference, reference, probes=case["probes"])
            self.assertEqual(metrics["psnr_db"], 99.0)
            self.assertAlmostEqual(metrics["ssim"], 1.0, places=12)
            self.assertEqual(metrics["edge_luma_mae"], 0.0)

    def test_evaluator_separates_png_and_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_suite(root, size=128, supersample=2)
            report = evaluate_suite(root, {"upload": root/"upload"})
            self.assertEqual(len(report["rows"]), 16)
            self.assertEqual({row["codec"] for row in report["rows"]}, {"png", "jpeg"})
            png = next(row for row in report["rows"] if row["codec"] == "png")
            self.assertAlmostEqual(png["ssim"], 1.0, places=12)

    def test_known_bad_controls_move_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate_suite(root, size=128, supersample=2)
            gradient = next(case for case in manifest["cases"] if case["name"] == "gradients")
            reference = root/gradient["reference"]
            identity = candidate_metrics(reference, reference, probes=gradient["probes"])
            banded = candidate_metrics(
                reference, root/"controls"/"banded"/"png__gradients.png",
                probes=gradient["probes"],
            )
            geometry = next(case for case in manifest["cases"] if case["name"] == "geometry")
            geometry_reference = root/geometry["reference"]
            blurred = candidate_metrics(
                geometry_reference, root/"controls"/"blur"/"png__geometry.png",
                probes=geometry["probes"],
            )
            self.assertGreater(banded["gradient_false_plateau_fraction"], identity["gradient_false_plateau_fraction"])
            self.assertLess(blurred["edge_psnr_db"], 99.0)
            self.assertLess(blurred["ssim"], 1.0)

    def test_shape_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root/"a.png", root/"b.png"
            Image.new("RGB", (32, 32)).save(a)
            Image.new("RGB", (31, 32)).save(b)
            with self.assertRaises(ValueError):
                candidate_metrics(a, b)


if __name__ == "__main__":
    unittest.main()
