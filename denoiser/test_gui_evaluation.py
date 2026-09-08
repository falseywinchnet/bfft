"""Headless invariants for the Dear PyGui evaluation plumbing."""

import unittest

import numpy as np
from scipy import sparse

from .gui import (
    DenoiserLab,
    IMAGE_METHODS,
    _diagnostic_summary,
    _evaluation_table,
    _safe_filename,
)


class _FakeDPG:
    def __init__(self):
        self.values = {
            "image_method": "Best — rebuilt FMMT transport support",
            "image_corruption": "Gaussian additive",
            "image_noise_amount": 0.1,
            "image_noise_density": 0.0,
            "image_scoreboard": "",
            "image_status": "",
            "image_diagnostics": "",
        }

    def get_value(self, key):
        return self.values[key]

    def set_value(self, key, value):
        self.values[key] = value


class GuiEvaluationTests(unittest.TestCase):
    def test_export_names_are_portable(self):
        self.assertEqual(
            _safe_filename("evaluation suite — stopped causal / FMMT"),
            "evaluation_suite_stopped_causal_FMMT",
        )

    def test_interface_exposes_only_best_freeze_and_canon_compare(self):
        self.assertEqual(IMAGE_METHODS, (
            "Best — rebuilt FMMT transport support",
            "Last canon freeze — causal population Chambolle",
            "Canon compare — OEM Chambolle / TV / mean / FMMT",
        ))

    def test_diagnostics_summarize_dense_and_sparse_state(self):
        dense = np.arange(12, dtype=np.float64).reshape(3, 4)
        matrix = sparse.eye(5, format="csr")
        result = _diagnostic_summary({"dense": dense, "sparse": matrix})
        self.assertEqual(result["dense"]["shape"], [3, 4])
        self.assertEqual(result["dense"]["maximum"], 11.0)
        self.assertEqual(result["sparse"]["shape"], [5, 5])
        self.assertEqual(result["sparse"]["nonzero_count"], 5)

    def test_scoreboard_measures_all_outputs_against_reference(self):
        yy, xx = np.mgrid[:8, :8]
        truth = 0.4 + 0.1 * np.sin(xx) + 0.05 * np.cos(yy)
        shifted = truth + 0.02
        table, scores = _evaluation_table(
            truth,
            {"observation": shifted, "identity": truth},
            {"identity": 0.25},
        )
        self.assertIn("observation", table)
        self.assertIn("identity", table)
        self.assertEqual(scores["identity"]["mse"], 0.0)
        self.assertGreater(scores["observation"]["mse"], 0.0)

    def test_best_runs_headlessly_without_peer_candidates(self):
        dpg = _FakeDPG()
        lab = DenoiserLab(dpg)
        yy, xx = np.indices((12, 12), dtype=float)
        lab.clean_image = 0.25 + 0.01 * xx - 0.006 * yy
        rng = np.random.default_rng(19)
        lab.image = np.clip(
            lab.clean_image + 0.05 * rng.normal(size=lab.clean_image.shape),
            0.0, 1.0)
        lab.source_name = "headless causal population"
        rendered = {}
        lab.render_images = lambda images: rendered.update(images)
        lab.run_2d()
        self.assertTrue(dpg.values["image_status"].startswith(
            "Finished Best — rebuilt FMMT transport support"))
        self.assertIn(
            "BEST — rebuilt FMMT transport support",
            lab.last_evaluation_report["reference_metrics"],
        )
        self.assertEqual(set(rendered), {
            "clean source", "corrupted", "BEST — rebuilt FMMT transport support",
        })

    def test_last_canon_freeze_runs_headlessly(self):
        dpg = _FakeDPG()
        dpg.values["image_method"] = (
            "Last canon freeze — causal population Chambolle")
        lab = DenoiserLab(dpg)
        yy, xx = np.indices((12, 12), dtype=float)
        lab.clean_image = 0.3 + 0.009 * xx - 0.005 * yy
        rng = np.random.default_rng(23)
        lab.image = np.clip(
            lab.clean_image + 0.06 * rng.normal(size=lab.clean_image.shape),
            0.0, 1.0)
        lab.source_name = "headless last canon"
        rendered = {}
        lab.render_images = lambda images: rendered.update(images)
        lab.run_2d()
        self.assertTrue(dpg.values["image_status"].startswith(
            "Finished Last canon freeze"))
        self.assertIn(
            "LAST CANON FREEZE — causal population Chambolle", rendered)

    def test_canon_compare_contains_only_declared_controls(self):
        dpg = _FakeDPG()
        dpg.values["image_method"] = (
            "Canon compare — OEM Chambolle / TV / mean / FMMT")
        lab = DenoiserLab(dpg)
        yy, xx = np.indices((12, 12), dtype=float)
        lab.clean_image = 0.3 + 0.009 * xx - 0.005 * yy
        rng = np.random.default_rng(23)
        lab.image = np.clip(
            lab.clean_image + 0.06 * rng.normal(size=lab.clean_image.shape),
            0.0, 1.0)
        lab.source_name = "headless canon compare"
        rendered = {}
        lab.render_images = lambda images: rendered.update(images)
        lab.run_2d()
        self.assertTrue(dpg.values["image_status"].startswith(
            "Finished Canon compare"))
        for method in (
            "BEST — rebuilt FMMT transport support",
            "CANON — OEM Chambolle 0.10",
            "CANON — reconstructed TV 0.10",
            "CANON — mean 3x3",
            "CANON — integrated FMMT",
        ):
            self.assertIn(
                method, lab.last_evaluation_report["reference_metrics"])
            self.assertIn(method, rendered)
        self.assertEqual(len(rendered), 7)


if __name__ == "__main__":
    unittest.main()
