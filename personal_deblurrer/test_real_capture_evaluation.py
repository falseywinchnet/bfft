from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from denoiser.run_2d_denoiser_battery import sources

from .real_capture_evaluation import (
    evaluate_capture_files,
    evaluate_capture_pair,
)
from .run_visibility_benchmark import _layered_pair


class RealCaptureEvaluationTests(unittest.TestCase):
    def test_identical_pair_abstains_and_closes_exactly(self) -> None:
        image = sources(32)["cameraman"]
        evaluation = evaluate_capture_pair(image, image, passes=8)
        np.testing.assert_array_equal(evaluation.result.image, image)
        self.assertEqual(evaluation.diagnostics["forward_closure_rms"], 0.0)
        self.assertEqual(
            evaluation.diagnostics["truth_status"],
            "no_reference_not_ground_truth",
        )

    def test_positive_atlas_forward_closure_beats_shared_average(self) -> None:
        _, observations = _layered_pair(
            sources(40)["cameraman"], 3.0, noise_sigma=0.0, seed=61000)
        evaluation = evaluate_capture_pair(*observations, passes=24)
        self.assertLess(
            evaluation.diagnostics[
                "forward_closure_over_pair_disagreement"],
            0.8,
        )
        self.assertEqual(
            evaluation.predicted_observations.shape,
            (2, *observations[0].shape),
        )

    def test_file_evaluation_preserves_source_bytes(self) -> None:
        first = np.zeros((24, 24), dtype=np.uint8)
        first[:, 8:16] = 190
        second = np.roll(first, 2, axis=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.png"
            second_path = root / "second.png"
            Image.fromarray(first).save(first_path)
            Image.fromarray(second).save(second_path)
            output = root / "audit"
            report = evaluate_capture_files(
                first_path, second_path, output, passes=8)
            self.assertTrue(report["all_sources_unchanged"])
            self.assertTrue((output / "deblurred.png").is_file())
            stored = json.loads((output / "evaluation.json").read_text())
            self.assertTrue(stored["all_sources_unchanged"])
            self.assertEqual(
                stored["source_provenance"][0]["sha256_before"],
                stored["source_provenance"][0]["sha256_after"],
            )


if __name__ == "__main__":
    unittest.main()
