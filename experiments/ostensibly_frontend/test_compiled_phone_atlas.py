from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.ostensibly_frontend.compiled_phone_atlas import (
    compile_phone_atlas,
    load_phone_atlas,
    save_phone_atlas,
)
from experiments.ostensibly_frontend.occupation_point_cloud import CloudFitConfig


class CompiledPhoneAtlasTests(unittest.TestCase):
    def test_nearest_occurrence_label_and_roundtrip(self) -> None:
        frame = np.linspace(0.0, 1.0, 80)

        def cloud(offset):
            return np.column_stack((offset + frame**2, frame, np.ones_like(frame)))

        config = CloudFitConfig(
            distance_mode="sliced_wasserstein",
            sliced_projection_count=8,
            row_metric_scale=1.0,
            frame_metric_scale=1.0,
            height_metric_scale=1.0,
        )
        atlas = compile_phone_atlas(
            (("A", "u:0", cloud(0.0)), ("A", "u:1", cloud(0.5)), ("B", "u:2", cloud(2.0))),
            config,
            quantile_count=24,
        )
        ranking = atlas.rank(cloud(0.1), chunk_size=2)
        self.assertEqual(ranking[0]["phone"], "A")
        self.assertEqual(ranking[0]["witness"], "u:0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phone_atlas.npz"
            save_phone_atlas(path, atlas)
            self.assertEqual(load_phone_atlas(path).rank(cloud(0.1)), ranking)


if __name__ == "__main__":
    unittest.main()
