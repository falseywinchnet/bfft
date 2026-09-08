from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.ostensibly_frontend.compiled_boundary_atlas import (
    compile_boundary_atlas,
    load_boundary_atlas,
    save_boundary_atlas,
)
from experiments.ostensibly_frontend.occupation_point_cloud import CloudFitConfig


class CompiledBoundaryAtlasTests(unittest.TestCase):
    def test_roundtrip_and_nearest_occurrence_pair_ranking(self) -> None:
        frame = np.linspace(0.0, 1.0, 96)

        def cloud(offset: float) -> np.ndarray:
            return np.column_stack(
                (offset + 0.1 * np.sin(7.0 * frame), frame, np.ones_like(frame))
            )

        occurrences = (
            (("B", "AH"), "u:0", cloud(0.0)),
            (("B", "AH"), "u:1", cloud(0.4)),
            (("K", "AH"), "u:2", cloud(2.0)),
        )
        config = CloudFitConfig(
            distance_mode="sliced_wasserstein",
            sliced_projection_count=8,
            row_metric_scale=1.0,
            frame_metric_scale=1.0,
            height_metric_scale=1.0,
        )
        atlas = compile_boundary_atlas(occurrences, config, quantile_count=32)
        ranking = atlas.rank(cloud(0.05), chunk_size=2)
        self.assertEqual(ranking[0]["phones"], ["B", "AH"])
        self.assertEqual(ranking[0]["witness"], "u:0")
        occurrences = atlas.rank_occurrences(cloud(0.05), chunk_size=2)
        self.assertEqual([item["witness"] for item in occurrences[:2]], ["u:0", "u:1"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.npz"
            save_boundary_atlas(path, atlas)
            loaded = load_boundary_atlas(path)
            self.assertEqual(loaded.rank(cloud(0.05)), ranking)


if __name__ == "__main__":
    unittest.main()
