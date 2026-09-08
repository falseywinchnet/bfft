from __future__ import annotations

import unittest

from experiments.ostensibly_frontend.run_query_radio_boundary_atlas import (
    endpoint_ranking,
    fuse_rankings,
)


class RadioBoundaryFusionTests(unittest.TestCase):
    def test_endpoint_ranking_selects_best_pair_occurrence_per_phone(self) -> None:
        ranking = [
            {"phones": ["A", "B"], "distance": 0.3, "witness": "x"},
            {"phones": ["A", "C"], "distance": 0.1, "witness": "y"},
            {"phones": ["D", "B"], "distance": 0.2, "witness": "z"},
        ]
        left = endpoint_ranking(ranking, 0)
        self.assertEqual(left[0]["phone"], "A")
        self.assertEqual(left[0]["witness"], "y")
        right = endpoint_ranking(ranking, 1)
        self.assertEqual(right[0]["phone"], "C")

    def test_rank_fusions_preserve_channel_provenance(self) -> None:
        channels = {
            "center": [
                {"phone": "A", "rank": 1},
                {"phone": "B", "rank": 2},
            ],
            "boundary": [
                {"phone": "B", "rank": 1},
                {"phone": "A", "rank": 3},
            ],
        }
        mean = fuse_rankings(channels, "mean")
        conservative = fuse_rankings(channels, "conservative")
        self.assertEqual(mean[0]["phone"], "B")
        self.assertEqual(mean[0]["channel_ranks"], {"center": 2, "boundary": 1})
        self.assertEqual(conservative[0]["phone"], "B")


if __name__ == "__main__":
    unittest.main()
