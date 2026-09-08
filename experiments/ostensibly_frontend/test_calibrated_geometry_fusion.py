from __future__ import annotations

import unittest

from .calibrated_geometry_fusion import (
    fuse_rank_channels,
    rank_summary,
    target_rank,
)


def _ranking(order: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {"phone": phone, "rank": rank, "distance": float(rank)}
        for rank, phone in enumerate(order, start=1)
    ]


class CalibratedGeometryFusionTests(unittest.TestCase):
    def test_zero_weight_is_exact_topology_order(self) -> None:
        topology = _ranking(("A", "B", "C"))
        physical = _ranking(("C", "B", "A"))
        self.assertEqual(
            [row["phone"] for row in fuse_rank_channels(topology, physical, 0.0)],
            ["A", "B", "C"],
        )

    def test_equal_weight_rewards_jointly_supported_label(self) -> None:
        topology = _ranking(("A", "B", "D", "C"))
        physical = _ranking(("C", "B", "D", "A"))
        self.assertEqual(target_rank(topology, physical, "B", 1.0), 1)

    def test_summary_reports_retrieval_counts(self) -> None:
        summary = rank_summary((1, 2, 6, 10))
        self.assertEqual(summary["top1"], 1)
        self.assertEqual(summary["top5"], 2)
        self.assertEqual(summary["top10"], 4)

    def test_minimum_policy_preserves_either_strong_channel(self) -> None:
        topology = _ranking(("A", "B", "C", "D"))
        physical = _ranking(("D", "C", "B", "A"))
        fused = fuse_rank_channels(topology, physical, 1.0, policy="minimum")
        self.assertEqual({fused[0]["phone"], fused[1]["phone"]}, {"A", "D"})


if __name__ == "__main__":
    unittest.main()
