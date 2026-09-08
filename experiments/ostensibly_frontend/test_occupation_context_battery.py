import unittest

import numpy as np

from experiments.ostensibly_frontend.run_occupation_context_battery import (
    add_empirical_rank_fusion,
    boundary_transition_distance,
    boundary_transition_law,
    edge_context_windows,
    ordered_context_cloud,
    proposal_shortlist,
    silence_occupation_cloud,
)
from experiments.ostensibly_frontend.summarize_occupation_context_battery import (
    summarize_results,
)
from experiments.ostensibly_frontend.run_occupation_word_span_battery import (
    joint_gauge_chunks,
    ordered_span_cloud,
    span_transition_distance,
    span_transition_laws,
    timestamp_lane_chunks,
)


class OccupationContextBatteryTests(unittest.TestCase):
    def test_silence_chart_is_deterministic_zero_mass(self):
        cloud = silence_occupation_cloud(32)
        self.assertEqual(cloud.shape, (32, 3))
        self.assertTrue(np.all(cloud[:, 0] == 0.5))
        self.assertTrue(np.all(cloud[:, 2] == 0.0))
        self.assertTrue(np.all(np.diff(cloud[:, 1]) > 0.0))

    def test_edge_context_applies_query_topology(self):
        windows = ["left", "center", "right"]
        self.assertEqual(
            edge_context_windows(windows, 1, True, False),
            ("left", "center", None),
        )
        self.assertEqual(
            edge_context_windows(windows, 0, False, True),
            (None, "left", "center"),
        )

    def test_ordered_context_places_phones_in_disjoint_time_lanes(self):
        base = np.column_stack(
            (
                np.linspace(0.0, 1.0, 20),
                np.linspace(0.0, 1.0, 20),
                np.ones(20),
            )
        )
        context = ordered_context_cloud((base, base, base), points_per_phone=10)
        self.assertEqual(context.shape, (30, 3))
        self.assertLessEqual(float(np.max(context[:10, 1])), 1.0 / 3.0)
        self.assertGreaterEqual(float(np.min(context[10:20, 1])), 1.0 / 3.0)
        self.assertLessEqual(float(np.max(context[10:20, 1])), 2.0 / 3.0)
        self.assertGreaterEqual(float(np.min(context[20:, 1])), 2.0 / 3.0)

    def test_ordered_context_rejects_missing_phone(self):
        cloud = np.ones((10, 3))
        with self.assertRaises(ValueError):
            ordered_context_cloud((cloud, np.empty((0, 3)), cloud), 5)

    def test_boundary_transport_quotients_common_row_translation(self):
        frame = np.linspace(0.0, 1.0, 400)
        row = np.linspace(0.1, 0.8, 400)
        left = np.column_stack((row, frame, np.ones_like(frame)))
        right = np.column_stack((row + 0.12 * row**2, frame, np.ones_like(frame)))
        shifted_left = left.copy()
        shifted_right = right.copy()
        shifted_left[:, 0] += 3.0
        shifted_right[:, 0] += 3.0
        law = boundary_transition_law(left, right)
        shifted_law = boundary_transition_law(shifted_left, shifted_right)
        self.assertLess(boundary_transition_distance(law, shifted_law), 1e-12)
        flat_right = left.copy()
        flat_law = boundary_transition_law(left, flat_right)
        self.assertGreater(boundary_transition_distance(law, flat_law), 1e-3)

    def test_conservative_rank_fusion_requires_all_channels(self):
        rows = [
            {"key": "balanced", "a": 2.0, "b": 2.0, "c": 2.0},
            {"key": "a_only", "a": 1.0, "b": 4.0, "c": 4.0},
            {"key": "b_only", "a": 4.0, "b": 1.0, "c": 3.0},
            {"key": "c_only", "a": 3.0, "b": 3.0, "c": 1.0},
        ]
        add_empirical_rank_fusion(rows, ("a", "b", "c"))
        by_key = {row["key"]: row for row in rows}
        self.assertLess(
            by_key["balanced"]["conservative_rank_fusion"],
            by_key["a_only"]["conservative_rank_fusion"],
        )
        self.assertEqual(
            set(by_key["balanced"]["conservative_rank_fusion_channels"]),
            {"a", "b", "c"},
        )

    def test_conservative_rank_fusion_is_label_free(self):
        rows = [
            {"key": "x", "label": "R", "a": 1.0, "b": 2.0},
            {"key": "y", "label": "NOT_R", "a": 2.0, "b": 1.0},
        ]
        add_empirical_rank_fusion(rows, ("a", "b"))
        first = [row["conservative_rank_fusion"] for row in rows]
        rows[0]["label"], rows[1]["label"] = rows[1]["label"], rows[0]["label"]
        add_empirical_rank_fusion(rows, ("a", "b"))
        self.assertEqual(first, [row["conservative_rank_fusion"] for row in rows])

    def test_proposal_shortlist_preserves_channel_provenance(self):
        def ranking(*phones):
            return [{"phone": phone} for phone in phones]

        proposals = proposal_shortlist(
            {
                "shape": ranking("R", "L", "S"),
                "boundary": ranking("S", "R", "K"),
            },
            top_k_per_channel=2,
        )
        by_phone = {item["phone"]: item for item in proposals}
        self.assertEqual(set(by_phone), {"R", "L", "S"})
        self.assertEqual(by_phone["R"]["channel_ranks"], {"shape": 1, "boundary": 2})
        self.assertEqual(by_phone["R"]["supporting_channel_count"], 2)

    def test_holdout_summary_reports_recall_without_selecting_a_winner(self):
        document = {
            "summary": {
                "target_phone": "R",
                "center_only_target_rank": 4,
                "context_target_rank": 3,
                "relational_target_rank": 2,
                "support_target_rank": 8,
                "conservative_fusion_target_rank": 5,
                "corresponding_context_instance_rank": 6,
                "corresponding_relational_instance_rank": 4,
                "corresponding_support_instance_rank": 9,
                "corresponding_conservative_fusion_instance_rank": 7,
            },
            "proposal_shortlists": {
                "top_3": [{"phone": "R"}, {"phone": "L"}],
                "top_5": [{"phone": "L"}],
            },
        }
        result = summarize_results([document])
        self.assertEqual(result["channel_summary"]["boundary_transport"]["top_5_hits"], 1)
        self.assertEqual(result["proposal_summary"]["top_3"]["target_hits"], 1)
        self.assertEqual(result["proposal_summary"]["top_5"]["target_hits"], 0)

    def test_ordered_word_span_preserves_every_phone_lane(self):
        cloud = np.column_stack(
            (
                np.linspace(0.0, 1.0, 30),
                np.linspace(0.0, 1.0, 30),
                np.ones(30),
            )
        )
        span = ordered_span_cloud((cloud, cloud, cloud, cloud), 10)
        self.assertEqual(span.shape, (40, 3))
        for slot in range(4):
            lane = span[slot * 10 : (slot + 1) * 10, 1]
            self.assertGreaterEqual(float(np.min(lane)), slot / 4)
            self.assertLessEqual(float(np.max(lane)), (slot + 1) / 4)

    def test_word_transition_sequence_detects_changed_boundary(self):
        frame = np.linspace(0.0, 1.0, 400)
        base = np.column_stack((frame, frame, np.ones_like(frame)))
        bent = base.copy()
        bent[:, 0] += 0.2 * frame**2
        left = span_transition_laws((base, bent, base))
        right = span_transition_laws((base, base, base))
        self.assertGreater(span_transition_distance(left, right), 1e-3)

    def test_joint_gauge_preserves_between_phone_row_offset(self):
        frame = np.linspace(0.0, 1.0, 100)
        low = np.column_stack((0.2 + 0.1 * frame, frame, np.ones_like(frame)))
        high = np.column_stack((0.7 + 0.1 * frame, frame, np.ones_like(frame)))
        left, right = joint_gauge_chunks((low, high), 100)
        self.assertLess(float(np.max(left[:, 0])), float(np.min(right[:, 0])))

    def test_joint_gauge_is_common_monotone_row_invariant(self):
        frame = np.linspace(0.0, 1.0, 100)
        low = np.column_stack((0.2 + 0.1 * frame, frame, np.ones_like(frame)))
        high = np.column_stack((0.7 + 0.1 * frame, frame, np.ones_like(frame)))
        original = joint_gauge_chunks((low, high), 100)
        warped = tuple(chunk.copy() for chunk in (low, high))
        for chunk in warped:
            chunk[:, 0] = np.exp(3.0 * chunk[:, 0])
        transformed = joint_gauge_chunks(warped, 100)
        self.assertTrue(np.allclose(original[0][:, 0], transformed[0][:, 0]))
        self.assertTrue(np.allclose(original[1][:, 0], transformed[1][:, 0]))

    def test_timestamp_lanes_preserve_duration_and_gap(self):
        frame = np.linspace(0.0, 1.0, 32)
        chunk = np.column_stack((frame, frame, np.ones_like(frame)))
        placed = timestamp_lane_chunks(
            (chunk, chunk), ((10.0, 10.25), (10.50, 11.0))
        )
        first, second = placed[:32], placed[32:]
        self.assertAlmostEqual(float(np.max(first[:, 1])), 0.25)
        self.assertAlmostEqual(float(np.min(second[:, 1])), 0.50)


if __name__ == "__main__":
    unittest.main()
