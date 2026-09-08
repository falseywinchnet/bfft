from __future__ import annotations

import unittest

from .state_word_lattice import PronunciationTrie, compose_state_word_lattice
from .word_lattice import Pronunciation


def _candidate(phone: str, rank: int) -> dict[str, object]:
    return {"phone": phone, "rank": rank, "score": rank / 10.0}


class StateWordLatticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trie = PronunciationTrie(
            (
                Pronunciation("okay", ("OW", "K", "EY")),
                Pronunciation("ok", ("OW", "K")),
                Pronunciation("obey", ("OW", "B", "EY")),
                Pronunciation("okeh", ("OW", "K", "EY")),
            )
        )
        self.crop = {
            "landmarks": [0, 1, 2, 3, 4],
            "edges": [
                {"node0": 0, "node1": 1, "frame0": 0, "frame1": 1, "candidates": [_candidate("OW", 2)]},
                {"node0": 0, "node1": 2, "frame0": 0, "frame1": 2, "candidates": [_candidate("OW", 1)]},
                {"node0": 1, "node1": 2, "frame0": 1, "frame1": 2, "candidates": [_candidate("K", 8)]},
                {"node0": 2, "node1": 3, "frame0": 2, "frame1": 3, "candidates": [_candidate("K", 2), _candidate("B", 5)]},
                {"node0": 3, "node1": 4, "frame0": 3, "frame1": 4, "candidates": [_candidate("EY", 1)]},
            ],
        }

    def test_exact_trie_composition_recovers_best_segmentation(self) -> None:
        candidates = compose_state_word_lattice(self.crop, self.trie)
        okay = next(
            item
            for item in candidates
            if item.node0 == 0 and item.node1 == 4 and item.phones == ("OW", "K", "EY")
        )
        self.assertEqual(
            [(edge["node0"], edge["node1"]) for edge in okay.phone_edges],
            [(0, 2), (2, 3), (3, 4)],
        )
        self.assertEqual(okay.words, ("okay", "okeh"))
        self.assertEqual(okay.cost.worst_rank, 2)

    def test_acoustic_cost_orders_homophone_classes_on_same_span(self) -> None:
        candidates = compose_state_word_lattice(self.crop, self.trie)
        full_span = [item for item in candidates if (item.node0, item.node1) == (0, 4)]
        self.assertEqual(full_span[0].phones, ("OW", "K", "EY"))
        self.assertEqual(full_span[1].phones, ("OW", "B", "EY"))

    def test_missing_phone_edge_cannot_jump_a_dag_node(self) -> None:
        crop = {
            "landmarks": [0, 1, 2],
            "edges": [
                {"node0": 0, "node1": 1, "frame0": 0, "frame1": 1, "candidates": [_candidate("OW", 1)]},
            ],
        }
        candidates = compose_state_word_lattice(crop, self.trie)
        self.assertFalse(any(item.node1 == 2 for item in candidates))

    def test_duration_is_an_admission_stratum_not_a_geometry_distance(self) -> None:
        crop = {
            "landmarks": [0, 1, 2, 3],
            "edges": [
                {"node0": 0, "node1": 1, "frame0": 0, "frame1": 1, "candidates": [{**_candidate("OW", 1), "duration_rank": 1}]},
                {"node0": 1, "node1": 2, "frame0": 1, "frame1": 2, "candidates": [{**_candidate("B", 1), "duration_rank": 30}, {**_candidate("K", 5), "duration_rank": 2}]},
                {"node0": 2, "node1": 3, "frame0": 2, "frame1": 3, "candidates": [{**_candidate("EY", 1), "duration_rank": 1}]},
            ],
        }
        candidates = compose_state_word_lattice(
            crop, self.trie, maximum_duration_rank=24
        )
        full = [item for item in candidates if (item.node0, item.node1) == (0, 3)]
        self.assertEqual(full[0].phones, ("OW", "K", "EY"))
        self.assertEqual(full[0].cost.duration_uncovered, 0)
        self.assertEqual(full[1].cost.duration_uncovered, 1)

    def test_state_profile_is_an_admission_stratum_not_a_geometry_distance(self) -> None:
        crop = {
            "landmarks": [0, 1, 2, 3],
            "edges": [
                {"node0": 0, "node1": 1, "frame0": 0, "frame1": 1, "candidates": [{**_candidate("OW", 1), "state_profile_rank": 1}]},
                {"node0": 1, "node1": 2, "frame0": 1, "frame1": 2, "candidates": [{**_candidate("B", 1), "state_profile_rank": 30}, {**_candidate("K", 5), "state_profile_rank": 2}]},
                {"node0": 2, "node1": 3, "frame0": 2, "frame1": 3, "candidates": [{**_candidate("EY", 1), "state_profile_rank": 1}]},
            ],
        }
        candidates = compose_state_word_lattice(
            crop, self.trie, maximum_state_rank=23
        )
        full = [item for item in candidates if (item.node0, item.node1) == (0, 3)]
        self.assertEqual(full[0].phones, ("OW", "K", "EY"))
        self.assertEqual(full[0].cost.state_uncovered, 0)
        self.assertEqual(full[1].cost.state_uncovered, 1)

    def test_joint_proposal_is_a_union_not_an_intersection(self) -> None:
        crop = {
            "landmarks": [0, 1, 2, 3],
            "edges": [
                {"node0": 0, "node1": 1, "frame0": 0, "frame1": 1, "candidates": [{**_candidate("OW", 1), "duration_rank": 1, "state_profile_rank": 1}]},
                {"node0": 1, "node1": 2, "frame0": 1, "frame1": 2, "candidates": [{**_candidate("B", 1), "duration_rank": 2, "state_profile_rank": 30}, {**_candidate("K", 5), "duration_rank": 30, "state_profile_rank": 2}]},
                {"node0": 2, "node1": 3, "frame0": 2, "frame1": 3, "candidates": [{**_candidate("EY", 1), "duration_rank": 1, "state_profile_rank": 1}]},
            ],
        }
        candidates = compose_state_word_lattice(
            crop,
            self.trie,
            maximum_duration_rank=24,
            maximum_state_rank=23,
        )
        full = [item for item in candidates if (item.node0, item.node1) == (0, 3)]
        self.assertEqual(full[0].phones, ("OW", "B", "EY"))
        self.assertEqual(full[0].cost.proposal_uncovered, 0)
        self.assertEqual(full[1].cost.proposal_uncovered, 0)


if __name__ == "__main__":
    unittest.main()
