from __future__ import annotations

import unittest

from experiments.ostensibly_frontend.sentence_decoder import (
    decode_sentence_paths,
    edge_channel_scores,
    path_display,
)


def edge(phone0, phone1, score, phones, words, seconds0=None, seconds1=None):
    return {
        "phone0": phone0,
        "phone1": phone1,
        "seconds0": float(phone0 if seconds0 is None else seconds0),
        "seconds1": float(phone1 if seconds1 is None else seconds1),
        "ambiguity_classes": [
            {"score": score, "phones": list(phones), "words": list(words)}
        ],
    }


class SentenceDecoderTests(unittest.TestCase):
    def test_generic_channel_uses_robust_distance_not_route_score(self) -> None:
        classes = [
            {
                "score": 0.0,
                "phones": ["A"],
                "words": ["route"],
                "generic_word_channels": {
                    "maximum": {"rank": 2, "distance": 0.8}
                },
            },
            {
                "score": 5.0,
                "phones": ["B"],
                "words": ["geometry"],
                "generic_word_channels": {
                    "maximum": {"rank": 1, "distance": 0.2}
                },
            },
            {
                "score": 2.0,
                "phones": ["C"],
                "words": ["middle"],
                "generic_word_channels": {
                    "maximum": {"rank": 3, "distance": 1.4}
                },
            },
        ]
        scored = edge_channel_scores(classes, "generic_word:maximum")
        by_word = {row[0]["words"][0]: row[1] for row in scored}
        self.assertEqual(by_word["geometry"], 0.0)
        self.assertAlmostEqual(by_word["route"], 1.0)
        raw = edge_channel_scores(
            classes, "generic_word:maximum", generic_score_mode="raw"
        )
        self.assertEqual(
            {row[0]["words"][0]: row[1] for row in raw}["geometry"],
            0.2,
        )
        empirical = edge_channel_scores(
            classes,
            "generic_word:maximum",
            generic_score_mode="length_cdf",
            length_distribution=(0.1, 0.2, 0.8, 1.4),
        )
        empirical_by_word = {
            row[0]["words"][0]: row[1] for row in empirical
        }
        self.assertLess(
            empirical_by_word["geometry"], empirical_by_word["route"]
        )
        best = decode_sentence_paths(
            (
                {
                    "phone0": 0,
                    "phone1": 1,
                    "seconds0": 0.0,
                    "seconds1": 0.1,
                    "ambiguity_classes": classes,
                },
            ),
            1,
            evidence_channel="generic_word:maximum",
        )[0]
        self.assertEqual(best.segments[0].words, ("geometry",))

    def test_global_path_beats_locally_best_first_edge(self) -> None:
        edges = (
            edge(0, 1, 0.0, ("A",), ("a",)),
            edge(1, 3, 0.9, ("B", "C"), ("bad_tail",)),
            edge(0, 2, 0.15, ("A", "B"), ("good_head",)),
            edge(2, 3, 0.15, ("C",), ("good_tail",)),
        )
        best = decode_sentence_paths(edges, 3, path_count=4)[0]
        self.assertEqual(
            tuple(segment.words for segment in best.segments),
            (("good_head",), ("good_tail",)),
        )

    def test_homophone_class_is_not_hard_selected(self) -> None:
        path = decode_sentence_paths(
            (edge(0, 2, 0.1, ("F", "AO"), ("four", "for")),),
            2,
        )[0]
        self.assertEqual(path.segments[0].words, ("for", "four"))
        self.assertEqual(path_display(path), "{for|four}")

    def test_unknown_fallback_completes_disconnected_lattice(self) -> None:
        path = decode_sentence_paths(
            (edge(0, 1, 0.0, ("A",), ("a",)),),
            2,
            unknown_phone_penalty=0.8,
        )[0]
        self.assertEqual(path.cost.uncovered_phones, 1)
        self.assertTrue(path.segments[-1].unknown)

    def test_timing_gaps_render_punctuation_without_semantics(self) -> None:
        path = decode_sentence_paths(
            (
                edge(0, 1, 0.0, ("A",), ("one",), 0.0, 0.1),
                edge(1, 2, 0.0, ("B",), ("two",), 0.4, 0.5),
                edge(2, 3, 0.0, ("C",), ("three",), 1.1, 1.2),
            ),
            3,
        )[0]
        self.assertEqual(path_display(path), "one, two. three")


if __name__ == "__main__":
    unittest.main()
