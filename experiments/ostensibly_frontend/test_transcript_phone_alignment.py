from __future__ import annotations

import unittest

from experiments.ostensibly_frontend.transcript_phone_alignment import (
    align_phone_proposals,
    align_phone_proposals_trace,
    markdown_section,
    transcript_words,
)


class TranscriptPhoneAlignmentTests(unittest.TestCase):
    def test_alignment_separates_top1_topk_missing_and_surplus(self) -> None:
        result = align_phone_proposals(
            ("R", "AY", "T", "N"),
            (("R",), ("EH", "AY"), ("T",), ("S",), ("N",)),
        )
        self.assertEqual(result.edit_cost, 1)
        self.assertEqual(result.top1_hits, 3)
        self.assertEqual(result.topk_hits, 4)
        self.assertEqual(result.missing_reference_phones, 0)
        self.assertEqual(result.surplus_proposal_regions, 1)
        _, mapping = align_phone_proposals_trace(
            ("R", "AY", "T", "N"),
            (("R",), ("EH", "AY"), ("T",), ("S",), ("N",)),
        )
        self.assertEqual(mapping, (0, 1, 2, 4))

    def test_markdown_listener_extraction_removes_speaker_and_unclear(self) -> None:
        text = "# T\n## A\n**Speaker 1:** Right [unclear].\n## B\nWrong"
        self.assertEqual(transcript_words(markdown_section(text, "A")), ("right",))


if __name__ == "__main__":
    unittest.main()
