from __future__ import annotations

import unittest

from .run_oracle_lexicographic_sequence_audit import local_history_key


class LexicographicSequenceAuditTests(unittest.TestCase):
    def test_history_key_prefers_smaller_worst_then_next_worst(self) -> None:
        self.assertLess(local_history_key([1, 4, 2]), local_history_key([1, 5, 1]))
        self.assertLess(local_history_key([1, 4, 2]), local_history_key([3, 4, 2]))


if __name__ == "__main__":
    unittest.main()
