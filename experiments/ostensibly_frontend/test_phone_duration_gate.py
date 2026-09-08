from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .phone_duration_gate import (
    combine_phone_duration_gates,
    compile_phone_duration_gate,
    load_phone_duration_gate,
    save_phone_duration_gate,
)


class PhoneDurationGateTests(unittest.TestCase):
    def test_duration_only_admits_labels_and_preserves_geometric_order(self) -> None:
        gate = compile_phone_duration_gate(
            (("SHORT", 0.04), ("SHORT", 0.05), ("LONG", 0.18), ("LONG", 0.20))
        )
        ranking = [
            {"phone": "LONG", "distance": 0.1},
            {"phone": "SHORT", "distance": 0.2},
        ]
        self.assertEqual(gate.allowed_labels(0.045, 1), ("SHORT",))
        self.assertEqual(gate.rank_labels(0.045), ("SHORT", "LONG"))
        self.assertEqual(gate.filter_ranking(0.045, ranking, 1)[0]["phone"], "SHORT")

    def test_combined_gate_round_trips(self) -> None:
        first = compile_phone_duration_gate((("A", 0.1),))
        second = compile_phone_duration_gate((("B", 0.2),))
        combined = combine_phone_duration_gates((first, second))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.npz"
            save_phone_duration_gate(path, combined)
            loaded = load_phone_duration_gate(path)
        self.assertEqual(set(loaded.labels.tolist()), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
