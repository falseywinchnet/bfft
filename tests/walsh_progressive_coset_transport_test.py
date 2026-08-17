#!/usr/bin/env python3

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.walsh_progressive_coset_transport import audit_dimension


def test_progressive_matching_is_finite_and_balanced() -> None:
    row = audit_dimension(3, half_width=2, h=1, seed=5)
    assert row["states"] == 4**3
    assert 0.0 < row["target_prefix_probability"] <= 1.0
    for transport in row["transports"]:
        assert transport["final_renyi2_log2"] >= -1e-12
        assert transport["steps"][0]["active_states"] == row["states"] // 2


def test_renyi_assignment_beats_geometric_for_one_step() -> None:
    row = audit_dimension(4, half_width=2, h=1, seed=17)
    transports = {item["mode"]: item for item in row["transports"]}
    assert (
        transports["renyi"]["final_renyi2_log2"]
        <= transports["geometric"]["final_renyi2_log2"] + 1e-12
    )


if __name__ == "__main__":
    test_progressive_matching_is_finite_and_balanced()
    test_renyi_assignment_beats_geometric_for_one_step()
    print("walsh progressive-coset transport tests passed")
