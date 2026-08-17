#!/usr/bin/env python3
"""Regression tests for the two new walk research references."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.conjugate_pair_split_walk import (  # noqa: E402
    comparison_ledger,
    conjugate_pair_fft,
    conjugate_pair_leaf_order,
    conjugate_pair_rfft,
    conjugate_pair_rfft_residues,
    real_fold_ledger,
)
from experiments.dip2_phase_axis import (  # noqa: E402
    dip2_rfft,
    dip2_rfft_remainder_tree,
    phase_series,
    tree_ledger,
)
from experiments.dip2_phase_axis_subquadratic import (  # noqa: E402
    dip2_subquadratic_rfft,
)


def main():
    rng = np.random.default_rng(84)

    for n in (4, 8, 16, 32, 64, 256, 1024):
        x = rng.standard_normal(n)
        a, b = phase_series(x)
        assert np.isrealobj(a) and np.isrealobj(b)
        err = np.max(np.abs(dip2_rfft(x) - np.fft.rfft(x)))
        assert err < 2e-10 * n

        ba, bb = conjugate_pair_rfft_residues(x)
        assert np.isrealobj(ba) and np.isrealobj(bb)
        berr = np.max(np.abs((ba - 1j * bb) - np.fft.rfft(x)))
        assert berr < 2e-10 * n

    for n in (4, 8, 16, 32):
        x = rng.standard_normal(n)
        err = np.max(np.abs(dip2_rfft_remainder_tree(x) - np.fft.rfft(x)))
        assert err < 2e-8
        led = tree_ledger(n)
        assert led["natural_leaf_order"]
        assert led["all_subtrees_contiguous"]
        assert led["all_intervals_full_row_rank"]
        assert led["min_rank_margin"] == 0

    for n in (4, 8, 16, 32):
        x = rng.standard_normal(n)
        err = np.max(np.abs(dip2_subquadratic_rfft(x) - np.fft.rfft(x)))
        assert err < 2e-10 * n

    for n in (4, 8, 16, 64, 256):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        err = np.max(np.abs(conjugate_pair_fft(x) - np.fft.fft(x)))
        assert err < 2e-10 * n
        assert sorted(conjugate_pair_leaf_order(n)) == list(range(n))

    # The conjugate-pair rotation genuinely changes the ingest permutation,
    # but its improvement is transport-shape evidence, not a novelty proof.
    led = comparison_ledger(1024)
    assert not led["same_permutation"]
    assert led["travel_ratio"] < 0.90
    for n in (16, 64, 256, 1024):
        led = real_fold_ledger(n)
        assert led["real_multiplications"] > 0
        assert led["real_additions"] == 2 * led["real_multiplications"]

    # Larger exactness check exercises the logarithmic braid without invoking
    # the diagnostic quadratic remainder oracle.
    x = rng.standard_normal(4096)
    assert np.max(np.abs(conjugate_pair_rfft(x) - np.fft.rfft(x))) < 2e-8
    print("PASS DIP2 phase-axis Bruun invariant + conjugate-pair walk")


if __name__ == "__main__":
    main()
