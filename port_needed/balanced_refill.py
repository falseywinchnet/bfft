"""PORT 06: simultaneous mass-balanced refill along unstable directions.

The viewer uses the one-shot histogram form: two finite image passes and a
small fixed local scan per cell.  It replaces the older fourteen global
bisection passes.  Every unstable cell acts simultaneously; nothing is ranked
or admitted through a global budget.
"""

from __future__ import annotations

from experiments.wasserstein_allocation_tree import (
    _balanced_branch_histogram as _reference_balanced_branch_histogram,
)


def balanced_refill(*args, bins: int = 64, **kwargs):
    return _reference_balanced_branch_histogram(
        *args, max(int(bins), 8), **kwargs)
