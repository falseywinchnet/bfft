"""Orchestration of ports 02–06 for simultaneous transport refill.

This module is intentionally thin.  The current executable reference remains
the validated experiment while its hot stages are isolated beside this file.
Replacing those stages in C++ must preserve this public result contract.
"""

from __future__ import annotations

from experiments.wasserstein_allocation_tree import (
    bifurcate_allocation as _reference_bifurcate_allocation,
)


def allocate_supports(geometry: dict, **parameters):
    # Structural policy enforced by the canonical viewer.
    parameters["exact_branch_balance"] = False
    parameters["direct_metric_branches"] = False
    return _reference_bifurcate_allocation(geometry, **parameters)
