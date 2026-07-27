"""PORT 07: hard-cell affine/ridge readout.

The affine solve is independent 3x3 normal assembly per region.  Optional
measured ridges add a very small local basis; no global sparse factorization
or all-cell graph is constructed.
"""

from __future__ import annotations

from experiments.wasserstein_allocation_tree import (
    fit_hard_regions as _fit_hard_regions,
    fit_hard_regions_with_ridge as _fit_hard_regions_with_ridge,
)


def fit_regions(labels, centers, target_lab, objective, *, ridge_count=0):
    if int(ridge_count) <= 0:
        record, reconstruction = _fit_hard_regions(
            labels, target_lab, objective)
        return record, reconstruction, {"ridge_count": 0}
    return _fit_hard_regions_with_ridge(
        labels, centers, target_lab, objective,
        ridge_count=int(ridge_count))
