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


def fit_regions(
    labels,
    centers,
    target_lab,
    objective,
    *,
    ridge_count=0,
    affine_record=None,
    affine=None,
):
    if affine_record is None or affine is None:
        affine_record, affine = _fit_hard_regions(
            labels, target_lab, objective)
    if int(ridge_count) <= 0:
        return affine_record, affine, {
            "ridge_count": 0, "selected": "affine"}
    ridge_record, ridge, information = _fit_hard_regions_with_ridge(
        labels, centers, target_lab, objective,
        ridge_count=int(ridge_count),
        initial_affine=affine,
    )
    if affine_record["objective"] <= ridge_record["objective"]:
        # Restore the residual field to the actually selected readout.
        objective.evaluate(affine_record["rgb"])
        information["selected"] = "affine"
        information["rejected_ridge_objective"] = ridge_record["objective"]
        return affine_record, affine, information
    information["selected"] = "ridge"
    information["affine_objective"] = affine_record["objective"]
    return ridge_record, ridge, information
