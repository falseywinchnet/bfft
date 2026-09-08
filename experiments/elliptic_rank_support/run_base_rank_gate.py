"""Identify pencils that carry five independent sections at the record fiber."""

from __future__ import annotations

from elliptic_rank_support import (
    integral_record_points,
    reduction_dependency_masks,
    support_pencils,
)


def main() -> None:
    passing = []
    pencils = support_pencils(integral_record_points())
    for index, pencil in enumerate(pencils):
        model = pencil.integral_model(2)
        points = pencil.transported_points(2)
        dependencies = reduction_dependency_masks(model, points)
        if not dependencies:
            passing.append(index)
    print(f"pencils={len(pencils)}")
    print(f"base-rank-five divisors={len(passing)}")
    print("indices=", passing)


if __name__ == "__main__":
    main()
