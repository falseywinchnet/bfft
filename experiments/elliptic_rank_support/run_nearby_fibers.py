"""Emit the finite nearest-fiber audit for every six-section pencil."""

from __future__ import annotations

from math import gcd

from elliptic_rank_support import (
    RECORD_CONDUCTOR,
    integral_record_points,
    invariants,
    support_pencils,
    verify_fiber,
)


def main() -> None:
    pencils = support_pencils(integral_record_points())
    print(f"pencils={len(pencils)}")
    candidates = []
    for index, pencil in enumerate(pencils):
        for u in (1, 3):
            model = pencil.integral_model(u)
            points = pencil.transported_points(u)
            if model is None or points is None:
                continue
            c4, _, discriminant = invariants(model)
            if abs(discriminant) >= RECORD_CONDUCTOR:
                continue
            candidates.append(
                (abs(discriminant), index, u, gcd(abs(c4), abs(discriminant)), model, points)
            )
            if not verify_fiber(pencil, u):
                raise AssertionError("fiber certificate failed")
    for item in sorted(candidates):
        print(item)
    print(f"sub-record-discriminant fibers={len(candidates)}")


if __name__ == "__main__":
    main()
