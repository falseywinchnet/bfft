"""Deterministic checks for oriented HPWL interval stalks."""

import numpy as np

from .interval_connection import (
    decompose_interval_stalk,
    synthesize_interval_stalk,
)


def main() -> None:
    points = np.asarray(((5.0, 0.0), (5.0, 10.0)))
    weights = np.ones(2)
    diagonal = np.asarray(((1.0, 1.0), (1.0, -1.0)))
    diagonal /= np.linalg.norm(diagonal, axis=1)[:, None]
    even, odd, signs, info = decompose_interval_stalk(
        diagonal, weights, points
    )
    assert np.allclose(even, [np.sqrt(0.5), 0.0])
    assert np.allclose(odd, [0.0, np.sqrt(0.5)])
    assert np.allclose(signs, [[0.0, 1.0], [0.0, -1.0]])
    assert np.allclose(
        synthesize_interval_stalk(even, odd, signs), diagonal
    )
    assert info["paired_axis_count"] == 1
    assert not info["candidate_destinations_materialized"]

    # A fixed opposite face supplies geometry but no relative orientation.
    fixed = np.asarray(((5.0, 20.0),))
    _, one_sided_odd, _, one_sided = decompose_interval_stalk(
        diagonal, weights, points, fixed
    )
    assert np.allclose(one_sided_odd, 0.0)
    assert one_sided["paired_axis_count"] == 0
    print("oriented interval stalk checks: PASS")


if __name__ == "__main__":
    main()
