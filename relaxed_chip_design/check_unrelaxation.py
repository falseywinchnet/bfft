"""Deterministic checks for the standalone transport inverse."""

import numpy as np

from .unrelaxation import (
    conjugate_support_quantiles,
    reference_anchor_phases,
)


def check_identity() -> None:
    active = np.array([[2, 0, 1, -1]], dtype=np.int64)
    measure = np.array([[0.2, 0.2, 0.6, 0.0]], dtype=np.float64)
    result = conjugate_support_quantiles(
        active_segments=active,
        anchor_segments=np.array([1]),
        transported_support=measure,
        reference_support=measure,
        segment_y=np.array([0.0, 0.0, 0.0]),
        segment_x=np.array([0.0, 1.0, 2.0]),
        source_phase=np.array([0.5]),
    )
    np.testing.assert_array_equal(result.target_segments, np.array([1]))


def check_directed_transport() -> None:
    result = conjugate_support_quantiles(
        active_segments=np.array([[0, 1, 2]], dtype=np.int64),
        anchor_segments=np.array([1]),
        reference_support=np.array([[0.6, 0.3, 0.1]]),
        transported_support=np.array([[0.1, 0.2, 0.7]]),
        segment_y=np.zeros(3),
        segment_x=np.arange(3, dtype=np.float64),
        source_phase=np.array([0.5]),
    )
    np.testing.assert_allclose(result.quantiles, np.array([0.75]))
    np.testing.assert_array_equal(result.target_segments, np.array([2]))


def check_rank_prefix_phase() -> None:
    phase = reference_anchor_phases(
        cell_features=np.ones((3, 1)),
        site_features=np.array([[1.0], [2.0], [1.0]]),
        site_segments=np.zeros(3, dtype=np.int64),
        site_x=np.array([0.0, 1.0, 2.0]),
        anchor_segments=np.zeros(3, dtype=np.int64),
        anchor_x=np.array([0.0, 1.0, 1.5]),
        segment_count=1,
    )
    np.testing.assert_allclose(phase, np.array([0.125, 0.5, 0.75]))


def main() -> int:
    check_identity()
    check_directed_transport()
    check_rank_prefix_phase()
    print("relaxed_chip_design unrelaxation checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
