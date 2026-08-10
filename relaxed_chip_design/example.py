"""Small synthetic example of CDF-conjugate unrelaxation."""

import numpy as np

from .unrelaxation import conjugate_support_quantiles


def main() -> int:
    result = conjugate_support_quantiles(
        active_segments=np.array([[0, 1, 2]], dtype=np.int64),
        anchor_segments=np.array([1]),
        reference_support=np.array([[0.6, 0.3, 0.1]]),
        transported_support=np.array([[0.1, 0.2, 0.7]]),
        segment_y=np.zeros(3),
        segment_x=np.arange(3, dtype=np.float64),
        source_phase=np.array([0.5]),
    )
    print("source quantile:", result.quantiles[0])
    print("transport-selected segment:", result.target_segments[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
