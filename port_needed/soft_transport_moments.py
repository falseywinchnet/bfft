"""PORT 04: fused two-label Gibbs mass, centroid, covariance and metric.

The reference is a fixed number of reductions over the image.  A native
implementation should perform one parallel pixel pass into per-thread cell
blocks, then reduce those blocks.  It must not materialize pixel×cell data.
"""

from __future__ import annotations

from experiments.wasserstein_allocation_tree import (
    _soft_transport_moments as _reference_soft_transport_moments,
)


def accumulate_soft_moments(*args, **kwargs):
    return _reference_soft_transport_moments(*args, **kwargs)
