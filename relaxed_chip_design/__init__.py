"""Reusable primitives for transport-based relaxed circuit design."""

from .unrelaxation import (
    QuantileUnrelaxation,
    conjugate_support_quantiles,
    reference_anchor_phases,
)
from .basin_walk import (
    FarFieldFit,
    fit_chart_displacement,
    spherical_chart_walk,
)
from .preimage import (
    TransportSecant,
    graft_transported_rows,
    project_transport_residual,
    pullback_residual,
    requires_row_self_distillation,
)
from .vector_diffusion import diffuse_connection_orientations

__all__ = [
    "QuantileUnrelaxation",
    "conjugate_support_quantiles",
    "reference_anchor_phases",
    "FarFieldFit",
    "fit_chart_displacement",
    "spherical_chart_walk",
    "TransportSecant",
    "graft_transported_rows",
    "project_transport_residual",
    "pullback_residual",
    "requires_row_self_distillation",
    "diffuse_connection_orientations",
]
