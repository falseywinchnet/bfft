"""Single-observation Eikonal super-resolution laboratory."""

from .core import (
    PreparedObservation,
    SuperResolutionConfig,
    SuperResolutionResult,
    prepare_observation,
    run_eikonal_upscale,
)

__all__ = [
    "PreparedObservation",
    "SuperResolutionConfig",
    "SuperResolutionResult",
    "prepare_observation",
    "run_eikonal_upscale",
]
