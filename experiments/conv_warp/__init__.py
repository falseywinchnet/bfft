"""Direct geometric warping experiments for CONV/CONV*."""

from .geometry import (
    ProjectiveMap,
    affine_about_center,
    perspective_about_center,
    pullback_spd,
)
from .operators import (
    direct_warp,
    positive_warped_basin_average,
    source_validity_mask,
)
from .warped_pixels import (
    certified_projective_warped_pixel_average,
    exact_affine_warped_pixel_average,
)
from .synthetic import FourierField

__all__ = [
    "FourierField",
    "ProjectiveMap",
    "affine_about_center",
    "perspective_about_center",
    "pullback_spd",
    "direct_warp",
    "positive_warped_basin_average",
    "exact_affine_warped_pixel_average",
    "certified_projective_warped_pixel_average",
    "source_validity_mask",
]
