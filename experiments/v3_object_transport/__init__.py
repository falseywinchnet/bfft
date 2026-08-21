"""V3-native object recomposition research.

The package begins after segmentation.  Its first artifact is a read-only
region complex: immutable V3 compound regions, literal interfaces, and
separate structural/content coordinates.  No object decision is made here.
"""

from .region_complex import build_region_complex, summarize_region_complex
from .incidence_bundle import build_incidence_bundle, summarize_incidence_bundle

__all__ = (
    "build_incidence_bundle",
    "build_region_complex",
    "summarize_incidence_bundle",
    "summarize_region_complex",
)
