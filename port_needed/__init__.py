"""Executable Python contracts for the remaining native vision ports.

Each module is deliberately one algorithmic boundary.  The viewer imports
this package rather than reaching into research scripts, so a C++ kernel can
replace one reference at a time without changing the model.
"""

__all__ = [
    "PreparedSegmentingTarget",
    "SegmentingConfig",
    "build_segmenting_representation",
]


def __getattr__(name):
    """Keep leaf algorithm imports independent of the full viewer pipeline."""

    if name in __all__:
        from .pipeline import (
            PreparedSegmentingTarget,
            SegmentingConfig,
            build_segmenting_representation,
        )
        return {
            "PreparedSegmentingTarget": PreparedSegmentingTarget,
            "SegmentingConfig": SegmentingConfig,
            "build_segmenting_representation": build_segmenting_representation,
        }[name]
    raise AttributeError(name)
