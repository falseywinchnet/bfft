"""Executable Python contracts for the remaining native vision ports.

Each module is deliberately one algorithmic boundary.  The viewer imports
this package rather than reaching into research scripts, so a C++ kernel can
replace one reference at a time without changing the model.
"""

from .pipeline import SegmentingConfig, build_segmenting_representation

__all__ = ["SegmentingConfig", "build_segmenting_representation"]
