"""Forensic PDF representation optimizer.

The package root is deliberately lazy: representation studies that operate on
already-extracted images do not require the PDF object dependency.
"""

from importlib import import_module

__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "SignedPdfError",
    "analyze_pdf",
    "optimize_pdf",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(".core", __name__), name)
