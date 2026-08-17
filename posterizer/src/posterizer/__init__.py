"""Perceptual posterization descended from converter v2."""

from .core import PosterizerConfig, PosterizerResult, posterize_array, posterize_image

__all__ = [
    "PosterizerConfig",
    "PosterizerResult",
    "posterize_array",
    "posterize_image",
]
