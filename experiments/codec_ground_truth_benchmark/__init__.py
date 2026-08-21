"""Synthetic, reference-grounded JPEG/PNG optimizer benchmark."""

from .evaluate import evaluate_suite
from .generate import generate_suite

__all__ = ["evaluate_suite", "generate_suite"]
