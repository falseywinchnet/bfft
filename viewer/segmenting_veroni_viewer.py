#!/usr/bin/env python3
"""Canonical BFFT image-segmentation viewer.

This deliberately preserves the requested ``veroni`` filename.  The former
canopy laboratory remains directly runnable as
``viewer/transport_measure_app.py``.

Run from the repository root:

    python viewer/segmenting_veroni_viewer.py
"""

from __future__ import annotations

from segmenting_veroni_app import main


if __name__ == "__main__":
    raise SystemExit(main())
