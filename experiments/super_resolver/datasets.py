"""Dataset facade shared with the V3 segmenter application.

The catalogue deliberately remains owned by :mod:`viewer.gallery`; this
module only gives the super-resolution lab a small, stable interface to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from viewer import gallery


def available_entries() -> list[dict]:
    """Return loadable V3 gallery entries in the catalogue's UI order."""
    keys = set(gallery.available())
    return [gallery.describe(key) for key, *_rest in gallery.ENTRIES if key in keys]


def label_for(entry: dict) -> str:
    return f"[{entry['group'][:3]}] {entry['label']}"


def key_for_label(label: str) -> str:
    return gallery.key_for_label(label)


def load_gallery(key: str) -> np.ndarray:
    return np.asarray(gallery.load(key))


def load_file(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as image:
        return np.asarray(image.convert("RGB"))
