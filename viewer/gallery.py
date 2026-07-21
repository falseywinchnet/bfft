"""Test images for the Meyer decomposition demos.

Three groups:

  segregation  fields built from two or more real textures at matched
               first and second order statistics, so mean and contrast
               carry no information about the boundary and only scale
               geometry can find it.  These are the honest test of a
               cartoon + texture split: any method can separate a bright
               region from a dark one.

  texture      single natural textures, and photographs whose interest is
               oscillatory content at several scales at once.

  colour       colour photographs, for the per-channel decomposition.

Every loader returns float64: grey images in [0, 255], colour images as
(H, W, 3) in [0, 1] -- which is what bfft.effects expects.  Images come
from skimage.data (bundled, no network) plus synthetic constructions
here; a loader that cannot find its source raises and is skipped by
:func:`available`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ENTRIES", "GROUPS", "available", "load", "describe"]


def _grey(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3] @ [0.2125, 0.7154, 0.0721]
    if a.max() <= 1.5:
        a = a * 255.0
    return a


def _colour(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    return a / 255.0 if a.max() > 1.5 else a


def _sk(name):
    from skimage import data
    return getattr(data, name)()


def _match_stats(a, mean=128.0, std=45.0):
    """Force a texture to a given mean and standard deviation.

    This is what makes a segregation field honest: after matching, no
    pointwise statistic distinguishes the two sides."""
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return mean + (a - a.mean()) * (std / (s if s > 1e-9 else 1.0))


def _compose_matched(parts, masks, mean=128.0, std=45.0):
    """Assemble regions from different textures, matching each region's own
    statistics after assembly.

    Matching the textures globally and then cutting them up leaves the
    regions a few units apart, since each region samples its texture
    unevenly; matching what actually lands in each region makes the first
    and second order statistics identical by construction."""
    out = np.zeros_like(parts[0], dtype=np.float64)
    for a, m in zip(parts, masks):
        sel = m.astype(bool)
        vals = np.asarray(a, dtype=np.float64)[sel]
        s = vals.std()
        out[sel] = mean + (vals - vals.mean()) * (std / (s if s > 1e-9 else 1.0))
    return np.clip(out, 0, 255)


def _tile(a, h, w):
    """Mirror-tile a texture up to (h, w)."""
    ah, aw = a.shape
    ry, rx = int(np.ceil(h / ah)), int(np.ceil(w / aw))
    big = np.block([[a if (i + j) % 2 == 0 else a[::-1, ::-1]
                     for j in range(rx)] for i in range(ry)])
    return big[:h, :w]


# --- synthetic segregation fields ---------------------------------------

def _labels_of(masks):
    """Region masks -> an integer label map, for scoring a segmentation."""
    lab = np.zeros(masks[0].shape, dtype=np.int32)
    for i, m in enumerate(masks):
        lab[np.asarray(m, dtype=bool)] = i
    return lab


def _seg_curve(n=512):
    """Two textures split by a curved boundary, matched region by region."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    edge = n * 0.5 + n * 0.16 * np.sin(2 * np.pi * y / n * 1.5)
    m = x < edge
    masks = [m, ~m]
    return _compose_matched([_tile(_sk("brick"), n, n),
                             _tile(_sk("grass"), n, n)], masks), \
        _labels_of(masks)


def _seg_quads(n=512):
    """Four matched textures in quadrants: brick, grass, gravel, weave."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = n // 2
    weave = np.cos(2 * np.pi * (x + y) / 9)
    masks = [(y < h) & (x < h), (y < h) & (x >= h),
             (y >= h) & (x < h), (y >= h) & (x >= h)]
    return _compose_matched([_tile(_sk("brick"), n, n),
                             _tile(_sk("grass"), n, n),
                             _tile(_sk("gravel"), n, n), weave], masks), \
        _labels_of(masks)


def _seg_frequency(n=512):
    """One texture, two scales.  Both halves are the same weave; the right
    half runs at half the period.  Nothing but scale separates them."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    lo = np.cos(2 * np.pi * (x + y) / 12) + np.cos(2 * np.pi * (x - y) / 12)
    hi = np.cos(2 * np.pi * (x + y) / 6) + np.cos(2 * np.pi * (x - y) / 6)
    masks = [x < n / 2, x >= n / 2]
    return _compose_matched([lo, hi], masks), _labels_of(masks)


def _seg_orientation(n=512):
    """One texture, two orientations: the same band content, rotated 90
    degrees inside a disc."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    a = np.cos(2 * np.pi * x / 8) + 0.6 * np.cos(2 * np.pi * x / 13)
    b = np.cos(2 * np.pi * y / 8) + 0.6 * np.cos(2 * np.pi * y / 13)
    m = np.hypot(x - n / 2, y - n / 2) < n * 0.28
    masks = [m, ~m]
    return _compose_matched([a, b], masks), _labels_of(masks)


def _seg_contrast(n=512):
    """A texture at DECREASING contrast over a smooth illumination ramp:
    the case where a threshold on local variance fails and a scale split
    should not."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    amp = np.linspace(35.0, 3.0, n)[None, :]
    ramp = 60 + 120 * (y / n) * (1 - 0.4 * np.cos(2 * np.pi * x / n))
    return np.clip(ramp + amp * np.cos(2 * np.pi * (x + y) / 7), 0, 255)


def _rig():
    """The synthetic cartoon + two-texture rig with ground truth."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                          "experiments"))
    from meyer_bregman import rig_scene
    return np.asarray(rig_scene(256)[0], dtype=np.float64)


# --- the catalogue -------------------------------------------------------
#
# (key, group, label, loader, kind) with kind in {"grey", "colour"}

ENTRIES = [
    ("seg_curve", "segregation", "brick | grass, curved boundary, matched",
     lambda: _grey(_seg_curve()[0]), "grey"),
    ("seg_quads", "segregation", "four textures, matched quadrants",
     lambda: _grey(_seg_quads()[0]), "grey"),
    ("seg_freq", "segregation", "one weave, two periods (scale only)",
     lambda: _grey(_seg_frequency()[0]), "grey"),
    ("seg_orient", "segregation", "one texture, two orientations",
     lambda: _grey(_seg_orientation()[0]), "grey"),
    ("seg_contrast", "segregation", "texture fading over an illumination ramp",
     lambda: _grey(_seg_contrast()), "grey"),
    ("rig", "segregation", "synthetic cartoon + two-texture rig",
     lambda: _grey(_rig()), "grey"),

    ("brick", "texture", "brick", lambda: _grey(_sk("brick")), "grey"),
    ("grass", "texture", "grass", lambda: _grey(_sk("grass")), "grey"),
    ("gravel", "texture", "gravel", lambda: _grey(_sk("gravel")), "grey"),
    ("checker", "texture", "checkerboard",
     lambda: _grey(_sk("checkerboard")), "grey"),
    ("text", "texture", "handwriting on paper",
     lambda: _grey(_sk("text")), "grey"),
    ("page", "texture", "printed page", lambda: _grey(_sk("page")), "grey"),
    ("coins", "texture", "coins on a textured ground",
     lambda: _grey(_sk("coins")), "grey"),
    ("moon", "texture", "moon", lambda: _grey(_sk("moon")), "grey"),
    ("camera", "texture", "cameraman", lambda: _grey(_sk("camera")), "grey"),
    ("cell", "texture", "cell", lambda: _grey(_sk("cell")), "grey"),
    ("clock", "texture", "clock", lambda: _grey(_sk("clock")), "grey"),

    ("astronaut", "colour", "astronaut",
     lambda: _colour(_sk("astronaut")), "colour"),
    ("chelsea", "colour", "cat (fur at several scales)",
     lambda: _colour(_sk("chelsea")), "colour"),
    ("coffee", "colour", "coffee", lambda: _colour(_sk("coffee")), "colour"),
    ("ihc", "colour", "immunohistochemistry",
     lambda: _colour(_sk("immunohistochemistry")), "colour"),
    ("retina", "colour", "retina",
     lambda: _colour(_sk("retina"))[::2, ::2], "colour"),
    ("hubble", "colour", "hubble deep field",
     lambda: _colour(_sk("hubble_deep_field")), "colour"),
]

GROUPS = ("segregation", "texture", "colour")

# The synthetic segregation fields come with ground truth: an integer label
# map over the regions, so a segmentation from the decomposition can be
# scored rather than eyeballed.
TRUTH = {
    "seg_curve": _seg_curve,
    "seg_quads": _seg_quads,
    "seg_freq": _seg_frequency,
    "seg_orient": _seg_orientation,
}


def load_truth(key):
    """(image in [0, 255], integer label map) for a scored entry."""
    img, lab = TRUTH[key]()
    return _grey(img), lab

_BY_KEY = {e[0]: e for e in ENTRIES}


def describe(key):
    e = _BY_KEY[key]
    return {"key": e[0], "group": e[1], "label": e[2], "kind": e[4]}


def load(key):
    """Load one entry.  Grey entries return (H, W) in [0, 255]; colour
    entries return (H, W, 3) in [0, 1]."""
    return _BY_KEY[key][3]()


def available(group=None):
    """Keys that actually load on this machine, optionally in one group."""
    out = []
    for key, grp, _, fn, _kind in ENTRIES:
        if group is not None and grp != group:
            continue
        try:
            fn()
        except Exception:
            continue
        out.append(key)
    return out


def labels(keys=None):
    """UI labels, group-prefixed, in catalogue order."""
    keys = set(keys) if keys is not None else None
    return [f"[{e[1][:3]}] {e[2]}" for e in ENTRIES
            if keys is None or e[0] in keys]


def key_for_label(label):
    for e in ENTRIES:
        if f"[{e[1][:3]}] {e[2]}" == label:
            return e[0]
    raise KeyError(label)
