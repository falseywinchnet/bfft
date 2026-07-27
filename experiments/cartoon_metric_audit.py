#!/usr/bin/env python3
"""Auditing the 40x move in `cartoon_mse`.

`experiments/cartoon_stage_isotropy_cost.py` produced one cell that does not
behave like a measurement: Chelsea under an 8-pass anisotropic cartoon scored
`cartoon_mse` 3.0e-6 against 1.2e-4 everywhere else, while its PSNR and
`texture_mse` were indistinguishable from the control.

The suspect is the metric, not the reconstruction.  `native_components` does

    scale[i] = 255 / (plane.max() - plane.min())
    work     = (plane - plane.min()) * scale[i]
    split    = meyer_split(work)
    return     split.cartoon / scale[i]

and it is applied *separately* to the target and to the reconstruction.  Two
consequences, both bad for a comparison:

1. `scale` is an extreme-value statistic.  One clipped pixel moves it, and it
   is not a property of the content.
2. The target and the reconstruction are therefore stretched by **different**
   affine maps before being decomposed, so the two cartoons are produced by
   effectively different lambda, and dividing each by its own scale afterwards
   does not undo that.

This file measures the metric three ways on the same reconstructions:

* **as shipped** -- each divided by its own scale;
* **shared scale** -- both divided by the target's scale, which at least puts
  them in one unit;
* **commensurable** -- the reconstruction stretched by the *target's* affine
  map before being decomposed, which is the only version where the two
  decompositions are the same operator applied to two inputs.

    PYTHONPATH=.:viewer .venv/bin/python experiments/cartoon_metric_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import _to_working, lab_to_srgb  # noqa: E402
from transport_voronoi import Config  # noqa: E402

from cartoon_stage_isotropy_cost import run_arm  # noqa: E402


def split_with(image, cfg, offset=None, scale=None):
    """Decompose, optionally forcing a given affine pre-stretch."""
    planes, _names, _carried, _space = _to_working(image, "oklab_lc")
    k = planes.shape[2]
    own_offset = np.empty(k)
    own_scale = np.empty(k)
    for i in range(k):
        p = planes[..., i]
        low = float(p.min())
        span = float(p.max()) - low
        own_offset[i] = low
        own_scale[i] = 255.0 / span if span > 1e-12 else 1.0
    use_offset = own_offset if offset is None else offset
    use_scale = own_scale if scale is None else scale
    cartoon = np.empty_like(planes)
    texture = np.empty_like(planes)
    for i in range(k):
        work = (planes[..., i] - use_offset[i]) * use_scale[i]
        u, v = bfft.meyer_split(work, lam=cfg.lam, mu=cfg.mu,
                                passes=cfg.passes, threads=4)
        cartoon[..., i] = u
        texture[..., i] = v
    return cartoon, texture, own_offset, own_scale, use_scale


def audit(key, cfg):
    print(f"\n=== {key} ===")
    image = gallery.load(key)
    arms = (("isotropic, periodic (shipped)", None, ""),
            ("anisotropic, 24 passes", 24, "anisotropic"),
            ("anisotropic, 8 passes", 8, "anisotropic"))
    target_cartoon, _tt, t_offset, t_scale, _ = split_with(
        gallery.load(key), cfg)
    print(f"  target scale per channel: "
          f"{np.array2string(t_scale, precision=4)}")

    for label, passes, kind in arms:
        _record, model = run_arm(image, cfg, label, passes, kind)
        reconstruction = np.clip(
            lab_to_srgb(model.reconstruction), 0.0, 1.0)
        # normalise the target against the model's own resized rgb
        tgt_c, _tv, _to, tgt_scale, _ = split_with(model.rgb, cfg)
        rec_c, _rv, _ro, rec_scale, _ = split_with(reconstruction, cfg)
        shipped = float(np.mean(
            (tgt_c / tgt_scale - rec_c / rec_scale) ** 2))
        shared = float(np.mean(
            ((tgt_c - rec_c) / tgt_scale) ** 2))
        com_c, _cv, _co, _cs, _u = split_with(
            reconstruction, cfg, offset=_to, scale=tgt_scale)
        commensurable = float(np.mean(
            ((tgt_c - com_c) / tgt_scale) ** 2))
        drift = float(np.max(np.abs(rec_scale / tgt_scale - 1.0)))
        print(f"  {label:30s} scale drift {drift * 100:6.2f}%   "
              f"shipped {shipped:.3e}   shared {shared:.3e}   "
              f"commensurable {commensurable:.3e}")


def main():
    cfg = Config(max_side=128, initial_cells=120, max_cells=700,
                 split_batch=24, allocation_mode="Expected affine gain",
                 recursive_memory_stages=1, residual_memory_weight=0.0,
                 composition_discrepancy_weight=0.0)
    for key in ("chelsea", "pikachu"):
        audit(key, cfg)


if __name__ == "__main__":
    main()
