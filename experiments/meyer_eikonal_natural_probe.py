"""Natural-image visual probe for ordinary and Eikonal-flux Meyer splits."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage import data, transform


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from denoiser.continual_eikonal_noise_transport_2d import (  # noqa: E402
    continual_transport_metric,
    directional_noise_witnesses,
)
from experiments.meyer_eikonal_transport_split import (  # noqa: E402
    inverse_sqrt_metric,
    metric_meyer,
)


OUT = ROOT / "experiments" / "out" / "meyer_eikonal_transport_split"


def normalized(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 3:
        value = 0.2126*value[..., 0]+0.7152*value[..., 1]+0.0722*value[..., 2]
    value = transform.resize(value, (128, 128), order=1, anti_aliasing=True,
                             preserve_range=True)
    if np.max(value) <= 1.0:
        value *= 255.0
    return value


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    gx = np.roll(field, -1, 1)-field
    gy = np.roll(field, -1, 0)-field
    return np.hypot(gx, gy)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.ravel(a), np.ravel(b)
    x, y = x-np.mean(x), y-np.mean(y)
    return float(np.vdot(x, y).real/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30))


def ranged(field: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(field)), float(np.max(field))
    return np.clip((field-lo)/max(hi-lo, 1e-30), 0, 1)


def signed(field: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(0.5+0.5*field/max(scale, 1e-30), 0, 1)


def main() -> None:
    sources = {
        "camera": normalized(data.camera()),
        "coins": normalized(data.coins()),
        "page": normalized(data.page()),
        "moon": normalized(data.moon()),
    }
    records = {}
    arrays = {}
    rows = []
    identity = (np.ones((128,128)), np.zeros((128,128)), np.ones((128,128)))
    for name, image in sources.items():
        plan = bfft.MeyerPlan(image.shape, lam=0.05, mu=40.0, passes=96, threads=1)
        ordinary_u, ordinary_v = plan.split_legacy(image)
        _centre, variance, _radius, _ = directional_noise_witnesses(image, ordinary_u)
        metric = continual_transport_metric(ordinary_u, variance)
        matched_u, matched_v, _ = metric_meyer(image, identity)
        eikonal_u, eikonal_v, diagnostic = metric_meyer(
            image, identity, texture_inverse_sqrt=inverse_sqrt_metric(metric))
        edge = gradient_magnitude(ordinary_u)
        records[name] = {
            "ordinary_texture_rms": float(np.sqrt(np.mean(ordinary_v**2))),
            "matched_texture_rms": float(np.sqrt(np.mean(matched_v**2))),
            "eikonal_texture_rms": float(np.sqrt(np.mean(eikonal_v**2))),
            "ordinary_texture_edge_correlation": correlation(np.abs(ordinary_v), edge),
            "matched_texture_edge_correlation": correlation(np.abs(matched_v), edge),
            "eikonal_texture_edge_correlation": correlation(np.abs(eikonal_v), edge),
            "eikonal_vs_matched_cartoon_relative_l2": float(
                np.linalg.norm(eikonal_u-matched_u)/np.linalg.norm(matched_u)),
            "eikonal_vs_matched_texture_relative_l2": float(
                np.linalg.norm(eikonal_v-matched_v)/max(np.linalg.norm(matched_v),1e-30)),
            "metric_condition_p90": float(metric["metric_condition_p90"]),
            "recomposition_linf": diagnostic["recomposition_linf"],
        }
        for label,value in (("source",image),("ordinary_u",ordinary_u),
                            ("ordinary_v",ordinary_v),("matched_u",matched_u),
                            ("matched_v",matched_v),("eikonal_u",eikonal_u),
                            ("eikonal_v",eikonal_v)):
            arrays[f"{name}_{label}"] = value.astype(np.float32)
        rows.append((name,image,ordinary_u,ordinary_v,matched_u,matched_v,eikonal_u,eikonal_v))
    (OUT/"natural_metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    np.savez_compressed(OUT/"natural_probe.npz",**arrays)
    cell,label_width,header=128,80,30
    canvas=Image.new("RGB",(label_width+7*cell,header+len(rows)*cell),"white")
    draw=ImageDraw.Draw(canvas)
    titles=("source","ordinary U","ordinary V","matched U","matched V",
            "Eikonal-flux U","Eikonal-flux V")
    for j,title in enumerate(titles): draw.text((label_width+j*cell+3,8),title,fill="black")
    for i,row in enumerate(rows):
        name,*planes=row; y0=header+i*cell; draw.text((4,y0+8),name,fill="black")
        scale=max(float(np.percentile(np.abs(planes[2]),99.5)),1.0)
        panels=(ranged(planes[0]),ranged(planes[1]),signed(planes[2],scale),
                ranged(planes[3]),signed(planes[4],scale),ranged(planes[5]),
                signed(planes[6],scale))
        for j,panel in enumerate(panels):
            canvas.paste(Image.fromarray(np.uint8(panel*255)).convert("RGB"),
                         (label_width+j*cell,y0))
    canvas.save(OUT/"meyer_eikonal_natural.png")


if __name__ == "__main__": main()
