"""Shape Meyer texture flux with a transport state raised directly from CONV."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from denoiser.dcnt import transport_uncertainty  # noqa: E402
from denoiser.continual_eikonal_noise_transport_2d import (  # noqa: E402
    continual_transport_metric,
    directional_noise_witnesses,
)
from experiments.meyer_eikonal_transport_split import (  # noqa: E402
    inverse_sqrt_metric,
    metric_meyer,
    relative_mse,
    scenes,
)


OUT = ROOT / "experiments" / "out" / "meyer_conv_transport_split"


def _forward_gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(field, dtype=np.float64)
    gy = np.zeros_like(value)
    gx = np.zeros_like(value)
    gy[:-1] = value[1:]-value[:-1]
    gx[:, :-1] = value[:, 1:]-value[:, :-1]
    return gy, gx


def _positive_tensor(
    xx: np.ndarray, xy: np.ndarray, yy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = xx+yy
    gap = np.hypot(xx-yy, 2.0*xy)
    high = np.maximum(0.5*(trace+gap), 0.0)
    low = np.maximum(0.5*(trace-gap), 0.0)
    angle = 0.5*np.arctan2(2.0*xy, xx-yy)
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        high*cosine*cosine+low*sine*sine,
        (high-low)*cosine*sine,
        high*sine*sine+low*cosine*cosine,
    )


def conv_raised_transport_metric(
    image: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Raise CONV target-excluded current agreement to one SPD metric.

    Each target has three predictions from parity charts which did not sample
    that target.  Their mean first current is the transported structural
    witness.  Their full 2x2 current covariance is uncertainty about that
    witness.  Only the PSD excess of mean-current action over covariance is
    admitted as geometry.
    """

    source = np.asarray(image, dtype=np.float64)
    law = transport_uncertainty(source)
    finite = np.isfinite(law.family)
    completed = np.where(finite, law.family, law.centre[None, ...])
    currents_y = np.empty_like(completed)
    currents_x = np.empty_like(completed)
    for chart in range(completed.shape[0]):
        currents_y[chart], currents_x[chart] = _forward_gradient(completed[chart])
    mean_y = np.mean(currents_y, axis=0)
    mean_x = np.mean(currents_x, axis=0)
    centred_y = currents_y-mean_y
    centred_x = currents_x-mean_x
    covariance_yy = np.mean(centred_y*centred_y, axis=0)
    covariance_xy = np.mean(centred_x*centred_y, axis=0)
    covariance_xx = np.mean(centred_x*centred_x, axis=0)
    excess_xx, excess_xy, excess_yy = _positive_tensor(
        mean_x*mean_x-covariance_xx,
        mean_x*mean_y-covariance_xy,
        mean_y*mean_y-covariance_yy,
    )
    magnitude = max(float(np.max(np.abs(source))), float(np.ptp(source)), 1.0)
    numerical = np.finfo(float).eps*magnitude*magnitude
    persistent_action = mean_x*mean_x+mean_y*mean_y
    uncertainty_action = covariance_xx+covariance_yy
    excess_trace = excess_xx+excess_yy
    excess_gap = np.hypot(excess_xx-excess_yy, 2.0*excess_xy)
    # Two independent dimensionless witnesses are required: excess over
    # transport uncertainty, and directional shape within that excess.
    coherence = np.divide(
        excess_trace,
        persistent_action+uncertainty_action+numerical,
        out=np.zeros_like(excess_trace),
        where=(persistent_action+uncertainty_action)>0.0,
    )
    directional = np.divide(
        excess_gap,
        excess_trace+numerical,
        out=np.zeros_like(excess_trace),
        where=excess_trace>0.0,
    )
    strength = np.clip(coherence,0.0,1.0)*np.clip(directional,0.0,1.0)
    angle = 0.5*np.arctan2(2.0*excess_xy,excess_xx-excess_yy)
    cosine,sine=np.cos(angle),np.sin(angle)
    high,low=np.exp(strength),np.exp(-strength)
    mxx=high*cosine*cosine+low*sine*sine
    mxy=(high-low)*cosine*sine
    myy=high*sine*sine+low*cosine*cosine
    determinant = mxx*myy-mxy*mxy
    gap = np.hypot(mxx-myy, 2.0*mxy)
    condition = (mxx+myy+gap)/np.maximum(mxx+myy-gap, numerical)
    if np.any(mxx <= 0.0) or np.any(determinant <= 0.0):
        raise RuntimeError("CONV-raised metric left the SPD cone")
    return {
        "metric_xx": np.ascontiguousarray(mxx),
        "metric_xy": np.ascontiguousarray(mxy),
        "metric_yy": np.ascontiguousarray(myy),
        "metric_determinant_minimum": float(np.min(determinant)),
        "metric_condition_p90": float(np.percentile(condition, 90.0)),
        "metric_condition_maximum": float(np.max(condition)),
        "mean_current_action": float(np.mean(mean_x*mean_x+mean_y*mean_y)),
        "mean_current_uncertainty": float(np.mean(
            covariance_xx+covariance_yy)),
        "mean_admitted_excess": float(np.mean(excess_xx+excess_yy)),
        "mean_intrinsic_strength": float(np.mean(strength)),
        "maximum_intrinsic_strength": float(np.max(strength)),
        "chart_recomposition_count_minimum": int(np.min(np.sum(finite, axis=0))),
        "chart_recomposition_count_maximum": int(np.max(np.sum(finite, axis=0))),
    }


def ranged(array: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(array)), float(np.max(array))
    return np.clip((array-lo)/max(hi-lo, 1e-30), 0, 1)


def signed(array: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(0.5+0.5*array/max(scale, 1e-30), 0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = scenes()
    identity = (np.ones((128,128)), np.zeros((128,128)), np.ones((128,128)))
    records: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for name,(image,cartoon_truth,texture_truth) in cases.items():
        plan=bfft.MeyerPlan(image.shape,lam=.05,mu=40.0,passes=96,threads=1)
        ordinary_u,ordinary_v=plan.split_legacy(image)
        _centre,variance,_radius,_=directional_noise_witnesses(image,ordinary_u)
        meyer_metric=continual_transport_metric(ordinary_u,variance)
        conv_metric=conv_raised_transport_metric(image)
        matched_u,matched_v,matched_diag=metric_meyer(image,identity)
        meyer_u,meyer_v,meyer_diag=metric_meyer(
            image,identity,texture_inverse_sqrt=inverse_sqrt_metric(meyer_metric))
        conv_u,conv_v,conv_diag=metric_meyer(
            image,identity,texture_inverse_sqrt=inverse_sqrt_metric(conv_metric))
        records[name]={
            "ordinary_cartoon_relative_mse":relative_mse(ordinary_u,cartoon_truth),
            "ordinary_texture_relative_mse":relative_mse(ordinary_v,texture_truth),
            "matched_cartoon_relative_mse":relative_mse(matched_u,cartoon_truth),
            "matched_texture_relative_mse":relative_mse(matched_v,texture_truth),
            "meyer_metric_cartoon_relative_mse":relative_mse(meyer_u,cartoon_truth),
            "meyer_metric_texture_relative_mse":relative_mse(meyer_v,texture_truth),
            "conv_metric_cartoon_relative_mse":relative_mse(conv_u,cartoon_truth),
            "conv_metric_texture_relative_mse":relative_mse(conv_v,texture_truth),
            "conv_metric":{key:value for key,value in conv_metric.items()
                           if np.isscalar(value)},
            "matched_diagnostic":matched_diag,
            "meyer_metric_diagnostic":meyer_diag,
            "conv_metric_diagnostic":conv_diag,
        }
        key=name.replace(" ","_")
        for label,value in (
            ("source",image),("cartoon_truth",cartoon_truth),("texture_truth",texture_truth),
            ("ordinary_u",ordinary_u),("ordinary_v",ordinary_v),
            ("matched_u",matched_u),("matched_v",matched_v),
            ("meyer_metric_u",meyer_u),("meyer_metric_v",meyer_v),
            ("conv_metric_u",conv_u),("conv_metric_v",conv_v),
        ): arrays[f"{key}_{label}"]=value.astype(np.float32)
        rows.append((name,image,cartoon_truth,texture_truth,ordinary_u,ordinary_v,
                     matched_u,matched_v,meyer_u,meyer_v,conv_u,conv_v))
    (OUT/"metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    np.savez_compressed(OUT/"meyer_conv_transport_split.npz",**arrays)
    cell,label_width,header=128,135,30
    canvas=Image.new("RGB",(label_width+11*cell,header+len(rows)*cell),"white")
    draw=ImageDraw.Draw(canvas)
    titles=("source","truth U","truth V","ordinary U","ordinary V",
            "matched U","matched V","Meyer-metric U","Meyer-metric V",
            "CONV-state U","CONV-state V")
    for j,title in enumerate(titles):draw.text((label_width+j*cell+3,8),title,fill="black")
    for i,row in enumerate(rows):
        name,*planes=row;y0=header+i*cell;draw.text((4,y0+8),name,fill="black")
        scale=max(float(np.max(np.abs(planes[2]))),1.0)
        panels=[]
        for j,plane in enumerate(planes):
            panels.append(ranged(plane) if j in (0,1,3,5,7,9) else signed(plane,scale))
        for j,panel in enumerate(panels):
            canvas.paste(Image.fromarray(np.uint8(panel*255)).convert("RGB"),
                         (label_width+j*cell,y0))
    canvas.save(OUT/"meyer_conv_transport_comparison.png")


if __name__=="__main__":main()
