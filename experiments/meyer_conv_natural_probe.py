"""Natural-image probe of Meyer texture flux shaped by CONV transport state."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage import data


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import bfft  # noqa:E402
from denoiser.continual_eikonal_noise_transport_2d import (  # noqa:E402
    continual_transport_metric,directional_noise_witnesses)
from experiments.meyer_conv_transport_split import (  # noqa:E402
    conv_raised_transport_metric,ranged,signed)
from experiments.meyer_eikonal_natural_probe import (  # noqa:E402
    correlation,gradient_magnitude,normalized)
from experiments.meyer_eikonal_transport_split import (  # noqa:E402
    inverse_sqrt_metric,metric_meyer)


OUT=ROOT/"experiments"/"out"/"meyer_conv_transport_split"


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    sources={"camera":normalized(data.camera()),"coins":normalized(data.coins()),
             "page":normalized(data.page()),"moon":normalized(data.moon())}
    identity=(np.ones((128,128)),np.zeros((128,128)),np.ones((128,128)))
    records={};arrays={};rows=[]
    for name,image in sources.items():
        plan=bfft.MeyerPlan(image.shape,lam=.05,mu=40,passes=96,threads=1)
        ordinary_u,ordinary_v=plan.split_legacy(image)
        _centre,variance,_radius,_=directional_noise_witnesses(image,ordinary_u)
        meyer_metric=continual_transport_metric(ordinary_u,variance)
        conv_metric=conv_raised_transport_metric(image)
        matched_u,matched_v,_=metric_meyer(image,identity)
        meyer_u,meyer_v,_=metric_meyer(
            image,identity,texture_inverse_sqrt=inverse_sqrt_metric(meyer_metric))
        conv_u,conv_v,conv_diag=metric_meyer(
            image,identity,texture_inverse_sqrt=inverse_sqrt_metric(conv_metric))
        edge=gradient_magnitude(ordinary_u)
        record={}
        for label,u,v in (("ordinary",ordinary_u,ordinary_v),
                          ("matched",matched_u,matched_v),
                          ("meyer_metric",meyer_u,meyer_v),
                          ("conv_metric",conv_u,conv_v)):
            record[f"{label}_texture_rms"]=float(np.sqrt(np.mean(v*v)))
            record[f"{label}_texture_edge_correlation"]=correlation(np.abs(v),edge)
            record[f"{label}_cartoon_tv"]=float(np.sum(gradient_magnitude(u)))
        record.update({
            "conv_vs_matched_cartoon_relative_l2":float(
                np.linalg.norm(conv_u-matched_u)/np.linalg.norm(matched_u)),
            "conv_vs_matched_texture_relative_l2":float(
                np.linalg.norm(conv_v-matched_v)/max(np.linalg.norm(matched_v),1e-30)),
            "conv_metric_condition_p90":conv_metric["metric_condition_p90"],
            "conv_metric_condition_maximum":conv_metric["metric_condition_maximum"],
            "conv_recomposition_linf":conv_diag["recomposition_linf"],
        })
        records[name]=record
        for label,value in (("source",image),("ordinary_u",ordinary_u),
                            ("ordinary_v",ordinary_v),("matched_u",matched_u),
                            ("matched_v",matched_v),("meyer_u",meyer_u),
                            ("meyer_v",meyer_v),("conv_u",conv_u),("conv_v",conv_v)):
            arrays[f"{name}_{label}"]=value.astype(np.float32)
        rows.append((name,image,ordinary_u,ordinary_v,matched_u,matched_v,
                     meyer_u,meyer_v,conv_u,conv_v))
    (OUT/"natural_metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    np.savez_compressed(OUT/"natural_probe.npz",**arrays)
    cell,label_width,header=128,80,30
    canvas=Image.new("RGB",(label_width+9*cell,header+len(rows)*cell),"white")
    draw=ImageDraw.Draw(canvas)
    titles=("source","ordinary U","ordinary V","matched U","matched V",
            "Meyer-metric U","Meyer-metric V","CONV-state U","CONV-state V")
    for j,title in enumerate(titles):draw.text((label_width+j*cell+3,8),title,fill="black")
    for i,row in enumerate(rows):
        name,*planes=row;y0=header+i*cell;draw.text((4,y0+8),name,fill="black")
        scale=max(float(np.percentile(np.abs(planes[2]),99.5)),1.0)
        panels=[]
        for j,plane in enumerate(planes):
            panels.append(ranged(plane) if j in (0,1,3,5,7) else signed(plane,scale))
        for j,panel in enumerate(panels):
            canvas.paste(Image.fromarray(np.uint8(panel*255)).convert("RGB"),
                         (label_width+j*cell,y0))
    canvas.save(OUT/"meyer_conv_natural.png")


if __name__=="__main__":main()
