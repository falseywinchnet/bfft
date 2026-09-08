"""Render the current direct-warp evidence without resampled ground truth."""

from __future__ import annotations

import argparse,json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS=("joint_current","conv","lanczos3","bilinear")
DISPLAY={"joint_current":"CONV","joint_conv":"CONV",
         "conv":"factor-only ablation","lanczos3":"Lanczos-3","bilinear":"bilinear"}
COLORS={"joint_current":"#0072B2","joint_conv":"#0072B2","conv":"#009E73",
        "lanczos3":"#D55E00","bilinear":"#777777"}


def _mean(records: list[dict],method: str,metric: str)->float:
    return float(np.mean([r["metrics"][method][metric] for r in records]))


def render(root: Path,out: Path)->None:
    sizes=(9,13,17,25)
    curved=[json.loads((root/f"conv_warp_curved_canonical_{size}.json").read_text())
            for size in sizes]
    sharp=json.loads((root/"conv_warp_projective_range_compiled_17_13.json").read_text())
    smooth=json.loads((root/"conv_warp_canonical_range_25_17.json").read_text())
    figure,axes=plt.subplots(2,2,figsize=(10.8,7.7),constrained_layout=True)
    floor=2e-16
    for method in METHODS:
        wrong=[max(floor,_mean(item["records"],method,"normal_wrong_way_variation"))
               for item in curved]
        excursion=[max(floor,_mean(item["records"],method,"normal_range_excursion"))
                   for item in curved]
        axes[0,0].plot(sizes,wrong,"o-",label=DISPLAY[method],color=COLORS[method])
        axes[0,1].plot(sizes,excursion,"o-",label=DISPLAY[method],color=COLORS[method])
    for axis,title,ylabel in (
        (axes[0,0],"Curved interfaces: intrinsic normal reversal","mean wrong-way variation"),
        (axes[0,1],"Curved interfaces: admitted amplitude","mean range excursion"),
    ):
        axis.set_yscale("log");axis.set_xticks(sizes);axis.set_xlabel("source side")
        axis.set_ylabel(ylabel);axis.set_title(title);axis.grid(True,which="both",alpha=.22)
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2)
    for axis in axes[0]:
        axis.text(.02,.97,"values below $2\\times10^{-16}$ shown at floor",
                  transform=axis.transAxes,fontsize=7,color="#555555",va="top",
                  bbox={"facecolor":"white","alpha":.78,"edgecolor":"none","pad":1})
    labels=[DISPLAY[m] for m in ("joint_conv","conv","lanczos3","bilinear")]
    x=np.arange(4);width=.36
    for axis,data,title in (
        (axes[1,0],sharp,"Sharp affine/projective corpus"),
        (axes[1,1],smooth,"Band-admitted Fourier corpus"),
    ):
        point=[np.mean([r["point"][m]["mse"] for r in data["records"]])
               for m in ("joint_conv","conv","lanczos3","bilinear")]
        basin=[np.mean([r["basin"][m]["mse"] for r in data["records"]])
               for m in ("joint_conv","conv","lanczos3","bilinear")]
        axis.bar(x-width/2,point,width,label="point",color="#56B4E9")
        axis.bar(x+width/2,basin,width,label="warped pixel",color="#E69F00")
        axis.set_yscale("log");axis.set_xticks(x,labels,rotation=18,ha="right")
        axis.set_ylabel("mean exact-truth MSE");axis.set_title(title)
        axis.grid(True,axis="y",which="both",alpha=.22);axis.legend(frameon=False,fontsize=8)
    figure.suptitle(
        "Direct inverse warps: topology stress and exact procedural fidelity",
        fontsize=12,
    )
    out.parent.mkdir(parents=True,exist_ok=True);figure.savefig(out,dpi=300)
    plt.close(figure)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("output/support_geometry"))
    parser.add_argument("--out",type=Path,default=Path("/tmp/conv_warp_metrics.png"))
    args=parser.parse_args();render(args.root,args.out);print(args.out)


if __name__=="__main__":main()
