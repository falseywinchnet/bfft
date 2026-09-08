"""Render exact projective basin truth and matched direct-warp reconstructions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import perspective_about_center
from .joint_reference import finite_joint_control_nets
from .operators import positive_warped_basin_average,source_validity_mask
from .synthetic import canonical_polygon_fields,half_plane_at_angle
from .warped_pixels import certified_projective_warped_pixel_average


def _largest_true_rectangle(mask: np.ndarray)->tuple[slice,slice]:
    best=(0,0,0,0,0)
    height,width=mask.shape
    for top in range(height):
        active=np.ones(width,dtype=bool)
        for bottom in range(top,height):
            active &= mask[bottom]
            start=0
            while start<width:
                while start<width and not active[start]: start+=1
                stop=start
                while stop<width and active[stop]: stop+=1
                area=(bottom-top+1)*(stop-start)
                if area>best[0]: best=(area,top,bottom+1,start,stop)
                start=stop+1
    if best[0]==0:
        raise RuntimeError("no common valid projective display region")
    return slice(best[1],best[2]),slice(best[3],best[4])


def render(out: Path,source_side: int=33,target_side: int=25)->None:
    transform=perspective_about_center(
        perspective_x=.32,perspective_y=-.18,angle_degrees=9.0,
        scale_x=.67,scale_y=.72,
    )
    fields={"diagonal interface":half_plane_at_angle(37.0),
            "thin oblique bar":canonical_polygon_fields()["thin_oblique_bar"],
            "rotated box":canonical_polygon_fields()["rotated_box"]}
    methods=("conv","lanczos3","bilinear")
    figure,axes=plt.subplots(len(fields),6,figsize=(15.2,7.2),constrained_layout=True)
    for row,(name,field) in enumerate(fields.items()):
        source=field.sample_nodes((source_side,source_side)).astype(np.float32)
        truth=field.warped_basin_average(transform,(target_side,target_side))
        mask=source_validity_mask(transform,(target_side,target_side),source.shape)
        display_region=_largest_true_rectangle(mask)
        control,_=finite_joint_control_nets(source)
        joint,joint_diagnostic=certified_projective_warped_pixel_average(
            source,transform,(target_side,target_side),control=control,
            tolerance=2e-5,maximum_depth=4,
        )
        joint_residual=np.asarray(joint,dtype=np.float64)-truth
        joint_mse=float(np.mean(joint_residual[mask]**2))
        joint_excursion=max(float(np.max(-joint[mask],initial=0.0)),
                             float(np.max(joint[mask]-1.0,initial=0.0)))
        images=[source,truth[display_region],joint[display_region]];titles=[
            f"{name}\nsource","exact projective\nbasin truth (valid crop)",
            f"CONV\nMSE {joint_mse:.3e}, exc. {joint_excursion:.3g}\n"
            f"cert. {joint_diagnostic['maximum_certified_error']:.1e}",
        ]
        for method in methods:
            estimate=positive_warped_basin_average(
                source,transform,(target_side,target_side),method=method,quadrature_order=6
            )
            residual=np.asarray(estimate,dtype=np.float64)-truth
            mse=float(np.mean(residual[mask]**2))
            excursion=max(float(np.max(-estimate[mask],initial=0.0)),
                          float(np.max(estimate[mask]-1.0,initial=0.0)))
            label={"conv":"factor-only ablation","lanczos3":"Lanczos-3",
                   "bilinear":"bilinear"}[method]
            images.append(estimate[display_region]);titles.append(
                f"{label}\nMSE {mse:.3e}, exc. {excursion:.3f}"
            )
        for column,(image,title) in enumerate(zip(images,titles)):
            axes[row,column].imshow(image,cmap="gray",vmin=-.12,vmax=1.12,
                                    interpolation="nearest")
            axes[row,column].set_title(title,fontsize=8)
            axes[row,column].set_xticks([]);axes[row,column].set_yticks([])
    figure.suptitle(
        "One direct homography; exact procedural basin truth",
        fontsize=11,
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(out,dpi=300)
    plt.close(figure)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument(
        "--out",type=Path,default=Path("/tmp/conv_warp_projective_plate.png"))
    parser.add_argument("--source-side",type=int,default=33)
    parser.add_argument("--target-side",type=int,default=25)
    args=parser.parse_args();render(args.out,args.source_side,args.target_side);print(args.out)


if __name__=="__main__":main()
