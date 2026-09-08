"""Compare the integrable joint-control oracle with direct factor CONV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import affine_about_center
from .joint_reference import direct_joint_warp, joint_control_nets, evaluate_joint_atlas
from .operators import direct_warp, source_validity_mask, target_nodes
from .synthetic import canonical_fourier_fields


def _metrics(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str,float]:
    residual = np.asarray(estimate,dtype=np.float64)-np.asarray(truth,dtype=np.float64)
    selected = residual[mask]
    return {"mse":float(np.mean(selected*selected)),
            "linf":float(np.max(np.abs(selected)))}


def run(source_side: int=25,target_side: int=41) -> dict[str,object]:
    transforms = {
        "rotate_23":affine_about_center(angle_degrees=23.0,scale_x=0.72,scale_y=0.72),
        "skew":affine_about_center(scale_x=0.72,scale_y=0.72,shear_x=0.38),
        "anisotropic":affine_about_center(angle_degrees=-17.0,scale_x=0.55,scale_y=0.86),
    }
    records=[]
    for name,field in canonical_fourier_fields().items():
        source=field.sample_nodes((source_side,source_side)).astype(np.float32)
        control,diagnostic=joint_control_nets(source,enforce_range=False)
        for transform_name,transform in transforms.items():
            truth=field.warped_nodes(transform,(target_side,target_side))
            mask=source_validity_mask(transform,(target_side,target_side),source.shape[:2])
            qx,qy=target_nodes((target_side,target_side))
            px,py=transform.map(qx,qy)
            joint=evaluate_joint_atlas(control,px*(source_side-1),py*(source_side-1))
            estimates={
                "joint_reference":joint,
                "conv":direct_warp(source,transform,(target_side,target_side),method="conv"),
                "lanczos3":direct_warp(source,transform,(target_side,target_side),method="lanczos3"),
                "bilinear":direct_warp(source,transform,(target_side,target_side),method="bilinear"),
            }
            records.append({
                "field":name,"transform":transform_name,
                "valid_points":int(np.sum(mask)),"joint_diagnostic":diagnostic,
                "metrics":{method:_metrics(value,truth,mask)
                           for method,value in estimates.items()},
            })
    return {"source_side":source_side,"target_side":target_side,"records":records,
            "status":"reference polytope; not a proposed CONV replacement"}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-side",type=int,default=25)
    parser.add_argument("--target-side",type=int,default=41)
    parser.add_argument("--out",type=Path,default=Path("/tmp/conv_warp_joint.json"))
    args=parser.parse_args()
    result=run(args.source_side,args.target_side)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(args.out)


if __name__=="__main__":
    main()
