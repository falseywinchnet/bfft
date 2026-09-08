"""Exact-truth census for finite joint CONV direct warps."""

from __future__ import annotations

import argparse,json,time
from pathlib import Path

import numpy as np

from .geometry import affine_about_center
from .joint_reference import evaluate_joint_atlas,finite_joint_control_nets
from .operators import (
    direct_warp,positive_warped_basin_average,source_validity_mask,target_nodes,
)
from .synthetic import canonical_fourier_fields
from .warped_pixels import exact_affine_warped_pixel_average


def _metrics(estimate: np.ndarray,truth: np.ndarray,mask: np.ndarray,source: np.ndarray)->dict[str,float]:
    residual=np.asarray(estimate,dtype=np.float64)-np.asarray(truth,dtype=np.float64)
    selected=residual[mask]
    return {
        "mse":float(np.mean(selected*selected)),
        "linf":float(np.max(np.abs(selected))),
        "source_range_excursion":max(
            float(np.max(np.min(source,axis=(0,1))-estimate,initial=0.0)),
            float(np.max(estimate-np.max(source,axis=(0,1)),initial=0.0)),
        ),
    }


def run(source_side: int=17,target_side: int=13)->dict[str,object]:
    transforms={
        "rotate_23":affine_about_center(angle_degrees=23.0,scale_x=.72,scale_y=.72),
        "skew":affine_about_center(scale_x=.72,scale_y=.72,shear_x=.38),
        "anisotropic":affine_about_center(angle_degrees=-17.0,scale_x=.55,scale_y=.86),
    }
    records=[]
    for field_name,field in canonical_fourier_fields().items():
        source=field.sample_nodes((source_side,source_side)).astype(np.float32)
        start=time.perf_counter()
        control,admission=finite_joint_control_nets(source)
        analysis_seconds=time.perf_counter()-start
        for transform_name,transform in transforms.items():
            point_truth=field.warped_nodes(transform,(target_side,target_side))
            basin_truth=field.affine_basin_average(transform,(target_side,target_side))
            mask=source_validity_mask(transform,(target_side,target_side),source.shape[:2])
            qx,qy=target_nodes((target_side,target_side));px,py=transform.map(qx,qy)
            start=time.perf_counter()
            joint_point=evaluate_joint_atlas(
                control,px*(source_side-1),py*(source_side-1)
            )
            joint_point_seconds=time.perf_counter()-start
            start=time.perf_counter()
            joint_basin,basin_diagnostic=exact_affine_warped_pixel_average(
                source,transform,(target_side,target_side),control=control
            )
            joint_basin_seconds=time.perf_counter()-start
            record={
                "field":field_name,"transform":transform_name,
                "valid_points":int(np.sum(mask)),"admission":admission,
                "joint_analysis_seconds":analysis_seconds,
                "point":{
                    "joint_conv":_metrics(joint_point,point_truth,mask,source)
                },
                "basin":{
                    "joint_conv":_metrics(joint_basin,basin_truth,mask,source)
                },
                "timing_seconds":{
                    "joint_point":joint_point_seconds,
                    "joint_exact_basin":joint_basin_seconds,
                },
                "basin_diagnostic":basin_diagnostic,
            }
            for method in ("conv","lanczos3","bilinear"):
                start=time.perf_counter()
                point=direct_warp(source,transform,(target_side,target_side),method=method)
                record["timing_seconds"][f"{method}_point"]=time.perf_counter()-start
                start=time.perf_counter()
                basin=positive_warped_basin_average(
                    source,transform,(target_side,target_side),method=method,
                    quadrature_order=6,
                )
                record["timing_seconds"][f"{method}_basin_q6"]=time.perf_counter()-start
                record["point"][method]=_metrics(point,point_truth,mask,source)
                record["basin"][method]=_metrics(basin,basin_truth,mask,source)
            records.append(record)
    return {
        "source_shape":[source_side,source_side],
        "target_shape":[target_side,target_side],
        "coordinate_convention":"one direct target-to-source map",
        "joint_witness":"cone of Q1 currents on the exact two-jet support",
        "point_truth":"closed-form procedural field evaluation",
        "basin_truth":"closed-form affine Fourier basin integral",
        "joint_basin":"exact knot-split positive polynomial integration",
        "records":records,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-side",type=int,default=17)
    parser.add_argument("--target-side",type=int,default=13)
    parser.add_argument("--out",type=Path,default=Path("/tmp/conv_warp_canonical.json"))
    args=parser.parse_args();result=run(args.source_side,args.target_side)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n");print(args.out)


if __name__=="__main__":main()
