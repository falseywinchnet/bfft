"""Exact sharp-geometry census for direct affine and projective warps."""

from __future__ import annotations

import argparse,json,time
from pathlib import Path
import numpy as np

from .geometry import affine_about_center,perspective_about_center
from .joint_reference import evaluate_joint_atlas,finite_joint_control_nets
from .operators import (
    direct_warp,positive_warped_basin_average,source_validity_mask,target_nodes,
)
from .synthetic import canonical_polygon_fields,half_plane_at_angle
from .warped_pixels import (
    certified_projective_warped_pixel_average,exact_affine_warped_pixel_average,
)


def _metrics(estimate: np.ndarray,truth: np.ndarray,mask: np.ndarray,source: np.ndarray)->dict[str,float]:
    residual=np.asarray(estimate,dtype=np.float64)-truth
    selected=residual[mask]
    selected_estimate=np.asarray(estimate,dtype=np.float64)[mask]
    low,high=float(np.min(source)),float(np.max(source))
    return {"mse":float(np.mean(selected*selected)),"linf":float(np.max(np.abs(selected))),
            "source_range_excursion":max(float(np.max(low-selected_estimate,initial=0.0)),
                                         float(np.max(selected_estimate-high,initial=0.0)))}


def run(source_side: int=33,target_side: int=25)->dict[str,object]:
    transforms={
        "rotate_21":affine_about_center(angle_degrees=21.0,scale_x=0.72,scale_y=0.72),
        "skew":affine_about_center(scale_x=0.7,scale_y=0.75,shear_x=0.42),
        "anisotropic":affine_about_center(angle_degrees=-14.0,scale_x=0.5,scale_y=0.84),
        "perspective":perspective_about_center(perspective_x=0.32,perspective_y=-0.18,
                                               angle_degrees=9.0,scale_x=0.67,scale_y=0.72),
    }
    methods=("conv","lanczos3","bilinear")
    records=[]
    fields={"diagonal_interface":half_plane_at_angle(37.0),**canonical_polygon_fields()}
    for field_name,field in fields.items():
        source=field.sample_nodes((source_side,source_side)).astype(np.float32)
        start=time.perf_counter();control,admission=finite_joint_control_nets(source)
        analysis_seconds=time.perf_counter()-start
        for transform_name,transform in transforms.items():
            point_truth=field.warped_nodes(transform,(target_side,target_side))
            basin_truth=field.warped_basin_average(transform,(target_side,target_side))
            mask=source_validity_mask(transform,(target_side,target_side),source.shape)
            qx,qy=target_nodes((target_side,target_side));px,py=transform.map(qx,qy)
            start=time.perf_counter();joint_point=evaluate_joint_atlas(
                control,px*(source_side-1),py*(source_side-1)
            );joint_point_seconds=time.perf_counter()-start
            start=time.perf_counter()
            if transform.is_affine:
                joint_basin,basin_diagnostic=exact_affine_warped_pixel_average(
                    source,transform,(target_side,target_side),control=control
                )
            else:
                joint_basin,basin_diagnostic=certified_projective_warped_pixel_average(
                    source,transform,(target_side,target_side),control=control,
                    tolerance=2e-5,maximum_depth=4,
                )
            joint_basin_seconds=time.perf_counter()-start
            record={"field":field_name,"transform":transform_name,
                    "valid_points":int(np.sum(mask)),"point":{},"basin":{},
                    "admission":admission,"basin_diagnostic":basin_diagnostic,
                    "timing_seconds":{"joint_analysis":analysis_seconds,
                                      "joint_point":joint_point_seconds,
                                      "joint_basin":joint_basin_seconds}}
            record["point"]["joint_conv"]=_metrics(joint_point,point_truth,mask,source)
            record["basin"]["joint_conv"]=_metrics(joint_basin,basin_truth,mask,source)
            for method in methods:
                start=time.perf_counter();point=direct_warp(
                    source,transform,(target_side,target_side),method=method
                );record["timing_seconds"][f"{method}_point"]=time.perf_counter()-start
                start=time.perf_counter()
                basin=positive_warped_basin_average(
                    source,transform,(target_side,target_side),method=method,quadrature_order=6
                )
                record["timing_seconds"][f"{method}_basin_q6"]=time.perf_counter()-start
                record["point"][method]=_metrics(point,point_truth,mask,source)
                record["basin"][method]=_metrics(basin,basin_truth,mask,source)
            records.append(record)
    return {"source_shape":[source_side,source_side],"target_shape":[target_side,target_side],
            "point_truth":"exact polygon membership after one direct map",
            "basin_truth":"exact projective polygon coverage in target coordinates",
            "joint_basin":"exact affine or certified projective atlas integral",
            "records":records}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--source-side",type=int,default=33)
    parser.add_argument("--target-side",type=int,default=25)
    parser.add_argument("--out",type=Path,default=Path("/tmp/conv_warp_projective.json"))
    args=parser.parse_args();result=run(args.source_side,args.target_side)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(args.out)


if __name__=="__main__":main()
