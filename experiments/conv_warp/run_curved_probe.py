"""Intrinsic normal/tangent current probes on transformed conic interfaces."""

from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np

from .geometry import affine_about_center,perspective_about_center
from .operators import sample_inverse_warp
from .synthetic import EllipseField
from .joint_reference import (
    _experimental_finite_joint_control_nets,evaluate_joint_atlas,
    global_joint_control_nets,
)


def _sign_changes(difference: np.ndarray,tolerance: float=1e-5)->int:
    sign=np.sign(difference[np.abs(difference)>tolerance])
    return int(np.sum(sign[1:]!=sign[:-1])) if sign.size else 0


def run(source_side: int=41,samples: int=385,witness: str="completed_direction")->dict[str,object]:
    transforms={
        "rotate_21":affine_about_center(angle_degrees=21.0,scale_x=.78,scale_y=.78),
        "skew":affine_about_center(scale_x=.76,scale_y=.78,shear_x=.34),
        "perspective":perspective_about_center(perspective_x=.28,perspective_y=-.16,
                                               angle_degrees=8.0,scale_x=.72,scale_y=.76),
    }
    fields={
        "circle":EllipseField(radius_x=.27,radius_y=.27),
        "ellipse":EllipseField(radius_x=.31,radius_y=.19,angle_degrees=17.0),
    }
    methods=("joint_current","conv","lanczos3","bilinear");records=[]
    coordinate=np.linspace(-.11,.11,samples)
    tangent_coordinate=np.linspace(-.10,.10,samples)
    for field_name,field in fields.items():
        source=field.sample_nodes((source_side,source_side)).astype(np.float32)
        external=None
        witness_mode=witness
        if witness.startswith("finite_"):
            finite_modes={
                "finite_sum":"support_sum",
                "finite_cone":"support_cone",
                "finite_local_gradient_cone":"local_gradient_cone",
                "finite_support_gradient_cone":"support_gradient_cone",
            }
            if witness not in finite_modes:
                raise ValueError(f"unknown finite witness {witness!r}")
            joint_control,joint_diagnostic=_experimental_finite_joint_control_nets(
                source,witness_mode=finite_modes[witness]
            )
        elif witness=="analytic_conic":
            cell_coordinate=(np.arange(source_side-1)+.5)/(source_side-1)
            yy,xx=np.meshgrid(cell_coordinate,cell_coordinate,indexing="ij")
            homogeneous=np.stack((xx,yy,np.ones_like(xx)),axis=-1)
            gradient=2.0*np.einsum(
                "ij,...j->...i",field.source_conic(),homogeneous,optimize=True
            )[..., :2]
            external=-gradient
            witness_mode="phase_gradient"
        elif witness not in ("support_sum","phase_gradient","completed_direction"):
            raise ValueError(
                "witness must be 'support_sum', 'phase_gradient', "
                "'completed_direction', 'finite_entity', or 'analytic_conic'"
            )
        if not witness.startswith("finite_"):
            joint_control,joint_diagnostic=global_joint_control_nets(
                source,external_direction=external,witness_mode=witness_mode
            )
        for transform_name,transform in transforms.items():
            for phase_index,phase in enumerate(np.linspace(0.0,2*np.pi,12,endpoint=False)):
                point,inward,tangent=field.target_boundary_frame(transform,float(phase))
                normal_x=point[0]+coordinate*inward[0];normal_y=point[1]+coordinate*inward[1]
                tangent_point=point+.045*inward
                tangent_x=tangent_point[0]+tangent_coordinate*tangent[0]
                tangent_y=tangent_point[1]+tangent_coordinate*tangent[1]
                record={"field":field_name,"transform":transform_name,"phase_index":phase_index,
                        "joint_diagnostic":joint_diagnostic,"metrics":{}}
                for method in methods:
                    if method=="joint_current":
                        px,py=transform.map(normal_x,normal_y)
                        normal=evaluate_joint_atlas(joint_control,px*(source_side-1),py*(source_side-1))
                        px,py=transform.map(tangent_x,tangent_y)
                        along=evaluate_joint_atlas(joint_control,px*(source_side-1),py*(source_side-1))
                    else:
                        normal=sample_inverse_warp(source,transform,normal_x,normal_y,method=method)
                        along=sample_inverse_warp(source,transform,tangent_x,tangent_y,method=method)
                    difference=np.diff(normal.astype(np.float64))
                    record["metrics"][method]={
                        "normal_sign_changes":_sign_changes(difference),
                        "normal_wrong_way_variation":float(np.sum(np.maximum(-difference,0.0))),
                        "normal_range_excursion":max(float(np.max(-normal,initial=0.0)),
                                                     float(np.max(normal-1.0,initial=0.0))),
                        "tangent_ripple_rms":float(np.sqrt(np.mean((along-np.mean(along))**2))),
                        "tangent_total_variation":float(np.sum(np.abs(np.diff(along.astype(np.float64))))),
                    }
                records.append(record)
    return {"source_side":source_side,"samples_per_probe":samples,
            "joint_witness":witness,
            "normal_truth":"one positive jump at the probed conic point",
            "tangent_probe":"inside offset; curvature is part of truth and must be separated from ripple",
            "records":records}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--source-side",type=int,default=41)
    parser.add_argument("--samples",type=int,default=385);parser.add_argument(
        "--out",type=Path,default=Path("/tmp/conv_warp_curved.json"))
    parser.add_argument(
        "--witness",choices=(
            "support_sum","phase_gradient","completed_direction","finite_sum",
            "finite_cone","finite_local_gradient_cone",
            "finite_support_gradient_cone","analytic_conic"
        ), default="completed_direction"
    )
    args=parser.parse_args();result=run(args.source_side,args.samples,args.witness)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(args.out)


if __name__=="__main__":main()
