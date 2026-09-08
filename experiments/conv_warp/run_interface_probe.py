"""Directional current diagnostics on exactly transformed step interfaces."""

from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np

from .geometry import affine_about_center,perspective_about_center
from .joint_reference import evaluate_joint_atlas,finite_joint_control_nets
from .operators import sample_inverse_warp
from .synthetic import half_plane_at_angle


def _sign_changes(difference: np.ndarray,tolerance: float=1e-5)->int:
    value=np.asarray(difference,dtype=np.float64)
    sign=np.sign(value[np.abs(value)>tolerance])
    return int(np.sum(sign[1:]!=sign[:-1])) if sign.size else 0


def _probe_line(line: np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    normal=line[:2]/np.linalg.norm(line[:2]);tangent=np.array((-normal[1],normal[0]))
    centre=np.array((0.5,0.5));point=centre-(line@np.r_[centre,1.0])*line[:2]/np.dot(line[:2],line[:2])
    return point,normal,tangent


def run(source_side: int=33,samples: int=513)->dict[str,object]:
    transforms={
        "rotate_21":affine_about_center(angle_degrees=21.0,scale_x=.72,scale_y=.72),
        "skew":affine_about_center(scale_x=.7,scale_y=.75,shear_x=.42),
        "perspective":perspective_about_center(perspective_x=.32,perspective_y=-.18,
                                               angle_degrees=9.0,scale_x=.67,scale_y=.72),
    }
    methods=("joint_current","conv","lanczos3","bilinear")
    records=[]
    for angle in (0.0,15.0,30.0,45.0,60.0,75.0):
        field=half_plane_at_angle(angle);source=field.sample_nodes((source_side,source_side)).astype(np.float32)
        joint_control,joint_diagnostic=finite_joint_control_nets(source)
        for transform_name,transform in transforms.items():
            line=field.target_line(transform);point,normal,tangent=_probe_line(line)
            coordinate=np.linspace(-.24,.24,samples)
            normal_x=point[0]+coordinate*normal[0];normal_y=point[1]+coordinate*normal[1]
            # The positive normal side is exactly constant in the generator.
            tangent_point=point+.055*normal
            tangent_x=tangent_point[0]+coordinate*tangent[0]
            tangent_y=tangent_point[1]+coordinate*tangent[1]
            record={"source_angle_degrees":angle,"transform":transform_name,
                    "joint_diagnostic":joint_diagnostic,"metrics":{}}
            for method in methods:
                if method=="joint_current":
                    px,py=transform.map(normal_x,normal_y)
                    normal_value=evaluate_joint_atlas(
                        joint_control,px*(source_side-1),py*(source_side-1)
                    )
                    px,py=transform.map(tangent_x,tangent_y)
                    tangent_value=evaluate_joint_atlas(
                        joint_control,px*(source_side-1),py*(source_side-1)
                    )
                else:
                    normal_value=sample_inverse_warp(
                        source,transform,normal_x,normal_y,method=method
                    )
                    tangent_value=sample_inverse_warp(
                        source,transform,tangent_x,tangent_y,method=method
                    )
                difference=np.diff(normal_value.astype(np.float64))
                wrong_way=float(np.sum(np.maximum(-difference,0.0)))
                record["metrics"][method]={
                    "normal_sign_changes":_sign_changes(difference),
                    "normal_wrong_way_variation":wrong_way,
                    "normal_range_excursion":max(float(np.max(-normal_value,initial=0.0)),
                                                 float(np.max(normal_value-1.0,initial=0.0))),
                    "tangent_ripple_rms":float(np.sqrt(np.mean((tangent_value-np.mean(tangent_value))**2))),
                    "tangent_total_variation":float(np.sum(np.abs(np.diff(tangent_value.astype(np.float64))))),
                }
            records.append(record)
    return {"source_side":source_side,"samples_per_probe":samples,
            "normal_truth":"one positive jump; zero wrong-way current",
            "tangent_truth":"constant one; zero variation","records":records}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--source-side",type=int,default=33)
    parser.add_argument("--samples",type=int,default=513)
    parser.add_argument("--out",type=Path,default=Path("/tmp/conv_warp_interface.json"))
    args=parser.parse_args();result=run(args.source_side,args.samples)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(args.out)


if __name__=="__main__":main()
