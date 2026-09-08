#!/usr/bin/env python3
"""Compile ordered conditional ridge surfaces from cached generic phones."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from .conditional_ridge_atlas import compile_conditional_ridge_atlas, save_conditional_ridge_atlas
from .crazy_climber_geometry import CrazyClimberConfig, HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .run_occupation_context_battery import _cache_name, _window_list


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("arctic_root",type=Path); p.add_argument("--speaker",required=True); p.add_argument("--utterances",required=True); p.add_argument("--raw-cache",type=Path,required=True); p.add_argument("--time-bins",type=int,default=32); p.add_argument("--row-quantiles",type=int,default=16); p.add_argument("--maximum-shift",type=int,default=2); p.add_argument("--derivative-weight",type=float,default=.5); p.add_argument("--mass-weight",type=float,default=.25); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    payload={"rows":128,"noise_db":8.0,"morphology":asdict(DepthmapGeometryConfig()),"ridge":asdict(HessianRidgeConfig()),"climbers":asdict(CrazyClimberConfig()),"point_cloud":asdict(PointCloudConfig(point_count=32768))}
    occurrences=[]
    for utterance in (x for x in args.utterances.split(",") if x):
        for ordinal,window in enumerate(_window_list(args.arctic_root,args.speaker,utterance)):
            path=args.raw_cache/_cache_name(window,{**payload,"key":window.key,"representation":"raw_occupation_point_cloud_v1"})
            occurrences.append((window.label,f"{args.speaker}:{utterance}:{ordinal}",np.asarray(np.load(path),dtype=np.float64)))
    atlas=compile_conditional_ridge_atlas(occurrences,args.time_bins,args.row_quantiles,args.maximum_shift,args.derivative_weight,args.mass_weight); args.out.parent.mkdir(parents=True,exist_ok=True); save_conditional_ridge_atlas(args.out,atlas)
    print(json.dumps({"output":str(args.out),"occurrences":len(occurrences),"labels":len(set(atlas.labels.tolist())),"size_bytes":args.out.stat().st_size},indent=2))


if __name__=="__main__": main()
