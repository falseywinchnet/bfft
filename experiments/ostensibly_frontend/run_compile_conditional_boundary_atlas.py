#!/usr/bin/env python3
"""Compile ordered conditional ridge surfaces for generic diphones."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .conditional_ridge_atlas import compile_conditional_ridge_atlas,save_conditional_ridge_atlas
from .crazy_climber_geometry import CrazyClimberConfig,HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .run_occupation_context_battery import _cache_name,_window_list
from .run_occupation_word_span_battery import joint_gauge_chunks,lane_chunks


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("arctic_root",type=Path);p.add_argument("--speaker",required=True);p.add_argument("--utterances",required=True);p.add_argument("--raw-cache",type=Path,required=True);p.add_argument("--points-per-phone",type=int,default=2048);p.add_argument("--time-bins",type=int,default=32);p.add_argument("--row-quantiles",type=int,default=16);p.add_argument("--out",type=Path,required=True);args=p.parse_args()
    payload={"rows":128,"noise_db":8.0,"morphology":asdict(DepthmapGeometryConfig()),"ridge":asdict(HessianRidgeConfig()),"climbers":asdict(CrazyClimberConfig()),"point_cloud":asdict(PointCloudConfig(point_count=32768))};occurrences=[]
    def raw(w):
        path=args.raw_cache/_cache_name(w,{**payload,"key":w.key,"representation":"raw_occupation_point_cloud_v1"});return np.asarray(np.load(path),dtype=np.float64)
    for utterance in (x for x in args.utterances.split(",") if x):
        windows=_window_list(args.arctic_root,args.speaker,utterance)
        for i,(left,right) in enumerate(zip(windows,windows[1:])):
            cloud=lane_chunks(joint_gauge_chunks((raw(left),raw(right)),args.points_per_phone));occurrences.append((f"{left.label}|{right.label}",f"{args.speaker}:{utterance}:{i}",cloud))
    atlas=compile_conditional_ridge_atlas(occurrences,args.time_bins,args.row_quantiles);args.out.parent.mkdir(parents=True,exist_ok=True);save_conditional_ridge_atlas(args.out,atlas);print(json.dumps({"output":str(args.out),"occurrences":len(occurrences),"pair_types":len(set(atlas.labels.tolist())),"size_bytes":args.out.stat().st_size},indent=2))


if __name__=="__main__":main()
