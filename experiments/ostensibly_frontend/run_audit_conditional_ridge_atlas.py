#!/usr/bin/env python3
"""Audit conditional ridge surfaces on labeled cached ARCTIC phones."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .crazy_climber_geometry import CrazyClimberConfig,HessianRidgeConfig
from .depthmap_geometry import DepthmapGeometryConfig
from .occupation_point_cloud import PointCloudConfig
from .run_occupation_context_battery import _cache_name,_window_list


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("arctic_root",type=Path);p.add_argument("atlas",type=Path);p.add_argument("--query-speaker",required=True);p.add_argument("--utterances",required=True);p.add_argument("--raw-cache",type=Path,required=True);p.add_argument("--out",type=Path,required=True);args=p.parse_args()
    payload={"rows":128,"noise_db":8.0,"morphology":asdict(DepthmapGeometryConfig()),"ridge":asdict(HessianRidgeConfig()),"climbers":asdict(CrazyClimberConfig()),"point_cloud":asdict(PointCloudConfig(point_count=32768))};atlas=load_conditional_ridge_atlas(args.atlas);ranks=[];started=perf_counter()
    for utterance in (x for x in args.utterances.split(",") if x):
        for window in _window_list(args.arctic_root,args.query_speaker,utterance):
            path=args.raw_cache/_cache_name(window,{**payload,"key":window.key,"representation":"raw_occupation_point_cloud_v1"});ranking=atlas.rank(np.asarray(np.load(path),dtype=np.float64));ranks.append(next(i+1 for i,x in enumerate(ranking) if x["phone"]==window.label))
    result={"query_speaker":args.query_speaker,"elapsed_seconds":perf_counter()-started,"summary":{"count":len(ranks),"top1":sum(x==1 for x in ranks),"top5":sum(x<=5 for x in ranks),"top10":sum(x<=10 for x in ranks),"median_rank":float(np.median(ranks))},"ranks":ranks};args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.out),**result["summary"],"elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__=="__main__":main()
