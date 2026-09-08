#!/usr/bin/env python3
"""Classify cached radio regions by ordered conditional ridge trajectories."""

from __future__ import annotations
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from .conditional_ridge_atlas import load_conditional_ridge_atlas


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("region_clouds",type=Path);p.add_argument("region_lattice",type=Path);p.add_argument("atlas",type=Path);p.add_argument("--top-k",type=int,default=39);p.add_argument("--out",type=Path,required=True);args=p.parse_args();lattice=json.loads(args.region_lattice.read_text());
    with np.load(args.region_clouds) as d:clouds=np.asarray(d["clouds"],dtype=np.float64)
    if clouds.shape[0]!=len(lattice["phones"]):raise ValueError("region cloud and lattice counts disagree")
    atlas=load_conditional_ridge_atlas(args.atlas);rows=[];started=perf_counter()
    for index,(cloud,region) in enumerate(zip(clouds,lattice["phones"],strict=True)):
        ranking=atlas.rank(cloud);rows.append({**{k:v for k,v in region.items() if k!="top5"},"top5":ranking[:args.top_k]})
        if (index+1)%50==0:print(f"classified {index+1}/{clouds.shape[0]} conditional ridges",flush=True)
    result={"method":"ordered_conditional_ridge_trajectory_atlas","selection_status":"generic local trajectory channel; no radio labels","phone_count":len(rows),"elapsed_seconds":perf_counter()-started,"phones":rows};args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(args.out),"phone_count":len(rows),"elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__=="__main__":main()
