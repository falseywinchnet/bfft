#!/usr/bin/env python3
"""Fuse conditional phone and conditional diphone trajectory rankings."""

from __future__ import annotations
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from .conditional_ridge_atlas import load_conditional_ridge_atlas
from .run_occupation_word_span_battery import joint_gauge_chunks,lane_chunks
from .run_query_radio_boundary_atlas import endpoint_ranking,fuse_rankings


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("region_clouds",type=Path);p.add_argument("center_lattice",type=Path);p.add_argument("boundary_atlas",type=Path);p.add_argument("--points-per-phone",type=int,default=2048);p.add_argument("--report-count",type=int,default=39);p.add_argument("--retain-boundary-count",type=int,default=128);p.add_argument("--out",type=Path,required=True);args=p.parse_args();center=json.loads(args.center_lattice.read_text())
    with np.load(args.region_clouds) as d:clouds=np.asarray(d["clouds"],dtype=np.float64)
    if clouds.shape[0]!=len(center["phones"]):raise ValueError("region cloud and center lattice counts disagree")
    atlas=load_conditional_ridge_atlas(args.boundary_atlas);left=[None]*len(clouds);right=[None]*len(clouds);records=[];started=perf_counter()
    for i in range(len(clouds)-1):
        pair=lane_chunks(joint_gauge_chunks((clouds[i],clouds[i+1]),args.points_per_phone));raw=atlas.rank(pair);ranking=[{"phones":item["phone"].split("|"),"distance":item["distance"],"witness":item["witness"],"rank":item["rank"]} for item in raw];records.append({"boundary_index":i,"ranking":ranking[:args.retain_boundary_count]});right[i]=endpoint_ranking(ranking,0);left[i+1]=endpoint_ranking(ranking,1)
        if (i+1)%50==0:print(f"queried {i+1}/{len(clouds)-1} conditional boundaries",flush=True)
    rows=[]
    for i,row in enumerate(center["phones"]):
        center_rank=[{**item,"rank":rank} for rank,item in enumerate(row["top5"],1)];channels={}
        if left[i] is not None:channels["left_boundary"]=left[i]
        if right[i] is not None:channels["right_boundary"]=right[i]
        boundary=fuse_rankings(channels,"mean");all_channels={"center":center_rank,**channels};mean=fuse_rankings(all_channels,"mean");conservative=fuse_rankings(all_channels,"conservative");rows.append({**{k:v for k,v in row.items() if k!="top5"},"center_top5":center_rank[:args.report_count],"boundary_top5":boundary[:args.report_count],"mean_rank_top5":mean[:args.report_count],"conservative_top5":conservative[:args.report_count],"top5":mean[:args.report_count]})
    result={"method":"conditional_phone_plus_diphone_trajectory_rank_fusion","selection_status":"diagnostic variants retained independently","phone_count":len(rows),"boundary_query_seconds":perf_counter()-started,"joint_boundary_pair_swd_rankings":records,"phones":rows};args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(args.out),"boundary_query_seconds":result["boundary_query_seconds"]},indent=2))


if __name__=="__main__":main()
