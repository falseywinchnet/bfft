#!/usr/bin/env python3
"""Concatenate compatible conditional ridge atlases."""

from __future__ import annotations
import argparse,json
from pathlib import Path
from .conditional_ridge_atlas import combine_conditional_ridge_atlases,load_conditional_ridge_atlas,save_conditional_ridge_atlas


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("atlases",nargs="+",type=Path);p.add_argument("--out",type=Path,required=True);args=p.parse_args();atlas=combine_conditional_ridge_atlases(load_conditional_ridge_atlas(x) for x in args.atlases);args.out.parent.mkdir(parents=True,exist_ok=True);save_conditional_ridge_atlas(args.out,atlas);print(json.dumps({"output":str(args.out),"occurrences":atlas.surfaces.shape[0],"labels":len(set(atlas.labels.tolist())),"size_bytes":args.out.stat().st_size},indent=2))


if __name__=="__main__":main()
