"""Matched fixed-policy surface refinement with and without neighborhood terms."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .jet_bundle import refine_jets
from .scene_promotion import audit


def run_branch(args):
    paths,folder,maximum,coherent,retain_points=args
    e,pose,packet=[json.loads(Path(p).read_text()) for p in paths]
    result,receipt,candidate=refine_jets(packet,pose,pose.get('camera_models',e['cameras']),max_evaluations=maximum,coherent_surfaces=coherent,protect_cameras=True,point_evidence=e if retain_points else None)
    receipt['inputs_sha256']={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    receipt['sources_sha256']={n:hashlib.sha256(Path(__file__).with_name(n+'.py').read_bytes()).hexdigest() for n in ('coherence_study','surface_coherence','jet_bundle','point_support')}
    name='coherent' if coherent else 'control';folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for suffix,value in [('',result),('.candidate',candidate)]:
        if value is not None:(folder/(name+suffix+'.json')).write_text(json.dumps(audit(e,value),indent=2))
    (folder/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
    return name,receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--pose',required=True);p.add_argument('--jets',required=True);p.add_argument('--out',required=True);p.add_argument('--retain-point-support',action='store_true');p.add_argument('--max-evaluations',type=int,default=1000);a=p.parse_args()
    paths=[a.evidence,a.pose,a.jets]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for name,r in pool.map(run_branch,[(paths,a.out,a.max_evaluations,c,a.retain_point_support) for c in (False,True)]):
            print(json.dumps(dict(branch=name,applied=r['applied'],before={k:v for k,v in r['before'].items() if k!='cameras'},after={k:v for k,v in r['after'].items() if k!='cameras'},failed_cameras=[i for i,v in r['camera_checks'].items() if not v['passed']],failed_point_cameras=[i for i,v in r['point_support']['camera_checks'].items() if not v],seconds=r['seconds'],evaluations=r['evaluations'],coherence_before=r['coherence_before'],coherence_after=r['coherence_after'])),flush=True)


if __name__=='__main__':main()
