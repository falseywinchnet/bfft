"""Retain every supported panorama-seeded camera hypothesis for reconciliation."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .scene_pose import reconstruct
from .scene_promotion import audit

EVIDENCE=None


def initialize(path):
    global EVIDENCE
    EVIDENCE=json.loads(Path(path).read_text())


def solve(spec):
    a,b,folder,allow_photo_seeds=spec;path=Path(folder)/f'{a:03d}-{b:03d}.json'
    if path.exists():result=json.loads(path.read_text())
    else:
        start=time.perf_counter();result=audit(EVIDENCE,reconstruct(EVIDENCE,maximum_seed_trials=1,seed_pair=(a,b),allow_photo_seeds=allow_photo_seeds))
        result['seconds']=time.perf_counter()-start;path.write_text(json.dumps(result,separators=(',',':')))
    return dict(seed=[a,b],accepted=result['accepted'],cameras=sorted(map(int,result.get('poses',{}))),
                points=len(result.get('points',[])),promotion=result.get('promotion_counts',{}),seconds=result.get('seconds'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed-scope',choices=['panorama','all'],default='panorama');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    e=json.loads(args.evidence.read_text());pairs=[(r['a'],r['b']) for r in e['edges'] if r['status']=='translation_supported' and (args.seed_scope=='all' or all(e['records'][i]['panoramic'] for i in (r['a'],r['b'])))]
    policy=dict(parent_evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('scene_submaps.py','scene_pose.py','scene_promotion.py','multiview_geometry.py')},scope=args.seed_scope,selection='every translation-supported seed in the declared scope')
    receipt=args.out/'policy.json'
    if receipt.exists() and json.loads(receipt.read_text())!=policy:raise RuntimeError('Submap inputs or algorithm changed; choose a new output directory')
    receipt.write_text(json.dumps(policy,indent=2));start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(args.evidence),)) as pool:
        for row in pool.map(solve,[(a,b,str(args.out),args.seed_scope=='all') for a,b in pairs]):rows.append(row);print(json.dumps(dict(completed=len(rows),total=len(pairs),**row)),flush=True)
    packet=dict(submaps=rows,records=e['records'],policy=policy,seconds=time.perf_counter()-start,
                observed_cameras=sorted({i for r in rows if r['accepted'] for i in r['cameras']}),
                limitations='Alternative local camera hypotheses have independent coordinate frames and scales. Their union is not a registered shared scene. Preserve rejected seeds and require cross-map geometry checks before merging.')
    (args.out/'submaps.json').write_text(json.dumps(packet,indent=2));print(json.dumps(dict(accepted=sum(r['accepted'] for r in rows),observed_cameras=len(packet['observed_cameras']),seconds=packet['seconds'])),flush=True)


if __name__=='__main__':main()
