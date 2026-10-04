"""Run direct regional transport on the whole panorama candidate collection."""
import argparse,json,time,hashlib
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .scene_evidence import canonical_features
from .region_transport import transport,POLICY

@lru_cache(maxsize=16)
def feature(path):
    f=canonical_features(Path(path))
    with np.load(path) as packed:f['lab']=packed['lab'];f['rgb']=packed['rgb']
    return f


def pair(spec):
    edge,records,folder,out=spec;start=time.perf_counter();a,b=edge['a'],edge['b'];dest=Path(out)/f'{a:03d}-{b:03d}.npz'
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    fa=feature(str(Path(folder)/(Path(records[a]['name']).stem+'.npz')));fb=feature(str(Path(folder)/(Path(records[b]['name']).stem+'.npz')));cc=np.array(edge['candidates'])
    dest=Path(out)/f'{a:03d}-{b:03d}.npz'
    if not len(cc):return dict(a=a,b=b,candidates=0,accepted=0,localized=0,seconds=0)
    pa=fa['sites'][cc[:,0].astype(int)];pb=fb['sites'][cc[:,1].astype(int)]
    try:r=transport(fa['lab'],fb['lab'],pa,pb)
    except Exception as error:raise RuntimeError(f'regional transport failed for capture pair {a}, {b}') from error
    np.savez_compressed(dest,source=pa,target_site=pb,site_ids=cc[:,:2].astype(int),descriptor_distance=cc[:,2],**r)
    accepted=r['accepted'];n=int(accepted.sum());row=dict(a=a,b=b,candidates=len(cc),accepted=n,localized=int(r['well_localized'].sum()),
        initial_correlation=float(np.median(r['initial_correlation'][accepted])) if n else None,final_correlation=float(np.median(r['validation_correlation'][accepted])) if n else None,
        median_shift=float(np.median(np.linalg.norm(r['center'][accepted]-pb[accepted],axis=1))) if n else None,seconds=time.perf_counter()-start)
    dest.with_suffix('.json').write_text(json.dumps(row));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',type=int,default=0);p.add_argument('--all-captures',action='store_true');p.add_argument('--workers',type=int,default=3);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);e=json.loads(args.evidence.read_text());edges=[edge for edge in e['edges'] if args.all_captures or all(e['records'][i]['panoramic'] for i in (edge['a'],edge['b']))]
    if args.probe:edges=[edge for edge in edges if edge['b']==edge['a']+1][:args.probe]
    fingerprint=hashlib.sha256(args.evidence.read_bytes()+Path(__file__).read_bytes()+Path(__file__).with_name('region_transport.py').read_bytes())
    for record in e['records']:fingerprint.update(hashlib.sha256((args.features/(Path(record['name']).stem+'.npz')).read_bytes()).digest())
    policy_file=args.out/'run-policy.json';identity=fingerprint.hexdigest()
    if policy_file.exists() and json.loads(policy_file.read_text())['sha256']!=identity:raise RuntimeError('Inputs or policy changed; use a new output directory')
    policy_file.write_text(json.dumps(dict(sha256=identity,policy=POLICY)))
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(pair,[(edge,e['records'],str(args.features),str(args.out)) for edge in edges]):
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(policy=POLICY,records=e['records'],cameras=e['cameras'],pairs=rows,seconds=time.perf_counter()-start,parent_evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('capture_transport.py','region_transport.py')},
        limitations='Independent local affine region maps with pixel-level validation and reverse checks. No camera pose, scene depth, full alternative-track solution or image fusion is certified by these pair maps.')
    (args.out/'transport.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(pairs=len(rows),accepted=sum(r['accepted'] for r in rows),seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
