"""Bidirectional local-map frame evidence; connectivity is not global geometry."""
import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .submap_alignment import align

EVIDENCE=None
MAPS=None


def round_trip(forward,reverse,points):
    """Forward maps a to b; reverse maps b to a, all in source-frame units."""
    s=forward['scale'];r=np.asarray(forward['rotation']);t=np.asarray(forward['translation'])
    u=reverse['scale'];v=np.asarray(reverse['rotation']);w=np.asarray(reverse['translation'])
    x=np.asarray(points);returned=u*(s*x@r.T+t)@v.T+w
    # Centering makes the normalization independent of arbitrary chart origin.
    extent=max(float(np.median(np.linalg.norm(x-np.median(x,axis=0),axis=1))),1e-9)
    angle=float(np.rad2deg(Rotation.from_matrix(v@r).magnitude()))
    logscale=float(abs(np.log(s*u)));error=float(np.quantile(np.linalg.norm(returned-x,axis=1),.9)/extent)
    return dict(accepted=bool(angle<=1.2 and logscale<=.05 and error<=.05),rotation_degrees=angle,absolute_log_scale=logscale,relative_point_p90=error)


def components(count,edges):
    links=[set() for _ in range(count)]
    for e in edges:
        if e['accepted']:links[e['a']].add(e['b']);links[e['b']].add(e['a'])
    remaining=set(range(count));groups=[]
    while remaining:
        todo=[min(remaining)];group=set()
        while todo:
            i=todo.pop()
            if i in group:continue
            group.add(i);todo.extend(links[i]-group)
        remaining-=group;groups.append(sorted(group))
    return groups


def initialize(evidence_path,map_paths):
    global EVIDENCE,MAPS
    EVIDENCE=json.loads(Path(evidence_path).read_text());MAPS=[json.loads(Path(p).read_text()) for p in map_paths]


def solve(spec):
    a,b,folder=spec;path=Path(folder)/f'{a:03d}-{b:03d}.json'
    if path.exists():return json.loads(path.read_text())
    first=align(MAPS[a],MAPS[b],EVIDENCE);second=align(MAPS[b],MAPS[a],EVIDENCE)
    row=dict(a=a,b=b,forward=first,reverse=second,accepted=False)
    if first.get('accepted') and second.get('accepted'):
        cloud=[p['xyz'] for p in MAPS[a]['points'] if p['promotion']=='third_view_consistent']
        row['round_trip']=round_trip(first,second,cloud);row['accepted']=row['round_trip']['accepted']
    path.write_text(json.dumps(row,indent=2));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--submaps',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);meta=json.loads((args.submaps/'submaps.json').read_text())
    rows=sorted([r for r in meta['submaps'] if r['accepted']],key=lambda r:r['seed'])
    paths=[args.submaps/f"{r['seed'][0]:03d}-{r['seed'][1]:03d}.json" for r in rows]
    sets=[{p['track'] for p in json.loads(path.read_text())['points'] if p['promotion']=='third_view_consistent'} for path in paths]
    pairs=[(a,b) for a,b in itertools.combinations(range(len(rows)),2) if len(sets[a]&sets[b])>=20]
    policy=dict(evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),submaps_sha256=hashlib.sha256((args.submaps/'submaps.json').read_bytes()).hexdigest(),
                map_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('submap_graph.py','submap_alignment.py','scene_pose.py','multiview_geometry.py')},
                candidates='All accepted local map pairs sharing at least 20 consistent track IDs; both directions independently fitted and held-out checked; round-trip scale, orientation and point checks.')
    pp=args.out/'policy.json'
    if pp.exists() and json.loads(pp.read_text())!=policy:raise RuntimeError('Map graph policy or inputs changed; use a new output directory')
    pp.write_text(json.dumps(policy,indent=2));start=time.perf_counter();edges=[]
    print(json.dumps(dict(maps=len(rows),candidate_pairs=len(pairs))),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(args.evidence),[str(p) for p in paths])) as pool:
        for row in pool.map(solve,[(a,b,str(args.out)) for a,b in pairs],chunksize=4):
            edges.append(row)
            if len(edges)%100==0:print(json.dumps(dict(completed=len(edges),total=len(pairs),accepted=sum(r['accepted'] for r in edges))),flush=True)
    groups=components(len(rows),edges)
    packet=dict(maps=rows,edges=edges,components=[dict(maps=g,cameras=sorted({i for k in g for i in rows[k]['cameras']})) for g in groups],policy=policy,seconds=time.perf_counter()-start,
                limitations='Pairwise compatibility graph only. A connected component does not establish globally consistent frames, metric depth, complete visibility, or fusion. Keep rejected edges.')
    (args.out/'graph.json').write_text(json.dumps(packet,indent=2));print(json.dumps(dict(accepted_edges=sum(r['accepted'] for r in edges),components=len(groups),component_camera_counts=sorted([len(g['cameras']) for g in packet['components']],reverse=True),seconds=packet['seconds'])),flush=True)


if __name__=='__main__':main()
