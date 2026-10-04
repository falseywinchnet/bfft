"""Expand a panorama scene through automatically discovered capture triangles."""
import argparse,itertools,json,shutil,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .dense_transport import grow,check_cycles
import numpy as np


def supported_graph(folder,rows,minimum=12):
    graph={}
    for row in rows:
        a,b=row['a'],row['b'];counts=[]
        for x,y in ((a,b),(b,a)):
            with np.load(folder/f'{x:03d}-{y:03d}.npz') as f:counts.append(int(np.sum(f['third_view_support']>0)))
        if min(counts)>=minimum:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    return graph


def reachable(graph,seeds):
    seen=set(seeds);todo=list(seeds)
    while todo:
        a=todo.pop()
        for b in graph.get(a,set())-seen:seen.add(b);todo.append(b)
    return seen


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--transport',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);base=json.loads((args.base/'dense.json').read_text());meta=json.loads((args.transport/'transport.json').read_text());start=time.perf_counter()
    for name,digest in base['source_hashes'].items():
        if name in ('dense_transport.py','region_transport.py') and hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Parent field algorithm changed: {name}')
    for file in args.base.glob('*.npz'):shutil.copyfile(file,args.out/file.name)
    rows=list(base['pairs']);done={(r['a'],r['b']) for r in rows};eligible={(r['a'],r['b']):r for r in meta['pairs'] if r['localized']>=8};adj={}
    for a,b in eligible:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
    seeds={i for i,r in enumerate(meta['records']) if r['panoramic']};known=reachable(supported_graph(args.out,rows),seeds);rounds=[]
    for iteration in range(3):
        targets=set()
        for a in sorted(adj):
            for b in sorted(v for v in adj[a] if v>a):
                for c in sorted(v for v in adj[a]&adj.get(b,set()) if v>b):
                    nodes={a,b,c}
                    if nodes&known and nodes-known:targets.update(((a,b),(b,c),(a,c)))
        targets=sorted(targets-done,key=lambda ab:(-eligible[ab]['localized'],ab))
        if not targets:break
        print(json.dumps(dict(round=iteration,anchored_captures=len(known),candidate_edges=len(targets))),flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(grow,[(a,b,meta,str(args.transport),str(args.features),str(args.out)) for a,b in targets]):rows.append(row);done.add((row['a'],row['b']));print(json.dumps(row),flush=True)
        cycles=check_cycles(args.out,dict(pairs=rows,records=meta['records']));new=reachable(supported_graph(args.out,rows),seeds);joined=sorted(new-known);rounds.append(dict(round=iteration,candidate_edges=len(targets),joined=joined,anchored_captures=len(new),validated_directed_sites=cycles['validated_directed_sites']));print(json.dumps(rounds[-1]),flush=True);known=new
        if not joined:break
    cycles=check_cycles(args.out,dict(pairs=rows,records=meta['records']));final_graph=supported_graph(args.out,rows);unseen=set(range(len(meta['records'])));components=[]
    while unseen:
        group=reachable(final_graph,{min(unseen)});components.append(sorted(group));unseen-=group
    packet=dict(capture_components=sorted(components,key=lambda x:-len(x)),records=meta['records'],cameras=meta['cameras'],pairs=rows,cycles=cycles,attachment_rounds=rounds,anchored_captures=sorted(known),unresolved_captures=sorted(set(range(len(meta['records'])))-known),seconds=time.perf_counter()-start,
        policy=dict(minimum_proposal_seed_maps=8,minimum_closed_grid_sites=12,maximum_rounds=3,frontier='regional triangle containing at least one scene capture; no exclusive station assignment'),
        parent_dense_sha256=hashlib.sha256((args.base/'dense.json').read_bytes()).hexdigest(),parent_transport_sha256=hashlib.sha256((args.transport/'transport.json').read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('attach_fields.py','dense_transport.py','region_transport.py')})
    (args.out/'dense.json').write_text(json.dumps(packet,indent=2));print(json.dumps(dict(anchored=len(known),unresolved=packet['unresolved_captures'],pairs=len(rows),seconds=packet['seconds'])),flush=True)
if __name__=='__main__':main()
