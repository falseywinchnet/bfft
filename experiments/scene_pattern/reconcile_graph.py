"""Reconcile neighboring map components, rechecking against accumulated structure."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .scene_growth import grow
from .merge_submaps import merge

EVIDENCE=None


def initialize(path):
    global EVIDENCE
    EVIDENCE=json.loads(Path(path).read_text())


def reconcile(evidence,submaps,rows):
    selected=min(rows,key=lambda r:(-len(r['cameras']),-r['promotion'].get('third_view_consistent',0),r['seed']))
    a,b=selected['seed'];base=json.loads((submaps/f'{a:03d}-{b:03d}.json').read_text())
    # Normalize structure support before the retention comparison. Raw seed
    # clouds may contain duplicate image-site proposals or tentative points.
    base=grow(evidence,base,maximum_rounds=0)
    result=merge(evidence,base,submaps,maximum_rounds=len(rows),allowed_seeds={tuple(r['seed']) for r in rows})
    universe=sorted({i for r in rows for i in r['cameras']})
    result['component_reconciliation']=dict(seed=selected['seed'],initial_cameras=len(base['poses']),initial_points=len(base['points']),
        candidate_seeds=[r['seed'] for r in rows],candidate_cameras=universe,unresolved_cameras=sorted(set(universe)-set(map(int,result['poses']))),
        policy='Start from maximum camera coverage, then consistent-point support, then seed ID. Recheck every merge against accumulated geometry; no frame placement follows from graph connectivity alone.')
    return result


def solve(spec):
    number,rows,submaps,out=spec;path=Path(out)/f'{number:03d}.json';start=time.perf_counter()
    result=reconcile(EVIDENCE,Path(submaps),rows);path.write_text(json.dumps(result,indent=2))
    return dict(component=number,seed=result['seed'],cameras=sorted(map(int,result['poses'])),points=len(result['points']),
                initial_cameras=result['component_reconciliation']['initial_cameras'],
                merged_seeds=result['submap_merge']['merged_seeds'],unresolved_cameras=result['component_reconciliation']['unresolved_cameras'],seconds=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--submaps',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);graph=json.loads(args.graph.read_text());start=time.perf_counter();rows=[]
    specs=[(k,[graph['maps'][i] for i in c['maps']],str(args.submaps),str(args.out)) for k,c in enumerate(graph['components'])]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(args.evidence),)) as pool:
        for row in pool.map(solve,specs):rows.append(row);print(json.dumps(row),flush=True)
    packet=dict(components=rows,seconds=time.perf_counter()-start,
                source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('reconcile_graph.py','merge_submaps.py','scene_growth.py','submap_alignment.py')},
                parent_graph_sha256=hashlib.sha256(args.graph.read_bytes()).hexdigest(),parent_evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
                limitations='Components retain independent frames. Individual component joins are rechecked against accumulated image evidence; no global fusion, physical depth calibration or visibility claim follows from their union.')
    (args.out/'components.json').write_text(json.dumps(packet,indent=2));print(json.dumps(dict(components=len(rows),largest_camera_count=max(len(r['cameras']) for r in rows),merged=sum(len(r['merged_seeds']) for r in rows),seconds=packet['seconds'])),flush=True)


if __name__=='__main__':main()
