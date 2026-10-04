"""Bounded all-capture identity search without inherited correspondences or poses."""
import argparse,hashlib,json,time
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .identity_register import POLICY,texture_coordinates,signature,trace_peaks,verify,rerank
from .radial import unwind

@lru_cache(maxsize=8)
def feature(path):
    with np.load(path) as f:return {k:f[k] for k in ('lab','texture','points')}


def select_sites(f,maximum):
    p=np.unique(np.round(f['points'],4),axis=0);h,w=f['lab'].shape[:2]
    p=p[((p>27)&(p<np.array([w,h])-28)).all(1)]
    if not len(p):return p
    # Balance image coverage using cells, with texture energy selecting a region
    # center inside each cell. Sites still originate in the native segmentation.
    energy=ndimage.gaussian_filter(np.sum(f['texture']**2,axis=-1),4)
    values=ndimage.map_coordinates(energy,[p[:,1],p[:,0]],order=1)
    aspect=w/h;nx=max(1,int(np.sqrt(maximum*aspect)));ny=max(1,int(np.ceil(maximum/nx)))
    cells=np.minimum((p/np.array([w,h])*[nx,ny]).astype(int),[nx-1,ny-1]);chosen={}
    for i in np.argsort(-values,kind='stable'):chosen.setdefault(tuple(cells[i]),int(i))
    ids=list(chosen.values())[:maximum]
    return p[ids]


def retrieve(traces,nodes,radii):
    descriptors=np.array([signature(t) for t in traces]);mean=descriptors.mean(0)
    # PCA is only a retrieval accelerator. Full descriptor distance re-ranks its
    # neighbors; no identity or geometry is decided by a hash collision.
    _,_,vt=np.linalg.svd(descriptors-mean,full_matrices=False);projection=vt[:20]
    reduced=(descriptors-mean)@projection.T;tree=cKDTree(reduced)
    _,neighbors=tree.query(reduced,k=min(80,len(nodes)),workers=1)
    alternatives={}
    for i,js in enumerate(neighbors):
        src=tuple(map(int,nodes[i]));best={}
        for j in js:
            dst=tuple(map(int,nodes[j]))
            if src[0]==dst[0]:continue
            distance=float(np.linalg.norm(descriptors[i]-descriptors[j]))
            key=(src,dst);row=dict(a=list(src),b=list(dst),distance=distance,trace_a=int(i),trace_b=int(j),radius_ratio=float(radii[j]/radii[i]))
            if key not in best or distance<best[key]['distance']:best[key]=row
        for key,row in best.items():
            if key not in alternatives or row['distance']<alternatives[key]['distance']:alternatives[key]=row
    bysite={}
    for row in alternatives.values():bysite.setdefault(tuple(row['a']),[]).append(row)
    return [r for rows in bysite.values() for r in sorted(rows,key=lambda r:(r['distance'],r['b']))[:POLICY['retrieval_neighbors']]]


def work(spec):
    row,files,points,matrices=spec
    a,b=row['a'],row['b'];fa=feature(files[0]);fb=feature(files[1])
    trials=verify(fa['lab'],fb['lab'],fa['texture'][...,0],fb['texture'][...,0],points[0],points[1],matrices)
    # Keep distinct successful transforms, but cap duplicate starts that arrive
    # at the same local solution. Selection uses training correlation only.
    kept=[]
    for t in sorted(trials,key=lambda x:-x['train']):
        if not t['accepted']:continue
        if any(np.linalg.norm(np.array(t['q'])-k['q'])<1 and np.linalg.norm(np.array(t['A'])-k['A'])<.1 for k in kept):continue
        kept.append(t)
    if not kept:kept=[max(trials,key=lambda x:x['train'])]
    return dict(candidate=row,trials=len(trials),phase_accepted=any(t['accepted'] and t['initialization']=='phase' for t in trials),
        ringing_accepted=any(t['accepted'] and t['initialization']=='ringing' for t in trials),
        edges=[dict(**t,a=a,b=b) for t in kept])


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--verify-per-capture',type=int,default=24);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'summary.json').exists() or (args.out/'verification.jsonl').exists():raise ValueError('Use a fresh output directory; receipts must not mix runs')
    start=time.perf_counter();records=json.loads((args.data/'manifest.json').read_text());traces=[];nodes=[];radii=[];sites=[];files=[];hashes={}
    for i,r in enumerate(records):
        path=args.features/(Path(r['name']).stem+'.npz');files.append(str(path));hashes[r['name']]=hashlib.sha256(path.read_bytes()).hexdigest();f=feature(str(path));points=select_sites(f,POLICY['sites_per_capture']);sites.append(points)
        channels=texture_coordinates(f['lab'],f['texture'])
        for k,xy in enumerate(points):
            for radius in POLICY['radii']:
                trace,valid,meta=unwind(channels,xy,r_min=radius/4,r_max=radius,radial_count=16,angular_count=48)
                if not valid.all():raise ValueError('Invalid annular support')
                traces.append(trace.astype(np.float32));nodes.append((i,k));radii.append(radius)
    traces=np.asarray(traces);nodes=np.asarray(nodes);radii=np.asarray(radii)
    print(json.dumps(dict(stage='index',captures=len(records),sites=sum(map(len,sites)),traces=len(traces),seconds=time.perf_counter()-start)),flush=True)
    candidates=retrieve(traces,nodes,radii);(args.out/'candidates.json').write_text(json.dumps(candidates))
    (args.out/'sites.json').write_text(json.dumps([s.tolist() for s in sites]));jobs=[]
    for i in range(len(records)):
        # Verify all retrieved identities at a selected site before committing any.
        groups={}
        for r in candidates:
            if r['a'][0]==i:groups.setdefault(tuple(r['a']),[]).append(r)
        order=sorted(groups,key=lambda key:min(x['distance'] for x in groups[key]));selected=[]
        for key in order:
            selected.extend(sorted(groups[key],key=lambda x:x['distance']))
            if len(selected)>=args.verify_per_capture:break
        for row in selected[:args.verify_per_capture]:
            peaks=trace_peaks(traces[row['trace_a']],traces[row['trace_b']],np.log(4)/15);matrices=[]
            for peak in peaks:
                angle=peak['angle'];scale=peak['scale']*row['radius_ratio'];matrices.append(scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]))
            a,b=row['a'],row['b'];jobs.append((row,[files[a[0]],files[b[0]]],[sites[a[0]][a[1]],sites[b[0]][b[1]]],matrices))
    source_names=['identity_study.py','identity_register.py','radial.py','region_transport.py','transport_tracks.py']
    run=dict(policy=POLICY,verification_budget_per_capture=args.verify_per_capture,feature_hashes=hashes,
        source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names})
    (args.out/'run-policy.json').write_text(json.dumps(run,indent=2));print(json.dumps(dict(stage='retrieved',candidates=len(candidates),scheduled=len(jobs),seconds=time.perf_counter()-start)),flush=True)
    rows=[]
    with (args.out/'verification.jsonl').open('w') as ledger,ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work,jobs,chunksize=4):
            ledger.write(json.dumps(row)+'\n');ledger.flush();rows.append(row)
            if len(rows)%120==0:print(json.dumps(dict(stage='verification',done=len(rows),accepted=sum(any(e['accepted'] for e in r['edges']) for r in rows),seconds=time.perf_counter()-start)),flush=True)
    edges=rerank([e for row in rows for e in row['edges']]);(args.out/'identities.json').write_text(json.dumps(edges))
    accepted=[e for e in edges if e['accepted']]
    summary=dict(captures=len(records),sites=sum(map(len,sites)),retrieved_candidates=len(candidates),verified_candidates=len(rows),
        accepted_candidates=sum(any(e['accepted'] for e in r['edges']) for r in rows),accepted_transforms=len(accepted),
        captures_with_pair_evidence=len({i for e in accepted for i in (e['a'][0],e['b'][0])}),
        transforms_with_third_capture=sum(bool(e['third_capture_support']) for e in accepted),
        transforms_with_two_third_captures=sum(len(e['third_capture_support'])>=2 for e in accepted),
        phase_only_accepted=sum(r['phase_accepted'] for r in rows),ringing_accepted=sum(r['ringing_accepted'] for r in rows),
        ringing_added_candidates=sum(r['ringing_accepted'] and not r['phase_accepted'] for r in rows),
        seconds=time.perf_counter()-start,scene_promoted=False,
        limitations='Bounded annular identity retrieval and local affine verification. Relative weights are heuristic, not calibrated probabilities. Triangle closure cannot establish physical identity under repeated scenery. No station assignment, camera update, depth or fused scene is certified. Unverified candidates remain in the ledger.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
