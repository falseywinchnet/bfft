"""Grow competing image-verified regional maps without committing track identity."""
import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from .identity_study import feature
from .dense_transport import from_prior

POLICY=dict(step=12,margin=16,rounds=3,seed_distance=24.,alternatives=4,
    distinct_target_pixels=2.,distinct_jacobian=.15,
    selection='training correlation; distinct spatial hypotheses survive; no averaging')


def distinct(rows,maximum=4):
    kept=[]
    for row in sorted(rows,key=lambda r:-r['training']):
        if any(np.linalg.norm(row['q']-r['q'])<2 and np.linalg.norm(row['A']-r['A'])<.15 for r in kept):continue
        kept.append(row)
        if len(kept)>=maximum:break
    return kept


def grow_images(a,b,seeds,rounds=3):
    h,w=a.shape[:2];yy,xx=np.mgrid[16:h-16:12,16:w-16:12];grid=np.c_[xx.ravel(),yy.ravel()]
    bank={};history=[];frontier=seeds
    for iteration in range(rounds):
        if not frontier:break
        sp=np.array([s['p'] for s in frontier]);tree=cKDTree(sp);dd,kk=tree.query(grid,k=min(12,len(sp)),distance_upper_bound=24)
        dd=np.reshape(dd,(len(grid),-1));kk=np.reshape(kk,(len(grid),-1));proposals=[];sites=[]
        for site,(p,dist,ids) in enumerate(zip(grid,dd,kk)):
            choices=[]
            for d,k in zip(dist,ids):
                if not np.isfinite(d):continue
                seed=frontier[k];q=seed['q']+seed['A']@(p-seed['p'])
                choices.append(dict(p=p,q=q,A=seed['A'],training=seed['training'],root=seed['root']))
            for proposal in distinct(choices):
                # Previously fitted hypotheses remain in the bank. Only distinct
                # predictions spend a new image-fit evaluation.
                if any(np.linalg.norm(proposal['q']-old['q'])<1.5 and np.linalg.norm(proposal['A']-old['A'])<.12 for old in bank.get(site,[])):continue
                proposals.append(proposal);sites.append(site)
        if not proposals:break
        p=np.array([r['p'] for r in proposals]);q=np.array([r['q'] for r in proposals]);A=np.array([r['A'] for r in proposals]);f=from_prior(a,b,p,q,A)
        added=[]
        for k in np.flatnonzero(f['valid']):
            row=dict(p=p[k],q=f['target'][k],A=f['affine'][k],training=float(f['training'][k]),validation=float(f['correlation'][k]),root=proposals[k]['root'],site=sites[k])
            bank.setdefault(sites[k],[]).append(row);added.append(row)
        for site in bank:bank[site]=distinct(bank[site])
        frontier=[r for rows in bank.values() for r in rows]+seeds
        history.append(dict(round=iteration,attempted=len(proposals),passing=len(added),retained=sum(map(len,bank.values()))))
    rows=[r for site in sorted(bank) for r in bank[site]]
    return rows,history


def work(spec):
    a,b,seeds,files,out=spec;start=time.perf_counter();fa=feature(files[a]);fb=feature(files[b]);rows,history=grow_images(fa['lab'],fb['lab'],seeds)
    packet=dict(a=a,b=b,rows=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()} for row in rows],history=history,seconds=time.perf_counter()-start)
    path=Path(out)/f'{a:03d}-{b:03d}.json';path.write_text(json.dumps(packet))
    return dict(a=a,b=b,regions=len({r['site'] for r in rows}),hypotheses=len(rows),seconds=packet['seconds'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--identities',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if list(args.out.glob('*.json')):raise ValueError('Use a fresh output directory')
    start=time.perf_counter();records=json.loads((args.data/'manifest.json').read_text());files=[str(args.features/(Path(r['name']).stem+'.npz')) for r in records];edges=json.loads(args.identities.read_text());pairs={}
    for k,e in enumerate(edges):
        if not e['accepted']:continue
        a,b=e['a'][0],e['b'][0];row=dict(p=np.array(e['p']),q=np.array(e['q']),A=np.array(e['A']),training=e['train'],root=k)
        pairs.setdefault((a,b),[]).append(row)
        # Inversion is a proposal only; every grown reverse map is independently
        # fitted to the two original images before entering the field.
        inverse=dict(p=row['q'],q=row['p'],A=np.linalg.inv(row['A']),training=row['training'],root=k)
        pairs.setdefault((b,a),[]).append(inverse)
    # Both original directions retain their independently estimated seeds.
    source_names=['identity_fields.py','dense_transport.py','region_transport.py','identity_study.py']
    policy=dict(policy=POLICY,source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names},input_sha256=hashlib.sha256(args.identities.read_bytes()).hexdigest(),feature_hashes={r['name']:hashlib.sha256(Path(f).read_bytes()).hexdigest() for r,f in zip(records,files)})
    (args.out/'run-policy.json').write_text(json.dumps(policy,indent=2));rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work,[(a,b,s,files,str(args.out)) for (a,b),s in sorted(pairs.items())],chunksize=1):
            rows.append(row)
            if len(rows)%60==0:print(json.dumps(dict(done=len(rows),total=len(pairs),regions=sum(r['regions'] for r in rows),seconds=time.perf_counter()-start)),flush=True)
    summary=dict(pairs=rows,total_directed_regions=sum(r['regions'] for r in rows),hypotheses=sum(r['hypotheses'] for r in rows),captures=len({i for r in rows if r['regions'] for i in (r['a'],r['b'])}),seconds=time.perf_counter()-start,scene_promoted=False,limitations='Competing image-local affine maps. Neighboring samples and their descendants are correlated. No fixed tracks, common camera center, or scene geometry is inferred here.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='pairs'}),flush=True)
if __name__=='__main__':main()
