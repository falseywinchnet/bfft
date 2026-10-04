"""Alternating regional identity / relative-camera hypotheses with blocked checks."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .affine_camera_geometry import measurements,essential_linear,homography_linear,homography_poses,skew,epi_error,refine_essential,rotation_error,fit_rotation
from .multiview_geometry import essential_poses
from .scene_evidence import camera_models

POLICY=dict(version=1,trials=100,retained_models_per_kind=4,alternating_rounds=2,
    maximum_error_pixels=1.5,tile=48,training_guard_radius=14,
    minimum_training_sites=8,minimum_training_tiles=3,minimum_check_sites=4,
    minimum_check_tiles=2,minimum_check_fraction=.6,
    holdout='source image tiles; guard excludes training footprints touching check tiles',
    evidence='conditional geometry prediction on already image-matched regions; not unseen-image validation')


def tile_id(x,y,capture):return (int(x)*73856093+int(y)*19349663+int(capture)*83492791)%5


def roles(rows,capture):
    train=[];held=[];blocks=[]
    for r in rows:
        p=np.asarray(r['p']);cell=np.floor(p/48).astype(int);blocks.append(tuple(map(int,cell)));check=tile_id(*cell,capture)==0;held.append(check)
        lo=np.floor((p-14)/48).astype(int);hi=np.floor((p+14)/48).astype(int)
        train.append(not any(tile_id(x,y,capture)==0 for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1)))
    return np.array(train),np.array(held),blocks


def select(error,eligible,rows,threshold=1.5):
    used_a=set();used_b=set();good=[]
    for k in sorted(eligible,key=lambda k:(float(error[k])+.15*(1-rows[k]['training']),k)):
        if error[k]>threshold:continue
        a=rows[k]['site'];b=tuple(np.round(np.asarray(rows[k]['q'])/2).astype(int))
        if a in used_a or b in used_b:continue
        used_a.add(a);used_b.add(b);good.append(k)
    return np.array(good,int)


def fit_pair(rows,cameras,capture=0,trials=100,split=None,prior_models=()):
    if not rows:return dict(models=[],reason='no_image_maps',updates=[])
    m=measurements(rows,cameras);training,validation,blocks=roles(rows,capture) if split is None else split
    train=np.flatnonzero(training);held=np.flatnonzero(validation);rng=np.random.default_rng(107)
    groups={}
    for k in train:groups.setdefault(rows[k]['site'],[]).append(int(k))
    if len(groups)<3:return dict(models=[],reason='fewer_than_three_training_regions',updates=[],training_sites=len(groups))
    sites=sorted(groups);pools={'rotation':[],'essential':[]}
    def err(kind,M):return rotation_error(m,M) if kind=='rotation' else epi_error(m,M)
    def score(error):
        good=select(error,train,rows);return (len({blocks[k] for k in good}),len(good),-float(np.median(error[good])) if len(good) else -1e10),good
    def insert(kind,M,origin):
        if not np.isfinite(M).all():return
        errors=err(kind,M);value,good=score(errors)
        if len(good)<3:return
        normalized=M/max(np.linalg.norm(M),1e-12)
        for old in pools[kind]:
            other=old['matrix']/max(np.linalg.norm(old['matrix']),1e-12)
            distance=min(np.linalg.norm(normalized-other),np.linalg.norm(normalized+other)) if kind=='essential' else np.linalg.norm(normalized-other)
            if distance<.02:
                if value>old['score']:old.update(matrix=M,score=value,good=good,origin=origin)
                return
        pools[kind].append(dict(matrix=M,score=value,good=good,origin=origin));pools[kind].sort(key=lambda x:x['score'],reverse=True);del pools[kind][4:]
    for trial in range(trials):
        sample_sites=rng.choice(sites,3,replace=False);ids=[]
        for s in sample_sites:
            candidates=groups[int(s)]
            ids.append(max(candidates,key=lambda k:rows[k]['training']) if trial%2==0 else int(rng.choice(candidates)))
        insert('rotation',fit_rotation(m,ids),'regional_differentials')
        try:insert('essential',essential_linear(m,ids),'three_affine_regions')
        except (ValueError,np.linalg.LinAlgError):pass
        try:
            H=homography_linear(m,ids)
            for pose in homography_poses(H)[::2]:insert('essential',skew(pose['translation'])@pose['rotation'],'planar_initialization_only')
        except (ValueError,np.linalg.LinAlgError):pass
    # Replaying a training-selected branch is not a hard constraint. It is
    # refitted and checked against the new fields. Import ALL previous models,
    # including rejected ones, so past holdout outcomes cannot select seeds.
    for prior in prior_models:
        kind=prior['kind'];M=np.asarray(prior['matrix']);value,good=score(err(kind,M))
        if len(good)<3:continue
        if any(np.allclose(M,old['matrix'],atol=1e-12) for old in pools[kind]):continue
        pools[kind].append(dict(matrix=M,score=value,good=good,origin='replayed_training_hypothesis'))
    result=[]
    for kind,pool in pools.items():
        for candidate in sorted(pool,key=lambda c:c['score'],reverse=True):
            M=candidate['matrix'];initial_M=M.copy();history=[]
            for iteration in range(2):
                before=select(err(kind,M),train,rows)
                if len(before)<3:break
                proposed=fit_rotation(m,before) if kind=='rotation' else refine_essential(m,before,M)
                old_score,_=score(err(kind,M));new_score,after=score(err(kind,proposed))
                if new_score>=old_score:M=proposed
                chosen=select(err(kind,M),train,rows);history.append(dict(iteration=iteration,training_choices_changed=len(set(before)^set(chosen)),training_sites=len(chosen)))
            # Pair-local improvement may destroy a globally useful mode. Both
            # the starting proposal and its refined descendant can reach the
            # multi-camera optimizer; neither is held fixed there.
            stages=[('pair_refined',M)] if np.allclose(initial_M,M,rtol=1e-10,atol=1e-12) else [('proposal',initial_M),('pair_refined',M)]
            for stage,M in stages:
                errors=err(kind,M);good=select(errors,train,rows);check=select(errors,held,rows);nt=len({rows[k]['site'] for k in held})
                train_blocks=len({blocks[k] for k in good});check_blocks=len({blocks[k] for k in check})
                accepted=len(good)>=8 and train_blocks>=3 and len(check)>=4 and check_blocks>=2 and len(check)>=.6*nt
                record=dict(stage=stage,kind=kind,matrix=M.tolist(),origin=candidate['origin'],accepted=bool(accepted),train_count=len(good),train_tiles=train_blocks,check_count=len(check),check_tiles=check_blocks,check_total=nt,
                    train_median=float(np.median(errors[good])) if len(good) else None,check_median=float(np.median(errors[check])) if len(check) else None,training_inliers=good.tolist(),check_inliers=check.tolist(),iterations=history if stage=='pair_refined' else [])
                if kind=='essential' and len(good):
                    poses=essential_poses(M,m['a'][good],m['b'][good]);record['poses']=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items() if k not in ('points','positive')} for p in poses]
                    record['translation_observed']=bool(poses[0]['positive_fraction']>=.8 and np.rad2deg(poses[0]['median_parallax'])>=.7)
                result.append(record)
    # Geometry feedback remains reversible. Every original image hypothesis is
    # scored, including losers; no one-way identity union modifies the inputs.
    accepted=[r for r in result if r['accepted']];updates=[]
    if accepted:
        weights=np.array([r['train_count'] for r in accepted],float);weights/=weights.sum()
        errors=np.array([err(r['kind'],np.array(r['matrix'])) for r in accepted]);likelihood=np.sum(weights[:,None]*np.exp(-.5*np.minimum(errors/1.5,10)**2),axis=0)
        bysite={}
        for k,row in enumerate(rows):bysite.setdefault(row['site'],[]).append(k)
        for site,ids in bysite.items():
            old=max(ids,key=lambda k:rows[k]['training']);value={k:8*(rows[k]['training']-.78)+np.log(max(float(likelihood[k]),1e-12)) for k in ids};new=max(ids,key=lambda k:value[k])
            updates.append(dict(site=site,before=old,after=new,changed=old!=new,scores=[dict(row=k,score=value[k],geometric_likelihood=float(likelihood[k])) for k in ids]))
    return dict(models=result,updates=updates,training_sites=len(groups),check_sites=len({rows[k]['site'] for k in held}),guarded_rows=int((~training&~validation).sum()),input_rows=len(rows),accepted_models=len(accepted),scene_promoted=False)


def work(spec):
    path,cameras,out,prior_folder=spec;data=json.loads(Path(path).read_text());start=time.perf_counter();a,b=data['a'],data['b'];prior_path=Path(prior_folder)/Path(path).name if prior_folder else None
    prior=json.loads(prior_path.read_text())['models'] if prior_path and prior_path.exists() else []
    result=fit_pair(data['rows'],[cameras[a],cameras[b]],a,prior_models=prior)
    result.update(a=a,b=b,seconds=time.perf_counter()-start);(Path(out)/Path(path).name).write_text(json.dumps(result));return dict(a=a,b=b,models=len(result['models']),accepted=result.get('accepted_models',0),identity_changes=sum(r['changed'] for r in result['updates']),seconds=result['seconds'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--fields',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--prior-geometry',type=Path);p.add_argument('--scope',choices=['all','panoramas'],default='all');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if list(args.out.glob('*.json')):raise ValueError('Use a fresh output directory')
    records=json.loads((args.data/'manifest.json').read_text());metadata=json.loads((args.data/'camera-metadata.json').read_text());features=[]
    for r in records:
        with np.load(args.features/(Path(r['name']).stem+'.npz')) as f:features.append(dict(shape=f['shape']))
    cameras=camera_models(records,metadata,features);paths=sorted(args.fields.glob('[0-9][0-9][0-9]-[0-9][0-9][0-9].json'));start=time.perf_counter();rows=[]
    if args.scope=='panoramas':paths=[f for f in paths if all(records[int(i)]['panoramic'] for i in f.stem.split('-'))]
    sources=['identity_geometry.py','affine_camera_geometry.py','multiview_geometry.py','scene_evidence.py'];policy=dict(policy=POLICY,scope=args.scope,source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources},parent_summary_sha256=hashlib.sha256((args.fields/'summary.json').read_bytes()).hexdigest(),field_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},records=records,cameras=cameras)
    if args.prior_geometry:
        prior_policy=json.loads((args.prior_geometry/'run-policy.json').read_text())
        if prior_policy['cameras']!=cameras or prior_policy['records']!=records:raise ValueError('Prior camera calibration/capture identity differs')
        for key in ('tile','training_guard_radius','holdout'):
            if prior_policy['policy'][key]!=POLICY[key]:raise ValueError('Prior training/holdout roles differ')
        policy['prior_geometry']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.prior_geometry.glob('*.json'))}
    (args.out/'run-policy.json').write_text(json.dumps(policy,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work,[(str(f),cameras,str(args.out),str(args.prior_geometry) if args.prior_geometry else None) for f in paths],chunksize=1):
            rows.append(row)
            if len(rows)%60==0:print(json.dumps(dict(done=len(rows),accepted_pairs=sum(r['accepted']>0 for r in rows),identity_changes=sum(r['identity_changes'] for r in rows),seconds=time.perf_counter()-start)),flush=True)
    summary=dict(pairs=rows,accepted_directed_pairs=sum(r['accepted']>0 for r in rows),identity_changes=sum(r['identity_changes'] for r in rows),seconds=time.perf_counter()-start,scene_promoted=False,limitations='Multiple conditional relative-camera hypotheses with guarded source-tile checks. Image discovery used the entire pair. Geometry scores revise region choices but do not yet constitute a global scene or station membership.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='pairs'}),flush=True)
if __name__=='__main__':main()
