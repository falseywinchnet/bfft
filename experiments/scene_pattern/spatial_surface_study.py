"""Matched spatially blocked prediction of a shared differential surface field.

Camera poses and discovered image maps are conditional inputs. Every observation
center is reserved in exactly one fold. Entire affine footprints plus a two-pixel
guard are excluded from training when they touch a reserved tile.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .surface_jets import fit_jet,image_rays,project,is_training,is_validation
from .jet_bundle import refine_jets


FOLDS=4
TILE=32
GUARD=2.


def tile_fold(capture,x,y):
    key=f'{capture}:{int(x)}:{int(y)}'.encode()
    return int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')%FOLDS


def mark_observation(o,radius,fold):
    o=copy.deepcopy(o);xy=np.asarray(o['xy']);extent=radius*np.abs(o['affine']).sum(axis=1)+GUARD
    lo=np.floor((xy-extent)/TILE).astype(int);hi=np.floor((xy+extent)/TILE).astype(int)
    held=tile_fold(o['capture'],*np.floor(xy/TILE).astype(int))==fold
    touches=any(tile_fold(o['capture'],x,y)==fold for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1))
    o.update(training=not touches,validation=held,guarded=touches and not held)
    return o


def roles(packet,fold):
    result=[]
    for original in packet['results']:
        if 'observations' not in original:continue
        j=copy.deepcopy(original);j['observations']=[mark_observation(o,j['radius'],fold) for o in j['observations']];j['holdout_capture']=None;result.append(j)
    return result


def initialize(marked,pose,cameras):
    poses={int(i):p for i,p in pose['poses'].items()};rows=[];unsupported=[];ranges=[];pending=[]
    for j in marked:
        train=[o for o in j['observations'] if is_training(j,o)]
        if len(train)<2:unsupported.append(j['track']);continue
        fitted=fit_jet(j['observations'],cameras,poses,np.zeros(3),radius=j['radius'])
        if 'center' in fitted and fitted['training']['positive']:
            row=dict(track=j['track'],**fitted);rows.append(row)
            if fitted['training']['maximum_pixels']<=1.5:
                ranges.extend(np.linalg.norm(np.asarray(row['center'])-np.asarray(poses[o['capture']]['center'])) for o in train)
        else:pending.append(j)
    # This initialization is derived only from training fits and fixed cameras.
    # It supplies a starting point, not evidence of finite depth for weak patches.
    baseline=np.linalg.norm(np.asarray(poses[pose['seed'][0]]['center'])-poses[pose['seed'][1]]['center'])
    distance=float(np.median(ranges)) if ranges else max(float(baseline),1e-3)
    for j in pending:
        o=next(o for o in j['observations'] if is_training(j,o));i=o['capture'];xy=np.asarray(o['xy']);affine=np.asarray(o['affine']);rotation=np.asarray(poses[i]['rotation'])
        def ray(q):return image_rays([q],cameras[i])[0]@rotation
        center=np.asarray(poses[i]['center'])+distance*ray(xy)
        tangent=np.array([(ray(xy+affine[:,axis]*.001)-ray(xy-affine[:,axis]*.001))*(distance/.002) for axis in range(2)])
        row=copy.deepcopy(j);row.update(center=center.tolist(),tangents=tangent.tolist(),initialization='training_range_guess',training=dict(positive=True,maximum_pixels=1e99));rows.append(row)
    rows.sort(key=lambda j:j['track'])
    return dict(results=rows,retain_training_failures=True),dict(input_patches=len(marked),fitted_patches=len(rows),unsupported_tracks=unsupported,range_initializations=len(pending),training_range_median=distance)


def predict(marked,pose,cameras):
    solved={j['track']:j for j in pose.get('surface_jets',[])};result=[];delta=np.array([(u,v) for v in (-1.,0.,1.) for u in (-1.,0.,1.)])
    for j in marked:
        for o in j['observations']:
            if not is_validation(j,o):continue
            record=dict(track=j['track'],capture=o['capture'])
            if j['track'] not in solved:record.update(resolved=False,reason='fewer_than_two_training_views');result.append(record);continue
            s=solved[j['track']];world=np.asarray(s['center'])+s['radius']*delta@np.asarray(s['tangents']);camera=cameras[o['capture']];p=pose['poses'][str(o['capture'])]
            expected=np.asarray(o['xy'])+s['radius']*delta@np.asarray(o['affine']).T;error=project(world,camera,p)-expected
            if camera['projection']=='cylindrical':
                period=2*np.pi*camera.get('focal_x',camera['focal']);error[:,0]=(error[:,0]+period/2)%period-period/2
            q=world@np.asarray(p['rotation']).T+p['translation'];positive=bool(np.all(np.sum(q*image_rays(expected,camera),axis=1)>0));values=np.linalg.norm(error,axis=1)
            record.update(resolved=True,errors=values.tolist(),positive=positive,passed=positive and bool(values.max()<=1.5));result.append(record)
    return result


def stats(rows):
    solved=[r for r in rows if r['resolved']];errors=[v for r in solved for v in r['errors']]
    return dict(observations=len(rows),resolved=len(solved),unresolved=len(rows)-len(solved),passed=sum(r['passed'] for r in solved),nonpositive=sum(not r['positive'] for r in solved),median_pixels=float(np.median(errors)) if errors else None,p90_pixels=float(np.quantile(errors,.9)) if errors else None)


def run_branch(args):
    paths,folder,fold,coherent,maximum,joint=args;start=time.perf_counter();e,pose,packet=[json.loads(Path(p).read_text()) for p in paths];cameras=pose.get('camera_models',e['cameras'])
    marked=roles(packet,fold);prepared,initialization=initialize(marked,pose,cameras)
    result,receipt,candidate=refine_jets(prepared,pose,cameras,max_evaluations=maximum,coherent_surfaces=coherent,freeze_cameras=not joint,protect_cameras=True)
    if candidate is None:raise ValueError('Insufficient prepared surface field')
    predictions=predict(marked,candidate,cameras);name=f'{fold}-'+('coherent' if coherent else 'control');folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    receipt.update(initialization=initialization,fold=fold,coherent=coherent,joint_cameras=joint,conditional_spatial_prediction=True,policy=('Joint camera poses; ' if joint else 'Fixed camera poses; ')+'All patches with two training observations retained regardless of training fit. Four deterministic 32-pixel camera/image tile folds, full affine footprint plus two-pixel guard; each observation center scored once across folds. No additional point constraints in this study.',statistics=stats(predictions),wall_seconds=time.perf_counter()-start,
        sources_sha256={n:hashlib.sha256(Path(__file__).with_name(n+'.py').read_bytes()).hexdigest() for n in ('spatial_surface_study','surface_jets','jet_bundle','surface_coherence')},inputs_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    for suffix,value in [('.candidate',candidate),('.retained',result),('.receipt',receipt),('.predictions',predictions)]:
        (folder/(name+suffix+'.json')).write_text(json.dumps(value))
    print(json.dumps(dict(branch=name,statistics=receipt['statistics'],seconds=receipt['wall_seconds'],evaluations=receipt['evaluations'],status=receipt['status'])),flush=True)
    return name


def aggregate(folder,packet):
    expected={(j['track'],o['capture']) for j in packet['results'] if 'observations' in j for o in j['observations']};summary={};sets=[]
    for name in ('control','coherent'):
        rows=[r for fold in range(FOLDS) for r in json.loads((folder/f'{fold}-{name}.predictions.json').read_text())]
        counts=Counter((r['track'],r['capture']) for r in rows)
        if set(counts)!=expected or any(n!=1 for n in counts.values()):raise ValueError('Spatial folds do not score every original observation exactly once')
        summary[name]=dict(overall=stats(rows),by_camera={str(i):stats([r for r in rows if r['capture']==i]) for i in sorted({r['capture'] for r in rows})});sets.append({(r['track'],r['capture']) for r in rows if r['resolved']})
    if sets[0]!=sets[1]:raise ValueError('Control and coherent branch have different predictive support')
    summary['policy']='Matched conditional prediction across all four folds. Unresolved observations stay in coverage counts; residual quantiles describe resolved observations only. Existing camera poses and discovered image maps are not independently held out. No old acceptance decision is replaced.'
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser()
    for n in ('evidence','pose','jets','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--joint-cameras',action='store_true');p.add_argument('--max-evaluations',type=int,default=300);p.add_argument('--workers',type=int,default=2);a=p.parse_args();paths=[a.evidence,a.pose,a.jets]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_branch,[(paths,a.out,k,c,a.max_evaluations,a.joint_cameras) for k in range(FOLDS) for c in (False,True)]))
    summary=aggregate(Path(a.out),json.loads(Path(a.jets).read_text()));print(json.dumps({n:summary[n]['overall'] for n in ('control','coherent')}))


if __name__=='__main__':main()
