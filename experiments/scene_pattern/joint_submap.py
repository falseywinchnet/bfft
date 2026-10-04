"""Joint candidate refinement for map relations that cannot be rigidly joined.

A failed frame relation supplies initialization only. One variable per shared
camera ties the maps; prediction and per-camera regression checks decide whether
any output can replace the retained hypothesis.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .submap_alignment import transfer_pose
from .scene_bundle import refine
from .scene_pose import track_bearing
from .multiview_geometry import unit


def initialize_union(target,source,relation):
    result=copy.deepcopy(target);new=set(source['poses'])-set(target['poses'])
    for i in sorted(new,key=int):result['poses'][i]=transfer_pose(source['poses'][i],relation)
    existing={p['track'] for p in result['points']};s=relation['scale'];r=np.asarray(relation['rotation']);t=np.asarray(relation['translation'])
    for point in source['points']:
        if point['track'] in existing or point.get('promotion')!='third_view_consistent':continue
        row=copy.deepcopy(point);row['xyz']=(s*r@np.asarray(point['xyz'])+t).tolist();result['points'].append(row)
    return result


def heldout_errors(evidence,pose,tracks=None,cameras=None):
    lookup={t['id']:t for t in evidence['tracks']};models=pose.get('camera_models',evidence['cameras']);rows={};used=set()
    for point in sorted(pose['points'],key=lambda p:p['track']):
        if tracks is not None and point['track'] not in tracks:continue
        for obs in lookup[point['track']]['observations']:
            i=obs['capture'];key=(i,obs['site'])
            if str(i) not in pose['poses'] or obs['site']%7!=0 or key in used or (cameras is not None and i not in cameras):continue
            used.add(key);p=pose['poses'][str(i)];b=track_bearing(obs,models);q=unit(np.asarray(p['rotation'])@point['xyz']+p['translation'])
            error=float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(q,b)),q@b)))
            rows.setdefault(i,[]).append(error)
    return rows


def refresh_training_points(evidence,pose):
    """Refit every retained point in the final cameras, excluding holdout rays.

Joint fitting caps duplicate image-site votes and can omit points entirely.
Those points must not silently retain coordinates from before the cameras moved.
"""
    result=copy.deepcopy(pose);lookup={t['id']:t for t in evidence['tracks']};models=pose.get('camera_models',evidence['cameras']);refreshed=0;unsupported=[]
    for point in result['points']:
        obs=[o for o in lookup[point['track']]['observations'] if str(o['capture']) in pose['poses'] and o['site']%7!=0]
        if len(obs)<2:unsupported.append(point['track']);continue
        centers=np.array([pose['poses'][str(o['capture'])]['center'] for o in obs])
        rays=np.array([track_bearing(o,models)@np.asarray(pose['poses'][str(o['capture'])]['rotation']) for o in obs])
        projectors=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
        matrix=projectors.sum(0)
        if np.linalg.cond(matrix)>1e5:unsupported.append(point['track']);continue
        point['xyz']=np.linalg.solve(matrix,np.einsum('nij,nj->i',projectors,centers)).tolist();refreshed+=1
    result['training_point_refresh']=dict(refreshed=refreshed,unsupported_tracks=unsupported,policy='Final camera poses; only site-modulo-seven training rays; ill-conditioned or one-ray points retain their previous coordinates, are recorded as unsupported, and remain in the prediction gate.')
    return result


def validate(evidence,before,candidate):
    original=set(map(int,before['poses']));new=set(map(int,candidate['poses']))-original
    ids={p['track'] for p in before['points']};base=heldout_errors(evidence,before,tracks=ids,cameras=original);after=heldout_errors(evidence,candidate,tracks=ids,cameras=original)
    checks=[]
    for i in sorted(original):
        a=np.asarray(base.get(i,[]));b=np.asarray(after.get(i,[]));f=before.get('camera_models',evidence['cameras'])[i]['focal']
        if len(a)<4 or len(a)!=len(b):checks.append(dict(capture=i,accepted=False,reason='insufficient_or_changed_validation_support'));continue
        ma,mb=float(np.median(a)),float(np.median(b));qa,qb=float(np.quantile(a,.9)),float(np.quantile(b,.9))
        checks.append(dict(capture=i,heldout_sites=len(a),before_median_degrees=ma,after_median_degrees=mb,before_p90_degrees=qa,after_p90_degrees=qb,
                           accepted=bool(mb<=ma*1.1+np.rad2deg(.1/f) and qb<=qa*1.2+np.rad2deg(.2/f))))
    novel=heldout_errors(evidence,candidate,cameras=new);new_checks=[]
    for i in sorted(new):
        errors=np.asarray(novel.get(i,[]));agree=int(np.sum(errors<=1.2))
        new_checks.append(dict(capture=i,heldout_sites=len(errors),agree=agree,median_degrees=float(np.median(errors)) if len(errors) else None,accepted=bool(agree>=4 and agree>=.6*len(errors))))
    return dict(accepted=bool(new_checks and all(r['accepted'] for r in checks+new_checks)),existing_cameras=checks,new_cameras=new_checks,
                policy='Original track support retained for old-camera holdouts: median <= 1.1× baseline + 0.1 bearing-equivalent pixel, p90 <= 1.2× + 0.2 pixel. New cameras require four held-out sites and 60% within 1.2 degrees. Site modulo seven excluded from joint fit.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--submaps',type=Path,required=True);p.add_argument('--joins',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-evaluations',type=int,default=300);args=p.parse_args()
    e=json.loads(args.evidence.read_text());base=json.loads(args.pose.read_text());attempts=json.loads(args.joins.read_text())['submap_merge']['attempts'];lookup={t['id']:t for t in e['tracks']};base_tracks={p['track'] for p in base['points']};choices=[]
    for a in attempts:
        if a.get('round',0)!=0 or 'scale' not in a['alignment'] or a['alignment'].get('training_inliers',0)<20:continue
        x,y=a['seed'];source=json.loads((args.submaps/f'{x:03d}-{y:03d}.json').read_text());shared=base_tracks&{p['track'] for p in source['points'] if p.get('promotion')=='third_view_consistent'};new=set(map(int,source['poses']))-set(map(int,base['poses']))
        votes=0
        for point in source['points']:
            if point.get('promotion')!='third_view_consistent':continue
            obs=[o for o in lookup[point['track']]['observations'] if o['site']%7!=0]
            if len({o['capture'] for o in obs if str(o['capture']) in base['poses']})>=2:
                votes+=sum(o['capture'] in new for o in obs)
        choices.append((votes,a['alignment']['training_inliers'],tuple(-i for i in a['seed']),a,source))
    if not choices:raise RuntimeError('No training-supported frame initializer with new observations')
    votes,_,_,chosen,source=max(choices,key=lambda r:r[:3]);initial=initialize_union(base,source,chosen['alignment'])
    print(json.dumps(dict(seed=chosen['seed'],new_cameras=len(initial['poses'])-len(base['poses']),initial_points=len(initial['points']),training_link_observations=votes)),flush=True)
    effective=dict(e,cameras=base.get('camera_models',e['cameras']));result,receipt=refine(effective,initial,max_evaluations=args.max_evaluations)
    candidate=result.pop('bundle_candidate',None)
    if candidate is None:candidate=result
    from .scene_promotion import audit
    candidate=audit(effective,refresh_training_points(effective,candidate))
    gate=validate(effective,base,candidate);accepted=receipt['applied'] and gate['accepted']
    review=dict(accepted=bool(accepted),source_seed=chosen['seed'],training_link_observations=votes,initializer=chosen['alignment'],bundle=receipt,prediction_gate=gate,
                limitations='One automatically selected initialization, ranked by new training observations on tracks seen by at least two existing training cameras. Bundle metrics precede the training-only point refresh; per-camera gates follow it. Rejected frame is an optimizer starting point only. Collection-derived correspondences and starting maps are not unseen-scene validation.',
                source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('joint_submap.py','scene_bundle.py','submap_alignment.py')},parent_pose_sha256=hashlib.sha256(args.pose.read_bytes()).hexdigest())
    candidate['joint_map']=review;retained=candidate if accepted else copy.deepcopy(base);retained['joint_map']=review
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(retained,indent=2));args.out.with_suffix('.candidate.json').write_text(json.dumps(candidate,indent=2));print(json.dumps(review),flush=True)


if __name__=='__main__':main()
