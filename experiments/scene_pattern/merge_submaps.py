"""Reconcile independently seeded camera maps without overwriting established poses."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .submap_alignment import align,transfer_pose
from .scene_pose import track_bearing
from .multiview_geometry import unit
from .scene_growth import triangulated_pool
from .scene_promotion import audit


def existing_support(target,new_poses,evidence):
    lookup={t['id']:t for t in evidence['tracks']};tested=agreed=0
    for point in target['points']:
        obs=[o for o in lookup[point['track']]['observations'] if o['capture'] in new_poses]
        if not obs:continue
        tested+=1;good=True;x=np.asarray(point['xyz'])
        for o in obs:
            p=new_poses[o['capture']];b=track_bearing(o,evidence['cameras']);q=unit(np.asarray(p['rotation'])@x+p['translation'])
            angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(q,b)),q@b))
            good&=angle<=1.2
        agreed+=int(good)
    return dict(points_tested=tested,points_agree=agreed,accepted=bool(tested<12 or agreed>=.6*tested))


def merge(evidence,pose,folder,maximum_rounds=3,allowed_seeds=None):
    start=time.perf_counter();meta=json.loads((folder/'submaps.json').read_text());target=copy.deepcopy(pose);attempts=[];merged=[]
    candidates=[r for r in meta['submaps'] if r['accepted'] and (allowed_seeds is None or tuple(r['seed']) in allowed_seeds)]
    preferred={j['track']:j['center'] for j in pose.get('surface_jets',[]) if j['accepted']}
    for round_index in range(maximum_rounds):
        added=0
        order=sorted(candidates,key=lambda r:(-len(set(r['cameras'])-set(map(int,target['poses']))),-r['promotion'].get('third_view_consistent',0),r['seed']))
        for row in order:
            new=set(row['cameras'])-set(map(int,target['poses']))
            if not new:continue
            a,b=row['seed'];source=json.loads((folder/f'{a:03d}-{b:03d}.json').read_text());relation=align(source,target,evidence)
            receipt=dict(round=round_index,seed=row['seed'],new_cameras=sorted(new),alignment=relation)
            if not relation['accepted']:attempts.append(receipt);continue
            proposed={i:transfer_pose(source['poses'][str(i)],relation) for i in sorted(new)}
            models=copy.deepcopy(target.get('camera_models',evidence['cameras']))
            source_models=source.get('camera_models',evidence['cameras'])
            for i in new:models[i]=copy.deepcopy(source_models[i])
            candidate_evidence=dict(evidence,cameras=models)
            check=existing_support(target,proposed,candidate_evidence);receipt['existing_point_prediction']=check
            if not check['accepted']:receipt['rejected']='new_cameras_conflict_with_existing_points';attempts.append(receipt);continue
            all_poses={int(i):p for i,p in target['poses'].items()};all_poses.update(proposed)
            cloud,pool=triangulated_pool(candidate_evidence,all_poses,preferred);receipt['proposed_pool']=pool
            if len(cloud)<.8*len(target['points']):receipt['rejected']='structure_support_collapses';attempts.append(receipt);continue
            for i,p in proposed.items():p['role']='submap_transferred';p['source_seed']=row['seed']
            if 'camera_models' in target or 'camera_models' in source:target['camera_models']=models
            target['poses']={str(i):p for i,p in all_poses.items()};target['points']=[dict(track=tid,xyz=x.tolist()) for tid,x in cloud.items()];target=audit(evidence,target)
            receipt['merged']=True;attempts.append(receipt);merged.append(row['seed']);added+=len(proposed)
            print(json.dumps(dict(seed=row['seed'],added=sorted(new),cameras=len(target['poses']),points=len(target['points']))),flush=True)
        if not added:break
    target['submap_merge']=dict(attempts=attempts,merged_seeds=merged,seconds=time.perf_counter()-start,
        unresolved_local_cameras=sorted(set(meta['observed_cameras'])-set(map(int,target['poses']))),
        policy='Positive-scale frame alignment checked on unused common points and shared-camera orientation; new-camera predictions checked against established points; existing poses are not overwritten; reject severe point-support collapse.',
        limitations='Exploratory sequential reconciliation of collection-derived hypotheses. Accepted local frame relations do not certify a globally optimized or visibility-complete scene; alternative and rejected relations remain recorded.')
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--submaps',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    e=json.loads(args.evidence.read_text());pose=json.loads(args.pose.read_text());result=merge(e,pose,args.submaps)
    result['submap_merge'].update(source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('merge_submaps.py','submap_alignment.py','scene_growth.py')},parent_pose_sha256=hashlib.sha256(args.pose.read_bytes()).hexdigest(),parent_submaps_sha256=hashlib.sha256((args.submaps/'submaps.json').read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(cameras=len(result['poses']),points=len(result['points']),merged=len(result['submap_merge']['merged_seeds']),unresolved_local_cameras=result['submap_merge']['unresolved_local_cameras'],seconds=result['submap_merge']['seconds'])),flush=True)


if __name__=='__main__':main()
