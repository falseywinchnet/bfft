"""Expand a retained camera hypothesis using refreshed multi-view structure."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .multiview_geometry import unit
from .scene_pose import track_bearing,fit_pose
from .scene_promotion import audit


def triangulated_pool(evidence,poses,preferred=None):
    cameras=evidence['cameras'];proposals=[];preferred=preferred or {}
    for track in evidence['tracks']:
        obs=[o for o in track['observations'] if o['capture'] in poses]
        if len(obs)<3:continue
        centers=np.array([poses[o['capture']]['center'] for o in obs])
        rays=np.array([track_bearing(o,cameras)@np.asarray(poses[o['capture']]['rotation']) for o in obs])
        projector=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:];lhs=projector.sum(0)
        if np.linalg.cond(lhs)>1e5:continue
        if track['id'] in preferred:x=np.asarray(preferred[track['id']])
        else:x=np.linalg.solve(lhs,np.einsum('nij,nj->i',projector,centers))
        d=x-centers;depth=np.sum(d*rays,axis=1)
        error=np.arctan2(np.linalg.norm(np.cross(unit(d),rays),axis=1),np.sum(unit(d)*rays,axis=1))
        parallax=np.arccos(np.clip(np.min(rays@rays.T),-1,1))
        if np.all(depth>0) and error.max()<=np.deg2rad(1.2) and parallax>=np.deg2rad(.7):
            proposals.append((len(obs),float(error.max()),track['id'],x,obs))
    # A dense chart sampled again from another reference cannot multiply a
    # physical image-site vote in the structure used to recover new cameras.
    used=set();cloud={};rejected=0
    for _,error,tid,x,obs in sorted(proposals,key=lambda r:(-r[0],r[1],r[2])):
        sites={(o['capture'],o['site']) for o in obs}
        if sites&used:rejected+=1;continue
        used.update(sites);cloud[tid]=x
    return cloud,dict(consistent_proposals=len(proposals),retained_points=len(cloud),duplicate_site_proposals=rejected)


def grow(evidence,pose,maximum_rounds=8):
    evidence=dict(evidence,cameras=pose.get('camera_models',evidence['cameras']))
    start=time.perf_counter();result=copy.deepcopy(pose);poses={int(i):p for i,p in pose['poses'].items()}
    lookup={t['id']:{o['capture']:o for o in t['observations']} for t in evidence['tracks']}
    preferred={j['track']:j['center'] for j in pose.get('surface_jets',[]) if j['accepted']}
    rounds=[]
    for round_index in range(maximum_rounds):
        cloud,pool=triangulated_pool(evidence,poses,preferred);checks=[];proposals=[]
        for i in sorted(set(range(len(evidence['cameras'])))-set(poses)):
            ids=[tid for tid in cloud if i in lookup[tid]]
            if len(ids)<16:continue
            points=np.array([cloud[tid] for tid in ids]);directions=np.array([track_bearing(lookup[tid][i],evidence['cameras']) for tid in ids])
            fitted=fit_pose(points,directions,np.array([lookup[tid][i]['site'] for tid in ids]))
            if fitted is None:continue
            checks.append(dict(capture=i,**{k:v for k,v in fitted.items() if k not in ('rotation','translation','center','errors')}))
            if fitted['accepted']:proposals.append((i,fitted))
        for i,fitted in proposals:
            poses[i]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fitted.items() if k not in ('errors','accepted','median_degrees')};poses[i]['role']='heldout_pose'
        rounds.append(dict(round=round_index,pool=pool,checks=checks,added_cameras=[i for i,_ in proposals],cameras=len(poses)))
        if not proposals:break
    cloud,pool=triangulated_pool(evidence,poses,preferred)
    result['poses']={str(i):p for i,p in poses.items()};result['points']=[dict(track=tid,xyz=x.tolist()) for tid,x in cloud.items()]
    result['growth']=dict(rounds=rounds,final_pool=pool,seconds=time.perf_counter()-start,
                          policy='Rebuild structure from at least three recovered views; retain accepted differential centers; cap duplicate image sites; fixed independent target-site pose gates; existing cameras remain fixed.')
    return audit(evidence,result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    e=json.loads(args.evidence.read_text());pose=json.loads(args.pose.read_text())
    if 'camera_models' in pose:e['cameras']=pose['camera_models']
    result=grow(e,pose);result['growth'].update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_pose_sha256=hashlib.sha256(args.pose.read_bytes()).hexdigest(),parent_evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(cameras=len(result['poses']),points=len(result['points']),growth=result['growth'],promotion=result['promotion_counts'])),flush=True)


if __name__=='__main__':main()
