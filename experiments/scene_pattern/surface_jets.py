"""Infer local 3-D position and tangents from measured image-map Jacobians."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .dense_transport import Field
from .multiview_geometry import bearings, unit


def image_rays(xy, camera):
    return bearings(xy,camera['shape'],camera['focal'],camera['projection'],
                    focal_x=camera.get('focal_x'),focal_y=camera.get('focal_y'),principal_point=camera.get('principal_point'))


def project(points, camera, pose):
    q=np.asarray(points)@np.asarray(pose['rotation']).T+pose['translation']
    h,w=camera['shape'];fx=camera.get('focal_x',camera['focal']);fy=camera.get('focal_y',camera['focal']);center=camera.get('principal_point',[(w-1)/2,(h-1)/2])
    if camera['projection']=='perspective':p=q[:,:2]/np.maximum(q[:,2,None],1e-12)
    elif camera['projection']=='cylindrical':p=np.c_[np.arctan2(q[:,0],q[:,2]),q[:,1]/np.maximum(np.hypot(q[:,0],q[:,2]),1e-12)]
    else:raise ValueError('Unsupported projection for surface jet')
    return p*[fx,fy]+center


def is_training(jet,observation):
    return bool(observation.get('training',observation['capture']!=jet.get('holdout_capture')))


def is_validation(jet,observation):
    return bool(observation.get('validation',not is_training(jet,observation)))


def spatial_holdout(observations):
    """Vary the withheld camera by physical reference site, never by fit score."""
    if len(observations)<3:return observations
    ref=observations[0];x,y=np.rint(np.asarray(ref['xy'])*2).astype(int)
    key=f"{ref['capture']}:{x}:{y}".encode()
    digest=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')
    chosen=1+digest%(len(observations)-1)
    return [o for k,o in enumerate(observations) if k!=chosen]+[observations[chosen]]


def fit_jet(observations,cameras,poses,initial_point,radius=3.):
    """Reserve the last camera, fit a nine-parameter first-order surface jet.

Observations use a shared reference-image coordinate. Their affine matrices
map reference pixel offsets to offsets in each observing capture.
"""
    explicit=any('training' in o for o in observations)
    if len(observations)<3 and not explicit:return dict(accepted=False,reason='fewer_than_three_views')
    delta=np.array([(u,v) for v in (-1.,0.,1.) for u in (-1.,0.,1.)])
    basis=np.c_[np.ones(len(delta)),delta]
    train=[o for o in observations if o['training']] if explicit else observations[:-1]
    hold=[o for o in observations if o.get('validation',False)] if explicit else observations[-1:]
    if len(train)<2:return dict(accepted=False,reason='fewer_than_two_training_views')
    expected={o['capture']:np.asarray(o['xy'])+radius*delta@np.asarray(o['affine']).T for o in observations}
    position=np.tile(np.asarray(initial_point),(len(delta),1));jet=None;condition=None
    for iteration in range(3):
        rows=[];rhs=[]
        for o in train:
            i=o['capture'];p=poses[i];camera=cameras[i];r=np.asarray(p['rotation']);center=np.asarray(p['center'])
            rays=image_rays(expected[i],camera)@r;projector=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
            weights=np.ones(len(delta)) if explicit and iteration==0 else camera['focal']/np.maximum(np.linalg.norm(position-center,axis=1),1e-6)
            rows.append(np.concatenate([projector*basis[:,k,None,None] for k in range(3)],axis=2)*weights[:,None,None])
            rhs.append(np.einsum('nij,j->ni',projector,center)*weights[:,None])
        matrix=np.concatenate(rows).reshape(-1,9);target=np.concatenate(rhs).ravel()
        x,_,rank,s=np.linalg.lstsq(matrix,target,rcond=1e-10)
        condition=float(s[0]/max(s[-1],1e-30))
        if rank<9 or condition>1e7:return dict(accepted=False,reason='unresolved_depth_or_tangent',condition=condition)
        jet=x.reshape(3,3);position=basis@jet
    def errors(views):
        if not views:return dict(evaluated=False,median_pixels=0.,maximum_pixels=0.,positive=True)
        values=[];positive=[]
        for o in views:
            i=o['capture'];q=project(position,cameras[i],poses[i]);values.extend(np.linalg.norm(q-expected[i],axis=1).tolist())
            world_rays=image_rays(expected[i],cameras[i])@np.asarray(poses[i]['rotation'])
            positive.extend((np.sum((position-poses[i]['center'])*world_rays,axis=1)>0).tolist())
        return dict(median_pixels=float(np.median(values)),maximum_pixels=float(np.max(values)),positive=bool(all(positive)))
    training=errors(train);validation=errors(hold)
    accepted=bool(hold) and training['positive'] and validation['positive'] and training['maximum_pixels']<=1.5 and validation['maximum_pixels']<=1.5
    tangent=jet[1:]/radius;normal=unit(np.cross(tangent[0],tangent[1]))
    return dict(accepted=accepted,center=jet[0].tolist(),tangents=tangent.tolist(),normal=normal.tolist(),condition=condition,
                training=training,validation=validation,holdout_capture=hold[0]['capture'] if len(hold)==1 else None,
                center_change=float(np.linalg.norm(jet[0]-initial_point)),radius=radius,
                reference_capture=observations[0]['capture'],observations=observations)


def run(evidence,pose,folder):
    start=time.perf_counter();meta=json.loads((folder/'dense.json').read_text());poses={int(i):p for i,p in pose['poses'].items()}
    fields={};cameras=pose.get('camera_models',evidence['cameras'])
    for pair in meta['pairs']:
        a,b=pair['a'],pair['b']
        if a not in poses or b not in poses:continue
        for a,b in ((a,b),(b,a)):
            with np.load(folder/f'{a:03d}-{b:03d}.npz') as f:
                good=f['valid']&(f['third_view_support']>0)
                fields[a,b]=Field(f['grid'].reshape(-1,2)[good],f['target'][good],f['affine'][good])
    lookup={t['id']:t for t in evidence['tracks']};results=[]
    for point in pose['points']:
        if point['promotion']!='third_view_consistent':continue
        obs={o['capture']:o for o in lookup[point['track']]['observations'] if o['capture'] in poses};choices=[]
        for a in sorted(obs):
            rows=[dict(capture=a,xy=obs[a]['xy'],affine=np.eye(2).tolist())]
            for b in sorted(obs):
                if (a,b) not in fields:continue
                q,j,good=fields[a,b].at(np.array([obs[a]['xy']]))
                if not good[0] or np.linalg.norm(q[0]-obs[b]['xy'])>1.5:continue
                rows.append(dict(capture=b,xy=obs[b]['xy'],affine=j[0].tolist()))
            choices.append(rows)
        # Select by available measured maps only, before fitting any surface.
        observations=spatial_holdout(max(choices,key=len))
        result=fit_jet(observations,cameras,poses,np.asarray(point['xyz']))
        results.append(dict(track=point['track'],**result))
    accepted=[r for r in results if r['accepted']];tested=[r for r in results if 'validation' in r]
    return dict(results=results,summary=dict(points=len(results),tested=len(tested),accepted=len(accepted),
                validation_median_pixels=float(np.median([r['validation']['median_pixels'] for r in tested])) if tested else None,
                accepted_validation_median_pixels=float(np.median([r['validation']['median_pixels'] for r in accepted])) if accepted else None),
                seconds=time.perf_counter()-start,policy=dict(radius_pixels=3.,maximum_reprojection_pixels=1.5,
                    fit='position and two surface tangents from image-map Jacobians; cameras fixed',
                    holdout='spatial reference-site hash chooses one non-reference camera; excluded from jet fit',reference='maximum count of available measured maps; stable tie order'),
                limitations='Conditional local surface differential hypotheses. Fixed cameras and correspondence discovery used this collection; withheld-camera patch checks are not independent physical validation or global surface closure.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True)
    p.add_argument('--fields',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=run(json.loads(args.evidence.read_text()),json.loads(args.pose.read_text()),args.fields)
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['parent_pose_sha256']=hashlib.sha256(args.pose.read_bytes()).hexdigest()
    result['parent_evidence_sha256']=hashlib.sha256(args.evidence.read_bytes()).hexdigest();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
