"""Joint camera/point refinement with a fixed gauge and excluded image sites."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from .multiview_geometry import unit
from .scene_pose import track_bearing
from .scene_promotion import audit


def refine(evidence, pose, max_evaluations=100, calibrate_panoramas=False):
    start = time.perf_counter()
    ids = sorted(map(int, pose['poses']))
    index = {i:k for k,i in enumerate(ids)}
    seed, baseline = pose['seed']
    rotations = np.array([pose['poses'][str(i)]['rotation'] for i in ids])
    translations = np.array([pose['poses'][str(i)]['translation'] for i in ids])
    # Seed camera fixes frame; the second seed fixes scale, not its orientation
    # or baseline direction. The current seed frame has zero translation.
    if np.linalg.norm(translations[index[seed]]) > 1e-8:
        raise ValueError('Expected reconstruction in seed camera coordinates')
    scale = np.linalg.norm(translations[index[baseline]])
    moving = [i for i in ids if i != seed]
    cam_slice = {i:slice(6*k,6*k+6) for k,i in enumerate(moving)}
    lookup = {t['id']:t for t in evidence['tracks']}
    clouds = []; observations = []; used = set()
    for point in sorted(pose['points'], key=lambda p:(-len(lookup[p['track']]['observations']),p['track'])):
        obs = [o for o in lookup[point['track']]['observations']
               if o['capture'] in index and (o['capture'],o['site']) not in used]
        if len(obs)<3 or sum(o['site']%7!=0 for o in obs)<2:
            continue
        n = len(clouds); clouds.append(point)
        for o in obs:
            used.add((o['capture'],o['site']))
            observations.append((n,o['capture'],o['site'],track_bearing(o,evidence['cameras']),o['xy']))
    if len(clouds)<10:
        return copy.deepcopy(pose), dict(applied=False,reason='insufficient_multi_view_points')
    pi = np.array([o[0] for o in observations]); ci = np.array([index[o[1]] for o in observations])
    target = np.array([o[3] for o in observations]); train = np.array([o[2]%7!=0 for o in observations])
    focal = np.array([evidence['cameras'][o[1]]['focal'] for o in observations])
    pose_size=6*len(moving)
    calibrating=[i for i in ids if calibrate_panoramas and evidence['cameras'][i]['projection']=='cylindrical']
    intrinsic_slice={i:slice(pose_size+3*k,pose_size+3*k+3) for k,i in enumerate(calibrating)}
    nc = pose_size+3*len(calibrating)
    initial = np.r_[np.concatenate([np.r_[Rotation.from_matrix(rotations[index[i]]).as_rotvec(),translations[index[i]]] for i in moving]),
                    np.zeros(3*len(calibrating)),
                    np.asarray([p['xyz'] for p in clouds]).ravel()]
    def unpack(x):
        r=rotations.copy(); t=translations.copy()
        for i in moving:
            v=x[cam_slice[i]];r[index[i]]=Rotation.from_rotvec(v[:3]).as_matrix();t[index[i]]=v[3:]
        t[index[baseline]]=unit(t[index[baseline]])*scale
        return r,t,x[nc:].reshape(-1,3)
    def camera_points(x):
        r,t,p=unpack(x)
        return np.einsum('nij,nj->ni',r[ci],p[pi])+t[ci]
    def predictions(x):return unit(camera_points(x))
    image_xy=np.asarray([o[4] for o in observations])
    groups={i:np.flatnonzero(ci==index[i]) for i in ids}
    def camera_models(x):
        cameras=copy.deepcopy(evidence['cameras'])
        for i in calibrating:
            c=cameras[i];v=x[intrinsic_slice[i]];h,w=c['shape']
            c.update(focal_x=c['focal']*float(np.exp(v[0])),focal_y=c['focal']*float(np.exp(v[1])),
                     principal_point=[(w-1)/2,(h-1)/2+float(v[2])*h],calibration_status='joint_cylindrical_candidate')
        return cameras
    def pixel_errors(x):
        points=camera_points(x);errors=np.zeros((len(points),2));models=camera_models(x)
        for i,k in groups.items():
            c=models[i];h,w=c['shape'];f=c['focal'];fx=c.get('focal_x',f);fy=c.get('focal_y',f);cx,cy=c.get('principal_point',[(w-1)/2,(h-1)/2]);q=points[k]
            if c['projection']=='cylindrical':
                angle=np.arctan2(q[:,0],q[:,2])-(image_xy[k,0]-cx)/fx
                errors[k,0]=np.arctan2(np.sin(angle),np.cos(angle))*fx
                errors[k,1]=q[:,1]/np.maximum(np.hypot(q[:,0],q[:,2]),1e-9)*fy+cy-image_xy[k,1]
            elif c['projection']=='perspective':errors[k]=q[:,:2]/np.maximum(q[:,2,None],1e-9)*[fx,fy]+[cx,cy]-image_xy[k]
            else:raise ValueError('Calibrated bundle currently supports cylindrical and perspective cameras')
        return errors
    def residual(x):
        if calibrate_panoramas:
            prior=x[pose_size:nc]/np.tile([.2,.2,.15],len(calibrating))
            return np.r_[pixel_errors(x)[train].ravel(),prior]
        return ((predictions(x)-target)*focal[:,None])[train].ravel()
    training=np.flatnonzero(train);channels=2 if calibrate_panoramas else 3
    sparsity=lil_matrix((channels*len(training)+3*len(calibrating),len(initial)),dtype=int)
    for j,k in enumerate(training):
        camera=observations[k][1]
        if camera!=seed:sparsity[channels*j:channels*j+channels,cam_slice[camera]]=1
        if camera in intrinsic_slice:sparsity[channels*j:channels*j+channels,intrinsic_slice[camera]]=1
        sparsity[channels*j:channels*j+channels,nc+3*pi[k]:nc+3*pi[k]+3]=1
    for k in range(3*len(calibrating)):sparsity[channels*len(training)+k,pose_size+k]=1
    def metrics(x):
        pred=predictions(x)
        actual=target
        if calibrate_panoramas:
            models=camera_models(x)
            actual=np.asarray([track_bearing(dict(capture=o[1],xy=o[4]),models) for o in observations])
        error=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(pred,actual),axis=1),np.sum(pred*actual,axis=1)))
        result=dict(train_median_degrees=float(np.median(error[train])),
                    holdout_median_degrees=float(np.median(error[~train])) if (~train).any() else None,
                    holdout_p90_degrees=float(np.quantile(error[~train],.9)) if (~train).any() else None,
                    holdout_within_1_2_degrees=int(np.sum(error[~train]<=1.2)),
                    positive_fraction=float(np.mean(np.sum(pred*actual,axis=1)>0)))
        if calibrate_panoramas:
            px=np.linalg.norm(pixel_errors(x),axis=1)
            result.update(train_median_pixels=float(np.median(px[train])),holdout_median_pixels=float(np.median(px[~train])),holdout_p90_pixels=float(np.quantile(px[~train],.9)))
        return result
    before=metrics(initial)
    lower=np.full(len(initial),-np.inf);upper=-lower
    for s in intrinsic_slice.values():lower[s]=[np.log(.6),np.log(.6),-.4];upper[s]=[np.log(1.8),np.log(1.8),.4]
    opt=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),x_scale='jac',
                      loss='soft_l1',f_scale=1.,max_nfev=max_evaluations,ftol=1e-5,bounds=(lower,upper))
    after=metrics(opt.x)
    suffix='pixels' if calibrate_panoramas else 'degrees'
    applied=bool(after['holdout_median_degrees'] is not None and
                 after['holdout_median_'+suffix]<=before['holdout_median_'+suffix]+1e-8 and
                 after['holdout_p90_'+suffix]<=before['holdout_p90_'+suffix]+1e-8 and
                 after['positive_fraction']>=before['positive_fraction'])
    receipt=dict(applied=applied,points=len(clouds),observations=len(observations),
                 training_observations=int(train.sum()),heldout_observations=int((~train).sum()),
                 before=before,after=after,evaluations=opt.nfev,solver_status=opt.status,optimality=float(opt.optimality),
                 maximum_evaluations=max_evaluations,comparison_roundoff=1e-8,
                 seconds=time.perf_counter()-start,
                 policy=('Cylindrical horizontal/vertical focal and vertical principal point; pixel residuals with fixed weak priors. ' if calibrate_panoramas else 'Fixed nominal intrinsics. ')+'Original seed frame and baseline length; physical image-site holdout excluded from joint fit. One candidate per declared basis, no holdout tuning.')
    result=copy.deepcopy(pose)
    # Keep the rejected candidate as evidence, not only its aggregate scores.
    if calibrate_panoramas:result['camera_models']=camera_models(opt.x)
    r,t,points=unpack(opt.x)
    for i in ids:
        camera=result['poses'][str(i)]
        camera.update(rotation=r[index[i]].tolist(),translation=t[index[i]].tolist(),center=(-r[index[i]].T@t[index[i]]).tolist())
        camera['joint_refinement']=True
    mapped={p['track']:x for p,x in zip(clouds,points)}
    for point in result['points']:
        if point['track'] in mapped:point['xyz']=mapped[point['track']].tolist()
    result['bundle']=receipt
    candidate=audit(evidence,result)
    retained=candidate if applied else audit(evidence,copy.deepcopy(pose))
    retained['bundle']=receipt
    if not applied:retained['bundle_candidate']=candidate
    return retained,receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--pose',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibrate-panoramas',action='store_true')
    p.add_argument('--max-evaluations',type=int,default=100)
    args=p.parse_args();e=json.loads(args.evidence.read_text());pose=json.loads(args.pose.read_text())
    result,receipt=refine(e,pose,max_evaluations=args.max_evaluations,calibrate_panoramas=args.calibrate_panoramas)
    result['bundle']['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['bundle']['parent_pose_sha256']=hashlib.sha256(args.pose.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    candidate=result.pop('bundle_candidate',None)
    if candidate is not None:args.out.with_suffix('.candidate.json').write_text(json.dumps(candidate,indent=2))
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
