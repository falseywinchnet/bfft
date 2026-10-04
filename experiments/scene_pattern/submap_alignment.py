"""Estimate positive-scale frame relations and check unused shared observations."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .multiview_geometry import unit
from .scene_pose import track_bearing


def similarity(source,target):
    a=np.asarray(source);b=np.asarray(target);ma=a.mean(0);mb=b.mean(0);x=a-ma;y=b-mb
    u,s,v=np.linalg.svd(y.T@x/len(x));sign=np.array([1.,1.,np.linalg.det(u@v)]);r=u@np.diag(sign)@v
    scale=float(np.sum(s*sign)/max(np.mean(np.sum(x*x,axis=1)),1e-12));t=mb-scale*r@ma
    return scale,r,t


def align(source,target,evidence,trials=300):
    a={p['track']:p['xyz'] for p in source['points'] if p.get('promotion')=='third_view_consistent'}
    b={p['track']:p['xyz'] for p in target['points'] if p.get('promotion')=='third_view_consistent'}
    ids=sorted(a.keys()&b.keys());lookup={t['id']:t for t in evidence['tracks']}
    if len(ids)<20:return dict(accepted=False,reason='insufficient_shared_points',shared_points=len(ids))
    site=np.array([lookup[i]['observations'][0]['capture']*1000003+lookup[i]['observations'][0]['site'] for i in ids])
    held=site%7==0;train=np.flatnonzero(~held);test=np.flatnonzero(held)
    if len(train)<12 or len(test)<4:return dict(accepted=False,reason='insufficient_split_support',shared_points=len(ids))
    x=np.array([a[i] for i in ids]);y=np.array([b[i] for i in ids]);bound=.05*np.maximum(np.linalg.norm(y,axis=1),1.);rng=np.random.default_rng(912);best=None
    for _ in range(trials):
        chosen=rng.choice(train,3,replace=False);singular=np.linalg.svd(x[chosen]-x[chosen].mean(0),compute_uv=False)
        if singular[1]<singular[0]*.01:continue
        scale,r,t=similarity(x[chosen],y[chosen]);error=np.linalg.norm(scale*x@r.T+t-y,axis=1);good=train[error[train]<=bound[train]]
        score=(len(good),-float(np.median(error[good]/bound[good])) if len(good) else -100.)
        if best is None or score>best[0]:best=(score,scale,r,t,good)
    if best is None or len(best[-1])<12:return dict(accepted=False,reason='no_supported_similarity',shared_points=len(ids))
    _,scale,r,t,good=best
    scale,r,t=similarity(x[good],y[good]);initial=np.r_[np.log(max(scale,1e-9)),Rotation.from_matrix(r).as_rotvec(),t]
    target_poses={int(i):p for i,p in target['poses'].items()};cameras=target.get('camera_models',evidence['cameras'])
    owners=[];rotations=[];translations=[];directions=[];focals=[]
    for k,tid in enumerate(ids):
        obs=[o for o in lookup[tid]['observations'] if o['capture'] in target_poses]
        for o in obs:
            p=target_poses[o['capture']];owners.append(k);rotations.append(p['rotation']);translations.append(p['translation']);directions.append(track_bearing(o,cameras));focals.append(cameras[o['capture']]['focal']/np.sqrt(len(obs)))
    owners=np.array(owners);rotations=np.array(rotations);translations=np.array(translations);directions=np.array(directions);focals=np.array(focals);fit_mask=np.isin(owners,good)
    def transformed(v):return np.exp(v[0])*x@Rotation.from_rotvec(v[1:4]).as_matrix().T+v[4:]
    def predicted(v):return unit(np.einsum('nij,nj->ni',rotations,transformed(v)[owners])+translations)
    def residual(v):return ((predicted(v)-directions)*focals[:,None])[fit_mask].ravel()
    fitted=least_squares(residual,initial,loss='soft_l1',f_scale=1.,max_nfev=100)
    predicted_rays=predicted(fitted.x);errors=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(predicted_rays,directions),axis=1),np.sum(predicted_rays*directions,axis=1)))
    per_point=np.zeros(len(ids));np.maximum.at(per_point,owners,errors);valid=test[per_point[test]<=1.2]
    scale=float(np.exp(fitted.x[0]));r=Rotation.from_rotvec(fitted.x[1:4]).as_matrix();t=fitted.x[4:]
    shared_cameras=sorted(set(source['poses'])&set(target['poses']));angles=[]
    for i in shared_cameras:
        actual=np.asarray(source['poses'][i]['rotation'])@r.T;expected=np.asarray(target['poses'][i]['rotation'])
        angles.append(float(np.rad2deg(Rotation.from_matrix(actual@expected.T).magnitude())))
    rotation_median=float(np.median(angles)) if angles else None
    accepted=len(valid)>=4 and len(valid)>=.6*len(test) and (rotation_median is None or rotation_median<=1.2)
    return dict(accepted=bool(accepted),shared_points=len(ids),training_inliers=len(good),heldout_points=len(test),heldout_agree=len(valid),
                heldout_median_max_degrees=float(np.median(per_point[test])),shared_camera_rotation_median_degrees=rotation_median,
                scale=scale,rotation=r.tolist(),translation=t.tolist(),
                policy='Train-only robust similarity and bearing refinement; withheld physical reference-site observations; shared-camera orientation check; positive scale.')


def transfer_pose(pose,alignment):
    scale=alignment['scale'];r=np.asarray(alignment['rotation']);t=np.asarray(alignment['translation']);local=np.asarray(pose['rotation'])
    world=local@r.T;translation=scale*np.asarray(pose['translation'])-world@t
    return dict(**{k:v for k,v in pose.items() if k not in ('rotation','translation','center')},rotation=world.tolist(),translation=translation.tolist(),center=(-world.T@translation).tolist())
