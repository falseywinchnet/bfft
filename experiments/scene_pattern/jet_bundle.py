"""Jointly fit camera poses and first-order surfaces to regional image maps."""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from .multiview_geometry import unit
from .surface_jets import image_rays,project,is_training,is_validation


def refine_jets(packet,pose,cameras,max_evaluations=1000,calibrate_panoramas=False,coherent_surfaces=False,protect_cameras=False,point_evidence=None,freeze_cameras=False):
    start=time.perf_counter()
    # Selection sees training evidence only. Failed third-view jets remain in
    # this optimization and its final predictive score.
    jets=[r for r in packet['results'] if 'center' in r and (packet.get('retain_training_failures',False) or ('training' in r and r['training']['positive'] and r['training']['maximum_pixels']<=1.5))]
    if point_evidence is not None and any('training' in o for j in jets for o in j['observations']):
        raise ValueError('Explicit spatial roles require matching point exclusions; point support is not enabled for this path')
    if len(jets)<10:return copy.deepcopy(pose),dict(applied=False,reason='insufficient_training_supported_jets'),None
    ids=sorted(map(int,pose['poses']));ix={i:k for k,i in enumerate(ids)};seed,baseline=pose['seed']
    rotations=np.array([pose['poses'][str(i)]['rotation'] for i in ids]);translations=np.array([pose['poses'][str(i)]['translation'] for i in ids])
    training_coverage=Counter(o['capture'] for j in jets for o in j['observations'] if is_training(j,o))
    heldout_coverage=Counter(o['capture'] for j in jets for o in j['observations'] if is_validation(j,o))
    from .point_support import prepare
    support=prepare(point_evidence,pose,jets,cameras);support_rows=support['observations']
    pose_training_coverage=training_coverage.copy();pose_training_coverage.update(o['capture'] for o in support_rows if o['training'])
    support_ci=np.array([o['capture'] for o in support_rows],int);support_pi=np.array([o['point'] for o in support_rows],int)
    support_training=np.array([o['training'] for o in support_rows],bool);support_expected=np.array([o['xy'] for o in support_rows]).reshape(-1,2)
    support_groups={i:np.flatnonzero(support_ci==i) for i in ids}
    moving=[i for i in ids if not freeze_cameras and i!=seed and pose_training_coverage[i]>0];slices={i:slice(6*k,6*k+6) for k,i in enumerate(moving)};pose_size=6*len(moving)
    calibrating=[i for i in ids if calibrate_panoramas and cameras[i]['projection']=='cylindrical' and pose_training_coverage[i]>0]
    intrinsic_slices={i:slice(pose_size+3*k,pose_size+3*k+3) for k,i in enumerate(calibrating)}
    nc=pose_size+3*len(calibrating)
    scale=np.linalg.norm(translations[ix[baseline]])
    delta=np.array([(u,v) for v in (-1.,0.,1.) for u in (-1.,0.,1.)]);basis=np.c_[np.ones(9),delta]
    ji=[];ci=[];samples=[];expected=[];training=[];validation=[]
    initial_surfaces=[]
    for k,r in enumerate(jets):
        initial_surfaces.append(np.r_[r['center'],(np.asarray(r['tangents'])*r['radius']).ravel()])
        for o in r['observations']:
            ji.extend([k]*9);ci.extend([o['capture']]*9);samples.extend(range(9));training.extend([is_training(r,o)]*9);validation.extend([is_validation(r,o)]*9)
            expected.extend(np.asarray(o['xy'])+r['radius']*delta@np.asarray(o['affine']).T)
    ji=np.array(ji);ci=np.array(ci);samples=np.array(samples);expected=np.asarray(expected);training=np.array(training);validation=np.array(validation)
    initial=np.r_[np.asarray([np.r_[Rotation.from_matrix(rotations[ix[i]]).as_rotvec(),translations[ix[i]]] for i in moving]).ravel(),np.zeros(3*len(calibrating)),np.asarray(initial_surfaces).ravel(),np.asarray([p['xyz'] for p in support['points']]).ravel()]
    ns=9*len(jets);support_start=nc+ns
    groups={i:np.flatnonzero(ci==i) for i in ids}
    def unpack(x):
        r=rotations.copy();t=translations.copy()
        for i in moving:
            v=x[slices[i]];r[ix[i]]=Rotation.from_rotvec(v[:3]).as_matrix();t[ix[i]]=v[3:]
        if not freeze_cameras:t[ix[baseline]]=unit(t[ix[baseline]])*scale
        return r,t,x[nc:support_start].reshape(-1,3,3)
    def camera_models(x):
        models=[dict(c) for c in cameras]
        for i in calibrating:
            c=models[i];v=x[intrinsic_slices[i]];h,w=c['shape']
            center=c.get('principal_point',[(w-1)/2,(h-1)/2])
            c.update(focal_x=c.get('focal_x',c['focal'])*float(np.exp(v[0])),
                     focal_y=c.get('focal_y',c['focal'])*float(np.exp(v[1])),
                     principal_point=[center[0],center[1]+float(v[2])*h],
                     calibration_status='differential_cylindrical_candidate')
        return models
    def pixel_residual(x):
        r,t,jets=unpack(x);points=np.einsum('ni,nij->nj',basis[samples],jets[ji]);result=np.zeros_like(expected);positive=np.zeros(len(points),bool);models=camera_models(x)
        for i,k in groups.items():
            if not len(k):continue
            camera=models[i];p=dict(rotation=r[ix[i]],translation=t[ix[i]])
            result[k]=project(points[k],camera,p)-expected[k]
            if camera['projection']=='cylindrical':
                period=2*np.pi*camera.get('focal_x',camera['focal']);result[k,0]=(result[k,0]+period/2)%period-period/2
            q=points[k]@r[ix[i]].T+t[ix[i]];positive[k]=np.sum(q*image_rays(expected[k],camera),axis=1)>0
        return result,positive
    def support_residual(x):
        r,t,_=unpack(x);points=x[support_start:].reshape(-1,3);errors=np.zeros_like(support_expected);positive=np.zeros(len(support_rows),bool);models=camera_models(x)
        for i,k in support_groups.items():
            if not len(k):continue
            camera=models[i];world=points[support_pi[k]];p=dict(rotation=r[ix[i]],translation=t[ix[i]])
            errors[k]=project(world,camera,p)-support_expected[k]
            if camera['projection']=='cylindrical':
                period=2*np.pi*camera.get('focal_x',camera['focal']);errors[k,0]=(errors[k,0]+period/2)%period-period/2
            q=world@r[ix[i]].T+t[ix[i]];positive[k]=np.sum(q*image_rays(support_expected[k],camera),axis=1)>0
        return errors,positive
    from .surface_coherence import neighbors,operator,summary as coherence_summary
    edges=neighbors(jets) if coherent_surfaces else []
    coherence=operator(jets,edges,cameras,pose['poses'])
    def residual(x):
        prior=x[pose_size:nc]/np.tile([.2,.2,.15],len(calibrating))
        return np.r_[pixel_residual(x)[0][training].ravel()/3,prior,(coherence@x[nc:support_start])/3,support_residual(x)[0][support_training].ravel()]
    train_ids=np.flatnonzero(training);sparse=lil_matrix((2*len(train_ids)+3*len(calibrating)+coherence.shape[0]+2*int(support_training.sum()),len(initial)),dtype=int)
    for n,k in enumerate(train_ids):
        if ci[k] in slices:sparse[2*n:2*n+2,slices[ci[k]]]=1
        if ci[k] in intrinsic_slices:sparse[2*n:2*n+2,intrinsic_slices[ci[k]]]=1
        sparse[2*n:2*n+2,nc+9*ji[k]:nc+9*ji[k]+9]=1
    for k in range(3*len(calibrating)):sparse[2*len(train_ids)+k,pose_size+k]=1
    coherence_row=2*len(train_ids)+3*len(calibrating)
    sparse[coherence_row:coherence_row+coherence.shape[0],nc:support_start]=coherence.astype(bool)
    for n,k in enumerate(np.flatnonzero(support_training)):
        row=coherence_row+coherence.shape[0]+2*n;i=support_ci[k]
        if i in slices:sparse[row:row+2,slices[i]]=1
        if i in intrinsic_slices:sparse[row:row+2,intrinsic_slices[i]]=1
        sparse[row:row+2,support_start+3*support_pi[k]:support_start+3*support_pi[k]+3]=1
    def metrics(x):
        residual,positive=pixel_residual(x);error=np.linalg.norm(residual,axis=1)
        return dict(training_median_pixels=float(np.median(error[training])),holdout_median_pixels=float(np.median(error[validation])) if validation.any() else None,
                    holdout_p90_pixels=float(np.quantile(error[validation],.9)) if validation.any() else None,positive_fraction=float(positive.mean()),
                    cameras={str(i):dict(patches=int(heldout_coverage[i]),median_pixels=float(np.median(error[(ci==i)&validation])),p90_pixels=float(np.quantile(error[(ci==i)&validation],.9))) for i in ids if heldout_coverage[i]})
    def support_metrics(x):
        residual,positive=support_residual(x);error=np.linalg.norm(residual,axis=1)
        return dict(positive_fraction=float(positive.mean()) if len(positive) else None,
                    cameras={str(i):dict(patches=int(np.sum((support_ci==i)&~support_training)),median_pixels=float(np.median(error[(support_ci==i)&~support_training])),p90_pixels=float(np.quantile(error[(support_ci==i)&~support_training],.9))) for i in ids if np.any((support_ci==i)&~support_training)})
    lower=np.full(len(initial),-np.inf);upper=-lower
    for sl in intrinsic_slices.values():lower[sl]=[np.log(.6),np.log(.6),-.4];upper[sl]=[np.log(1.8),np.log(1.8),.4]
    loss_scales=np.ones(sparse.shape[0]);loss_scales[coherence_row+coherence.shape[0]:]=9.
    def weighted_loss(z):
        root=np.sqrt(1+z/loss_scales)
        return np.array([2*loss_scales*(root-1),1/root,-.5/(loss_scales*root**3)])
    before=metrics(initial);fit=least_squares(residual,initial,jac_sparsity=sparse.tocsr(),x_scale='jac',loss=weighted_loss if len(support_rows) else 'soft_l1',f_scale=1/3,
                                            max_nfev=max_evaluations,ftol=1e-5,bounds=(lower,upper))
    after=metrics(fit.x);applied=bool(validation.any()) and after['holdout_median_pixels']<=before['holdout_median_pixels'] and after['holdout_p90_pixels']<=before['holdout_p90_pixels'] and after['positive_fraction']>=before['positive_fraction']
    camera_checks={i:dict(before=b,after=after['cameras'][i],passed=bool(after['cameras'][i]['median_pixels']<=1.1*b['median_pixels']+.1 and after['cameras'][i]['p90_pixels']<=1.2*b['p90_pixels']+.2)) for i,b in before['cameras'].items()}
    if protect_cameras:applied=applied and all(row['passed'] for row in camera_checks.values())
    support_before=support_metrics(initial);support_after=support_metrics(fit.x)
    support_checks={i:bool(support_after['cameras'][i]['median_pixels']<=1.1*b['median_pixels']+.1 and support_after['cameras'][i]['p90_pixels']<=1.2*b['p90_pixels']+.2) for i,b in support_before['cameras'].items()}
    if len(support_rows):applied=applied and all(support_checks.values()) and support_after['positive_fraction']>=support_before['positive_fraction']
    receipt=dict(applied=bool(applied),jets=len(jets),training_samples=int(training.sum()),holdout_samples=int(validation.sum()),
                 training_patches_by_camera=dict(sorted(training_coverage.items())),heldout_patches_by_camera=dict(sorted(heldout_coverage.items())),
                 before=before,after=after,evaluations=fit.nfev,status=fit.status,optimality=float(fit.optimality),seconds=time.perf_counter()-start,
                 calibrated_cameras=calibrating,maximum_evaluations=max_evaluations,freeze_cameras=freeze_cameras,
                 coherent_surfaces=coherent_surfaces,protect_cameras=protect_cameras,camera_checks=camera_checks,prediction_evaluated=bool(validation.any()),
                 point_support=dict(points=len(support['points']),training_observations=int(support_training.sum()),heldout_observations=int((~support_training).sum()),excluded_holdout_sites=support['excluded_holdout_sites'],unsupported_tracks=support.get('unsupported_tracks',[]),before=support_before,after=support_after,camera_checks=support_checks),
                 cameras_without_patch_holdouts=[i for i in ids if not heldout_coverage[i]],
                 camera_gate='Every camera with patch holdouts: median <= 1.1*before + 0.1 pixel; p90 <= 1.2*before + 0.2 pixel. Cameras without holdouts have no predictive certificate.',
                 coherence_before=coherence_summary(jets,edges,coherence,initial[nc:support_start]),coherence_after=coherence_summary(jets,edges,coherence,fit.x[nc:support_start]),
                 policy=('Cylindrical horizontal/vertical scale and vertical crop with fixed weak priors and bounds. ' if calibrate_panoramas else 'Fixed intrinsics. ')+'Original seed frame and baseline length; one withheld camera per jet; selection uses training fit only; nine samples weighted as one regional observation.')
    r,t,surfaces=unpack(fit.x);candidate=copy.deepcopy(pose)
    if calibrate_panoramas:candidate['camera_models']=camera_models(fit.x)
    for i in ids:candidate['poses'][str(i)].update(rotation=r[ix[i]].tolist(),translation=t[ix[i]].tolist(),center=(-r[ix[i]].T@t[ix[i]]).tolist(),differential_refinement=True)
    centers={j['track']:s[0].tolist() for j,s in zip(jets,surfaces)}
    centers.update({p['track']:x.tolist() for p,x in zip(support['points'],fit.x[support_start:].reshape(-1,3))})
    for p in candidate['points']:
        if p['track'] in centers:p['xyz']=centers[p['track']]
    out_jets=[];errors,positive=pixel_residual(fit.x)
    for k,(j,surface) in enumerate(zip(jets,surfaces)):
        row=copy.deepcopy(j);row.update(center=surface[0].tolist(),tangents=(surface[1:]/j['radius']).tolist(),normal=unit(np.cross(surface[1],surface[2])).tolist())
        for split,mask in [('training',training),('validation',validation)]:
            chosen=(ji==k)&mask;err=np.linalg.norm(errors[chosen],axis=1);row[split]=dict(evaluated=bool(len(err)),median_pixels=float(np.median(err)) if len(err) else 0.,maximum_pixels=float(np.max(err)) if len(err) else 0.,positive=bool(positive[chosen].all()))
        row['training_supported']=row['training']['positive'] and row['training']['maximum_pixels']<=1.5
        row['accepted']=row['validation']['evaluated'] and all(row[s]['positive'] and row[s]['maximum_pixels']<=1.5 for s in ('training','validation'));out_jets.append(row)
    receipt['training_supported_jets_after']=sum(r['training_supported'] for r in out_jets);receipt['accepted_jets_after']=sum(r['accepted'] for r in out_jets);candidate['jet_bundle']=receipt
    candidate['surface_jets']=out_jets
    result=candidate if applied else copy.deepcopy(pose);result['jet_bundle']=receipt
    return result,receipt,candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--jets',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--pose',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retain-point-support',action='store_true');p.add_argument('--coherent-surfaces',action='store_true');p.add_argument('--protect-cameras',action='store_true');p.add_argument('--calibrate-panoramas',action='store_true');p.add_argument('--max-evaluations',type=int,default=1000);args=p.parse_args()
    e=json.loads(args.evidence.read_text());pose=json.loads(args.pose.read_text());packet=json.loads(args.jets.read_text())
    result,receipt,candidate=refine_jets(packet,pose,pose.get('camera_models',e['cameras']),max_evaluations=args.max_evaluations,calibrate_panoramas=args.calibrate_panoramas,coherent_surfaces=args.coherent_surfaces,protect_cameras=args.protect_cameras,point_evidence=e if args.retain_point_support else None)
    from .scene_promotion import audit
    result=audit(e,result);candidate=audit(e,candidate) if candidate is not None else None
    receipt.update(point_support_source_sha256=hashlib.sha256(Path(__file__).with_name('point_support.py').read_bytes()).hexdigest(),coherence_source_sha256=hashlib.sha256(Path(__file__).with_name('surface_coherence.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_jets_sha256=hashlib.sha256(args.jets.read_bytes()).hexdigest(),parent_pose_sha256=hashlib.sha256(args.pose.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    if candidate is not None:args.out.with_suffix('.candidate.json').write_text(json.dumps(candidate,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
