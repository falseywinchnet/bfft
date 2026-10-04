"""Compare joint point predictions with a fixed-camera, training-only control.

The original stored points were fit before these holdouts were reserved. Their
residual is a stability reference, not a fair training-only prediction baseline.
This audit does not change the existing candidate acceptance decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .point_support import prepare
from .surface_jets import image_rays,project


def errors(x,observations,cameras,poses):
    result=[]
    for o in observations:
        i=o['capture'];camera=cameras[i];v=project([x],camera,poses[str(i)])[0]-o['xy']
        if camera['projection']=='cylindrical':
            period=2*np.pi*camera.get('focal_x',camera['focal']);v[0]=(v[0]+period/2)%period-period/2
        result.append(v)
    return np.asarray(result)


def fit_training_point(observations,cameras,poses):
    rows=[o for o in observations if o['training']]
    centers=np.array([poses[str(o['capture'])]['center'] for o in rows])
    rays=np.array([image_rays([o['xy']],cameras[o['capture']])[0]@np.asarray(poses[str(o['capture'])]['rotation']) for o in rows])
    projector=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
    x=np.linalg.solve(projector.sum(0),np.einsum('nij,nj->i',projector,centers))
    fitted=least_squares(lambda x:errors(x,rows,cameras,poses).ravel(),x,loss='soft_l1',f_scale=1.,max_nfev=100,ftol=1e-6)
    return fitted.x,dict(evaluations=fitted.nfev,status=fitted.status,optimality=float(fitted.optimality))


def run(evidence,pose,packet,candidate):
    cameras=pose.get('camera_models',evidence['cameras']);jets=[r for r in packet['results'] if 'training' in r and r['training']['positive'] and r['training']['maximum_pixels']<=1.5]
    support=prepare(evidence,pose,jets,cameras);after={p['track']:p['xyz'] for p in candidate['points']};rows=[];fits=[]
    grouped={k:[] for k in range(len(support['points']))}
    for o in support['observations']:grouped[o['point']].append(o)
    for k,p in enumerate(support['points']):
        obs=grouped[k];x,receipt=fit_training_point(obs,cameras,pose['poses']);fits.append(dict(track=p['track'],**receipt));held=[o for o in obs if not o['training']]
        if not held:continue
        before=errors(p['xyz'],held,cameras,pose['poses']);control=errors(x,held,cameras,pose['poses']);joint=errors(after[p['track']],held,candidate.get('camera_models',cameras),candidate['poses'])
        for o,a,b,c in zip(held,before,control,joint):rows.append(dict(track=p['track'],capture=o['capture'],site=o['site'],stored=float(np.linalg.norm(a)),training_control=float(np.linalg.norm(b)),joint=float(np.linalg.norm(c))))
    def stats(rows,key):
        a=[r[key] for r in rows];return dict(median=float(np.median(a)),p90=float(np.quantile(a,.9)))
    by_camera={str(i):{key:stats([r for r in rows if r['capture']==i],key) for key in ('stored','training_control','joint')} for i in sorted({r['capture'] for r in rows})}
    for c in by_camera.values():c['passes_training_control']=bool(c['joint']['median']<=1.1*c['training_control']['median']+.1 and c['joint']['p90']<=1.2*c['training_control']['p90']+.2)
    return dict(points=len(fits),heldout_observations=len(rows),capped_fits=sum(f['status']==0 for f in fits),summary={key:stats(rows,key) for key in ('stored','training_control','joint')},by_camera=by_camera,point_fits=fits,observations=rows,
                policy='Fixed original cameras; point initialization and robust refinement use training rays only. Same point support and heldout observations for all columns. Diagnostic only: original candidate rejection stays unchanged.')


def main():
    p=argparse.ArgumentParser()
    for n in ('evidence','pose','jets','candidate','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();paths=[a.evidence,a.pose,a.jets,a.candidate];r=run(*[json.loads(p.read_text()) for p in paths]);r['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};r['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(dict(summary=r['summary'],points=r['points'],capped_fits=r['capped_fits'],failed_cameras=[i for i,v in r['by_camera'].items() if not v['passes_training_control']])));


if __name__=='__main__':main()
