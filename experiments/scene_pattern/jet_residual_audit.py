"""Separate center and differential disagreement in retained patch predictions.

This reports every fitted observation; it does not select or warp any pixels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .surface_jets import project


def audit(pose,cameras):
    cameras=pose.get('camera_models',cameras);rows=[]
    delta=np.array([(u,v) for v in (-1.,0.,1.) for u in (-1.,0.,1.)])
    for jet in pose['surface_jets']:
        world=np.asarray(jet['center'])+jet['radius']*delta@np.asarray(jet['tangents'])
        for obs in jet['observations']:
            i=obs['capture'];camera=cameras[i]
            expected=np.asarray(obs['xy'])+jet['radius']*delta@np.asarray(obs['affine']).T
            error=project(world,camera,pose['poses'][str(i)])-expected
            if camera['projection']=='cylindrical':
                period=2*np.pi*camera.get('focal_x',camera['focal']);error[:,0]=(error[:,0]+period/2)%period-period/2
            rows.append(dict(track=jet['track'],capture=i,heldout=i==jet['holdout_capture'],
                             center_pixels=float(np.linalg.norm(error[4])),
                             differential_rms_pixels=float(np.sqrt(np.mean(np.sum((error-error[4])**2,axis=1)))),
                             patch_maximum_pixels=float(np.max(np.linalg.norm(error,axis=1))),
                             center_vector_pixels=error[4].tolist(),xy=obs['xy']))
    def stats(items):
        return dict(patches=len(items),**{key:dict(median=float(np.median([r[key] for r in items])),p90=float(np.quantile([r[key] for r in items],.9))) for key in ('center_pixels','differential_rms_pixels','patch_maximum_pixels')}) if items else dict(patches=0)
    by_camera={str(i):{label:stats([r for r in rows if r['capture']==i and r['heldout']==held]) for label,held in [('training',False),('heldout',True)]} for i in sorted({r['capture'] for r in rows})}
    return dict(summary={label:stats([r for r in rows if r['heldout']==held]) for label,held in [('training',False),('heldout',True)]},by_camera=by_camera,observations=rows,
                policy='One vote per patch observation; center residual and offset-removed differential RMS; no filtering by accepted status.',
                limitations='Conditional residual decomposition at processing resolution. Heldout patches have been used to select development candidates; this is not unseen-scene validation.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--pose',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=audit(json.loads(args.pose.read_text()),json.loads(args.evidence.read_text())['cameras'])
    result['parent_pose_sha256']=hashlib.sha256(args.pose.read_bytes()).hexdigest();result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']))


if __name__=='__main__':main()
