"""Test whether a surface's training evidence also admits infinite depth.

A passing infinity hypothesis establishes a missing finite-depth constraint;
a failing hypothesis alone does not establish accurate or unique finite depth.
No withheld observation enters this diagnostic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .surface_jets import image_rays,project
from .surface_coherence import neighbors


def infinity_error(jet,cameras,poses):
    rows=[o for o in jet['observations'] if o['capture']!=jet['holdout_capture']]
    reference=next(o for o in rows if o['capture']==jet['reference_capture'])
    delta=np.array([(u,v) for v in (-1.,0.,1.) for u in (-1.,0.,1.)])*jet['radius']
    xy=np.asarray(reference['xy'])+delta@np.asarray(reference['affine']).T
    rays=image_rays(xy,cameras[reference['capture']])@np.asarray(poses[str(reference['capture'])]['rotation'])
    errors=[]
    for o in rows:
        i=o['capture'];camera=cameras[i];expected=np.asarray(o['xy'])+delta@np.asarray(o['affine']).T
        residual=project(rays,camera,dict(rotation=poses[str(i)]['rotation'],translation=np.zeros(3)))-expected
        if camera['projection']=='cylindrical':
            period=2*np.pi*camera.get('focal_x',camera['focal']);residual[:,0]=(residual[:,0]+period/2)%period-period/2
        errors.extend(np.linalg.norm(residual,axis=1))
    return float(max(errors))


def analyze(jets,cameras,poses):
    edges=neighbors(jets);a=[e['a'] for e in edges];b=[e['b'] for e in edges]
    _,labels=connected_components(coo_matrix((np.ones(2*len(a)),(a+b,b+a)),shape=(len(jets),len(jets))))
    rows=[dict(track=j['track'],component=int(labels[k]),maximum_training_infinity_pixels=infinity_error(j,cameras,poses)) for k,j in enumerate(jets)]
    components=[]
    for label in sorted(set(labels)):
        ids=np.flatnonzero(labels==label);compatible=[k for k in ids if rows[k]['maximum_training_infinity_pixels']<=1.5]
        components.append(dict(component=int(label),patches=len(ids),infinity_compatible_patches=len(compatible),all_patches_admit_infinity=len(ids)==len(compatible),
            training_cameras=sorted({o['capture'] for k in ids for o in jets[k]['observations'] if o['capture']!=jets[k]['holdout_capture']}),tracks=[jets[k]['track'] for k in ids]))
    return dict(patches=rows,components=components,summary=dict(patches=len(jets),infinity_compatible_patches=sum(r['maximum_training_infinity_pixels']<=1.5 for r in rows),components_admitting_infinity=sum(c['all_patches_admit_infinity'] for c in components)),
        policy='Project nine reference-region rays with camera rotation only, at infinite depth. Compare only measured training centers and affines at the unchanged 1.5-pixel tolerance. All-infinity component flag is a local image-evidence diagnostic; it does not test the finite-world continuity regularizer or certify other components.')


def main():
    p=argparse.ArgumentParser()
    for n in ('evidence','pose','jets','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();e=json.loads(a.evidence.read_text());pose=json.loads(a.pose.read_text());packet=json.loads(a.jets.read_text());jets=packet.get('surface_jets',packet.get('results'));jets=[j for j in jets if 'training' in j and j['training']['positive'] and j['training']['maximum_pixels']<=1.5]
    r=analyze(jets,pose.get('camera_models',e['cameras']),pose['poses']);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r['summary']))


if __name__=='__main__':main()
