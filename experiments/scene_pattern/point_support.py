"""Retain measured point constraints outside the differential surface subset."""
import numpy as np
from .surface_jets import image_rays


def prepare(evidence,pose,jets,cameras):
    if evidence is None:return dict(points=[],observations=[],excluded_holdout_sites=0)
    tracks={t['id']:t for t in evidence['tracks']};represented={j['track'] for j in jets};blocked=set()
    for j in jets:
        blocked.update((o['capture'],o['site']) for o in tracks[j['track']]['observations'] if o['capture']==j['holdout_capture'])
    points=[];rows=[];unsupported=[]
    for point in pose['points']:
        if point['track'] in represented:continue
        observations=[o for o in tracks[point['track']]['observations'] if str(o['capture']) in pose['poses']]
        train=[o for o in observations if o['site']%7!=0 and (o['capture'],o['site']) not in blocked]
        if len(train)<2:unsupported.append(point['track']);continue
        rays=np.array([image_rays([o['xy']],cameras[o['capture']])[0]@np.asarray(pose['poses'][str(o['capture'])]['rotation']) for o in train])
        matrix=(np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]).sum(0)
        if np.linalg.cond(matrix)>1e5:unsupported.append(point['track']);continue
        k=len(points);points.append(point)
        for o in observations:
            rows.append(dict(point=k,capture=o['capture'],site=o['site'],xy=o['xy'],training=o['site']%7!=0 and (o['capture'],o['site']) not in blocked))
    return dict(points=points,observations=rows,excluded_holdout_sites=len(blocked),unsupported_tracks=unsupported,
                policy='Original retained points outside the surface subset; at least two training rays with condition <=1e5. Physical site modulo seven and every surface-heldout camera/site are excluded from point fitting. Same-collection conditional prediction only.')
