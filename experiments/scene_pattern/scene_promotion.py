"""Keep third-view evidence separate from two-view depth proposals."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .scene_pose import track_bearing
from .multiview_geometry import unit


def audit(evidence,reconstruction):
    lookup={t['id']:t for t in evidence['tracks']};poses={int(i):p for i,p in reconstruction.get('poses',{}).items()};counts={};cameras=reconstruction.get('camera_models',evidence['cameras'])
    for point in reconstruction.get('points',[]):
        obs=[];x=np.asarray(point['xyz'])
        for observation in lookup[point['track']]['observations']:
            i=observation['capture']
            if i not in poses:continue
            p=poses[i];r=np.asarray(p['rotation']);t=np.asarray(p['translation']);b=track_bearing(observation,cameras);q=r@x+t
            error=float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(unit(q),b)),unit(q)@b)));positive=bool(q@b>0)
            obs.append(dict(capture=i,error_degrees=error,positive_depth=positive,agrees=bool(positive and error<=1.2),pose_holdout=bool(observation['site']%7==0 and p['role']=='heldout_pose')))
        support=sum(o['agrees'] for o in obs);point['observations']=obs;point['support_count']=support
        point['promotion']='third_view_consistent' if reconstruction.get('accepted') and len(obs)>=3 and all(o['agrees'] for o in obs) else 'tentative_depth'
        counts[point['promotion']]=counts.get(point['promotion'],0)+1
    reconstruction['promotion_counts']=counts
    reconstruction['promotion_limitations']='Third-view consistency is conditional on the fitted camera hypothesis and the discovered tracks. It does not establish a resolved surface, visibility, independent validation or safe image fusion.'
    return reconstruction


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=audit(json.loads(args.evidence.read_text()),json.loads(args.pose.read_text()));result['promotion_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['parent_pose_sha256']=hashlib.sha256(args.pose.read_bytes()).hexdigest();args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result['promotion_counts']))
if __name__=='__main__':main()
