"""Test camera geometry from dense, continuously closed regional observations.

Reference-grid tracklets sample nearby physical points independently. Quantized
image sites cap the vote count; they are not allowed to multiply support merely
because another reference chart sampled the same small image neighborhood.
"""
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from .multiview_geometry import bearings,hypotheses
from .scene_evidence import model_support,associations
from .scene_pose import reconstruct
from .scene_promotion import audit


def build(scene):
    cameras=scene['cameras'];tracks=[];pairs={}
    for source in sorted(scene['tracks'],key=lambda t:(-len(t['observations']),t['id'])):
        obs=[]
        for o in source['observations']:
            h,w=cameras[o['capture']]['shape'];q=np.rint(np.array(o['xy'])/2).astype(int);site=int(q[1]*(int(np.ceil(w/2))+1)+q[0]);obs.append(dict(capture=o['capture'],site=site,xy=o['xy'],centroid=o['centroid']))
        obs.sort(key=lambda o:o['capture']);ids=[o['capture'] for o in obs];tracks.append(dict(id=len(tracks),observations=obs,cycle_components=[ids],independent_cycles=(len(ids)-1)*(len(ids)-2)//2))
        for a,b in itertools.combinations(obs,2):pairs.setdefault((a['capture'],b['capture']),[]).append((a,b))
    edges=[]
    for (a,b),rows in sorted(pairs.items()):
        sa=set();sb=set();accepted=[]
        for x,y in rows:
            if x['site'] in sa or y['site'] in sb:continue
            accepted.append((x,y));sa.add(x['site']);sb.add(y['site'])
        ca,cb=cameras[a],cameras[b];pa=[x['xy'] for x,y in accepted];pb=[y['xy'] for x,y in accepted];ia=np.array([x['site'] for x,y in accepted]);ib=np.array([y['site'] for x,y in accepted]);aa=bearings(pa,ca['shape'],ca['focal'],ca['projection']);bb=bearings(pb,cb['shape'],cb['focal'],cb['projection']);models=hypotheses(aa,bb,ia,ib,trials=400) if len(accepted)>=20 else {};status,selected=model_support(models)
        edges.append(dict(a=a,b=b,candidates=[[int(x),int(y),0.,0] for x,y in zip(ia,ib)],a_points=pa,b_points=pb,models=models,status=status,selected_model=selected,candidate_measure='transported points; third candidate column unused'))
    return dict(records=scene['records'],cameras=cameras,tracks=tracks,edges=edges,associations=associations(scene['records'],tracks),policy=dict(site_cell_pixels=2,source='dense closed regional charts'),limitations='Conditional camera hypothesis from dense regional transports under nominal camera calibration. Track discovery and source selection use the collection; no independent physical validation follows.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();e=build(json.loads(args.scene.read_text()));e['source_hashes']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('dense_points.py','scene_pose.py','multiview_geometry.py')};(args.out/'evidence.json').write_text(json.dumps(e));print(json.dumps(dict(tracks=len(e['tracks']),supported_pairs=sum(x['selected_model'] is not None for x in e['edges']),seconds=time.perf_counter()-start)),flush=True);pose=audit(e,reconstruct(e));(args.out/'pose.json').write_text(json.dumps(pose,indent=2));print(json.dumps(dict(accepted=pose['accepted'],cameras=len(pose.get('poses',{})),points=len(pose.get('points',[])),promotion=pose.get('promotion_counts'),seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
