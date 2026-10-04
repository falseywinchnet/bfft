"""Fixed panorama projection-family screen with predictive pose checks.

This is model development on the existing collection, not an independent audit.
The only reused graph information is its 2-D regional candidate inventory.
"""
import argparse,copy,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .scene_evidence import canonical_features,model_support,build_tracks
from .multiview_geometry import bearings,hypotheses
from .scene_pose import reconstruct


def evaluate(spec):
    evidence_path,features_path,out,projection,factor=spec
    start=time.perf_counter();base=json.loads(Path(evidence_path).read_text());n=sum(r['panoramic'] for r in base['records'])
    assert all(r['panoramic'] for r in base['records'][:n]),'explicit panorama index remapping required'
    features=[canonical_features(Path(features_path)/(Path(r['name']).stem+'.npz')) for r in base['records'][:n]]
    e=dict(records=base['records'][:n],cameras=copy.deepcopy(base['cameras'][:n]),edges=[])
    for c in e['cameras']:c['focal']*=factor;c['projection']=projection
    for edge in base['edges']:
        a,b=edge['a'],edge['b']
        if max(a,b)>=n:continue
        cc=edge['candidates'];models={}
        if len(cc)>=20:
            ca,cb=e['cameras'][a],e['cameras'][b];sa=np.array([c[0] for c in cc]);sb=np.array([c[1] for c in cc]);aa=bearings(features[a]['sites'][sa],ca['shape'],ca['focal'],ca['projection']);bb=bearings(features[b]['sites'][sb],cb['shape'],cb['focal'],cb['projection']);models=hypotheses(aa,bb,sa,sb,trials=400)
        status,selected=model_support(models);e['edges'].append(dict(a=a,b=b,candidates=cc,models=models,status=status,selected_model=selected))
    e['tracks'],e['conflicts'],e['alternative_candidates']=build_tracks(e['edges'],features)
    pose=reconstruct(e);name=f'{projection}-{factor:g}';result=dict(name=name,projection=projection,focal_factor=factor,seconds=time.perf_counter()-start,
      supported_pairs=sum(r['selected_model'] is not None for r in e['edges']),translation_pairs=sum(r['status']=='translation_supported' for r in e['edges']),cycle_tracks=sum(bool(t['cycle_components']) for t in e['tracks']),pose=pose)
    target=Path(out)/name;target.mkdir(parents=True,exist_ok=True);(target/'evidence.json').write_text(json.dumps(e));(target/'result.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);args=p.parse_args();Path(args.out).mkdir(parents=True,exist_ok=True)
    specs=[(args.evidence,args.features,args.out,projection,factor) for projection in ('cylindrical','equirectangular') for factor in (.8,1.,1.25,1.6)]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(evaluate,specs):
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='pose'}|dict(accepted=row['pose']['accepted'],cameras=len(row['pose'].get('poses',{})))),flush=True)
    packet=dict(policy=dict(projections=['cylindrical','equirectangular'],focal_factors=[.8,1.,1.25,1.6],pair_trials=400,angular_threshold_degrees=1.2,stage='panoramas_only'),results=rows,
      source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('scene_calibration.py','scene_pose.py','scene_evidence.py','multiview_geometry.py')},limitations='Development comparison of eight nominal global camera families. No per-object or per-capture tuning. A selected family requires further independent validation; phone panorama stitching can require a noncentral or spatially varying model.')
    (Path(args.out)/'screen.json').write_text(json.dumps(packet,indent=2))
if __name__=='__main__':main()
