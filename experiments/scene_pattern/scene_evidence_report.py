"""Private source-observation reviewer; never substitutes a graph for fusion."""
import argparse,json,shutil
from pathlib import Path
from collections import Counter


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--calibration',type=Path);args=p.parse_args()
    e=json.loads(args.evidence.read_text());pose=json.loads(args.pose.read_text());args.out.mkdir(parents=True,exist_ok=True)
    images=args.out/'images';images.mkdir(exist_ok=True)
    for record in e['records']:shutil.copyfile(args.data/record['name'],images/record['name'])
    rows=[]
    lookup={(o['capture'],o['site']):o['xy'] for t in e['tracks'] for o in t['observations']}
    for edge in e['edges']:
        a,b=edge['a'],edge['b'];selected=edge['selected_model'];m=edge['models'].get(selected,{})
        pairs=[]
        for k in m.get('train_inliers',[])+m.get('heldout_inliers',[]):
            ca,cb,_,_=edge['candidates'][k]
            if 'a_points' in edge and 'b_points' in edge:pairs.append([edge['a_points'][k],edge['b_points'][k]])
            elif (a,ca) in lookup and (b,cb) in lookup:pairs.append([lookup[a,ca],lookup[b,cb]])
        rows.append(dict(a=a,b=b,status=edge['status'],pairs=pairs,models={name:{k:v for k,v in model.items() if k not in ('matrix','train_inliers','heldout_inliers','poses')} for name,model in edge['models'].items()}))
    summary=dict(captures=len(e['records']),panoramas=sum(r['panoramic'] for r in e['records']),tested_pairs=len(e['edges']),pair_status=dict(Counter(r['status'] for r in rows)),tracks=len(e['tracks']),cycle_tracks=sum(bool(t['cycle_components']) for t in e['tracks']),conflicts=len(e['conflicts']) if 'conflicts' in e else None,seconds=e.get('seconds'))
    packet=dict(summary=summary,records=e['records'],cameras=e['cameras'],edges=rows,tracks=e['tracks'],associations=e['associations'],pose=pose,policy=e['policy'],limitations=e['limitations'])
    if args.calibration:packet['calibration']=json.loads(args.calibration.read_text())
    (args.out/'data.json').write_text(json.dumps(packet,separators=(',',':')));shutil.copyfile(Path(__file__).with_name('scene-evidence-viewer.html'),args.out/'index.html');print(json.dumps(summary))
if __name__=='__main__':main()
