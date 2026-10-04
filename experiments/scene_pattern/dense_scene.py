"""Build source-view fusion charts from continuously closed transport fields."""
import argparse,json,time,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from .transport_fusion import render

@lru_cache(maxsize=16384)
def largest_clique(vertices,edge_rows):
    adjacent={v:set(row) for v,row in zip(vertices,edge_rows)};best=[]
    def grow(chosen,candidates):
        nonlocal best
        if len(chosen)+len(candidates)<=len(best):return
        if not candidates:
            if len(chosen)>len(best):best=chosen
            return
        todo=sorted(candidates,key=lambda v:(-len(adjacent[v]&set(candidates)),v))
        while todo:
            if len(chosen)+len(todo)<=len(best):break
            v=todo.pop(0);grow(chosen+[v],[u for u in todo if u in adjacent[v]])
        if len(chosen)>len(best):best=chosen
    grow([],list(vertices));return tuple(sorted(best))


def build(folder):
    meta=json.loads((folder/'dense.json').read_text());records=meta['records'];rows={};anchors={}
    for pair in meta['pairs']:
        for a,b in ((pair['a'],pair['b']),(pair['b'],pair['a'])):
            with np.load(folder/f'{a:03d}-{b:03d}.npz') as f:rows[a,b]=dict(f)
            anchors.setdefault(a,[]).append(b)
    tracks=[];per_anchor=[]
    for a,neighbors in sorted(anchors.items()):
        neighbors=sorted(neighbors);grid=rows[a,neighbors[0]]['grid'].reshape(-1,2);used=0;covered=set()
        for k,point in enumerate(grid):
            available=[b for b in neighbors if rows[a,b]['valid'][k] and rows[a,b]['third_view_support'][k]>0]
            if len(available)<2:continue
            adjacency=tuple(tuple(c for c in available if rows[a,b]['compatible_cameras'][k,c]) for b in available);chosen=largest_clique(tuple(available),adjacency)
            if len(chosen)<2:continue
            observations=[dict(capture=a,site=k,xy=point.tolist(),centroid=point.tolist())];links=[]
            for b in chosen:
                row=rows[a,b];q=row['target'][k];observations.append(dict(capture=b,site=k,xy=q.tolist(),centroid=row['prior'][k].tolist()));links.append(dict(a=[a,k],b=[b,k],affine=row['affine'][k].tolist()));covered.add(b)
            tracks.append(dict(id=len(tracks),render_anchor=a,support_radius=6.,window_power=.5,observations=observations,transport_links=links));used+=1
        per_anchor.append(dict(anchor=a,regions=used,contributors=sorted(covered)))
    return dict(records=records,cameras=meta['cameras'],tracks=tracks,per_anchor=per_anchor,kind='source_view_regional_charts_not_global_point_tracks',
        summary=dict(regions=len(tracks),anchors=len(anchors),observed_captures=len({o['capture'] for t in tracks for o in t['observations']})),
        policy=dict(region_radius=6,window_power=.5,view_selection='largest mutually cycle-compatible source set at each grid point; deterministic graph rule'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dense',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,default=1536);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();e=build(args.dense);(args.out/'regional-scene.json').write_text(json.dumps(e,separators=(',',':')));print(json.dumps(e['summary']),flush=True)
    anchors=sorted([r['anchor'] for r in e['per_anchor'] if e['records'][r['anchor']]['panoramic']],key=lambda a:-next(r['regions'] for r in e['per_anchor'] if r['anchor']==a));views=[]
    for a in anchors:
        row=render(e,args.data,args.out,a,width=args.width);views.append(row);print(json.dumps({k:v for k,v in row.items() if k!='participation'}),flush=True)
    packet=dict(records=e['records'],views=views,summary=e['summary'],seconds=time.perf_counter()-start,source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('dense_scene.py','dense_transport.py','region_transport.py','transport_fusion.py')},
      limitations='Ordinary-average fusion inside measured local support in multiple source-camera charts. Source sets are mutually compatible under local three-view cycles. No full surface reconstruction, free virtual camera, super-resolution or global physical validation is claimed. The control uses translated region predictions; corrected maps include affine shape and image-fitted displacement.')
    (args.out/'fusion.json').write_text(json.dumps(packet,indent=2));print(json.dumps(dict(rendered=len(views),seconds=packet['seconds'])),flush=True)
if __name__=='__main__':main()
