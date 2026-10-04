"""Fixed-policy coarse-scale search for components not tied to the sweeps."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from .sweep_features import describe,match
from .sweep_graph import load_features,components
from .sweep_verify import verify_edge
from .run import image


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();g=json.loads(args.graph.read_text());records=g['records'];groups=components(len(records),g['edges']);panos={i for i,r in enumerate(records) if r['panoramic']};mainset=set(max(groups,key=lambda c:len(set(c)&panos)));pending=set(range(len(records)))-mainset
    cache={};planes={};edges=[];audit=[]
    for i in sorted(pending):
        a=describe(image(args.data/records[i]['name']),side=128);l=a['lab'][...,0];plane=l-ndimage.gaussian_filter(l,3)
        for j in sorted(mainset):
            if j not in cache:
                path=args.features/(Path(records[j]['name']).stem+'.npz');cache[j]=load_features(path)
                with np.load(path) as f:ll=f['lab'][...,0]
                planes[j]=ll-ndimage.gaussian_filter(ll,3)
            e=match(a,cache[j]);e.update(a=i,b=j,stage=2);v=verify_edge(e,plane,planes[j]);audit.append(dict(a=i,b=j,accepted=v['accepted'],inliers=v.get('inliers',0)))
            if v['accepted']:
                with np.load(args.features/(Path(records[i]['name']).stem+'.npz')) as f:full=f['shape']
                factor=(np.array(full)[::-1]-1)/(a['shape'][::-1]-1)
                v['a_points']=(np.array(v['a_points'])*factor).tolist();v['matrix']=(np.array(v['matrix'])@np.diag([1/factor[0],1/factor[1],1])).tolist();v['scale_expansion']=128
                edges.append(v);print(json.dumps({'expanded_a':i,'b':j,'inliers':v['inliers']}),flush=True)
    g['edges']+=edges;g['components']=components(len(records),g['edges']);g['expansion_audit']=audit;g['policy']['unanchored_expansion']='128-pixel representation of each unanchored image, compared against every anchored image, with signed local verification';g['expansion_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(g,indent=2));print(json.dumps({'expanded_edges':len(edges),'components':g['components']}),flush=True)
if __name__=='__main__':main()
