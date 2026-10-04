"""Verify regional hypotheses against signed local image structure.

Descriptor similarity proposes correspondences. It cannot promote information
without local spatial agreement in the actual source observations.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from .core import samples,map_points
from .sweep_graph import components


def verify_edge(e,a,b):
    if e.get('inliers',0)<8 or 'matrix' not in e:return dict(e,accepted=False,verification='no_candidate_support')
    h=np.array(e['matrix']);pa=np.array(e['a_points']);height,width=a.shape;hb,wb=b.shape
    if len(pa)>120:pa=pa[np.linspace(0,len(pa)-1,120).astype(int)]
    offsets=np.array([(x,y) for y in np.linspace(-6,6,9) for x in np.linspace(-6,6,9)])
    shifts=np.array([(x,y) for y in (-4,-2,0,2,4) for x in (-4,-2,0,2,4)])
    q=pa[:,None,:]+offsets;qa=map_points(h,q);pb=map_points(h,pa)
    reference=samples(a,q/[width-1,height-1]);reference-=reference.mean(1,keepdims=True);norm=np.linalg.norm(reference,axis=1)
    moving=qa[None,:,:,:]+shifts[:,None,None,:];inside=np.all((moving>=0)&(moving<=[wb-1,hb-1]),axis=-1).mean(-1)>.98
    observed=samples(b,moving/[wb-1,hb-1]);observed-=observed.mean(-1,keepdims=True)
    corr=(observed*reference[None,:,:]).sum(-1)/np.maximum(np.linalg.norm(observed,axis=-1)*norm[None,:],1e-8)
    corr[~inside]=-1;choice=corr.argmax(0);best=corr[choice,np.arange(len(pa))]
    den=np.c_[pa,np.ones(len(pa))]@h[2];orientation=np.linalg.det(h)/(den**3)
    good=(best>=.6)&(norm>.025)&(orientation>0)&np.all((q>=0)&(q<=[width-1,height-1]),axis=(1,2))
    x=pa[good];y=(pb+shifts[choice])[good];count=len(x);cells=len(np.unique((x/[width,height]*[8,4]).astype(int),axis=0)) if count else 0
    span=np.ptp(x,axis=0)/[width,height] if count else np.zeros(2)
    accepted=count>=8 and cells>=3 and min(span)>.08
    result=dict(e,accepted=bool(accepted),descriptor_inliers=e.get('inliers',0),inliers=count,
        verification='signed_local_structure',verified_correlations=best[good].tolist(),verification_median=float(np.median(best)),
        a_points=x.tolist(),b_points=y.tolist(),cells=cells,span=span.tolist())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();g=json.loads(args.graph.read_text());planes=[]
    for r in g['records']:
        with np.load(args.features/(Path(r['name']).stem+'.npz')) as f:l=f['lab'][...,0]
        planes.append(l-ndimage.gaussian_filter(l,3))
    edges=[]
    for e in g['edges']:
        v=verify_edge(e,planes[e['a']],planes[e['b']]);edges.append(v)
        if len(edges)%100==0:print(json.dumps({'verified_pairs':len(edges),'accepted':sum(r['accepted'] for r in edges)}),flush=True)
    g['edges']=edges;g['components']=components(len(g['records']),edges);g['policy']['verification']=dict(min_correlation=.6,patch_side=9,patch_radius=6,search_shifts=[-4,-2,0,2,4],minimum_inliers=8,minimum_cells=3)
    g['parent_graph_sha256']=hashlib.sha256(args.graph.read_bytes()).hexdigest();g['verification_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(g,indent=2));print(json.dumps({'accepted':sum(r['accepted'] for r in edges),'components':g['components']}),flush=True)
if __name__=='__main__':main()
