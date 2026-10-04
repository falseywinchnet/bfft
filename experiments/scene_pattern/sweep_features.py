"""Multiscale texture descriptors at automatically transported region centers."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.feature import daisy
from skimage.measure import ransac
from skimage.transform import AffineTransform,ProjectiveTransform
from .core import analyze, samples, map_points
from .run import image


def describe(rgb,side=None):
    s=analyze(rgb,side=side or (768 if rgb.shape[1]/rgb.shape[0]>2.4 else 384))
    h,w=s.rgb.shape[:2]; xy=s.centers*np.array([w-1,h-1]); points=[];descs=[]
    # Gradient distributions are comparison descriptors, not an interest-point
    # detector: all sites originate in the five causal-density segmentations.
    carrier=s.cartoon[...,0]+s.texture[...,0]
    for radius in (8,16,24):
        d=daisy(carrier,step=1,radius=radius,rings=2,histograms=4,orientations=8,normalization='l1')
        valid=np.all((xy>radius+1)&(xy<np.array([w,h])-radius-2),axis=1)
        p=xy[valid]; z=p-radius
        desc=samples(d,z/np.array([d.shape[1]-1,d.shape[0]-1]))
        desc=np.sqrt(np.maximum(desc,0)); desc/=np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-8)
        color=samples(s.cartoon,s.centers[valid])*[.2,1.5,1.5]
        desc=np.c_[desc,color].astype(np.float32)
        points.append(p);descs.append(desc)
    return dict(points=np.concatenate(points),descriptors=np.concatenate(descs),
        shape=np.array([h,w]),rgb=s.rgb,lab=s.lab,texture=s.texture,receipt=np.array(json.dumps(s.receipt)))


def match(a,b):
    da,db=a['descriptors'],b['descriptors']
    if min(len(da),len(db))<12:return dict(accepted=False,reason='insufficient_regions')
    dist=cdist(da,db,'sqeuclidean');j=np.argmin(dist,axis=1);back=np.argmin(dist,axis=0)
    # Ratio compares different spatial sites, not duplicate scales at one site.
    best=dist[np.arange(len(j)),j].copy()
    for k in range(len(j)):
        near=np.linalg.norm(b['points']-b['points'][j[k]],axis=1)<3
        dist[k,near]=np.inf
    second=np.min(dist,axis=1)
    ii=np.flatnonzero((back[j]==np.arange(len(j)))&(best<.82**2*second))
    if len(ii)<10:return dict(accepted=False,reason='few_distinct_matches',matches=len(ii))
    pa,pb=a['points'][ii],b['points'][j[ii]]
    # Collapse duplicate reference sites across radii before consensus.
    _,keep=np.unique(np.rint(pa),axis=0,return_index=True);pa,pb=pa[keep],pb[keep]
    if len(pa)<10:return dict(accepted=False,reason='few_unique_matches',matches=len(pa))
    try:model,good=ransac((pa,pb),ProjectiveTransform,min_samples=4,residual_threshold=4,max_trials=1200,random_state=29)
    except TypeError:model,good=ransac((pa,pb),ProjectiveTransform,min_samples=4,residual_threshold=4,max_trials=1200,rng=np.random.default_rng(29))
    if model is None:return dict(accepted=False,reason='no_consensus',matches=len(pa))
    n=int(good.sum());x,y=pa[good],pb[good];h,w=a['shape'];hb,wb=b['shape']
    cells=len(np.unique((x/np.array([w,h])*[8,4]).astype(int),axis=0))
    span=np.ptp(x,axis=0)/[w,h] if n else np.zeros(2)
    error=float(np.median(np.linalg.norm(model(x)-y,axis=1))) if n else 999.
    # Only certify the measured domain; do not impose full-frame homography.
    accepted=n>=10 and cells>=4 and min(span)>.08 and np.all(np.isfinite(model.params))
    return dict(accepted=bool(accepted),matches=len(pa),inliers=n,cells=cells,span=span.tolist(),error=error,
        matrix=model.params.tolist(),a_points=x.tolist(),b_points=y.tolist(),reason='supported' if accepted else 'weak_support')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data/'manifest.json').read_text());start=time.perf_counter()
    for r in manifest:
        dest=args.out/(Path(r['name']).stem+'.npz')
        if not dest.exists():
            features=describe(image(args.data/r['name']));np.savez_compressed(dest,**features)
        else:
            with np.load(dest) as f: features=dict(f)
        print(json.dumps({'features':r['name'],'sites':len(features['points']),'seconds':time.perf_counter()-start}),flush=True)
        if args.probe and manifest.index(r)>=1:break
    if args.probe:
        a=np.load(args.out/(Path(manifest[0]['name']).stem+'.npz'));b=np.load(args.out/(Path(manifest[1]['name']).stem+'.npz'))
        print(json.dumps(match(a,b)),flush=True)
if __name__=='__main__':main()
