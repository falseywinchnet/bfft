"""Refine region seeds on fixed overlap without rewarding frame collapse."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy import ndimage,optimize
from .core import grid,map_points,samples,matrix_params,params_matrix,warp
from .run import save


def feature(scene):
    lab=scene.lab
    f=lab-ndimage.gaussian_filter(lab,(2,2,0))
    return f/np.maximum(np.std(f,axis=(0,1),keepdims=True),.008)


def fit(a,b,initial):
    q=grid((80,60)).reshape(-1,2);mapped=map_points(initial,q)
    valid=np.all((q>.03)&(q<.97)&(mapped>.06)&(mapped<.94),axis=1);q=q[valid]
    if len(q)<200:return initial,dict(accepted=False,count=len(q))
    aa,bb=feature(a),feature(b);target=samples(aa,q)
    # Split spatial cells, rather than letting an optimizer select its own mask.
    train=np.arange(len(q))%3!=0
    def residual(p):
        h=matrix_params(p);r=map_points(h,q)
        return (samples(bb,r)-target)*[1,.35,.35]
    def objective(p):return residual(p)[train].ravel()
    def jac(p):return np.column_stack([(objective(p+d)-objective(p-d))/(2e-4) for d in np.eye(8)*1e-4])
    start=params_matrix(initial);res=optimize.least_squares(objective,start,jac=jac,loss='soft_l1',f_scale=.4,max_nfev=70)
    before=float(np.mean(np.minimum(residual(start)[~train]**2,4)))
    after=float(np.mean(np.minimum(residual(res.x)[~train]**2,4)))
    candidate=matrix_params(res.x);motion=np.linalg.norm(map_points(candidate,q)-mapped[valid],axis=1)
    accepted=bool(after<before and np.quantile(motion,.95)<.12)
    return candidate if accepted else initial,dict(accepted=accepted,count=len(q),before=before,after=after)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    r=json.loads((args.base/'registration.json').read_text())
    with (args.base/'scenes.pkl').open('rb') as f:scenes=pickle.load(f)
    for i,pair in enumerate(r['pairs']):
        # Start from matched scene regions; the old final's >45% overlap score
        # excluded valid narrow-overlap seeds on these freely captured images.
        initial=np.asarray(pair['matrices']['regional_coarse'])
        h,rec=fit(scenes[i],scenes[i+1],initial);pair['overlap_refinement']=rec;pair['matrices']['previous_final']=pair['matrices']['final'];pair['matrices']['final']=h.tolist()
        warped,valid=warp(scenes[i+1].rgb,h,scenes[i].rgb.shape[:2],method='linear');save(np.where(valid[...,None],.5*(warped+scenes[i].rgb),scenes[i].rgb*.25),args.out/f'join{i+1:02}.jpg')
        print(json.dumps(dict(pair=i+1,**rec)),flush=True)
    (args.out/'registration.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
