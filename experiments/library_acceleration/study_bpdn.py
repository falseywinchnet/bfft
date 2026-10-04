"""Image-patch sparse coding against the actual SPORCO BPDN class."""
import argparse,json,inspect,hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
from skimage import data
import sporco
from sporco.admm.bpdn import BPDN
from .bpdn_adapter import make_solver,SparseCoding
from .certified_transport import run_certified
from .engine import run


def problem(side,signals,source,seed):
    # Standard separable overcomplete cosine dictionary, normalized columns.
    freq=int(1.5*side)
    D1=np.cos(np.pi*(np.arange(side)[:,None]+.5)*np.arange(freq)[None,:]/freq)
    D=np.kron(D1,D1);D/=np.linalg.norm(D,axis=0)
    im=np.asarray(getattr(data,source)(),dtype=float)/255
    rng=np.random.default_rng(seed);patches=[]
    for _ in range(signals):
        i=rng.integers(0,im.shape[0]-side+1);j=rng.integers(0,im.shape[1]-side+1)
        patch=im[i:i+side,j:j+side].ravel();patch-=patch.mean()
        patches.append(patch+.03*rng.normal(size=patch.shape))
    return D,np.column_stack(patches)


def gap(D,S,Y,lam):
    r=D@Y-S;corr=D.T@r
    theta=r/np.maximum(1,np.max(np.abs(corr),axis=0,keepdims=True)/lam)
    primal=.5*np.sum(r*r)+lam*np.sum(np.abs(Y))
    dual=-.5*np.sum(theta*theta)-np.sum(theta*S)
    return float(primal),float(primal-dual)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--sides',default='8,16')
    p.add_argument('--signals',default='1,16');p.add_argument('--sources',default='camera,coins')
    p.add_argument('--passes',type=int,default=500);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--tolerance',type=float,default=.02);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--depths',default='2,4,8');a=p.parse_args();rows=[];dest=Path(a.out)
    metadata=dict(protocol=vars(a),sporco=sporco.__version__,
                  source_sha256=hashlib.sha256(inspect.getsource(BPDN).encode()).hexdigest(),
                  certificate='exact-arithmetic cumulative state/output bound; float64 evaluation',
                  options=dict(AutoRho=False,RelaxParam=1,FastSolve=True,ReturnX=False))
    for side in map(int,a.sides.split(',')):
      for signals in map(int,a.signals.split(',')):
       for source in a.sources.split(','):
        D,S=problem(side,signals,source,a.seed)
        for lam in [.02,.1]:
            def model():return SparseCoding(make_solver(D,S,lam,a.passes))
            funcs=dict(library=lambda:(make_solver(D,S,lam,a.passes).solve(),{}),
                       quotient=lambda:run(model(),a.passes,False),
                       certified=lambda:run_certified(model(),a.passes,tolerance=a.tolerance,
                                                      depths=tuple(map(int,a.depths.split(',')))))
            for fn in funcs.values():fn()
            timing={k:[] for k in funcs};out={};stats={}
            for rep in range(a.repeats):
              for key in np.random.default_rng(rep).permutation(list(funcs)):
                start=perf_counter();out[key],stats[key]=funcs[key]();timing[key].append(perf_counter()-start)
            ref=out['library'];got=out['certified'];e=float(np.linalg.norm(got-ref));f,g=gap(D,S,got,lam);rf,rg=gap(D,S,ref,lam)
            row=dict(side=side,signals=signals,source=source,lmbda=lam,seed=a.seed,timing=timing,stats=stats['certified'],
                quotient_max_error=float(np.max(np.abs(out['quotient']-ref))),relative_l2=e/max(float(np.linalg.norm(ref)),1e-300),
                error_norm=e,bound_excess=e-stats['certified']['output_bound'],objective=f,reference_objective=rf,
                duality_gap=g,reference_duality_gap=rg,initial_objective=.5*float(np.sum(S*S)),
                speedup=float(np.median(timing['library'])/np.median(timing['certified'])),
                engine_only_speedup=float(np.median(timing['quotient'])/np.median(timing['certified'])))
            rows.append(row);dest.write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2))
            print({k:v for k,v in row.items() if k not in ('timing','stats')},
                  {k:v for k,v in row['stats'].items() if k!='events'},flush=True)
    print('saved',dest,flush=True)

if __name__=='__main__':main()
