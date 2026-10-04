"""Matched fixed-rho SPORCO TV deconvolution finite trajectories."""
import argparse,json,platform,hashlib,inspect
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy,skimage,sporco
from skimage import data
from skimage.transform import resize
from sporco.admm.tvl2 import TVL2Deconv
from .sporco_adapter import make_solver,TVDeconv
from .certified_transport import run_certified
from .engine import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--sizes',default='128,256');p.add_argument('--sources',default='camera,coins')
    p.add_argument('--passes',type=int,default=200);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--tolerance',type=float,default=.02);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lambdas',default='.02,.1');a=p.parse_args()
    rows=[];dest=Path(a.out)
    metadata=dict(protocol=vars(a),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
                  sporco=sporco.__version__,machine=platform.machine(),
                  source_sha256=hashlib.sha256(inspect.getsource(TVL2Deconv).encode()).hexdigest(),
                  certificate='exact-arithmetic Euclidean bound; float64 numerical evaluation',
                  options=dict(AutoRho=False,RelaxParam=1,FastSolve=True),
                  scope='same fixed pass count; not default adaptive-rho stopping behavior')
    for size in map(int,a.sizes.split(',')):
      for name in a.sources.split(','):
        clean=resize(np.asarray(getattr(data,name)(),dtype=float)/255,(size,size),preserve_range=True,anti_aliasing=True)
        for width in [1,9]:
          psf=np.ones((1,1))
          if width>1:
            t=np.arange(width)-(width-1)/2
            psf=np.exp(-(t[:,None]**2+t[None,:]**2)/(2*(width/6)**2));psf/=psf.sum()
          # Match the actual library's circular convolution and PSF origin.
          af=np.fft.rfftn(psf,s=clean.shape)
          blurred=np.fft.irfftn(af*np.fft.rfftn(clean),s=clean.shape)
          noisy=blurred+.04*np.random.default_rng(a.seed).standard_normal(clean.shape)
          for lam in map(float,a.lambdas.split(',')):
            def model():return TVDeconv(make_solver(noisy,psf,lam,a.passes))
            funcs=dict(library=lambda:(make_solver(noisy,psf,lam,a.passes).solve(),{}),
                       quotient=lambda:run(model(),a.passes-1,False),
                       certified=lambda:run_certified(model(),a.passes-1,tolerance=a.tolerance))
            for fun in funcs.values():fun()
            timing={k:[] for k in funcs};outputs={};stats={}
            for rep in range(a.repeats):
              for key in np.random.default_rng(rep).permutation(list(funcs)):
                start=perf_counter();outputs[key],stats[key]=funcs[key]();timing[key].append(perf_counter()-start)
            ref=outputs['library'];got=outputs['certified'];record=stats['certified']
            error=float(np.linalg.norm(got-ref))
            row=dict(size=size,source=name,width=width,lmbda=lam,seed=a.seed,timing=timing,stats=record,
                     quotient_max_error=float(np.max(np.abs(outputs['quotient']-ref))),
                     error_norm=error,relative_l2=error/max(float(np.linalg.norm(ref)),1e-300),
                     rmse=float(np.sqrt(np.mean((got-ref)**2))),max_error=float(np.max(np.abs(got-ref))),
                     bound_excess=error-record['output_bound'],
                     speedup=float(np.median(timing['library'])/np.median(timing['certified'])),
                     engine_only_speedup=float(np.median(timing['quotient'])/np.median(timing['certified'])),
                     quotient_speedup=float(np.median(timing['library'])/np.median(timing['quotient'])))
            rows.append(row);dest.write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2))
            print({k:v for k,v in row.items() if k not in ('timing','stats')},
                  {k:v for k,v in record.items() if k!='events'},flush=True)
    print('saved',dest,flush=True)


if __name__=='__main__':main()
