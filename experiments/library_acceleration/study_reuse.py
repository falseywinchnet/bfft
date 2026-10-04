"""Default adaptive TV deconvolution: independent exact Fourier-state reuse."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from skimage import data
from skimage.transform import resize
from sporco.admm.tvl2 import TVL2Deconv
from .sporco_reuse import TVL2DeconvReuse


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();rows=[];dest=Path(a.out)
    for size in [128,256]:
     for source in ['camera','coins']:
      image=resize(getattr(data,source)()/255.,(size,size),anti_aliasing=True)
      t=np.arange(9)-4;psf=np.exp(-(t[:,None]**2+t[None,:]**2)/4.5);psf/=psf.sum()
      f=np.fft.irfftn(np.fft.rfftn(psf,s=image.shape)*np.fft.rfftn(image),s=image.shape)
      f=f+.04*np.random.default_rng(3).normal(size=f.shape)
      for lam in [.02,.1]:
        opts=dict(Verbose=False,MaxMainIter=2000,RelStopTol=1e-4,AbsStopTol=1e-6)
        def run(cls):
            solver=cls(psf,f,lam,TVL2Deconv.Options(opts));out=solver.solve();return out,solver
        funcs=dict(library=TVL2Deconv,reuse=TVL2DeconvReuse)
        for cls in funcs.values():run(cls)
        timing={k:[] for k in funcs};out={};solvers={}
        for rep in range(a.repeats):
         for k in np.random.default_rng(rep).permutation(list(funcs)):
          start=perf_counter();out[k],solvers[k]=run(funcs[k]);timing[k].append(perf_counter()-start)
        old=solvers['library'];new=solvers['reuse'];oldstat=old.getitstat();newstat=new.getitstat()
        row=dict(size=size,source=source,lmbda=lam,timing=timing,iterations=dict(library=old.k,reuse=new.k),
                 speedup=float(np.median(timing['library'])/np.median(timing['reuse'])),
                 max_error=float(np.max(np.abs(out['library']-out['reuse']))),
                 relative_error=float(np.linalg.norm(out['library']-out['reuse'])/np.linalg.norm(out['library'])),
                 converged=bool(oldstat.PrimalRsdl[-1]<oldstat.EpsPrimal[-1] and oldstat.DualRsdl[-1]<oldstat.EpsDual[-1] and
                                newstat.PrimalRsdl[-1]<newstat.EpsPrimal[-1] and newstat.DualRsdl[-1]<newstat.EpsDual[-1]),
                 rho_history_max_error=float(np.max(np.abs(np.asarray(oldstat.Rho)-np.asarray(newstat.Rho)))) if old.k==new.k else None)
        rows.append(row);dest.write_text(json.dumps(dict(protocol=dict(repeats=a.repeats,options=opts),rows=rows),indent=2))
        print({k:v for k,v in row.items() if k!='timing'},flush=True)
    print('saved',dest,flush=True)

if __name__=='__main__':main()
