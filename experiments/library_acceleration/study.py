"""Paired public-library benchmarks; all accelerator work is timed."""
import argparse
import inspect
import hashlib
import json
from pathlib import Path
from time import perf_counter
import platform
import numpy as np
import scipy
import skimage
from scipy.signal import convolve
from skimage import data
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle, richardson_lucy
from .adapters import Chambolle, RichardsonLucy
from .engine import run


def cases(size, sources, seed):
    rng=np.random.default_rng(seed)
    for name in sources:
        image=np.asarray(getattr(data,name)(),dtype=float)
        image/=255
        image=resize(image,(size,size),anti_aliasing=True,preserve_range=True)
        for weight in [.1,.3]:
            noisy=image+.08*rng.standard_normal(image.shape)
            yield dict(kind='tv',source=name,size=size,parameter=weight,seed=seed), image,noisy,None
        for width in [9,25]:
            coord=np.arange(width)-(width-1)/2
            psf=np.exp(-(coord[:,None]**2+coord[None,:]**2)/(2*(width/6)**2));psf/=psf.sum()
            blurred=np.maximum(convolve(image,psf,mode='same'),0)
            noisy=rng.poisson(blurred*100)/100
            yield dict(kind='rl',source=name,size=size,parameter=width,seed=seed),image,noisy,psf


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--sizes',default='128,256');p.add_argument('--sources',default='camera,coins')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--passes',type=int,default=200)
    p.add_argument('--tolerance',type=float,default=.005);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();rows=[];root=Path(a.out);root.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
              machine=platform.machine(),python=platform.python_version(),
              library_hashes={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                              for f in [denoise_tv_chambolle,richardson_lucy]},
              protocol=vars(a), admission='existing sampled defect screen; no path certificate',
              threads='OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1')
    for size in map(int,a.sizes.split(',')):
      for label,clean,noisy,psf in cases(size,a.sources.split(','),a.seed):
        if label['kind']=='tv':
            factory=lambda:Chambolle(noisy,label['parameter'])
            steps=a.passes-1
            lib=lambda:denoise_tv_chambolle(noisy,weight=label['parameter'],eps=0,max_num_iter=a.passes)
        else:
            factory=lambda:RichardsonLucy(noisy,psf)
            steps=a.passes
            lib=lambda:richardson_lucy(noisy,psf,num_iter=a.passes,clip=False)
        functions=dict(library=lambda:(lib(),{}),ordinary_adapter=lambda:run(factory(),steps,False),
                       engine=lambda:run(factory(),steps,True,tolerance=a.tolerance))
        # Warm each path. Reference outputs are used only after the runs.
        for fun in functions.values():fun()
        timing={k:[] for k in functions};outputs={};statistics={}
        rng=np.random.default_rng(123)
        for repeat in range(a.repeats):
            for key in rng.permutation(list(functions)):
                start=perf_counter();out,stats=functions[key]();elapsed=perf_counter()-start
                timing[key].append(elapsed);outputs[key]=out;statistics[key]=stats
        ref=outputs['library'];got=outputs['engine']
        row=dict(**label,timing=timing,stats=statistics['engine'],
                 adapter_max_error=float(np.max(np.abs(outputs['ordinary_adapter']-ref))),
                 relative_l2=float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-300)),
                 rmse=float(np.sqrt(np.mean((got-ref)**2))),max_error=float(np.max(np.abs(got-ref))),
                 reference_rmse_to_clean=float(np.sqrt(np.mean((ref-clean)**2))),
                 engine_rmse_to_clean=float(np.sqrt(np.mean((got-clean)**2))),
                 speedup=float(np.median(timing['library'])/np.median(timing['engine'])),
                 adapter_speedup=float(np.median(timing['ordinary_adapter'])/np.median(timing['engine'])))
        rows.append(row)
        root.write_text(json.dumps(dict(metadata=meta,rows=rows),indent=2))
        print({k:v for k,v in row.items() if k not in ('timing','stats')},
              {k:v for k,v in row['stats'].items() if k!='events'},flush=True)
    print('saved',root,flush=True)


if __name__=='__main__':main()
