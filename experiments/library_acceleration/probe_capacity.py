"""Diagnostic of relation dimension, separately from end-to-end timing."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from skimage import data
from skimage.transform import resize
from .sporco_adapter import make_solver,TVDeconv
from .certified_transport import discover_certified


def main():
    rows=[];image=resize(data.camera()/255.,(64,64),anti_aliasing=True)
    t=np.arange(9)-4;psf=np.exp(-(t[:,None]**2+t[None,:]**2)/4.5);psf/=psf.sum()
    af=np.fft.rfftn(psf,s=image.shape)
    f=np.fft.irfftn(af*np.fft.rfftn(image),s=image.shape)+.04*np.random.default_rng(0).normal(size=image.shape)
    m=TVDeconv(make_solver(f,psf,.02,2000));z=m.initial
    for prefix in [0,32,128,512]:
        for _ in range(prefix-(rows[-1]['prefix'] if rows else 0)):z=m.step(z)
        for depth in [2,4,8,16,24]:
          for tol in [.02,.1]:
            start=perf_counter();got,e=discover_certified(m,z,128,depths=(depth,),tolerance=tol);seconds=perf_counter()-start
            start=perf_counter();truth=z.copy()
            for _ in range(e['horizon']):truth=m.step(truth)
            ordinary=perf_counter()-start
            row=dict(prefix=prefix,depth=depth,tolerance=tol,seconds=seconds,ordinary_seconds=ordinary,
                     speedup=ordinary/seconds,error=float(np.linalg.norm(got-truth)),**{k:v for k,v in e.items() if k!='depth'})
            rows.append(row);print(row,flush=True)
    Path('/tmp/library_capacity.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
