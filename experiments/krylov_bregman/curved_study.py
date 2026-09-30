"""Measured coverage and full cost of curved Meyer validity descriptions.

The future trajectory is used only for independent verification, never for
chart acquisition or horizon selection. Every result uses the original map.
"""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
from PIL import Image
from .curved_meyer import MeyerQuotient,CurvedChart

ROOT=Path(__file__).resolve().parents[2]


def scenes(size):
    for name,file in [('camera','personal_deblurrer/source_assets/v3_skimage/camera.png'),
                      ('barbara','paper/fast_meyer_bregman/assets/barbara_512.tif')]:
        a=np.asarray(Image.open(ROOT/file).convert('L').resize((size,size),Image.Resampling.BOX),dtype=float)
        yield name,a
        if name=='camera':yield 'camera_permuted',np.random.default_rng(42).permutation(a.ravel()).reshape(a.shape)
    y,x=np.mgrid[:size,:size]/size
    yield 'straight_carrier',100+25*np.sin(2*np.pi*5*x)
    yield 'crossing',100+20*np.sin(2*np.pi*(5*x+3*y))+20*np.sin(2*np.pi*(3*x-7*y))+50*(x>.5)


def main(args):
    data={'configuration':vars(args),'norm':'sqrt(||u+w||²+2||t_u||²+10||t_w||²)',
          'protocol':'original-map quotient, exact local curvature remainder, global nonexpansive path bound',
          'rows':[]}
    for name,im in scenes(args.size):
        model=MeyerQuotient(im);z=model.initial.copy();last=0
        for prefix in [4,32,128,512]:
            for _ in range(prefix-last):z=model.step(z)
            last=prefix
            actual={};baseline=[]
            for rep in range(args.repeats):
                x=z.copy();start=time.perf_counter()
                for m in range(1,65):
                    x=model.step(x)
                    if m in [8,16,32,64]:
                        if rep==0:actual[m]=x.copy()
                        baseline.append((m,time.perf_counter()-start))
            for depth in [2,4,8]:
                samples=[]
                for rep in range(args.repeats):
                    start=time.perf_counter();chart=CurvedChart.acquire(model,z,depth)
                    acquired=time.perf_counter()-start
                    scan=chart.scan(64)
                    # Include reconstruction, not only reduced coordinates.
                    candidates={m:chart.candidate(scan,m) for m in [8,16,32,64]}
                    samples.append((acquired,time.perf_counter()-start))
                for m in [8,16,32,64]:
                    err=float(np.linalg.norm(actual[m]-candidates[m]));displ=float(np.linalg.norm(actual[m]-z))
                    # Output after settling: emitted texture is f-s_next.
                    settled=model.step(candidates[m]);settled_actual=model.step(actual[m])
                    texture_error=float(np.linalg.norm((settled-settled_actual).reshape(5,*model.shape)[0]))
                    row={'scene':name,'prefix':prefix,'depth':chart.basis.actions,'horizon':m,
                         'curvature_gain':model.curvature_gain,'bound':float(scan['bound'][m-1]),
                         'relative_bound_predicted':float(scan['relative_bound'][m-1]),
                         'actual_error':err,'relative_error_actual':err/max(displ,1e-300),
                         'texture_error_after_settling':texture_error,
                         'curvature_bound_sum':float(sum(scan['curvature'][:m])),
                         'compression_bound_sum':float(sum(scan['compression'][:m])),
                         'seconds_acquire':statistics.median(t[0] for t in samples),
                         'seconds_acquire_scan64_reconstruct4':statistics.median(t[1] for t in samples),
                         'seconds_ordinary_horizon':statistics.median(t for h,t in baseline if h==m),
                         'acquisition_map_calls':1,'acquisition_tangent_actions':chart.basis.actions,
                         'scan_screened_solves':0}
                    data['rows'].append(row)
            Path(args.out).write_text(json.dumps(data,indent=2))
            selected=[r for r in data['rows'][-12:] if r['horizon']==16]
            print(name,prefix,[(r['depth'],round(r['relative_bound_predicted'],3),round(r['relative_error_actual'],3)) for r in selected],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=64);p.add_argument('--repeats',type=int,default=3)
    main(p.parse_args())
