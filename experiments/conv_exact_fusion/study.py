import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.conv_exact_fusion.core import operator

def run(out,repeats=31,methods='base,projection,ledger,stream,all'):
    ops={name:operator(name) for name in methods.split(',')}
    rng=np.random.default_rng(9251);checks=[]
    # Exact current comparison on ordinary and cancellation/tie cases.
    cases=[]
    for n in [5,9,33,129]:
        x=np.arange(n,dtype=np.float32)
        cases += [(f'random{n}',rng.normal(size=(n,97)).astype(np.float32)),
                  (f'quantized{n}',rng.integers(-3,4,size=(n,97)).astype(np.float32)),
                  (f'ramps{n}',np.stack([x,-x,x*0,(x>=n//2).astype(float),np.sin(x*2.1)],axis=1).astype(np.float32))]
    for scale in [1e-20,1e-10,1e-3,1,1e3,1e10,1e20]:
        cases.append((f'scale{scale}',(rng.normal(size=(65,33))*scale).astype(np.float32)))
    # Wide dynamic range within the same fibre stresses roundoff guards.
    cases.append(('dynamic',(rng.normal(size=(129,97))*10**rng.uniform(-15,15,size=(129,97))).astype(np.float32)))
    for label,source in cases:
        reference=ops['base'].profile(source)
        for key,op in ops.items():
            z=op.profile(source)
            row=dict(case=label,method=key,equal=bool(np.array_equal(z.view(np.uint32),reference.view(np.uint32))),
                     different=int(np.sum(z!=reference)),max_difference=float(np.max(abs(z.astype(float)-reference))))
            checks.append(row)
            if not row['equal']:raise AssertionError(row)
    images={name:np.asarray(Image.open('personal_deblurrer/source_assets/v3_skimage/'+name+'.png').convert('L'),dtype=np.float32)/255 for name in ['camera','text','brick','grass','coins']}
    jobs=[]
    for name,source in images.items():
        shape=tuple(x//8 for x in source.shape)
        jobs.append((name+'_roundtrip',source,shape,True))
    coarse=ops['base'].reduce(images['camera'],(64,64))
    for shape in [(256,256),(512,512),(1024,1024)]:jobs.append((f'camera64_to_{shape[0]}',coarse,shape,False))
    jobs.append(('noise64_to512',rng.random((64,64),dtype=np.float32),(512,512),False))
    jobs.append(('rgb64_to512',rng.random((64,64,3),dtype=np.float32),(512,512),False))
    result=dict(checks=checks,images=[],timing=[])
    for label,source,shape,roundtrip in jobs:
        def call(op):
            return op.synthesize(op.reduce(source,shape),source.shape) if roundtrip else op.synthesize(source,shape)
        reference=call(ops['base'])
        for key,op in ops.items():
            z=call(op)
            row=dict(case=label,method=key,equal=bool(np.array_equal(z.view(np.uint32),reference.view(np.uint32))),different=int(np.sum(z!=reference)))
            result['images'].append(row)
            if not row['equal']:raise AssertionError(row)
        timings={k:[] for k in ops};keys=list(ops)
        for rep in range(repeats):
            for key in (keys if rep%2==0 else keys[::-1]):
                start=time.perf_counter();call(ops[key]);timings[key].append(time.perf_counter()-start)
        result['timing'].append(dict(case=label,methods={k:dict(median=float(np.median(v)),seconds=v) for k,v in timings.items()}))
        print(label,{k:round(np.median(v)/np.median(timings['base']),3) for k,v in timings.items()},flush=True)
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
    print('saved',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_exact_fusion.json');p.add_argument('--repeats',type=int,default=31);p.add_argument('--methods',default='base,projection,ledger,stream,all');a=p.parse_args();run(a.out,a.repeats,a.methods)
