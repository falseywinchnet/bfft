"""Admission-boundary continuity and one-worker cost checks."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.conv_admission_band.core import Operator

def main(out):
    ops={k:Operator(k) for k in ['conv','uniform1','word1','word2','ray1']}
    ops['jet6']=Operator('conv',3);ops['jet8']=Operator('conv',4)
    u=np.linspace(0,1,4097);bern=np.stack([math.comb(5,k)*u**k*(1-u)**(5-k) for k in range(6)],axis=1)
    tails=np.cumsum(bern[:,1:][:,::-1],axis=1)[:,::-1]
    rows=[]
    for epsilon in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        for name,op in ops.items():
            values=[]
            for s in [-1,1]:
                a=(np.array([.25,0,-1/12,0,.25])+s*epsilon)/5
                values.append(tails@op.fibre(a,np.ones(5),float(a.sum())))
            rows.append(dict(epsilon=epsilon,method=name,output_difference=float(np.max(abs(values[1]-values[0]))),input_difference=2*epsilon))
    source=np.asarray(Image.open('personal_deblurrer/source_assets/v3_skimage/camera.png').convert('L'),dtype=np.float32)/255
    timings={k:[] for k in ops};names=list(ops)
    for op in ops.values():op.resize(op.reduce(source,(64,64)),source.shape)
    for r in range(31):
        for k in (names if r%2==0 else names[::-1]):
            op=ops[k];start=time.perf_counter();op.resize(op.reduce(source,(64,64)),source.shape);timings[k].append(time.perf_counter()-start)
    Path(out).write_text(json.dumps(dict(boundary=rows,timing=[dict(method=k,median=float(np.median(v)),seconds=v) for k,v in timings.items()]),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_admission_boundary.json');a=p.parse_args();main(a.out)
