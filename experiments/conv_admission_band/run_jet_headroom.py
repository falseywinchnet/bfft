"""Does the same admission law admit a tighter, canonically refined jet?"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.conv_admission_band.core import Operator
from experiments.conv_admission_band.run_study import quality,fourier

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ops={'conv':Operator('conv')}
    for r in (3,4):
        ops['jet'+str(2*r)]=Operator('conv',r)
        ops['jet'+str(2*r)+'_word1']=Operator('word1',r)
        ops['jet'+str(2*r)+'_raw']=Operator('raw',r)
    result=dict(natural=[],waves=[],edges=[],timing=[],threads=os.getenv('CONV_NATIVE_THREADS'))
    arrays={}
    for name in ['camera','text','brick','coins','grass','moon']:
        source=np.asarray(Image.open('personal_deblurrer/source_assets/v3_skimage/'+name+'.png').convert('L'),dtype=np.float32)/255
        shape=tuple(s//8 for s in source.shape);coarse=ops['conv'].reduce(source,shape)
        for key,op in ops.items():
            up=op.synthesize(coarse,source.shape);full=op.resize(op.reduce(source,shape),source.shape)
            result['natural'].append(dict(source=name,method=key,both=quality(full,source),up_only=quality(up,source)))
            if name=='camera':arrays[key]=full;arrays[key+'_up']=up;arrays['original']=source;arrays['coarse']=coarse
        print('natural',name,flush=True)
    n=33;m=257;y,x=np.mgrid[:n,:n];fy,fx=np.meshgrid(np.linspace(0,n-1,m),np.linspace(0,n-1,m),indexing='ij')
    crop=np.s_[40:-40,40:-40]
    for angle in [0,15,30,45,75]:
        ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        for f in [.25,.5,.7,.85,.95,1.05]:
            for p in [.17,1.03]:
                source=.5+.4*np.cos(np.pi*f*(ct*x+st*y)+p);phase=np.pi*f*(ct*fx+st*fy)+p;truth=.5+.4*np.cos(phase)
                for key,op in ops.items():
                    z=op.synthesize(source,(m,m))
                    result['waves'].append(dict(angle=angle,frequency=f,phase0=p,method=key,**quality(z[crop],truth[crop]),**fourier(z[crop],phase[crop])))
        for offset in [-.31,0,.23]:
            for kind in ['step','strip','sigmoid']:
                d=ct*(x-16)+st*(y-16)-offset;df=ct*(fx-16)+st*(fy-16)-offset
                fn=(lambda t:(t>=0).astype(float)) if kind=='step' else ((lambda t:(abs(t)<.8).astype(float)) if kind=='strip' else (lambda t:.5+.5*np.tanh(t/.45)))
                source=fn(d);truth=fn(df)
                for key,op in ops.items():
                    z=op.synthesize(source,(m,m))
                    result['edges'].append(dict(angle=angle,offset=offset,kind=kind,method=key,**quality(z[crop],truth[crop])))
    source=arrays['original'];coarse=arrays['coarse'];methods=[k for k in ops if not k.endswith('raw')]
    timing={k:[] for k in methods}
    for k in methods:ops[k].synthesize(coarse,source.shape)
    for rep in range(21):
        for k in (methods if rep%2==0 else methods[::-1]):
            op=ops[k];t=time.perf_counter();op.resize(op.reduce(source,(64,64)),source.shape);timing[k].append(time.perf_counter()-t)
    result['timing']=[dict(method=k,seconds=v,median=float(np.median(v))) for k,v in timing.items()]
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'arrays.npz',**arrays)
    print('saved',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_jet_headroom');a=p.parse_args();run(a.out)
