"""Fixed CONV source/control-order census: spectrum, images, and native cost."""
import argparse,json,math,os,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.conv_fixed_orders.core import CONFIGS,FixedOrder,bank,HERE,Legacy
from experiments.conv_admission_band.run_study import quality


def carrier(z,phase):
    z=np.asarray(z,dtype=float).ravel();c=np.cos(phase).ravel();s=np.sin(phase).ravel()
    c-=c.mean();s-=s.mean();v=z-z.mean()
    cc=c@c;ss=s@s;cs=c@s;vc=v@c;vs=v@s;det=cc*ss-cs*cs
    if det<1e-15:return {'gain':None,'generated':None}
    a=(vc*ss-vs*cs)/det;b=(vs*cc-vc*cs)/det
    return {'gain':float(np.hypot(a,b)/.4),'generated':float(np.mean((v-a*c-b*s)**2))}


def raw_spectrum(taps,degree,frequencies,scale=8):
    a=np.array(bank(taps,degree),float);r=(taps-2)//2
    offsets=np.arange(-r,r+2);g=np.zeros(len(frequencies),complex)
    for phase in range(scale):
        u=phase/scale
        basis=np.array([math.comb(degree,j)*u**j*(1-u)**(degree-j) for j in range(degree+1)])
        tails=np.array([basis[k+1:].sum() for k in range(degree)])
        weights=tails@a;weights[r]+=1
        g+=np.exp(1j*np.pi*frequencies[:,None]*(offsets-u))@weights/scale
    return g.real


def crossing(f,g,level):
    idx=np.flatnonzero((g[:-1]>=level)&(g[1:]<level))
    if not len(idx):return None
    i=idx[0];return float(f[i]+(f[i+1]-f[i])*(g[i]-level)/(g[i]-g[i+1]))


def run(out,repeats):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frozen=(HERE/'v1_native.c').read_text()
    ops={'v1':Legacy('conv',source_transform=lambda _:frozen)}
    ops.update({k:FixedOrder(*v) for k,v in CONFIGS.items()})
    result={'baseline':'Version 1.0: original six-source/five-current two-order CONV plus rolling ledger; no terminal fusion',
            'source_sha256':hashlib.sha256(frozen.encode()).hexdigest(),
            'threads':os.getenv('CONV_NATIVE_THREADS'),'configs':CONFIGS,
            'spectra':{},'carriers':[],'waves2d':[],'edges':[],'images':[],'timing':[]}
    f=np.linspace(0,2,1001)
    for name,(taps,d) in CONFIGS.items():
        gain=raw_spectrum(taps,d,f)
        f90=crossing(f,gain,.9);f10=crossing(f,gain,.1)
        result['spectra'][name]={'f':f.tolist(),'gain':gain.tolist(),'f90':f90,'f50':crossing(f,gain,.5),'f10':f10,
                                 'width90_10':f10-f90,'gain_at_0_9':float(np.interp(.9,f,gain))}
    # Actual admitted nonlinear carrier response; fixed phases, no learned fit.
    n=129;s=8;x=np.arange(n);xf=np.arange((n-1)*s+1)/s;keep=slice(16*s,-16*s)
    for freq in np.linspace(.1,1.9,73):
        for p in (.17,1.03):
            src=.5+.4*np.cos(np.pi*freq*x+p)
            for name,op in ops.items():
                z=op.axis(src,len(xf),0)
                result['carriers'].append(dict(method=name,frequency=float(freq),phase=p,
                    **carrier(z[keep],(np.pi*freq*xf+p)[keep])))
    print('carrier sweep complete',flush=True)
    n=33;m=257;y,x=np.mgrid[:n,:n];yf,xf=np.mgrid[:m,:m]/8;crop=np.s_[48:-48,48:-48]
    for angle in (0,30,45,60,90):
        ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        for freq in (.5,.7,.85,.95,1.05,1.2):
            for p in (.17,1.03):
                phase=np.pi*freq*(ct*xf+st*yf)+p
                src=.5+.4*np.cos(np.pi*freq*(ct*x+st*y)+p)
                truth=.5+.4*np.cos(phase)
                for name,op in ops.items():
                    z=op.synthesize(src,(m,m))
                    result['waves2d'].append(dict(method=name,angle=angle,frequency=freq,phase=p,
                        **quality(z[crop],truth[crop]),**carrier(z[crop],phase[crop])))
        for offset in (-.31,0,.23):
            for kind in ('step','strip'):
                d=ct*(x-16)+st*(y-16)-offset;df=ct*(xf-16)+st*(yf-16)-offset
                fn=(lambda a:(a>=0).astype(float)) if kind=='step' else (lambda a:(abs(a)<.8).astype(float))
                for name,op in ops.items():
                    z=op.synthesize(fn(d),(m,m));result['edges'].append(dict(method=name,angle=angle,offset=offset,kind=kind,**quality(z[crop],fn(df)[crop])))
        print('2D angle',angle,flush=True)
    arrays={};sources={}
    for image in ('camera','text','brick','coins','grass','moon'):
        src=np.asarray(Image.open('personal_deblurrer/source_assets/v3_skimage/'+image+'.png').convert('L'),dtype=np.float32)/255
        sources[image]=src;small=tuple(k//8 for k in src.shape);coarse=ops['v1'].reduce(src,small)
        for name,op in ops.items():
            up=op.synthesize(coarse,src.shape)
            full=op.synthesize(op.reduce(src,small),src.shape)
            if not(np.all(np.isfinite(up)) and np.all(np.isfinite(full))):raise RuntimeError('nonfinite '+image+name)
            result['images'].append(dict(method=name,image=image,synthesis_only=quality(up,src),roundtrip=quality(full,src)))
            if image=='camera':arrays[name]=full;arrays[name+'_up']=up
        if image=='camera':arrays['original']=src;arrays['coarse']=coarse
        print('image',image,flush=True)
    src=sources['camera'];coarse=arrays['coarse'];names=list(ops)
    jobs={'camera_roundtrip':lambda op:op.synthesize(op.reduce(src,(64,64)),(512,512)),
          'camera_synthesis':lambda op:op.synthesize(coarse,(512,512)),
          'camera_reduction':lambda op:op.reduce(src,(64,64))}
    for label,call in jobs.items():
        times={k:[] for k in names}
        for k in names:call(ops[k])
        for rep in range(repeats):
            for k in (names if rep%2==0 else names[::-1]):
                t=time.perf_counter();call(ops[k]);times[k].append(time.perf_counter()-t)
        result['timing'].append({'case':label,'methods':{k:{'median':float(np.median(v)),'seconds':v} for k,v in times.items()}})
        print(label,{k:round(np.median(v)/np.median(times['v1']),3) for k,v in times.items()},flush=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'arrays.npz',**arrays)
    coefficients={k:{'source_samples':t,'degree':d,'currents':d,'controls':d+1,'jet_order':(d-1)//2,
                     'raw_bank':[[str(x) for x in row] for row in bank(t,d)]} for k,(t,d) in CONFIGS.items()}
    (out/'coefficients.json').write_text(json.dumps(coefficients,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=31);a=p.parse_args();run(a.out,a.repeats)
