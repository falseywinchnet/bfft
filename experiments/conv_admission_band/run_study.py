"""Reproducible admission-only census with the actual native demo baseline."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from experiments.conv_admission_band.core import Operator,VARIANTS,subdivide,word,subsequence
from experiments.convstar import ordered_sign_ledger

def quality(z,truth):
    z=np.asarray(z,dtype=float);truth=np.asarray(truth,dtype=float)
    mse=float(np.mean((z-truth)**2))
    lo=float(truth.min());hi=float(truth.max())
    return dict(mse=mse,psnr=-10*math.log10(max(mse,1e-300)),
                excursion=float(max(0,lo-z.min(),z.max()-hi)),
                outside=int(np.sum((z<lo-1e-6)|(z>hi+1e-6))))

def fourier(z,phase):
    # Measurement only: exact 3-coordinate least-squares fit removes DC and
    # separates fundamental amplitude/phase from generated energy.
    design=np.stack([np.ones(phase.size),np.cos(phase).ravel(),np.sin(phase).ravel()],axis=1)
    coefficient=np.linalg.lstsq(design,np.asarray(z,dtype=float).ravel(),rcond=None)[0]
    return dict(gain=float(np.hypot(*coefficient[1:])/.4),
                phase=float(np.arctan2(-coefficient[2],coefficient[1])),
                generated=float(np.mean((design@coefficient-z.ravel())**2)))

def diagnose(source,ops):
    rows=[]
    for axis in (0,1):
        lines=np.ascontiguousarray(np.moveaxis(source,axis,0))
        raw=ops['raw'].profile(lines).astype(float)
        delta=np.diff(lines.astype(float),axis=0)
        ledger=ordered_sign_ledger(raw,delta)
        base=ops['conv'].profile(lines).astype(float)
        active=np.max(abs(base-raw),axis=1)>1e-6
        uniform=np.all(ledger==ledger[:,0:1,:],axis=1)
        recovery={}
        for name in ops:
            z=ops[name].profile(lines).astype(float)
            movement=np.sum((z-raw)**2,axis=1)
            recovery[name]=dict(current_error=float(movement.sum()),
                               uniform_error=float(movement[uniform].sum()),
                               mixed_error=float(movement[~uniform].sum()),
                               retained_active=int(np.sum(active&(np.max(abs(z-raw),axis=1)<1e-6))))
        rows.append(dict(axis=axis,cells=int(active.size),active=int(active.sum()),
                         active_uniform=int((active&uniform).sum()),
                         active_mixed=int((active&~uniform).sum()),variants=recovery))
    return rows

def run(out,quick=False,repeats=7):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ops={name:Operator(name) for name in VARIANTS}
    result=dict(threads=os.getenv('CONV_NATIVE_THREADS','default'),variants=VARIANTS,
                natural=[],waves=[],edges=[],mixtures=[],roundtrip_waves=[],diagnostics={},timing=[])
    arrays={}
    names=['camera','text','brick'] if quick else ['camera','text','brick','coins','grass','moon']
    for name in names:
        source=np.asarray(Image.open('personal_deblurrer/source_assets/v3_skimage/'+name+'.png').convert('L'),dtype=np.float32)/255
        # No pre-resize: use each source's native shape, with floor division
        # stated explicitly for non-divisible validation images.
        shape=tuple(s//8 for s in source.shape);base_small=ops['conv'].reduce(source,shape)
        if name=='camera':result['diagnostics']['camera64']=diagnose(base_small,ops)
        for key,op in ops.items():
            small=op.reduce(source,shape)
            same_analysis=op.synthesize(base_small,source.shape)
            full=op.synthesize(small,source.shape)
            reduction_only=ops['conv'].synthesize(small,source.shape)
            result['natural'].append(dict(source=name,shape=list(source.shape),small=list(shape),method=key,
                synthesis_only=quality(same_analysis,source),both=quality(full,source),
                analysis_only=quality(reduction_only,source),coarse_delta=float(np.max(abs(small-base_small)))))
            if name=='camera':
                arrays['original']=source;arrays['coarse']=base_small
                arrays[key]=full;arrays[key+'_synthesis']=same_analysis
        print('natural',name,flush=True)
    n=33;m=257
    y,x=np.mgrid[:n,:n];fy,fx=np.meshgrid(np.linspace(0,n-1,m),np.linspace(0,n-1,m),indexing='ij')
    crop=np.s_[24:-24,24:-24]
    frequencies=[.25,.5,.7,.85,.95,1.05] if quick else [.1,.25,.4,.55,.7,.85,.95,1.05]
    angles=[0,30,45] if quick else [0,15,30,45,60,75,90]
    phases=[.17,1.03] if quick else [.0,.37,1.03,2.11]
    for angle in angles:
        ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        for frequency in frequencies:
            for phase in phases:
                source=.5+.4*np.cos(np.pi*frequency*(ct*x+st*y)+phase)
                p=np.pi*frequency*(ct*fx+st*fy)+phase
                truth=.5+.4*np.cos(p)
                reference=ops['conv'].synthesize(source,(m,m))
                for name,op in ops.items():
                    z=op.synthesize(source,(m,m))
                    result['waves'].append(dict(angle=angle,frequency=frequency,phase0=phase,method=name,
                        max_difference=float(np.max(abs(z[crop]-reference[crop]))),
                        **quality(z[crop],truth[crop]),**fourier(z[crop],p[crop])))
        print('waves angle',angle,flush=True)
    for angle in ([0,30] if quick else [0,15,30,45,75]):
        ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        for frequency in [.5,.7,.85,.95]:
            for kind in ['crossed','amplitude_modulated','chirp','edge_texture']:
                def field(xx,yy):
                    u=ct*xx+st*yy;v=-st*xx+ct*yy
                    carrier=np.cos(np.pi*frequency*u+.37)
                    if kind=='crossed':return .5+.22*carrier+.22*np.cos(np.pi*(frequency-.13)*v+1.1)
                    if kind=='amplitude_modulated':return .5+.4*(.5+.5*np.cos(np.pi*.12*v))*carrier
                    if kind=='chirp':return .5+.4*np.cos(np.pi*(.2*u+(frequency-.2)*u*u/(2*n))+.37)
                    return .5+.3*np.tanh((u-16)/.5)+.15*carrier
                source=field(x,y);truth=field(fx,fy)
                reference=ops['conv'].synthesize(source,(m,m))
                for name,op in ops.items():
                    z=op.synthesize(source,(m,m))
                    result['mixtures'].append(dict(angle=angle,frequency=frequency,kind=kind,method=name,
                        max_difference=float(np.max(abs(z[crop]-reference[crop]))),**quality(z[crop],truth[crop])))
    # The complete reduction/enlargement response, expressed relative to the
    # endpoint-aligned reduced lattice Nyquist, not the fine input Nyquist.
    big=257;small=33;yy,xx=np.mgrid[:big,:big]
    for angle in ([0,45] if quick else [0,15,30,45,75]):
        ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        for frequency in [.25,.5,.7,.85,.95,1.05,1.2]:
            phase=np.pi*frequency*(small-1)/(big-1)*(ct*xx+st*yy)+.37
            source=.5+.4*np.cos(phase)
            coarse=ops['conv'].reduce(source,(small,small))
            for name,op in ops.items():
                z=op.resize(op.resize(source,(small,small)),(big,big))
                up_only=op.synthesize(coarse,(big,big))
                result['roundtrip_waves'].append(dict(angle=angle,frequency=frequency,method=name,
                    **quality(z[crop],source[crop]),**fourier(z[crop],phase[crop]),
                    up_only=fourier(up_only[crop],phase[crop])))
        print('roundtrip angle',angle,flush=True)
    # No clipping, and multiple locations/orientations of exact discontinuities.
    for angle in ([0,30,45,75] if quick else [0,7,15,30,45,60,75,83,90]):
        ct,st=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        for offset in ([.23] if quick else [-.31,.0,.23]):
            for kind in ['step','strip','sigmoid']:
                d=ct*(x-16)+st*(y-16)-offset;df=ct*(fx-16)+st*(fy-16)-offset
                f=(lambda t:(t>=0).astype(float)) if kind=='step' else ((lambda t:(abs(t)<.8).astype(float)) if kind=='strip' else (lambda t:.5+.5*np.tanh(t/.45)))
                source=f(d);truth=f(df)
                for name,op in ops.items():
                    z=op.synthesize(source,(m,m))
                    result['edges'].append(dict(angle=angle,offset=offset,kind=kind,method=name,**quality(z[crop],truth[crop])))
                    if angle==30 and kind=='step':arrays['step_'+name]=z;arrays['step_truth']=truth
    # Paired alternating order, warmed native functions; include geometry,
    # allocations, reduction and both synthesis orders in the total.
    source=arrays['original'];shape=(64,64)
    methods=[k for k in ops if k!='raw']
    for key in methods:ops[key].resize(source,shape);ops[key].synthesize(arrays['coarse'],source.shape)
    times={k:[] for k in methods};ups={k:[] for k in methods}
    for rep in range(repeats):
        for key in (methods if rep%2==0 else methods[::-1]):
            op=ops[key];start=time.perf_counter();small=op.reduce(source,shape);op.synthesize(small,source.shape)
            times[key].append(time.perf_counter()-start)
            start=time.perf_counter();op.synthesize(arrays['coarse'],source.shape);ups[key].append(time.perf_counter()-start)
    for key in methods:result['timing'].append(dict(method=key,total_seconds=times[key],up_seconds=ups[key],
        total_median=float(np.median(times[key])),up_median=float(np.median(ups[key]))))
    np.savez_compressed(out/'arrays.npz',**arrays)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('saved',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_admission_band');p.add_argument('--quick',action='store_true');p.add_argument('--repeats',type=int,default=7);a=p.parse_args();run(a.out,a.quick,a.repeats)
