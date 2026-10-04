"""Analytic coverage battery, GPU exchange format, and honest per-family reports."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from .core import halfplane_coverage, tangent_integral, conv_box

MSAA4=np.array([[-.125,-.375],[.375,-.125],[-.375,.125],[.125,.375]])


def dataset(size=64):
    y,x=np.indices((size,size),dtype=float); x-=(size-1)/2; y-=(size-1)/2
    rows=[];sources=[];references=[];sample4=[];masks=[]
    def add(family,key,phase,fn,ref,mask):
        source=fn(x,y).astype(float)
        multi=sum(fn(x+dx,y+dy) for dx,dy in MSAA4)/4
        if source.ndim==2: source=np.repeat(source[...,None],3,axis=2)
        if ref.ndim==2: ref=np.repeat(ref[...,None],3,axis=2)
        if multi.ndim==2: multi=np.repeat(multi[...,None],3,axis=2)
        mask=mask.copy();mask[:4]=False;mask[-4:]=False;mask[:,:4]=False;mask[:,-4:]=False
        rows.append(dict(family=family,key=key,phase=phase));sources.append(source);references.append(ref);sample4.append(multi);masks.append(mask)
    for angle in np.linspace(0,np.pi,16,endpoint=False):
        nx,ny=np.cos(angle),np.sin(angle)
        for phase in np.arange(8)/8:
            d=nx*x+ny*y-phase
            fn=lambda xx,yy: (nx*xx+ny*yy-phase>=0).astype(float)
            add('step',f'step_{angle:.8f}',phase,fn,halfplane_coverage(d,nx,ny),np.abs(d)<2)
            if angle not in np.linspace(0,np.pi,16,endpoint=False)[::2]: continue
            for width in (.35,.75,1.5,3.):
                fn=lambda xx,yy: (np.abs(nx*xx+ny*yy-phase)<width/2).astype(float)
                ref=halfplane_coverage(d+width/2,nx,ny)-halfplane_coverage(d-width/2,nx,ny)
                add(f'strip_{width}',f'strip_{width}_{angle:.8f}',phase,fn,ref,np.abs(d)<width/2+2)
    offsets=(np.arange(32)+.5)/32-.5
    for kind in ('disk','ring','corner','checker','sine'):
        for phase in np.arange(8)/8:
            def fn(xx,yy):
                xx=xx-phase
                if kind=='disk': return ((xx*xx+(yy-.17)**2)<13.3**2).astype(float)
                if kind=='ring': return (np.abs(np.hypot(xx,yy-.17)-13.3)<.35).astype(float)
                if kind=='corner': return ((xx+.31*yy>0)&(-.6*xx+yy>0)).astype(float)
                if kind=='checker': return ((np.floor((xx+.4*yy)/1.3).astype(int)+np.floor((yy-.4*xx)/1.3).astype(int))%2).astype(float)
                return .5+.5*np.sin(2*np.pi*(.43*xx+.21*yy))
            ref=np.zeros_like(x)
            for dx in offsets:
                for dy in offsets: ref+=fn(x+dx,y+dy)/len(offsets)**2
            add(kind,kind,phase,fn,ref,np.ones_like(x,dtype=bool))
    # Isoluminant in Rec.709: geometry alone must not hide the lost color edge.
    low=np.array([1.,0.,0.]);high=np.array([0.,.2126/.7152,0.])
    for phase in np.arange(8)/8:
        d=.8*x+.6*y-phase
        fn=lambda xx,yy: low+(high-low)*(.8*xx+.6*yy-phase>=0)[...,None]
        ref=low+(high-low)*halfplane_coverage(d,.8,.6)[...,None]
        add('isoluminant','isoluminant',phase,fn,ref,np.abs(d)<2)
    # Already covered source: target is identity, includes ambiguous axis-aligned lines.
    for width in (.35,1.,2.):
        for phase in np.arange(8)/8:
            d=.8*x+.6*y-phase
            ref=halfplane_coverage(d+width/2,.8,.6)-halfplane_coverage(d-width/2,.8,.6)
            fn=lambda xx,yy: halfplane_coverage(.8*xx+.6*yy-phase+width/2,.8,.6)-halfplane_coverage(.8*xx+.6*yy-phase-width/2,.8,.6)
            add('already_covered',f'covered_{width}',phase,fn,ref,np.abs(d)<width/2+2)
    return rows,np.array(sources),np.array(references),np.array(sample4),np.array(masks)


def generate(out):
    out.mkdir(parents=True,exist_ok=True)
    rows,source,reference,multi,mask=dataset()
    np.savez_compressed(out/'dataset.npz',source=source,reference=reference,multi=multi,mask=mask)
    (out/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    rgba=np.concatenate((source,(source@np.array([.2126,.7152,.0722]))[...,None]),axis=3).astype('<f4')
    with (out/'input.bin').open('wb') as f:
        np.array([source.shape[2],source.shape[1],len(rows)],dtype='<u4').tofile(f);rgba.tofile(f)
    print(json.dumps({'cases':len(rows),'shape':source.shape,'out':str(out)}),flush=True)


def analyze(out):
    data=np.load(out/'dataset.npz');src,ref,multi,mask=(data[k] for k in ('source','reference','multi','mask'))
    rows=json.loads((out/'cases.json').read_text()); n,h,w,_=src.shape
    outputs={'point':src,'coverage4':multi}
    checks={}
    for name in ('copyImage','fxaa12','tangentBox','tangentQuintic','tangentTensor'):
        a=np.fromfile(out/f'gpu_{name}.bin',dtype='<f4').reshape(n,h,w,4)[...,:3]
        if name=='copyImage':checks['gpu_copy_max_error']=float(np.max(np.abs(a-src)));continue
        outputs[name]=a
        if name.startswith('tangent'):
            p='quintic' if name=='tangentQuintic' else 'box'
            # Exclude exact isoluminance: float RGB/luma quantization can change a
            # nearly zero direction. Retain its quality failure in the main table.
            errors=[np.max(np.abs(a[i]-tangent_integral(src[i].astype(np.float32).astype(float),profile=p,tensor=name=='tangentTensor',luma_override=(src[i]@np.array([.2126,.7152,.0722])).astype(np.float32)))) for i in range(n)]
            checks[name+'_numpy_gpu_max_error']=float(max(errors))
            checks[name+'_numpy_gpu_pass']=bool(max(errors)<1e-5)
    # Original admitted CONV: fixed representative phase, every geometry family.
    subset=[i for i,r in enumerate(rows) if r['phase']==.375]
    conv=np.array([conv_box(src[i]) for i in subset])
    np.save(out/'conv_subset.npy',conv)
    summaries={};per_case=[]
    for family in sorted(set(r['family'] for r in rows)):
        ids=[i for i,r in enumerate(rows) if r['family']==family]
        summaries[family]={}
        for method,a in outputs.items():
            squared=(a-ref)**2
            mse=[float(squared[i][mask[i]].mean()) for i in ids]
            temporal=[]
            for i in ids:
                if i>0 and rows[i]['key']==rows[i-1]['key'] and rows[i]['phase']>rows[i-1]['phase']:
                    change=(a[i]-ref[i])-(a[i-1]-ref[i-1]); common=mask[i]|mask[i-1]
                    temporal.append(float(np.mean(change[common]**2)))
            # Groups are interleaved for step/strip, so use key-ordered motion.
            temporal=[]
            for key in sorted(set(rows[i]['key'] for i in ids)):
                sequence=sorted([i for i in ids if rows[i]['key']==key],key=lambda i:rows[i]['phase'])
                for i,j in zip(sequence[:-1],sequence[1:]):
                    change=(a[j]-ref[j])-(a[i]-ref[i]); temporal.append(float(np.mean(change[mask[i]|mask[j]]**2)))
            summaries[family][method]={'mean_roi_mse':float(np.mean(mse)),'worst_case_roi_mse':float(max(mse)),
                'phase_error_delta_mse':float(np.mean(temporal)) if temporal else None,
                'range_min':float(a[ids].min()),'range_max':float(a[ids].max())}
            for i,e in zip(ids,mse):per_case.append(dict(rows[i],index=i,method=method,roi_mse=e))
        ci=[j for j,i in enumerate(subset) if rows[i]['family']==family]
        if ci: summaries[family]['conv_box_subset']={'mean_roi_mse':float(np.mean([np.mean((conv[j]-ref[subset[j]])[mask[subset[j]]]**2) for j in ci])),'cases':len(ci)}
    checks['all_outputs_finite']=all(bool(np.isfinite(a).all()) for a in outputs.values())
    # Exact information ambiguity: two vertical steps have identical sampled image.
    a=(np.arange(8)>=3.1).astype(float);b=(np.arange(8)>=3.4).astype(float)
    checks['ambiguous_step_identical_samples']=bool(np.array_equal(a,b))
    checks['ambiguous_step_coverages']=[float(halfplane_coverage(3-z,1,0)) for z in (3.1,3.4)]
    result={'scope':'Scalar/RGB coverage diagnostic in linear numerical coordinates. FXAA12 official shader, default quality parameters, Rec709 luma in alpha. coverage4 uses geometry at four rotated samples, not a measured full MSAA renderer. Disk/ring/corner/checker/sine reference uses 32x32 midpoint quadrature. Straight edges/strips exact. ROI excludes 4px image boundary. No candidate receives geometric data. CONV subset uses phase .375 only; do not compare its aggregate to full-phase aggregates.',
        'python':platform.python_version(),'cases':n,'checks':checks,'families':summaries,'per_case':per_case,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.is_file()}}
    (out/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':checks,'families':summaries},indent=2),flush=True)
    assert checks['all_outputs_finite'] and all(v for k,v in checks.items() if k.endswith('_pass'))


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=np.load(out/'dataset.npz');src,ref=(data[k] for k in ('source','reference'))
    rows=json.loads((out/'cases.json').read_text());n,h,w,_=src.shape
    methods=[('Point',src),('FXAA 3.11',np.fromfile(out/'gpu_fxaa12.bin',dtype='<f4').reshape(n,h,w,4)[...,:3]),
        ('Tangent box',np.fromfile(out/'gpu_tangentBox.bin',dtype='<f4').reshape(n,h,w,4)[...,:3]),
        ('RGB tensor tangent',np.fromfile(out/'gpu_tangentTensor.bin',dtype='<f4').reshape(n,h,w,4)[...,:3]),
        ('4 coverage samples',data['multi']),('Area reference',ref)]
    chosen=[next(i for i,r in enumerate(rows) if r['family']==family and r['phase']==.375 and ('0.39269908' in r['key'] if family in ('step','strip_0.75') else True)) for family in ('step','strip_0.75','ring','checker','already_covered','isoluminant')]
    fig,axes=plt.subplots(len(chosen),len(methods),figsize=(15,14))
    for row,i in enumerate(chosen):
        for col,(name,a) in enumerate(methods):
            ax=axes[row,col];ax.imshow(np.clip(a[i,12:52,12:52],0,1),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(name)
            if col==0:ax.set_ylabel(rows[i]['family'])
    fig.suptitle('CONV first-jet tangent contraction: fixed-policy discovery battery\nCoverage-space diagnostic; no geometry enters the postprocess filters',fontsize=15)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','analyze','plot']);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    globals()[args.action](args.out)
