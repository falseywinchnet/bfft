"""Independent polygon clipping oracle for the retained boundary shader."""
import argparse
import json
from pathlib import Path
import numpy as np


def cross(a,b): return a[0]*b[1]-a[1]*b[0]


def pixel_area(triangle,x,y):
    # Sutherland-Hodgman vertices, independent of shader boundary-segment sum.
    poly=[np.array([x,y]),np.array([x+1,y]),np.array([x+1,y+1]),np.array([x,y+1])]
    for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
        edge=b-a;clipped=[]
        for p,q in zip(poly,poly[1:]+poly[:1]):
            fp,fq=cross(edge,p-a),cross(edge,q-a)
            if fp>=0:clipped.append(p)
            if (fp<0)!=(fq<0):clipped.append(p+(q-p)*(fp/(fp-fq)))
        poly=clipped
        if not poly:return 0.
    poly=[p-np.array([x,y]) for p in poly]
    return abs(sum(cross(p,q) for p,q in zip(poly,poly[1:]+poly[:1])))/2


def render_reference(triangles,w,h):
    out=np.zeros((h,w));sample=np.zeros((h,w));overlap=np.zeros((h,w),dtype=int)
    for tri in triangles:
        lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),[w,h])
        for y in range(lo[1],hi[1]):
            for x in range(lo[0],hi[0]):
                v=tri-np.array([x+.5,y+.5]);e=np.roll(v,-1,axis=0)-v
                d=e[:,0]*(-v[:,1])-e[:,1]*(-v[:,0]);r=(np.abs(e[:,0])+np.abs(e[:,1]))/2
                if np.any(d<=-r):continue
                area=1. if np.all(d>=r) else pixel_area(tri,x,y)
                out[y,x]+=area;overlap[y,x]+=area>0
    return out,int(overlap.max())


def analyze(out):
    result={'scope':'Disjoint opaque constant-white triangles on black; exact polygon-square intersection reference, independent of shader boundary-segment integration. GPU timing includes instanced primitive draws, clears, MSAA resolve, and FXAA when present. No textures, depth overlap, transparency, or dynamic upload timing.','scenes':[]}
    for scene in range(6):
        if not (out/f'triangles_{scene}.bin').exists():continue
        triangles=np.fromfile(out/f'triangles_{scene}.bin',np.float32).reshape(-1,3,2).astype(float)
        ref,max_overlap=render_reference(triangles,512,512)
        if scene==5: ref/=2  # Exact duplicate foreground: union is one triangle.
        np.save(out/f'geometry_reference_{scene}.npy',ref)
        expected=sum(abs(cross(t[1]-t[0],t[2]-t[0]))/2 for t in triangles)
        if scene==5:expected/=2
        mask=(ref>0)&(ref<1)
        record={'scene':scene,'triangles':len(triangles),'max_contributing_triangles_per_pixel':max_overlap,
            'reference_mass_error':float(abs(ref.sum()-expected)),'methods':{}}
        for name in ('point','point_fxaa12','msaa4','analytic_boundary','boundary_sum'):
            if not (out/f'render_{scene}_{name}.bin').exists():continue
            a=np.fromfile(out/f'render_{scene}_{name}.bin',np.uint8).reshape(512,512,4)[...,0]/255
            error=a-ref
            record['methods'][name]={'whole_image_mse':float(np.mean(error**2)),
                'boundary_mse':float(np.mean(error[mask]**2)),'max_abs_error':float(np.max(abs(error))),
                'mass_error':float(a.sum()-ref.sum())}
        result['scenes'].append(record)
        record['analytic_8bit_gate']=bool(record['methods']['analytic_boundary']['max_abs_error']<1/255+1e-5)
        if 'boundary_sum' in record['methods']:
            record['additive_8bit_gate']=bool(record['methods']['boundary_sum']['max_abs_error']<1/255+1e-5)
        record['expected_scope']= 'overlap/visibility counterexample' if scene==5 else 'shared-edge compositing counterexample' if scene==4 else 'disjoint primitives'
    (out/'geometry_quality.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert all(s['analytic_8bit_gate'] for s in result['scenes'] if s['scene']<4)
    assert all(s['methods']['boundary_sum']['max_abs_error']<1/255+1e-5 for s in result['scenes'] if s['scene']<=4 and 'boundary_sum' in s['methods'])


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=('point','point_fxaa12','msaa4','analytic_boundary','boundary_sum')
    scenes=[0,1,4,5]
    labels=['Point','Point + FXAA','4× MSAA','Boundary + alpha','Boundary + sum','Exact area']
    fig,axes=plt.subplots(4,6,figsize=(17,12))
    for row,scene in enumerate(scenes):
        ref=np.load(out/f'geometry_reference_{scene}.npy')
        arrays=[np.fromfile(out/f'render_{scene}_{m}.bin',np.uint8).reshape(512,512,4)[...,0]/255 for m in methods]+[ref]
        for col,(ax,a,name) in enumerate(zip(axes[row],arrays,labels)):
            crop=a[8:112,8:112] if scene==0 else a[8:60,8:60] if scene==1 else a[4:60,4:60]
            ax.imshow(crop,vmin=0,vmax=1,cmap='gray',interpolation='nearest')
            if row==0:ax.set_title(name)
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(['Sparse triangles','Dense triangles','Shared mesh edges','Coincident overlap'][row])
    fig.suptitle('Retained boundary measure on M4: hardware 4× MSAA comparison\nAdditive coverage fixes mesh seams; both analytic composites fail the overlap case')
    fig.tight_layout();fig.savefig(out/'geometry_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','plot']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.action](a.out)
