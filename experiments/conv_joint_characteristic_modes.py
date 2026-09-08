"""Finite, source-certified characteristic potential modes F = phi(G).

The current catalog recognizes monotone straight and radial geometry. It
rejects unsupported geometry instead of fitting a chart or changing a cone.
"""
import json
from pathlib import Path
import numpy as np
from experiments.conv_joint_ridge_modes import (
    compile_mode as compile_ridge,compile_profile,evaluate,evaluate_profile,support_normals,
)
from experiments.probe_conv_compatible_2d import resize


def compile_radial(source):
    source=np.asarray(source,dtype=float);h,w=source.shape
    scale=float(np.ptp(source));offset=float(source.min())
    if scale==0:
        return {'constant':offset},{'accepted':True,'chart':'constant'}
    z=(source-offset)/scale;yy,xx=np.mgrid[:h,:w]
    points=np.stack((xx.ravel(),yy.ravel()),axis=1);values=z.ravel()
    eps=64*np.finfo(float).eps
    centers=[]
    for mask in (values<=eps,values>=1-eps):
        center=np.mean(points[mask],axis=0)
        if not any(np.max(abs(center-c))<1e-12 for c in centers):
            centers.append(center)
    attempts=[]
    for center in centers:
        for polarity in (1.,-1.):
            phase=polarity*np.sum((points-center)**2,axis=1)
            model,diag=compile_profile(phase,values,128*np.finfo(float).eps*max(h,w)**2,flat_ends=True)
            if model is None:
                attempts.append(diag['reason']);continue
            model.update({'chart':'radial','center':center,'polarity':polarity,'offset':offset,'scale':scale})
            direction_margin=float('inf');range_margin=float('inf')
            for cy in range(h-1):
                for cx in range(w-1):
                    corners=np.array([[cx,cy],[cx+1,cy],[cx,cy+1],[cx+1,cy+1]],dtype=float)
                    gradients=2*polarity*(corners-center)
                    for normal in support_normals(z,cy,cx):
                        direction_margin=min(direction_margin,float(np.min(gradients@normal)))
                    nearest=np.clip(center,[cx,cy],[cx+1,cy+1])
                    low=float(np.sum((nearest-center)**2));high=float(np.max(np.sum((corners-center)**2,axis=1)))
                    phase_range=np.sort(polarity*np.array([low,high]))
                    attained=evaluate_profile(model,phase_range)
                    support=source[max(0,cy-2):min(h,cy+4),max(0,cx-2):min(w,cx+4)]
                    range_margin=min(range_margin,float(attained[0]-support.min()),float(support.max()-attained[1]))
            if direction_margin<-2e-11 or range_margin<-2e-11*scale:
                attempts.append('original direction or range certificate failed');continue
            recovered=evaluate(model,points[:,0],points[:,1])
            diag.update({'chart':'radial','center':center.tolist(),'polarity':polarity,
                         'direction_margin':direction_margin if np.isfinite(direction_margin) else None,
                         'range_margin':range_margin,'cardinality_error':float(np.max(abs(recovered-source.ravel())))})
            return model,diag
    return None,{'accepted':False,'reason':'no certified radial phase among extremum-center witnesses','attempts':attempts}


def compile_mode(source):
    model,diag=compile_ridge(source)
    if model is not None:
        diag['chart']=model.get('chart','constant');return model,diag
    radial,rd=compile_radial(source)
    rd['ridge_rejection']=diag['reason']
    return radial,rd


def measure(name,source,truth,root,rows,images,family=None):
    n=source.shape[0];q=np.linspace(0,n-1,truth.shape[0]);yy,xx=np.meshgrid(q,q,indexing='ij')
    mask=(xx>.2*(n-1))&(xx<.8*(n-1))&(yy>.2*(n-1))&(yy<.8*(n-1))
    model,d=compile_mode(source)
    if model is None:
        r={'case':name,'family':family,**d};rows.append(r);print(json.dumps(r),flush=True)
    else:
        data={'source':source.tolist(),'truth':truth.tolist()}
        for method in ('CONV','Lanczos-3','characteristic'):
            z=evaluate(model,xx,yy) if method=='characteristic' else resize(source,3,method)
            g=np.array(np.gradient(z));gt=np.array(np.gradient(truth))
            r={'case':name,'family':family,'method':method,'accepted':True,
               'mse':float(np.mean((z[mask]-truth[mask])**2)),
               'full_mse':float(np.mean((z-truth)**2)),
               'sampled_gradient_mse':float(np.mean(np.sum((g-gt)**2,axis=0)[mask])),
               'excursion':float(max(0,z.max()-source.max(),source.min()-z.min()))}
            if method=='characteristic':
                r.update(d)
            rows.append(r);data[method]=z.tolist();print(json.dumps(r),flush=True)
        images[name]=data
    Path(root).write_text(json.dumps({'rows':rows,'images':images},indent=2)+'\n')


def run(out):
    out=Path(out);out.mkdir(exist_ok=True)
    previous=json.loads(Path('output/support_geometry/conv_compatible_current/conv_compatible_2d.json').read_text())
    rows=[];images={}
    for name,data in previous['images'].items():
        measure(name,np.array(data['source']),np.array(data['truth']),out/'matched.json',rows,images)
    n=13;yy,xx=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
    v,u=np.meshgrid(np.linspace(0,1,37),np.linspace(0,1,37),indexing='ij')
    fields=[]
    for slope in (.35,2**-.5,-.4,1.1):
        for width in (18,32):
            fields.append((f'edge {slope} {width}','straight',
                           lambda x,y,s=slope,w=width:.5*(1+np.tanh(w*(x+s*(y-.5)-.48)))))
    for center in ((.5,.5),(.5+.5/12,.5-.5/12),(.43,.57)):
        for width in (25,45):
            fields.append((f'radial {center} {width}','radial',
                           lambda x,y,c=center,w=width:.5*(1+np.tanh(w*(np.hypot(x-c[0],y-c[1])-.23)))))
    fields.append(('ellipse','unsupported',lambda x,y:.5*(1+np.tanh(10*(np.hypot((x-.5)/.28,(y-.5)/.19)-1)))))
    fields.append(('crossing waves','unsupported',lambda x,y:.5+.2*np.sin(2*np.pi*(3*x+2*y))+.2*np.cos(2*np.pi*(2*x-3*y))))
    rows=[];images={}
    for name,family,f in fields:
        measure(name,f(xx,yy),f(u,v),out/'validation.json',rows,images,family)


if __name__=='__main__':
    run('/tmp/conv_joint_characteristic_modes')
