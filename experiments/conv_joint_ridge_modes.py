"""Source-witnessed monotone ridge modes, constructed without any solve.

This is a certified special-case mode, not a universal image interpolator.
The direction comes from finite sample-order and original cone inequalities.
"""
import json
import math
from pathlib import Path
import numpy as np
from experiments.conv_warp.joint_reference import _q1_corner_gradients,_dual_generators_of_planar_cone


def support_normals(y,cy,cx):
    """Original CONV support-cone witness, without importing the QP prototype."""
    h,w=y.shape;vectors=[]
    for iy in range(max(0,cy-2),min(h-1,cy+3)):
        for ix in range(max(0,cx-2),min(w-1,cx+3)):
            vectors.extend(_q1_corner_gradients(y[iy:iy+2,ix:ix+2]))
    return _dual_generators_of_planar_cone(np.asarray(vectors),float(np.max(abs(y))))


def direction_arc(vectors):
    """Intersect homogeneous planar halfspaces by circular ordering."""
    v=np.asarray(vectors,dtype=float).reshape(-1,2)
    v=v[np.sum(v*v,axis=1)>1e-28]
    if not len(v):
        return None
    angles=np.sort(np.mod(np.arctan2(v[:,1],v[:,0]),2*np.pi))
    gaps=np.diff(np.r_[angles,angles[0]+2*np.pi]);k=int(np.argmax(gaps))
    start=float(angles[(k+1)%len(angles)]);width=float(2*np.pi-gaps[k])
    if width>np.pi+2e-12:
        return None
    return start+width/2,max(0.,(np.pi-width)/2)


def compile_mode(source,selection='ties'):
    """Recognize and compile a source-compatible ridge; no parameter fit."""
    y=np.asarray(source,dtype=float)
    if y.ndim!=2 or min(y.shape)<2 or not np.all(np.isfinite(y)):
        raise ValueError('finite scalar rectangle required')
    scale=float(np.ptp(y));offset=float(y.min())
    if scale==0:
        return {'constant':offset},{'accepted':True,'constant':True}
    z=(y-offset)/scale;h,w=z.shape
    yy,xx=np.mgrid[:h,:w];points=np.stack((xx.ravel(),yy.ravel()),axis=1)
    values=z.ravel();i,j=np.triu_indices(len(values),1)
    diff=values[j]-values[i];eps=64*np.finfo(float).eps
    strict=abs(diff)>eps
    displacement=points[j]-points[i]
    order=displacement[strict]*np.sign(diff[strict,None])
    normals=[]
    for cy in range(h-1):
        for cx in range(w-1):
            normals.extend(support_normals(z,cy,cx))
    constraints=order
    arc=direction_arc(order)
    if arc is None:
        return None,{'accepted':False,'reason':'no global direction satisfies sample order'}
    center,half=arc
    phase_eps=128*np.finfo(float).eps*max(h,w)
    # The midpoint is canonical. Boundary flux is an optional direct selector.
    theta=center;tie_count=0
    if selection=='flux':
        wy=np.ones(h);wy[[0,-1]]=.5;wx=np.ones(w);wx[[0,-1]]=.5
        flux=np.array([wy@(z[:,-1]-z[:,0]),wx@(z[-1]-z[0])])
        angle=math.atan2(flux[1],flux[0]);relative=(angle-center+np.pi)%(2*np.pi)-np.pi
        theta=center+float(np.clip(relative,-half,half))
        trial=np.array([math.cos(theta),math.sin(theta)])
        if np.min(order@trial)<=phase_eps:
            theta=center
    elif selection=='ties':
        # Equal-value source pairs supply finite candidate normal directions.
        # Choose their most repeated admissible lattice direction. No truth,
        # angle supplied by the case, or reconstruction error is consulted.
        d=displacement[~strict].copy()
        if len(d):
            gcd=np.gcd(abs(d[:,0]),abs(d[:,1]));d=d//gcd[:,None]
            flip=(d[:,0]<0)|((d[:,0]==0)&(d[:,1]<0));d[flip]*=-1
            directions,counts=np.unique(d,axis=0,return_counts=True)
            candidates=[]
            for v,count in zip(directions,counts):
                angle=math.atan2(-v[0],v[1]);relative=(angle-center+np.pi)%(2*np.pi)-np.pi
                if abs(relative)>np.pi/2:
                    angle+=np.pi;relative=(angle-center+np.pi)%(2*np.pi)-np.pi
                if abs(relative)<=half+2e-12:
                    trial=np.array([math.cos(center+relative),math.sin(center+relative)])
                    if np.min(order@trial)>phase_eps:
                        candidates.append((int(count),-abs(relative),center+relative))
            if candidates:
                tie_count,_,theta=max(candidates)
    elif selection!='center':
        raise ValueError('selection must be center, flux, or ties')
    # Magnitudes, not only ordering, determine a witnessed affine field.
    affine_gradient=np.array([z[0,1]-z[0,0],z[1,0]-z[0,0]])
    affine=z[0,0]+points@affine_gradient
    affine_witness=bool(np.max(abs(affine-values))<=128*np.finfo(float).eps)
    if affine_witness:
        theta=math.atan2(affine_gradient[1],affine_gradient[0])
    n=np.array([math.cos(theta),math.sin(theta)])
    margin=float(np.min(constraints@n))
    if margin<-2e-11:
        return None,{'accepted':False,'reason':'direction fails final certificate','direction_margin':margin}
    model,diag=compile_profile(points@n,values,phase_eps)
    if model is None:
        return None,diag
    model.update({'direction':n,'offset':offset,'scale':scale,'chart':'ridge'})
    certificate=certify_ridge_current(z,model)
    if certificate['current_certificate_margin']<-2e-11 or certificate['range_margin']<-2e-11:
        return None,{'accepted':False,'reason':'original joint current or range certificate failed',**certificate}
    reconstructed=evaluate(model,points[:,0],points[:,1])
    diag.update({'selection':selection,'direction':n.tolist(),
                 'direction_interval_degrees':2*half*180/np.pi,'tie_witness_pairs':tie_count,
                 'order_margin':margin,**certificate,
                 'direction_only_cone_compatible':bool(not normals or np.min(np.asarray(normals)@n)>=-2e-11),
                 'affine_witness':affine_witness,
                 'cardinality_error':float(np.max(abs(reconstructed-y.ravel())))})
    return model,diag


def certify_ridge_current(source,model):
    """Certify the actual current phi'(G) grad G, including zero amplitude.

    On each intersecting phase piece, derivative Bernstein coefficients give
    a finite upper bound. A wrong direction with identically zero amplitude
    does not violate the current law. Original cones are never enlarged.
    """
    h,w=source.shape;n=model['direction'];knots=model['knots']
    derivative=5*np.diff(model['controls'],axis=1)/np.diff(knots)[:,None]
    upper=np.max(derivative,axis=1)
    current_margin=0.;range_margin=float('inf')
    normalized={**model,'offset':0.,'scale':1.}
    for cy in range(h-1):
        for cx in range(w-1):
            corners=np.array([[cx,cy],[cx+1,cy],[cx,cy+1],[cx+1,cy+1]])
            phase=corners@n;lo=float(phase.min());hi=float(phase.max())
            begin=max(0,int(np.searchsorted(knots,lo,side='right'))-1)
            end=min(len(upper),int(np.searchsorted(knots,hi,side='left')))
            maximum=float(np.max(upper[begin:end]))
            for normal in support_normals(source,cy,cx):
                q=float(n@normal)
                if q<0:
                    current_margin=min(current_margin,q*maximum)
            attained=evaluate_profile(normalized,np.array([lo,hi]))
            support=source[max(0,cy-2):min(h,cy+4),max(0,cx-2):min(w,cx+4)]
            range_margin=min(range_margin,float(attained[0]-support.min()),float(support.max()-attained[1]))
    return {'current_certificate_margin':current_margin,'range_margin':range_margin}


def compile_profile(phase,values,phase_eps,flat_ends=False):
    index=np.argsort(phase,kind='stable');t=np.asarray(phase)[index];f=np.asarray(values)[index]
    cut=np.r_[0,np.flatnonzero(np.diff(t)>phase_eps)+1,len(t)]
    knots=[];samples=[];merge_error=0.
    for a,b in zip(cut[:-1],cut[1:]):
        knots.append(float(np.mean(t[a:b])));samples.append(float(np.mean(f[a:b])))
        merge_error=max(merge_error,float(np.ptp(f[a:b])))
    t=np.array(knots);f=np.array(samples)
    if len(t)<2 or merge_error>2e-11 or np.min(np.diff(f))<-2e-11:
        return None,{'accepted':False,'reason':'observations incompatible along recovered phase',
                     'merge_error':merge_error}
    # Only roundoff-sized order violations can reach this operation.
    original=f.copy();f=np.maximum.accumulate(f)
    widths=np.diff(t);secants=np.diff(f)/widths
    m=np.zeros_like(f);m[0]=secants[0];m[-1]=secants[-1]
    if flat_ends:
        m[[0,-1]]=0
    a=secants[:-1];b=secants[1:];positive=(a>0)&(b>0)
    # Quintic Hermite profile with shared slopes and zero second derivatives
    # at phase knots. Its derivative controls are
    # [m_i, m_i, 5*d_i-2*m_i-2*m_{i+1}, m_{i+1}, m_{i+1}].
    # Capping each shared slope at 1.25 adjacent secants makes them nonnegative.
    harmonic=np.zeros_like(a);harmonic[positive]=2/(1/a[positive]+1/b[positive])
    m[1:-1]=np.minimum(harmonic,1.25*np.minimum(a,b))
    controls=np.stack((f[:-1],f[:-1]+widths*m[:-1]/5,f[:-1]+2*widths*m[:-1]/5,
                       f[1:]-2*widths*m[1:]/5,f[1:]-widths*m[1:]/5,f[1:]),axis=1)
    derivative=np.stack((m[:-1],m[:-1],5*secants-2*m[:-1]-2*m[1:],m[1:],m[1:]),axis=1)
    model={'knots':t,'controls':controls}
    diag={'accepted':True,'continuity':2,'phase_knots':len(t),
          'minimum_scalar_derivative_coefficient':float(derivative.min()),
          'roundoff_order_adjustment':float(np.max(abs(f-original))),
          'phase_merge_error':merge_error}
    return model,diag


def evaluate(model,x,y,gradient=False):
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    if 'constant' in model:
        return np.zeros(x.shape+(2,)) if gradient else np.full(x.shape,model['constant'])
    if model.get('chart')=='radial':
        dx=x-model['center'][0];dy=y-model['center'][1]
        t=model['polarity']*(dx*dx+dy*dy)
        phase_gradient=2*model['polarity']*np.stack((dx,dy),axis=-1)
    else:
        n=model['direction'];t=n[0]*x+n[1]*y;phase_gradient=n
    result=evaluate_profile(model,t,derivative=gradient)
    return result[...,None]*phase_gradient if gradient else result


def evaluate_profile(model,t,derivative=False):
    t=np.asarray(t,dtype=float);knots=model['knots']
    idx=np.clip(np.searchsorted(knots,t,side='right')-1,0,len(knots)-2)
    width=knots[idx+1]-knots[idx];u=np.clip((t-knots[idx])/width,0,1)
    p=model['controls'][idx]
    if derivative==2:
        d=20*np.diff(p,n=2,axis=-1)/width[...,None]**2
        curvature=sum(math.comb(3,k)*u**k*(1-u)**(3-k)*d[...,k] for k in range(4))
        return model['scale']*np.where((t<knots[0])|(t>knots[-1]),0.,curvature)
    if derivative:
        d=5*np.diff(p,axis=-1)/width[...,None]
        slope=sum(math.comb(4,k)*u**k*(1-u)**(4-k)*d[...,k] for k in range(5))
        slope=np.where((t<knots[0])|(t>knots[-1]),0.,slope)
        return model['scale']*slope
    z=sum(math.comb(5,k)*u**k*(1-u)**(5-k)*p[...,k] for k in range(6))
    return model['offset']+model['scale']*z


def evaluate_jet(model,x,y):
    """Value, gradient, and Hessian of the same characteristic potential."""
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    if 'constant' in model:
        return evaluate(model,x,y),np.zeros(x.shape+(2,)),np.zeros(x.shape+(2,2))
    if model.get('chart')=='radial':
        dx=x-model['center'][0];dy=y-model['center'][1];s=model['polarity']
        t=s*(dx*dx+dy*dy);v=2*s*np.stack((dx,dy),axis=-1);h=2*s*np.eye(2)
    else:
        v=model['direction'];t=v[0]*x+v[1]*y;h=np.zeros((2,2))
    first=evaluate_profile(model,t,True);second=evaluate_profile(model,t,2)
    gradient=first[...,None]*v
    hessian=second[...,None,None]*np.einsum('...i,...j->...ij',v,v)+first[...,None,None]*h
    return evaluate_profile(model,t),gradient,hessian


def run(out):
    previous=json.loads(Path('output/support_geometry/conv_compatible_current/conv_compatible_2d.json').read_text())
    q=np.linspace(0,16,49);yy,xx=np.meshgrid(q,q,indexing='ij')
    mask=(xx>3.2)&(xx<12.8)&(yy>3.2)&(yy<12.8)
    rows=[];images={}
    for name,data in previous['images'].items():
        source=np.array(data['source']);truth=np.array(data['truth'])
        for selection in ('center','flux','ties'):
            model,diag=compile_mode(source,selection);r={'case':name,**diag}
            if model is not None:
                z=evaluate(model,xx,yy);g=np.array(np.gradient(z));gt=np.array(np.gradient(truth))
                r.update({'mse':float(np.mean((z[mask]-truth[mask])**2)),
                          'full_mse':float(np.mean((z-truth)**2)),
                          'gradient_mse':float(np.mean(np.sum((g-gt)**2,axis=0)[mask])),
                          'excursion':float(max(0,z.max()-source.max(),source.min()-z.min()))})
                images[name+' / '+selection]=z.tolist()
            else:
                r['selection']=selection
            rows.append(r);print(json.dumps(r),flush=True)
    Path(out).write_text(json.dumps({'rows':rows,'images':images},indent=2)+'\n')


if __name__=='__main__':
    run('/tmp/conv_joint_ridge_modes.json')
