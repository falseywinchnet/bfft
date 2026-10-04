"""Direct affine transport of a region, rather than equating its centroids.

All coordinates are source-image pixels. A correspondence carries a local map,
photometric validation, reverse transport, and a two-dimensional information
matrix. No output atlas or camera-origin constraint enters this module.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

POLICY=dict(radius=12,step=2,iterations=16,maximum_shift=12.,minimum_correlation=.78,
    minimum_validation=.73,maximum_roundtrip=1.5,minimum_singular=.28,maximum_singular=3.5,
    minimum_information_ratio=.012,local_radius=65.,local_trials=32)


def sample(plane,xy):
    return np.stack([ndimage.map_coordinates(plane[...,k],[xy[...,1],xy[...,0]],order=1,mode='nearest') for k in range(plane.shape[-1])],axis=-1)


def normalize(values):
    centered=values-values.mean(axis=1,keepdims=True)
    norm=np.sqrt(np.sum(centered*centered,axis=(1,2),keepdims=True))
    return centered/np.maximum(norm,1e-9),norm


def objective(a,b):
    x,_=normalize(a);y,_=normalize(b);return np.sum(x*y,axis=(1,2))


def local_initializations(pa,pb,trials=32):
    """Centroids propose a neighborhood map; direct image fitting relocates it."""
    rng=np.random.default_rng(31);out=np.tile(np.eye(2),(len(pa),1,1));support=np.zeros(len(pa),int)
    distances=cdist(pa,pa)
    for k in range(len(pa)):
        near=np.flatnonzero((distances[k]>4)&(distances[k]<65)&(np.linalg.norm(pb-pb[k],axis=1)<150))
        if len(near)<4:continue
        near=near[np.argsort(distances[k,near])[:32]];x=pa[near]-pa[k];y=pb[near]-pb[k];best=(0,-1e9)
        for _ in range(trials):
            sel=rng.choice(len(x),2,replace=False)
            if abs(np.linalg.det(x[sel]))<25:continue
            a=np.linalg.solve(x[sel],y[sel]).T;sv=np.linalg.svd(a,compute_uv=False)
            if np.linalg.det(a)<=0 or sv.min()<.28 or sv.max()>3.5:continue
            errors=np.linalg.norm(x@a.T-y,axis=1);good=errors<5
            # A physical source site has at most one vote among its alternatives.
            votes=len(np.unique(np.round(pa[near[good]],4),axis=0));score=(votes,-float(np.median(errors[good])) if good.any() else -1e9)
            if score>best:best=score;out[k]=a;support[k]=votes
    return out,support


def fit_affine_regions(reference,moving,pa,pb,initial=None,iterations=16,radius=12):
    pa=np.asarray(pa,float);pb=np.asarray(pb,float);n=len(pa)
    if n==0:return {}
    a0=np.tile(np.eye(2),(n,1,1)) if initial is None else np.asarray(initial,float)
    axis=np.arange(-radius,radius+1,2);yy,xx=np.meshgrid(axis,axis,indexing='ij');offsets=np.c_[xx.ravel(),yy.ravel()];uv=offsets/radius
    held=((xx.ravel()/2+3*yy.ravel()/2).astype(int)%5)==0;train=~held
    channels=np.array([1.,2.,2.]);ref=np.asarray(reference,float)*channels;mov=np.asarray(moving,float)*channels
    grad=np.stack(np.gradient(mov,axis=(0,1)),axis=-1)[...,[1,0]]
    points=pa[:,None,:]+offsets;values=sample(ref,points);rvec,rnorm=normalize(values[:,train]);target0=pb[:,None,:]+np.einsum('nij,pj->npi',a0,offsets)
    params=np.zeros((n,6));basis=np.c_[uv,np.ones(len(uv))]
    def evaluate(p,jac=False):
        xy=target0+np.einsum('nij,pj->npi',p.reshape(n,2,3),basis)
        vals=sample(mov,xy);vec,norm=normalize(vals[:,train]);error=(vec-rvec).reshape(n,-1)
        if not jac:return xy,vals,error
        gg=np.stack([sample(grad[...,k],xy) for k in range(2)],axis=-1)[:,train]
        jj=np.concatenate([gg[...,k,None]*basis[None,train,None,:] for k in range(2)],axis=-1)
        jj-=jj.mean(axis=1,keepdims=True)
        jj=(jj-vec[...,None]*np.sum(vec[...,None]*jj,axis=(1,2),keepdims=True))/np.maximum(norm[...,None],1e-9)
        jj*= (norm>1e-5)[...,None]
        return xy,vals,error,jj.reshape(n,-1,6)
    for _ in range(iterations):
        xy,vals,err,j=evaluate(params,True);h=np.einsum('npi,npj->nij',j,j);g=np.einsum('npi,np->ni',j,err)
        damping=2e-5+1e-4*np.trace(h,axis1=1,axis2=2)/6
        step=np.linalg.solve(h+np.eye(6)[None]*damping[:,None,None],g[...,None])[...,0]
        step*=np.minimum(1,2.5/np.maximum(np.linalg.norm(step,axis=1),1e-8))[:,None]
        trial=params-step;_,_,te=evaluate(trial);keep=np.sum(te*te,axis=1)<np.sum(err*err,axis=1)
        params[keep]=trial[keep]
        if np.max(np.linalg.norm(step,axis=1))<.005:break
    xy,vals,error,j=evaluate(params,True);affine=a0+params.reshape(n,2,3)[:,:,:2]/radius;center=pb+params[:,[2,5]]
    sv=np.linalg.svd(affine,compute_uv=False);inside=np.all((points>=1)&(points<np.array(reference.shape[1::-1])-2),axis=(1,2))&np.all((xy>=1)&(xy<np.array(moving.shape[1::-1])-2),axis=(1,2))
    # Black panorama borders are missing support, not dark scene texture.
    valid_ref=sample(np.any(reference>.002,axis=-1)[...,None].astype(float),points)[...,0]
    valid_mov=sample(np.any(moving>.002,axis=-1)[...,None].astype(float),xy)[...,0];inside&=(valid_ref>.98).all(1)&(valid_mov>.98).all(1)
    h=np.einsum('npi,npj->nij',j,j);translation=[2,5];shape=[0,1,3,4];htt=h[:,translation][:,:,translation];hta=h[:,translation][:,:,shape];haa=h[:,shape][:,:,shape]
    info=htt-hta@np.linalg.pinv(haa,rcond=1e-6)@np.swapaxes(hta,1,2);eigs=np.linalg.eigvalsh(info);ratio=np.maximum(eigs[:,0],0)/np.maximum(eigs[:,1],1e-12)
    corr=objective(values[:,train],vals[:,train]);validation=objective(values[:,held],vals[:,held])
    admissible=inside&(np.linalg.det(affine)>0)&(sv.min(1)>.28)&(sv.max(1)<3.5)&(np.linalg.norm(center-pb,axis=1)<=12)&(rnorm[:,0,0]>.03)
    return dict(center=center,affine=affine,train_correlation=corr,validation_correlation=validation,initial_correlation=objective(values,sample(mov,target0)),information=info,information_ratio=ratio,admissible=admissible)


def transport(reference,moving,pa,pb):
    initial,neighborhood=local_initializations(pa,pb)
    # Identity remains an alternative when a neighborhood affine is wrong.
    aa=np.r_[initial,np.tile(np.eye(2),(len(pa),1,1))];x=np.r_[pa,pa];y=np.r_[pb,pb]
    fit=fit_affine_regions(reference,moving,x,y,aa);n=len(pa)
    score=fit['train_correlation'].copy();score[~fit['admissible']]=-2
    choice=np.where(score[:n]>=score[n:],np.arange(n),np.arange(n)+n)
    forward={k:v[choice] for k,v in fit.items()}
    # Reverse starts at the original moving centroid, not at the fitted endpoint.
    reverse=fit_affine_regions(moving,reference,pb,pa,np.linalg.inv(initial))
    delta=forward['center']-pb;back=reverse['center']+np.einsum('nij,nj->ni',reverse['affine'],delta)
    closure=np.linalg.norm(back-pa,axis=1);jac=np.linalg.norm(reverse['affine']@forward['affine']-np.eye(2),axis=(1,2))
    accepted=forward['admissible']&reverse['admissible']&(forward['train_correlation']>=.78)&(forward['validation_correlation']>=.73)&(reverse['validation_correlation']>=.73)&(closure<=1.5)&(jac<.6)
    return dict(**forward,accepted=accepted,well_localized=accepted&(forward['information_ratio']>=.012),roundtrip=closure,jacobian_roundtrip=jac,neighborhood_support=neighborhood,reverse_correlation=reverse['validation_correlation'],reverse_center=reverse['center'],reverse_affine=reverse['affine'])
