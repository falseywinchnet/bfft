"""Bearing geometry with distinct camera origins. No fused atlas coordinates."""
import numpy as np


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def bearings(points,shape,focal,projection='perspective',focal_x=None,focal_y=None,principal_point=None):
    h,w=shape;center=[(w-1)/2,(h-1)/2] if principal_point is None else principal_point
    p=(np.asarray(points)-center)/[focal if focal_x is None else focal_x,focal if focal_y is None else focal_y]
    if projection=='perspective':return unit(np.c_[p,np.ones(len(p))])
    if projection=='cylindrical':return unit(np.c_[np.sin(p[:,0]),p[:,1],np.cos(p[:,0])])
    if projection=='equirectangular':return np.c_[np.sin(p[:,0])*np.cos(p[:,1]),np.sin(p[:,1]),np.cos(p[:,0])*np.cos(p[:,1])]
    raise ValueError(projection)


def rotation_fit(a,b):
    u,_,v=np.linalg.svd(b.T@a);return u@np.diag([1,1,np.linalg.det(u@v)])@v


def rotation_errors(a,b,r):
    return np.arctan2(np.linalg.norm(np.cross(a@r.T,b),axis=1),np.sum((a@r.T)*b,axis=1))


def essential_fit(a,b):
    d=np.einsum('ni,nj->nij',b,a).reshape(-1,9)
    _,_,v=np.linalg.svd(d,full_matrices=len(d)<9);e=v[-1].reshape(3,3)
    u,s,v=np.linalg.svd(e);return u@np.diag([(s[0]+s[1])/2]*2+[0])@v


def essential_errors(a,b,e):
    na=a@e.T;nb=b@e;value=np.abs(np.sum(b*na,axis=1))
    # Two great-circle distances, both evaluated on unit bearing spheres.
    return np.maximum(np.arcsin(np.clip(value/np.maximum(np.linalg.norm(na,axis=1),1e-12),0,1)),np.arcsin(np.clip(value/np.maximum(np.linalg.norm(nb,axis=1),1e-12),0,1)))


def triangulate(a,b,r,t):
    ar=a@r.T;c=np.sum(ar*b,axis=1);ta=ar@t;tb=b@t;den=np.maximum(1-c*c,1e-12)
    da=(-ta+c*tb)/den;db=(-c*ta+tb)/den
    x=a*da[:,None];miss=np.linalg.norm(x@r.T+t-b*db[:,None],axis=1)
    angles=np.arctan2(np.linalg.norm(np.cross(ar,b),axis=1),c)
    return x,da,db,miss,angles


def essential_poses(e,a,b):
    if not len(a):return []
    u,_,v=np.linalg.svd(e)
    if np.linalg.det(u)<0:u[:,-1]*=-1
    if np.linalg.det(v)<0:v[-1]*=-1
    w=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    result=[]
    for r in (u@w@v,u@w.T@v):
        for sign in (1,-1):
            t=u[:,2]*sign;x,da,db,miss,angles=triangulate(a,b,r,t);positive=(da>0)&(db>0)
            result.append(dict(rotation=r,translation=t,positive_fraction=float(np.mean(positive)),median_parallax=float(np.median(angles[positive])) if positive.any() else 0.,points=x,positive=positive))
    return sorted(result,key=lambda r:(-r['positive_fraction'],-r['median_parallax']))


def hypotheses(a,b,site_a,site_b,seed=71,trials=320,threshold=np.deg2rad(1.2)):
    """Fit on fixed training sites, compare on unused regional sites.

Each model is scored with at most one vote per physical site in either image.
Alternative descriptor matches therefore do not multiply support.
"""
    n=len(a);split=(np.asarray(site_a)*2654435761%7)==0;train=np.flatnonzero(~split);held=np.flatnonzero(split)
    rng=np.random.default_rng(seed)
    def unique(error,eligible,other_sites=None):
        other_sites=site_b if other_sites is None else other_sites
        useda=set();usedb=set();accepted=[]
        for k in eligible[np.argsort(error[eligible])]:
            if error[k]>threshold:break
            if int(site_a[k]) not in useda and int(other_sites[k]) not in usedb:
                accepted.append(int(k));useda.add(int(site_a[k]));usedb.add(int(other_sites[k]))
        return np.array(accepted,dtype=int)
    result={}
    for name,minimum,fit,errors in [('rotation',3,rotation_fit,rotation_errors),('essential',8,essential_fit,essential_errors)]:
        best=[]
        if len(train)<minimum:continue
        _,first=np.unique(np.asarray(site_a)[train],return_index=True)
        preferred=train[np.sort(first)]
        for trial in range(trials):
            pool=preferred if trial%2==0 and len(preferred)>=minimum else train
            sample=rng.choice(pool,minimum,replace=False)
            if len(set(site_a[sample]))<minimum or len(set(site_b[sample]))<minimum:continue
            try:m=fit(a[sample],b[sample]);err=errors(a,b,m)
            except np.linalg.LinAlgError:continue
            good=unique(err,train);score=(len(good),-float(np.median(err[good])) if len(good) else -100.)
            if not best or score>best[0]:best=[score,m,good]
        if not best or len(best[2])<minimum:continue
        model=best[1]
        for _ in range(2):
            good=unique(errors(a,b,model),train)
            if len(good)>=minimum:model=fit(a[good],b[good])
        err=errors(a,b,model);good=unique(err,train);validation=unique(err,held)
        # Holdout acceptance is evidence from the preselected candidate set,
        # not independent scene truth or a calibrated depth certificate.
        null=[]
        for _ in range(63):
            permutation=rng.permutation(held);shuffled=b.copy();shuffled[held]=b[permutation];other=np.asarray(site_b).copy();other[held]=np.asarray(site_b)[permutation]
            null.append(len(unique(errors(a,shuffled,model),held,other)))
        null_p=(1+sum(v>=len(validation) for v in null))/(len(null)+1)
        result[name]=dict(heldout_permutation_p=null_p,heldout_null_median=float(np.median(null)),heldout_null_p95=float(np.quantile(null,.95)),matrix=model.tolist(),train_inliers=good.tolist(),heldout_inliers=validation.tolist(),train_count=len(good),heldout_count=len(validation),heldout_candidates=len(held),
            train_median_degrees=float(np.rad2deg(np.median(err[good]))) if len(good) else None,
            heldout_median_degrees=float(np.rad2deg(np.median(err[validation]))) if len(validation) else None)
    if 'essential' in result:
        e=result['essential'];good=np.array(e['train_inliers']+e['heldout_inliers'],int)
        poses=essential_poses(np.array(e['matrix']),a[good],b[good]);e['poses']=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items() if k not in ('points','positive')} for p in poses]
    return result
