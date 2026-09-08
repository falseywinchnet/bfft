"""Direct shared-jet capacity experiment. No optimization or linear solves.

The zero-jet baseline is range-feasible and C2, but need not satisfy the
original support-gradient cones. Such cases are explicitly rejected.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from experiments.convstar import _local_two_jet
from experiments.conv_joint_potential import (
    tensor_certificate, support_normals, _directional_derivative_coefficients, evaluate,
)


def hermite_map(n):
    out=np.zeros((5*(n-1)+1,3*n))
    for i in range(n-1):
        a=3*i;b=a+3;t=5*i
        out[t,a]=1
        out[t+1,a:a+3]=[1,1/5,0]
        out[t+2,a:a+3]=[1,2/5,1/20]
        out[t+3,b:b+3]=[1,-2/5,1/20]
        out[t+4,b:b+3]=[1,-1/5,0]
        out[t+5,b]=1
    return out


def jet_map(n):
    first,second=_local_two_jet(np.eye(n))
    out=np.empty((3*n,n));out[::3]=np.eye(n)
    out[1::3]=first;out[2::3]=second
    return out


def constraint_matrix(y,depth=1):
    h,w=y.shape;gw=5*(w-1)+1;gh=5*(h-1)+1
    cert=tensor_certificate(depth)
    rows=[];cols=[];data=[];bounds=[];kinds=[]
    def add(index,coeff,bound,kind):
        begin=len(bounds);r,c=np.nonzero(abs(coeff)>1e-15)
        rows.extend(begin+r);cols.extend(index[c]);data.extend(coeff[r,c])
        bounds.extend(np.broadcast_to(bound,len(coeff)));kinds.extend([kind]*len(coeff))
    for iy in range(h-1):
        for ix in range(w-1):
            index=((5*iy+np.arange(6)[:,None])*gw+5*ix+np.arange(6)[None,:]).ravel()
            support=y[max(0,iy-2):min(h,iy+4),max(0,ix-2):min(w,ix+4)]
            add(index,cert,float(support.min()),'range')
            add(index,-cert,-float(support.max()),'range')
            for dx,dy in support_normals(y,iy,ix):
                derivative=np.array(_directional_derivative_coefficients(dx,dy)).reshape(36,36)
                add(index,cert@derivative,0.,'direction')
    return sparse.coo_matrix((data,(rows,cols)),shape=(len(bounds),gh*gw)).tocsr(),np.array(bounds),np.array(kinds)


def finite_capacity(matrix,lower,base,target,owners):
    """One simultaneous capacity allocation and one closed-form ray.

    A shared nodal jet bundle is one entity, even when its polynomial support
    overlaps that of other entities. Each constraint uses the summed response
    of the complete bundle before measuring harmful contributions.
    """
    margin=matrix@base-lower;candidate=matrix@target-lower
    eps=2e-11
    diagnostic={'minimum_baseline_margin':float(margin.min()),
                'minimum_candidate_margin':float(candidate.min())}
    if candidate.min()>=-eps:
        return target.copy(),{**diagnostic,'accepted':True,'candidate_passed':True,
                              'minimum_admitted_margin':float(candidate.min())}
    if margin.min()<-eps:
        return None,{**diagnostic,'accepted':False,'reason':'baseline violates original constraints'}
    delta=target-base;count=int(owners.max())+1
    group=sparse.coo_matrix((np.ones(len(owners)),(np.arange(len(owners)),owners)),
                           shape=(len(owners),count)).tocsr()
    response=(matrix.multiply(delta)@group).tocsr()
    harmful=response.copy();harmful.data=np.maximum(-harmful.data,0);harmful.eliminate_zeros()
    loss=np.asarray(harmful.sum(axis=1)).ravel()
    ratio=np.ones(len(loss));active=loss>0
    ratio[active]=np.clip(np.maximum(margin[active],0)/loss[active],0,1)
    capacities=np.ones(count);coo=harmful.tocoo()
    np.minimum.at(capacities,coo.col,ratio[coo.row])
    admitted=base+capacities[owners]*delta
    margin=matrix@admitted-lower;motion=matrix@(target-admitted)
    negative=motion<0
    ray=min(1.,float(np.min(np.maximum(margin[negative],0)/(-motion[negative])))) if negative.any() else 1.
    admitted+=ray*(target-admitted)
    return admitted,{**diagnostic,'accepted':True,'candidate_passed':False,
                      'minimum_admitted_margin':float((matrix@admitted-lower).min()),
                      'minimum_capacity':float(capacities.min()),
                      'mean_capacity':float(capacities.mean()),'completion_ray':ray}


def admit(source,depth=1,grouping='node'):
    start=time.perf_counter();source=np.asarray(source,dtype=float)
    if source.ndim!=2 or min(source.shape)<5 or not np.all(np.isfinite(source)):
        raise ValueError('finite scalar source, at least five nodes per axis required')
    offset=float(source.min());scale=float(np.ptp(source))
    if scale==0:
        return np.full((5*(source.shape[0]-1)+1,5*(source.shape[1]-1)+1),offset),{'accepted':True,'seconds':0.}
    y=(source-offset)/scale;h,w=y.shape
    target=jet_map(h)@y@jet_map(w).T
    base=np.zeros_like(target);base[::3,::3]=y
    lift=sparse.kron(hermite_map(h),hermite_map(w),format='csr')
    a,lower,kinds=constraint_matrix(y,depth)
    matrix=(a@lift).tocsr()
    owner=(np.arange(3*h)[:,None]//3*w+np.arange(3*w)[None,:]//3).ravel()
    if grouping=='order':
        order=(np.arange(3*h)[:,None]%3+np.arange(3*w)[None,:]%3).ravel()
        owner=owner*5+order
    elif grouping=='component':
        owner=np.arange(target.size)
    elif grouping!='node':
        raise ValueError('unknown shared-jet entity grouping')
    jets,diag=finite_capacity(matrix,lower,base.ravel(),target.ravel(),owner)
    bm=matrix@base.ravel()-lower
    diag.update({'depth':depth,'continuity':2,'grouping':grouping,'seconds':time.perf_counter()-start,
                 'baseline_range_margin':float(bm[kinds=='range'].min()),
                 'baseline_direction_margin':float(bm[kinds=='direction'].min()) if np.any(kinds=='direction') else None})
    if jets is None:
        return None,diag
    lattice=(lift@jets).reshape((5*(h-1)+1,5*(w-1)+1))*scale+offset
    lattice[::5,::5]=source
    return lattice,diag


def run(out,grouping='node'):
    previous=json.loads(Path('output/support_geometry/conv_compatible_current/conv_compatible_2d.json').read_text())
    q=np.linspace(0,16,49);yy,xx=np.meshgrid(q,q,indexing='ij')
    mask=(xx>3.2)&(xx<12.8)&(yy>3.2)&(yy<12.8)
    rows=[];images={}
    for name,data in previous['images'].items():
        source=np.array(data['source']);truth=np.array(data['truth'])
        p,d=admit(source,grouping=grouping);r={'case':name,**d}
        if p is not None:
            z=evaluate(p,xx,yy)
            r.update({'mse':float(np.mean((z[mask]-truth[mask])**2)),
                      'full_mse':float(np.mean((z-truth)**2)),
                      'excursion':float(max(0,z.max()-source.max(),source.min()-z.min()))})
            images[name]={'lattice':p.tolist(),'image':z.tolist()}
        rows.append(r);print(json.dumps(r),flush=True)
        Path(out).write_text(json.dumps({'rows':rows,'images':images},indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--grouping',choices=['node','order','component'],default='node')
    p.add_argument('--out',default='/tmp/conv_joint_finite.json');a=p.parse_args()
    run(a.out,a.grouping)
