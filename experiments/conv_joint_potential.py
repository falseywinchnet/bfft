"""One source-defined tensor-quintic potential with sparse joint admission.

The variables are shared Bernstein controls, not two sequential rasters.
Subdivision certifies continuous range and support-gradient-cone membership.
An ADMM QP oracle minimizes the exact integrated squared gradient correction.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.conv_compatible_current import compatible_proposal
from experiments.conv_warp.joint_reference import (
    _q1_corner_gradients, _dual_generators_of_planar_cone,
    _directional_derivative_coefficients,
)


@lru_cache(None)
def line_map(n,boundary='polynomial'):
    y=np.eye(n)
    if boundary=='polynomial':
        c,_=compatible_proposal(y)
    elif boundary=='natural':
        from scipy.interpolate import make_interp_spline
        zero=np.zeros(n)
        spl=make_interp_spline(np.arange(n),y,k=5,
                              bc_type=([(3,zero),(4,zero)],[(3,zero),(4,zero)]))
        m=spl(np.arange(n),1);q=spl(np.arange(n),2);delta=np.diff(y,axis=0)
        c=np.stack((m[:-1]/5,m[:-1]/5+q[:-1]/20,
                    delta-.4*(m[:-1]+m[1:])+(q[1:]-q[:-1])/20,
                    m[1:]/5-q[1:]/20,m[1:]/5),axis=1)
    else:
        raise ValueError('boundary must be polynomial or natural')
    controls=np.concatenate((y[:-1,None,:],y[:-1,None,:]+np.cumsum(c,axis=1)),axis=1)
    out=np.zeros((5*(n-1)+1,n))
    for i in range(n-1):
        out[5*i:5*i+6]=controls[i]
    out[::5]=y
    return out


def proposal(values,boundary='polynomial'):
    y=np.asarray(values,dtype=float)
    return line_map(y.shape[0],boundary)@y@line_map(y.shape[1],boundary).T


def gram(n):
    return np.array([[math.comb(n,i)*math.comb(n,j)/((2*n+1)*math.comb(2*n,i+j))
                      for j in range(n+1)] for i in range(n+1)])


@lru_cache(None)
def local_gradient_gram():
    d=5*np.diff(np.eye(6),axis=0)
    m=gram(5);k=d.T@gram(4)@d
    return np.kron(m,k)+np.kron(k,m)


@lru_cache(None)
def subdivision(depth):
    maps=[np.eye(6)]
    for _ in range(depth):
        refined=[]
        for mat in maps:
            levels=[mat]
            for k in range(5):
                levels.append(.5*(levels[-1][:-1]+levels[-1][1:]))
            left=np.array([level[0] for level in levels])
            right=np.array([level[-1] for level in levels][::-1])
            refined.extend((left,right))
        maps=refined
    return tuple(maps)


@lru_cache(None)
def tensor_certificate(depth):
    return np.concatenate([np.kron(y,x) for y in subdivision(depth)
                           for x in subdivision(depth)],axis=0)


def support_normals(y,cy,cx):
    h,w=y.shape
    vectors=[]
    for iy in range(max(0,cy-2),min(h-1,cy+3)):
        for ix in range(max(0,cx-2),min(w-1,cx+3)):
            vectors.extend(_q1_corner_gradients(y[iy:iy+2,ix:ix+2]))
    return _dual_generators_of_planar_cone(np.asarray(vectors),float(np.max(abs(y))))


def continuity_lift(shape,continuity,free):
    """Eliminate normal trace equalities in each axis before optimization.

    Shared source values remain independent coordinates. Corrections at them
    are fixed to zero; C1/C2 continuity then holds structurally, to roundoff.
    """
    maps=[];retained=[]
    for n in shape:
        excluded={b+j for b in range(5,n-1,5) for j in range(1,continuity+1)}
        keep=np.array([i for i in range(n) if i not in excluded])
        lookup={v:k for k,v in enumerate(keep)}
        rows=list(keep);cols=list(range(len(keep)));vals=[1.]*len(keep)
        for b in range(5,n-1,5):
            if continuity>=1:
                rows.extend([b+1]*2);cols.extend([lookup[b-1],lookup[b]]);vals.extend([-1.,2.])
            if continuity>=2:
                rows.extend([b+2]*3);cols.extend([lookup[b-2],lookup[b-1],lookup[b]]);vals.extend([1.,-4.,4.])
        maps.append(sparse.coo_matrix((vals,(rows,cols)),shape=(n,len(keep))).tocsr())
        retained.append(keep)
    fixed=(retained[0][:,None]%5==0)&(retained[1][None,:]%5==0)
    return sparse.kron(maps[0],maps[1],format='csr')[free][:,~fixed.ravel()]


def qp_admm(h,a,lower,upper,*,tol=2e-7,maxiter=30000):
    """Solve min .5 x'Hx subject to lower <= A x <= upper.

    Proximal ADMM with row/column scaling and sparse direct factorization.
    No regularization term is added to the requested objective.
    """
    n=h.shape[0]
    column=1/np.sqrt(np.maximum(h.diagonal(),1e-12))
    d=sparse.diags(column)
    hs=(d@h@d).tocsc()
    a=(a@d).tocsr()
    row=1/np.maximum(np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel()),1e-12)
    a=(sparse.diags(row)@a).tocsc()
    lo=lower*row;hi=upper*row
    ata=(a.T@a).tocsc()
    sigma=1e-6;rho=.1;relax=1.6
    x=np.zeros(n);z=np.clip(np.zeros(a.shape[0]),lo,hi);dual=np.zeros_like(z)
    def factor():
        return splu(hs+rho*ata+sigma*sparse.eye(n,format='csc'))
    solve=factor();pr=dr=float('inf');refactors=0
    for iteration in range(1,maxiter+1):
        x=solve.solve(sigma*x+a.T@(rho*z-dual))
        ax=a@x
        relaxed=relax*ax+(1-relax)*z
        z=np.clip(relaxed+dual/rho,lo,hi)
        dual+=rho*(relaxed-z)
        if iteration%25==0:
            pr=float(np.max(abs(ax-z)))
            dr=float(np.max(abs(hs@x+a.T@dual)))
            if pr<=tol and dr<=tol:
                break
            if iteration%250==0:
                ratio=math.sqrt(max(pr,1e-20)/max(dr,1e-20))
                new_rho=float(np.clip(rho*ratio,1e-5,1e4))
                if new_rho/rho>3 or new_rho/rho<1/3:
                    rho=new_rho;solve=factor();refactors+=1
    return column*x,{'iterations':iteration,'primal_scaled':pr,'dual_scaled':dr,
                     'converged':bool(pr<=tol and dr<=tol),'rho':rho,'refactors':refactors}


def build_problem(y,depth=1,continuity=1,cones=True,boundary='polynomial'):
    h,w=y.shape;gh,gw=5*(h-1)+1,5*(w-1)+1
    target=proposal(y,boundary);flat=target.ravel()
    fixed=np.zeros((gh,gw),dtype=bool);fixed[::5,::5]=True
    free=np.flatnonzero(~fixed.ravel())
    lookup=np.full(gh*gw,-1,dtype=int);lookup[free]=np.arange(len(free))
    hr=[];hc=[];hv=[];ar=[];ac=[];av=[];lower=[];upper=[]
    cert=tensor_certificate(depth);local_h=local_gradient_gram()
    witnessed=0
    def add_rows(index,coeff,lo,hi):
        coeff=np.atleast_2d(coeff)
        lo=np.broadcast_to(lo,(len(coeff),));hi=np.broadcast_to(hi,(len(coeff),))
        mask=lookup[index]>=0
        rhs=coeff@flat[index]
        for k,b in enumerate(coeff):
            v=b[mask];active=np.flatnonzero(abs(v)>1e-15)
            if not len(active):
                if rhs[k]<lo[k]-1e-9 or rhs[k]>hi[k]+1e-9:
                    raise ValueError('fixed-only constraint conflict')
                continue
            rownum=len(lower)
            ar.extend([rownum]*len(active));ac.extend(lookup[index[mask]][active]);av.extend(v[active])
            lower.append(lo[k]-rhs[k]);upper.append(hi[k]-rhs[k])
    for cy in range(h-1):
        for cx in range(w-1):
            index=((5*cy+np.arange(6)[:,None])*gw+5*cx+np.arange(6)[None,:]).ravel()
            active=lookup[index]>=0;ind=lookup[index[active]]
            block=local_h[np.ix_(active,active)]
            hr.extend(np.repeat(ind,len(ind)));hc.extend(np.tile(ind,len(ind)));hv.extend(block.ravel())
            support=y[max(0,cy-2):min(h,cy+4),max(0,cx-2):min(w,cx+4)]
            add_rows(index,cert,float(support.min()),float(support.max()))
            if cones:
                normals=support_normals(y,cy,cx)
                witnessed+=bool(normals)
                for dx,dy in normals:
                    derivative=np.array(_directional_derivative_coefficients(dx,dy)).reshape(36,36)
                    add_rows(index,cert@derivative,0,np.inf)
    H=sparse.coo_matrix((hv,(hr,hc)),shape=(len(free),len(free))).tocsc()
    A=sparse.coo_matrix((av,(ar,ac)),shape=(len(lower),len(free))).tocsr()
    lift=continuity_lift((gh,gw),continuity,free)
    H=(lift.T@H@lift).tocsc();A=(A@lift).tocsr()
    A.eliminate_zeros()
    lo=np.array(lower);hi=np.array(upper)
    nonzero=np.diff(A.indptr)>0
    if np.any(lo[~nonzero]>1e-10) or np.any(hi[~nonzero]<-1e-10):
        raise ValueError('fixed trace constraint conflict')
    return target,free,H,A[nonzero],lo[nonzero],hi[nonzero],witnessed,lift


def admit(values,depth=1,continuity=1,cones=True,tol=2e-7,maxiter=60000,boundary='polynomial'):
    source=np.asarray(values,dtype=float)
    if source.ndim!=2 or min(source.shape)<6 or not np.all(np.isfinite(source)):
        raise ValueError('finite scalar HxW samples, at least 6 per axis required')
    if depth not in (0,1,2,3) or continuity not in (0,1,2) or tol<=0 or maxiter<25:
        raise ValueError('invalid certificate depth, continuity, or solver tolerance/budget')
    start=time.perf_counter();offset=float(source.min());scale=float(np.ptp(source))
    if scale==0:
        lattice=np.full((5*(source.shape[0]-1)+1,5*(source.shape[1]-1)+1),offset)
        return lattice,{'converged':True,'iterations':0,'maximum_violation':0.,'seconds':time.perf_counter()-start}
    y=(source-offset)/scale
    target,free,H,A,lo,hi,witnessed,lift=build_problem(y,depth,continuity,cones,boundary)
    initial=max(float(np.max(np.maximum(lo,0))),float(np.max(np.maximum(-hi,0))))
    if initial<=1e-11:
        correction=np.zeros(H.shape[0]);diagnostic={'converged':True,'iterations':0}
    else:
        correction,diagnostic=qp_admm(H,A,lo,hi,tol=tol,maxiter=maxiter)
    v=A@correction
    violation=max(float(np.max(np.maximum(lo-v,0))),float(np.max(np.maximum(v-hi,0))))
    result=target.copy();result.ravel()[free]+=lift@correction
    result=result*scale+offset;result[::5,::5]=source
    diagnostic.update({'maximum_violation':violation,'initial_violation':initial,
                       'gradient_objective':float(.5*correction@H@correction),
                       'variables':H.shape[0],'constraints':len(lo),'witnessed_cells':witnessed,
                       'depth':depth,'continuity':continuity,'cones':cones,'boundary':boundary,
                       'seconds':time.perf_counter()-start})
    return result,diagnostic


def evaluate(lattice,x,y,derivative=None):
    from experiments.conv_functional_current import basis
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    h,w=(np.array(lattice.shape)-1)//5
    ix=np.minimum(np.floor(x).astype(int),w-1);iy=np.minimum(np.floor(y).astype(int),h-1)
    if np.any(x<0) or np.any(x>w) or np.any(y<0) or np.any(y>h):
        raise ValueError('queries outside source rectangle')
    u=x-ix;v=y-iy
    patch=lattice[(5*iy)[...,None,None]+np.arange(6)[:,None],
                  (5*ix)[...,None,None]+np.arange(6)[None,:]]
    if derivative=='x':
        return 5*np.einsum('...i,...j,...ij->...',basis(v,5),basis(u,4),np.diff(patch,axis=-1))
    if derivative=='y':
        return 5*np.einsum('...i,...j,...ij->...',basis(v,4),basis(u,5),np.diff(patch,axis=-2))
    return np.einsum('...i,...j,...ij->...',basis(v,5),basis(u,5),patch)


def run(out,depth=1,continuity=1,size=17,cases=None,boundary='polynomial',maxiter=60000):
    # Reuse the exact previous truth/source arrays for a matched comparison.
    previous=json.loads(Path('output/support_geometry/conv_compatible_current/conv_compatible_2d.json').read_text())
    if size!=17:
        raise ValueError('matched saved battery uses size 17')
    x=np.linspace(0,16,49);yy,xx=np.meshgrid(x,x,indexing='ij')
    mask=(xx>3.2)&(xx<12.8)&(yy>3.2)&(yy<12.8)
    rows=[];images={}
    for name,data in previous['images'].items():
        if cases and name not in cases:
            continue
        source=np.array(data['source']);truth=np.array(data['truth'])
        lattice,diag=admit(source,depth,continuity,boundary=boundary,maxiter=maxiter)
        z=evaluate(lattice,xx,yy)
        row={'case':name,**diag,'mse':float(np.mean((z[mask]-truth[mask])**2)),
             'full_mse':float(np.mean((z-truth)**2)),
             'excursion':float(max(0,z.max()-source.max(),source.min()-z.min())),
             'cardinality_error':float(np.max(abs(z[::3,::3]-source)))}
        rows.append(row);images[name]={'image':z.tolist(),'lattice':lattice.tolist()}
        Path(out).write_text(json.dumps({'rows':rows,'images':images},indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_joint_potential.json')
    p.add_argument('--depth',type=int,default=1);p.add_argument('--continuity',type=int,default=1)
    p.add_argument('--cases',help='comma-separated matched case names')
    p.add_argument('--boundary',choices=['polynomial','natural'],default='polynomial')
    p.add_argument('--maxiter',type=int,default=60000)
    a=p.parse_args();run(a.out,a.depth,a.continuity,cases=a.cases.split(',') if a.cases else None,
                        boundary=a.boundary,maxiter=a.maxiter)
