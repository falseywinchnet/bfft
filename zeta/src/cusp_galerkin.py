"""Cusp-fitted continuous piecewise-linear Galerkin operator.

Each tensor cell is clipped along every u+v=log(n) interface before high-order
triangle quadrature. This removes the diagonal cusp from every integration
region instead of hoping a tensor rule resolves it.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import eigh
from numpy.polynomial.legendre import leggauss
from suzuki_kernel import k_omega


def _clip(poly, c, keep_below):
    if not poly:return []
    out=[]
    def inside(p): return (p[0]+p[1] <= c+1e-15) if keep_below else (p[0]+p[1] >= c-1e-15)
    for p,q in zip(poly,poly[1:]+poly[:1]):
        ip,iq=inside(p),inside(q)
        if ip:out.append(p)
        if ip != iq:
            fp=p[0]+p[1]-c;fq=q[0]+q[1]-c
            t=fp/(fp-fq);out.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
    return out


def _regions(x0,x1,y0,y1,cusps):
    square=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    bounds=[-float("inf")]+[c for c in cusps if x0+y0<c<x1+y1]+[float("inf")]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        p=square
        if math.isfinite(lo):p=_clip(p,lo,False)
        if math.isfinite(hi):p=_clip(p,hi,True)
        if len(p)>=3:yield p


def _integrate_polygon(poly,x0,h,y0,order,omega):
    z,w=leggauss(order); z=(z+1)/2;w=w/2
    total=np.zeros((2,2))
    p0=np.asarray(poly[0],float)
    for j in range(1,len(poly)-1):
        p1,p2=np.asarray(poly[j],float),np.asarray(poly[j+1],float)
        cross=abs(np.cross(p1-p0,p2-p0))
        rr,ss=np.meshgrid(z,z,indexing="ij");ww=np.outer(w,w)
        u=p0[0]+rr*(p1[0]-p0[0])+(1-rr)*ss*(p2[0]-p0[0])
        v=p0[1]+rr*(p1[1]-p0[1])+(1-rr)*ss*(p2[1]-p0[1])
        weight=ww*cross*(1-rr)
        kval=k_omega(omega,(u+v).ravel()).reshape(u.shape)
        xi=(u-x0)/h;eta=(v-y0)/h
        bx=(1-xi,xi);by=(1-eta,eta)
        for a in range(2):
            for b in range(2):total[a,b]+=np.sum(weight*kval*bx[a]*by[b])
    return total


def cusp_fitted_matrices(omega:float,a:float,n_cells:int,order:int=5):
    A0=math.log(a);edges=np.linspace(-A0,A0,n_cells+1);h=edges[1]-edges[0]
    cusps=[math.log(j) for j in range(1,int(math.floor(a*a))+1)]
    K=np.zeros((n_cells+1,n_cells+1));M=np.zeros_like(K)
    for i in range(n_cells):
        M[i:i+2,i:i+2]+=h*np.array([[2,1],[1,2]])/6
    # Uniform translated hats plus a sum kernel make the cell-pair block depend
    # only on i+j. Resolve each of the 2N-1 physical blocks once.
    blocks=[]
    for s in range(2*n_cells-1):
        i=min(s,n_cells-1);j=s-i
        local=np.zeros((2,2))
        for poly in _regions(edges[i],edges[i+1],edges[j],edges[j+1],cusps):
            local+=_integrate_polygon(poly,edges[i],h,edges[j],order,omega)
        blocks.append(local)
    for i in range(n_cells):
        for j in range(n_cells):
            K[i:i+2,j:j+2]+=blocks[i+j]
    return edges,(K+K.T)/2,M


def cusp_fitted_spectrum(omega,a,n_cells,order=5,vectors=False):
    edges,K,M=cusp_fitted_matrices(omega,a,n_cells,order)
    vals,vecs=eigh(K,M)
    # vecs are M-orthonormal. Standard-coordinate representatives use a
    # Cholesky factor and are only needed for subspace comparisons.
    plus=1+vals;minus=1-vals
    log_m=float(np.log(plus).sum()-np.log(minus).sum())
    result={"N_cells":n_cells,"dofs":n_cells+1,"lambda_min":float(vals[0]),
            "lambda_max":float(vals[-1]),"spectral_radius":float(np.max(abs(vals))),
            "log_m":log_m,"m":float(np.exp(log_m))}
    return (result,vals,vecs,M) if vectors else result
