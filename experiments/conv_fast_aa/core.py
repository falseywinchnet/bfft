"""Direct contractions of existing CONV source jets; experimental, not CONV*."""
import math
import numpy as np
from experiments.convstar import raw_current_jet_bank, ordered_sign_ledger, project_signed_fibres


def shifted(a, dx, dy):
    y = np.clip(np.arange(a.shape[0]) + dy, 0, a.shape[0] - 1)
    x = np.clip(np.arange(a.shape[1]) + dx, 0, a.shape[1] - 1)
    return a[y[:, None], x[None, :]]


def jet(a):
    """CONV fourth-order interior first jet, clamp extension at image boundary."""
    gx = (8*(shifted(a,1,0)-shifted(a,-1,0))-(shifted(a,2,0)-shifted(a,-2,0)))/12
    gy = (8*(shifted(a,0,1)-shifted(a,0,-1))-(shifted(a,0,2)-shifted(a,0,-2)))/12
    return gx, gy


def tangent_integral(a, radius=1., profile='box', tensor=False, luma_override=None):
    """Exact line measure of Q1 reconstruction, using a frozen nodal tangent.

    radius <= 1 in L-infinity coordinates. Beta profile is normalized (1-s²)².
    It inherits the derivative of CONV's quintic smoothstep as its measure.
    No threshold, classifier, search, or source-dependent radius.
    """
    if not 0 <= radius <= 1:
        raise ValueError('radius must be in [0,1]')
    a = np.asarray(a, dtype=float)
    if a.ndim == 3:
        luma = a @ np.array([.2126,.7152,.0722])
    else:
        luma = a
    if luma_override is not None: luma=np.asarray(luma_override,dtype=float)
    gx, gy = jet(luma)
    den = np.maximum(np.abs(gx), np.abs(gy))
    variation=np.maximum.reduce([np.abs(shifted(luma,dx,dy)-luma) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))])
    # A float32 conditioning rule, not a perceptual edge threshold. Exact
    # stationary points must not acquire arbitrary directions after quantization.
    den=np.where(den>8*np.finfo(np.float32).eps*variation,den,0)
    tx = np.divide(-gy,den,out=np.zeros_like(gx),where=den>0)
    ty = np.divide(gx,den,out=np.zeros_like(gy),where=den>0)
    coherence=np.ones_like(gx)
    if tensor:
        rgb=a if a.ndim==3 else a[...,None]
        rx,ry=jet(rgb)
        xx,xy,yy=np.sum(rx*rx,axis=-1),np.sum(rx*ry,axis=-1),np.sum(ry*ry,axis=-1)
        gap=np.hypot(xx-yy,2*xy);trace=xx+yy
        coherence=np.divide(gap,trace,out=np.zeros_like(gap),where=trace>0)
        variation=np.maximum.reduce([np.max(np.abs(shifted(rgb,dx,dy)-rgb),axis=-1) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))])
        coherence=np.where(trace>(8*np.finfo(np.float32).eps*variation)**2,coherence,0)
        cosine=np.divide(xx-yy,gap,out=np.zeros_like(gap),where=gap>0)
        tx=np.sqrt(np.maximum(0,(1-cosine)/2));ty=-np.where(xy>=0,1,-1)*np.sqrt(np.maximum(0,(1+cosine)/2))
        den=np.maximum(np.abs(tx),np.abs(ty));tx/=den;ty/=den
    ax, ay = np.abs(tx)*radius, np.abs(ty)*radius
    m1,m2 = (.5,1/3) if profile == 'box' else (5/16,1/7)
    wx,wy,wd = ax*m1/2-ax*ay*m2/2, ay*m1/2-ax*ay*m2/2, ax*ay*m2/2
    wx*=coherence;wy*=coherence;wd*=coherence
    same = tx*ty >= 0
    if a.ndim == 3:
        wx,wy,wd,same = (v[...,None] for v in (wx,wy,wd,same))
    diag = np.where(same,shifted(a,1,1)+shifted(a,-1,-1),
                    shifted(a,1,-1)+shifted(a,-1,1))
    return a + wx*(shifted(a,1,0)+shifted(a,-1,0)-2*a) + wy*(shifted(a,0,1)+shifted(a,0,-1)-2*a) + wd*(diag-2*a)


def conv_axis_average(a, axis):
    """Exact half-cell quadrature of original admitted 1-D CONV, no resampling."""
    src = np.moveaxis(a,axis,0)
    shape=src.shape
    lines=src.reshape(shape[0],-1)
    raw,delta=raw_current_jet_bank(lines)
    current=project_signed_fibres(raw,ordered_sign_ledger(raw,delta),delta)
    controls=np.concatenate((lines[:-1,None,:],lines[:-1,None,:]+np.cumsum(current,axis=1)),axis=1)
    nodes,weights=np.polynomial.legendre.leggauss(3)
    halves=[]
    for lo in (0.,.5):
        t=lo+(nodes+1)*.25
        basis=np.array([[math.comb(5,k)*u**k*(1-u)**(5-k) for k in range(6)] for u in t])
        halves.append(np.einsum('q,qk,ikc->ic',weights*.25,basis,controls))
    out=lines.copy()
    out[1:-1]=halves[1][:-1]+halves[0][1:]
    return np.moveaxis(out.reshape(shape),0,axis)


def conv_box(a):
    """Symmetric mean of the two full sequential CONV box-average orders."""
    return .5*(conv_axis_average(conv_axis_average(a,0),1)+conv_axis_average(conv_axis_average(a,1),0))


def halfplane_coverage(d, nx, ny):
    """Exact unit-square coverage of d + nx*x + ny*y >= 0.

    d is the signed centre value. This requires geometric information, and
    is an oracle here, never supplied to a postprocessing candidate.
    """
    a,b=sorted((abs(nx),abs(ny)),reverse=True)
    d=np.asarray(d,dtype=float)
    if a == 0:
        return (d>=0).astype(float)
    if b < 1e-12*a:
        return np.clip(.5+d/a,0,1)
    z=np.abs(d)
    tail=np.maximum((a+b)/2-z,0)**2/(2*a*b)
    tail=np.where(z <= (a-b)/2,.5-z/a,tail)
    return np.where(d>=0,1-tail,tail)
