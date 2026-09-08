"""Exact finite transport of the complete six-field increment.

The finite projection-transfer matrix is symmetric and between 0 and I.
It maps this actual increment exactly; it is not a Jacobian for arbitrary
other increments. It is rebuilt from the coupled state and increment.
"""
import numpy as np
from .model import State,div,grad,solve_screened

def secant_matrix(tx,ty,hx,hy,radius):
    yx=tx+hx;yy=ty+hy;r0=np.hypot(tx,ty);r1=np.hypot(yx,yy)
    sigma0=np.minimum(1.,radius/np.maximum(r0,1e-300))
    sigma1=np.minimum(1.,radius/np.maximum(r1,1e-300))
    sigma=.5*(sigma0+sigma1)
    lo=np.minimum(r0,r1);hi=np.maximum(r0,r1)
    rho=np.where(hi<=radius,1.,0.)
    crossed=(lo<radius)&(hi>radius)
    rho=np.where(crossed,np.divide(radius-lo,hi-lo,out=np.zeros_like(lo),where=crossed),rho)
    denom=r0+r1
    sx=np.divide(tx+yx,denom,out=np.zeros_like(tx),where=denom>0)
    sy=np.divide(ty+yy,denom,out=np.zeros_like(ty),where=denom>0)
    delta=rho-sigma
    return sigma+delta*sx*sx,delta*sx*sy,sigma+delta*sy*sy

def apply_matrix(B,hx,hy):
    xx,xy,yy=B
    return xx*hx+xy*hy,xy*hx+yy*hy

def increment_transfer(m,h,Bu,Bw):
    pu=apply_matrix(Bu,h.tux,h.tuy);pw=apply_matrix(Bw,h.twx,h.twy)
    u=solve_screened(m.cu*(h.u+h.w)-m.etau*div(h.tux-2*pu[0],h.tuy-2*pu[1]),m.cu,m.etau,m.symbol)
    w=solve_screened(-m.cw*u-m.etaw*div(h.twx-2*pw[0],h.twy-2*pw[1]),m.cw,m.etaw,m.symbol)
    gu=grad(u);gw=grad(w)
    return State(u,w,gu[0]+pu[0],gu[1]+pu[1],gw[0]+pw[0],gw[1]+pw[1])

def lifted_step(m,z,h):
    Bu=secant_matrix(z.tux,z.tuy,h.tux,h.tuy,m.ru)
    Bw=secant_matrix(z.twx,z.twy,h.twx,h.twy,m.rw)
    hn=increment_transfer(m,h,Bu,Bw)
    zn=State(*(a+b for a,b in zip(z.fields(),h.fields())))
    return zn,hn
