"""Exact quotient of SPORCO TVL2Deconv for fixed rho, RelaxParam=1.

Let S=prox_(lambda/rho)||.||_(2,1), L=A*A+rho G*G,
B=rho G L^-1 G*, and b=G L^-1 A*f. With v=Gx+u_old,
y=S(v), u=v-S(v), the complete future-driving state is
F(v)=v-S(v)+b+B(2S(v)-v).

B is self-adjoint with 0<=B<=I. Hence F=(I+R_B R_S)/2 plus b
is nonexpansive. The frozen derivative remainder is (2B-I) R_S,
whose norm is <=||R_S||. R_S is minus the Euclidean disk-projection
remainder, exactly the curved feature formula used by the Meyer engine.
This does not assert affine closure of a fixed exterior mask.
"""
import numpy as np
from experiments.krylov_bregman.core import Linearization
from experiments.krylov_bregman.curved_meyer import DiskRelation


def make_solver(image, psf, lmbda, passes, rho=None):
    from sporco.admm.tvl2 import TVL2Deconv
    options=TVL2Deconv.Options(dict(Verbose=False,MaxMainIter=passes,
        AutoRho=dict(Enabled=False),RelaxParam=1.,FastSolve=True,
        rho=(2*lmbda+.1 if rho is None else rho)))
    return TVL2Deconv(psf,image,lmbda,options)


class TVDeconv:
    def __init__(self, solver):
        self.solver=solver
        if solver.opt['AutoRho','Enabled'] or solver.rlx != 1:
            raise ValueError('fixed rho and unit relaxation required')
        if not np.isrealobj(solver.S) or not np.all(solver.Wtv == 1):
            raise ValueError('initial adapter requires real data and unit TV weights')
        if np.any(solver.Y) or np.any(solver.U):
            raise ValueError('initial TV adapter requires zero splitting state')
        self.shape=solver.S.shape
        if len(self.shape)!=2:raise ValueError('initial adapter supports 2-D scalar data')
        self.vshape=self.shape+(2,)
        self.initial=np.zeros(self.vshape).ravel()
        self.radius=float(solver.lmbda/solver.rho)
        self.den=(solver.AHAf+solver.rho*solver.GHGf).real
        if np.any(self.den<=0):raise ValueError('positive normal operator required')
        self.gf=solver.Gf
        self.fft=lambda x:solver.fftn(x,axes=solver.axes)
        self.ifft=lambda x:solver.ifftn(x,solver.axsz,axes=solver.axes)
        # Output Lipschitz constant of X(v)=L^-1(A*f+rho G*R_S(v)).
        self.output_gain=float(np.max(solver.rho*np.sqrt(solver.GHGf)/self.den))

    def shrink(self,v):
        v=v.reshape(self.vshape);length=np.linalg.norm(v,axis=-1,keepdims=True)
        return v*np.maximum(0,1-np.divide(self.radius,length,
                    out=np.full_like(length,np.inf),where=length>0))

    def xhat(self,w,affine=True):
        rhs=self.solver.rho*np.sum(np.conj(self.gf)*self.fft(w),axis=-1)
        if affine:rhs=rhs+self.solver.AHSf
        return rhs/self.den

    def B(self,w):
        return self.ifft(self.gf*self.xhat(w,False)[...,None])

    def step(self,v):
        v=v.reshape(self.vshape);s=self.shrink(v)
        gx=self.ifft(self.gf*self.xhat(2*s-v)[...,None])
        return (v-s+gx).ravel()

    def output(self,v):
        v=v.reshape(self.vshape)
        return self.ifft(self.xhat(2*self.shrink(v)-v))

    def linearize(self,v):
        z=v.reshape(self.vshape);length=np.linalg.norm(z,axis=-1,keepdims=True)
        outside=length>self.radius
        inv=np.divide(1.,length,out=np.zeros_like(length),where=outside)
        normal=z*inv;factor=1-self.radius*inv
        def action(h):
            h=h.reshape(self.vshape)
            ds=np.where(outside,factor*h+self.radius*inv*normal*np.sum(normal*h,axis=-1,keepdims=True),0.)
            return (h-ds+self.B(2*ds-h)).ravel()
        return Linearization(self.step(v),action)

    def valid(self,v):return np.isfinite(v).all()

    def feature_relation(self,z,Q):
        q=Q.reshape(-1,2,Q.shape[1]);v=z.reshape(-1,2)
        return DiskRelation.acquire(v[:,0],v[:,1],q[:,0,:],q[:,1,:],self.radius)
