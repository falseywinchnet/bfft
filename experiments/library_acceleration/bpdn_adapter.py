"""SPORCO BPDN sparse coding: the same nonexpansive quotient, scalar shrinkage."""
import numpy as np
from experiments.krylov_bregman.core import Linearization


def make_solver(D,S,lmbda,passes,rho=None,relaxation=1.):
    from sporco.admm.bpdn import BPDN
    options=BPDN.Options(dict(Verbose=False,MaxMainIter=passes,FastSolve=True,
        AutoRho=dict(Enabled=False),RelaxParam=relaxation,ReturnX=False,rho=rho))
    return BPDN(D,S,lmbda,options)


class ScalarRelation:
    def __init__(self,z,Q,radius):
        self.z=z;self.Q=Q;self.radius=radius
        self.s=np.sign(z)*np.maximum(np.abs(z)-radius,0.)
        self.active=np.abs(z)>radius
    def squared_remainder_norms(self,C):
        h=self.Q@C;new=self.z[:,None]+h
        remainder=np.sign(new)*np.maximum(np.abs(new)-self.radius,0.)-self.s[:,None]-self.active[:,None]*h
        return np.sum(remainder**2,axis=0)


class SparseCoding:
    def __init__(self,solver,allow_adaptive=False):
        from sporco.linalg import cho_solve_ATAI
        self.solver=solver;self.shape=solver.Y.shape
        if not 0 < solver.rlx < 2 or (solver.opt['AutoRho','Enabled'] and not allow_adaptive) or solver.opt['NonNegCoef']:
            raise ValueError('requires rho fixed within chart, relaxation in (0,2), signed coefficients')
        self.relaxation=float(solver.rlx);self.remainder_gain=self.relaxation
        if not np.isrealobj(solver.D) or not np.isrealobj(solver.S):
            raise ValueError('scalar real soft-threshold adapter required')
        if not np.all(solver.wl1 == 1) or solver.opt['ReturnX']:
            raise ValueError('initial adapter requires unit weights and ReturnX=False')
        self.radius=float(solver.lmbda/solver.rho)
        self.initial=(solver.Y+solver.U).ravel().copy();self.output_gain=1.
        self.solve=lambda rhs:cho_solve_ATAI(solver.D,solver.rho,rhs,solver.lu,solver.piv)
    def shrink(self,z):return np.sign(z)*np.maximum(np.abs(z)-self.radius,0.)
    def step(self,z):
        s=self.shrink(z)
        x=self.solve(self.solver.DTS+self.solver.rho*(2*s-z).reshape(self.shape))
        return z+self.relaxation*(x.ravel()-s)
    def linearize(self,z):
        active=np.abs(z)>self.radius
        def action(h):
            return h+self.relaxation*(self.solve((self.solver.rho*(2*active-1)*h).reshape(self.shape)).ravel()-active*h)
        return Linearization(self.step(z),action)
    def output(self,z):return self.shrink(z).reshape(self.shape)
    def valid(self,z):return np.isfinite(z).all()
    def feature_relation(self,z,Q):return ScalarRelation(z,Q,self.radius)
