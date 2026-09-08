"""Primal/dual bounds for discrete min TV(u)+lambda/2||f-u-v||², v in G_mu.

D is periodic forward difference; div=-D*. Projections give feasible fluxes
at ANY state, including an extrapolated state. The emitted v=f-u-w need not
be feasible; the bound uses the separately recovered feasible flux texture.
"""
import numpy as np
from .model import grad, div, project_disk

def tv(x):
    gx,gy=grad(x)
    return float(np.sum(np.hypot(gx,gy)))

def certificate(model,z):
    ux,uy=project_disk(z.tux,z.tuy,model.ru)
    wx,wy=project_disk(z.twx,z.twy,model.rw)
    q=-model.etau*div(ux,uy)  # q=D*p, |p|<=1: feasible TV dual.
    v=-(model.etaw/model.cw)*div(wx,wy) # v=D*g, |g|<=mu.
    residual=model.image-z.u-v
    primal=tv(z.u)+.5*model.lam*float(np.sum(residual**2))
    dual=float(np.sum(model.image*q))-.5/model.lam*float(np.sum(q*q))-model.mu*tv(q)
    return {'primal_upper':primal,'dual_lower':dual,'gap':primal-dual,
            'gap_per_pixel':(primal-dual)/model.count,
            'relative_gap':(primal-dual)/max(abs(primal),abs(dual),1e-30),
            'emitted_vs_feasible_texture_rms':float(np.sqrt(np.mean((model.image-z.u-z.w-v)**2)))}
