"""Research oracle: admit the derivative function, not its coefficient signs.

Uniform-sign cells use the convex cone of nonnegative quartics. Mixed-sign
cells retain CONV's original admission and moving-extremum semantics.
The continuous cone is solved by constraint exchange with exact polynomial
stationary-point separation (floating point, not a formal interval proof).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, lsq_linear
from experiments.convstar import (
    raw_current_jet_bank, ordered_sign_ledger, project_signed_fibres,
    synthesize_polyphase,
)


def basis(u, degree=4):
    u = np.asarray(u)
    return np.stack([math.comb(degree, j) * u**j * (1-u)**(degree-j)
                     for j in range(degree+1)], axis=-1)


POWER = np.zeros((5, 5))
for j in range(5):
    for k in range(5-j):
        POWER[j+k, j] = math.comb(4, j)*math.comb(4-j, k)*(-1)**k
SLOPE_GRAM = np.array([[25*math.comb(4, i)*math.comb(4, j)
                        / (9*math.comb(8, i+j)) for j in range(5)]
                      for i in range(5)])


def minimum(c):
    """Minimum of sum c_j B_j^4, including every real stationary point."""
    power = POWER @ c
    roots = np.polynomial.polynomial.polyroots(np.arange(1, 5)*power[1:])
    points = np.array([0., 1.] + [float(r.real) for r in roots
                      if abs(r.imag) < 1e-9 and 0 < r.real < 1])
    values = basis(points) @ c
    k = int(np.argmin(values))
    return float(values[k]), float(points[k])


def admit_cell(a, mass, sign, metric, functional):
    """Numerical unique projection, normalized to avoid amplitude tolerances."""
    # Fix the numerical orientation as well as the mathematical symmetry.
    # Near a double contact, mirrored optimizer paths can stop at slightly
    # different root positions. Reverse the metric with the coordinates.
    if tuple(sign*np.asarray(a)) > tuple(sign*np.asarray(a)[::-1]):
        return admit_cell(np.asarray(a)[::-1],mass,sign,metric[::-1,::-1],functional)[::-1]
    a=np.ascontiguousarray(a)
    metric=np.ascontiguousarray(metric)
    scale = max(float(np.max(np.abs(a))), abs(float(mass)))
    if scale == 0:
        return a.copy()
    a = sign*a/scale
    mass = sign*mass/scale
    if mass < -1e-12:
        raise ValueError('uniform ledger incompatible with cell mass')
    if mass <= 0:
        return np.zeros(5)
    if (minimum(a)[0] if functional else np.min(a)) >= -1e-13:
        return sign*scale*a
    rows = basis(np.linspace(0, 1, 9)) if functional else np.eye(5)
    # Optimize unit-mass coefficients. A global spline can propose an O(1)
    # oscillation in a cell with a 1e-20 secant; raw-amplitude scaling alone
    # makes its equality numerically indistinguishable from zero.
    c = np.full(5, .2)
    wa = metric@a
    for _ in range(80):
        result = minimize(lambda x: .5*mass*x@metric@x-x@wa, c,
                          jac=lambda x: mass*metric@x-wa, method='SLSQP',
                          constraints=[{'type':'eq', 'fun':lambda x: x.sum()-1,
                                        'jac':lambda x: np.ones(5)},
                                       {'type':'ineq', 'fun':lambda x: rows@x,
                                        'jac':lambda x: rows}],
                          options={'ftol':1e-13, 'maxiter':250})
        c = result.x
        # SLSQP can report a line-search failure at a numerically solved QP.
        # Accept that status only with an independent primal/KKT check.
        if not result.success:
            active = rows[(rows@c)<1e-7]
            columns = np.column_stack((active.T, np.ones(5)))
            gradient=mass*metric@c-wa
            dual = lsq_linear(columns, gradient,
                             bounds=(np.r_[np.zeros(len(active)),-np.inf],
                                     np.full(len(active)+1,np.inf)),tol=1e-13)
            residual = np.max(abs(columns@dual.x-gradient))
            if residual>1e-7 or np.min(rows@c)<-1e-8 or abs(c.sum()-1)>1e-10:
                raise RuntimeError(f'{result.message}; KKT residual {residual}; primal {np.min(rows@c)}')
        low, where = minimum(c) if functional else (float(np.min(c)), 0.)
        if low >= -2e-8:
            # Remove residual feasibility error by a mass-preserving mix with
            # the strictly positive constant derivative; perturbation is tiny.
            if low < 0:
                t = (-low+2e-14)/(.2-low)
                c = (1-t)*c+t*.2
            return sign*scale*mass*c
        rows = np.vstack((rows, basis(where)))
    raise RuntimeError('continuous-cone exchange did not converge')


def currents(source, functional=True, slope_metric=True):
    raw, delta = raw_current_jet_bank(np.asarray(source)[:, None])
    signs = ordered_sign_ledger(raw, delta)
    paper = project_signed_fibres(raw, signs, delta)
    result = paper.copy()
    metric = SLOPE_GRAM if slope_metric else np.eye(5)
    for i in range(len(delta)):
        ledger = signs[i, :, 0]
        if np.all(ledger == ledger[0]) and ledger[0] != 0:
            result[i, :, 0] = admit_cell(raw[i, :, 0], delta[i, 0],
                                       ledger[0], metric, functional)
    return result, paper, raw, signs


def lanczos(source, x):
    nodes = np.arange(len(source))
    d = np.asarray(x)[:, None]-nodes[None, :]
    w = np.sinc(d)*np.sinc(d/3)*(np.abs(d)<3)
    return w@source / w.sum(axis=1)


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scale, n = 64, 25
    nodes = np.arange(n, dtype=float)
    x = np.linspace(0, n-1, (n-1)*scale+1)
    u = (np.arange(scale)+.5)/scale
    mids = (np.arange(n-1)[:, None]+u).reshape(-1)
    records, plot_data = [], {}
    cases = []
    for phase in (.1, .35, .5, .8):
        centre = 12+phase
        for width in (.35, .7, 1.5, 3.):
            cases.append(('sigmoid', f'w={width},p={phase}',
                          lambda z,c=centre,w=width: .5*(1+np.tanh((z-c)/w)),
                          lambda z,c=centre,w=width: .5/w/np.cosh((z-c)/w)**2))
        for eps in (0., .001, .02):
            cases.append(('shoulder', f'e={eps},p={phase}',
                          lambda z,c=centre,e=eps: (z-c)**3/3+e*(z-c),
                          lambda z,c=centre,e=eps: (z-c)**2+e))
        for freq in (.08, .22, .4):
            cases.append(('sine', f'f={freq},p={phase}',
                          lambda z,f=freq,p=phase: np.sin(2*np.pi*(f*z+p)),
                          lambda z,f=freq,p=phase: 2*np.pi*f*np.cos(2*np.pi*(f*z+p))))
        cases.append(('step', f'p={phase}', lambda z,c=centre: (z>=c).astype(float), None))
    for family, label, f, df in cases:
        source, truth = f(nodes), f(x)
        fc, paper, raw, signs = currents(source)
        coeff = currents(source, functional=False)[0]
        fi = currents(source, slope_metric=False)[0]
        banks = {'CONV':paper, 'coefficient slope':coeff,
                 'functional Euclidean':fi, 'functional slope':fc, 'raw':raw}
        estimates = {name:synthesize_polyphase(source[:, None], bank, scale)[:,0]
                     for name,bank in banks.items()}
        estimates['Lanczos-3'] = lanczos(source, x)
        near = (x>10)&(x<15) if family!='sine' else (x>3)&(x<21)
        slope_near = (mids>10)&(mids<15) if family!='sine' else (mids>3)&(mids<21)
        amplitude = max(float(np.ptp(truth[near])), 1e-15)
        for name, estimate in estimates.items():
            if name in banks:
                slope = (5*np.einsum('ij,pj->ip', banks[name][:,:,0], basis(u))).reshape(-1)
                jumps = 5*(banks[name][:-1,4,0]-banks[name][1:,0,0])
            else:
                h = 1e-5
                slope = (lanczos(source,mids+h)-lanczos(source,mids-h))/(2*h)
                jumps = np.zeros(n-2)
            slope_sign = np.sign(slope[np.abs(slope)>1e-8*max(1,np.max(abs(slope)))])
            coarse_sign = np.sign(np.diff(source)[np.diff(source)!=0])
            row = {'family':family,'case':label,'method':name,
                   'value_nmse':float(np.mean((estimate[near]-truth[near])**2)/amplitude**2),
                   'slope_nmse':None if df is None else float(np.mean((slope[slope_near]-df(mids)[slope_near])**2)/amplitude**2),
                   'extra_sign_changes':int(np.sum(slope_sign[1:]!=slope_sign[:-1])-np.sum(coarse_sign[1:]!=coarse_sign[:-1])),
                   'max_interior_derivative_jump':float(np.max(abs(jumps[2:-2]))),
                   'range_excess':float(max(0,np.max(estimate)-max(source),min(source)-np.min(estimate)))}
            records.append(row)
        if (family=='shoulder' and label=='e=0.001,p=0.5') or (family=='sigmoid' and label=='w=0.7,p=0.5'):
            plot_data[family]=(truth,estimates)
    summary = {}
    for family in ('shoulder','sigmoid','sine','step'):
        summary[family] = {}
        for method in estimates:
            selected = [r for r in records if r['family']==family and r['method']==method]
            summary[family][method] = {k:None if selected[0][k] is None else float(np.mean([r[k] for r in selected]))
                                      for k in ('value_nmse','slope_nmse')}
    convergence=[]
    for h in (1.,.5,.25,.125,.0625):
        source=((nodes-12.5)*h)**3/3
        truth=((x-12.5)*h)**3/3
        fc,paper,_,_=currents(source)
        central=(x>=12)&(x<=13)
        convergence.append({'h':h,**{
            name:float(np.max(abs(synthesize_polyphase(source[:,None],c,scale)[:,0][central]-truth[central])))
            for name,c in [('CONV',paper),('functional slope',fc)]}})
    result={'cases':len(cases),'source_size':n,'refinement':scale,
            'stationary_inflection_convergence':convergence,
            'summary':summary,'records':records,
            'plot_x':x.tolist(),
            'plot_data':{k:[t.tolist(),{m:e.tolist() for m,e in es.items()}]
                         for k,(t,es) in plot_data.items()}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    data=json.loads((out/'results.json').read_text())
    x=np.array(data['plot_x'])
    plot_data={k:(np.array(t),{m:np.array(e) for m,e in es.items()})
               for k,(t,es) in data['plot_data'].items()}
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for row,family in enumerate(('shoulder','sigmoid')):
        truth, estimates = plot_data[family]
        for name in ('CONV','functional slope','Lanczos-3'):
            axes[row,0].plot(x,estimates[name],label=name)
            axes[row,1].plot(x,estimates[name]-truth,label=name)
        axes[row,0].plot(x,truth,'k--',label='analytic truth')
        for col in range(2):
            axes[row,col].set_xlim(11.5,13.5)
            axes[row,col].set_title(f'{family}: '+('profile' if col==0 else 'value error'))
            axes[row,col].grid(alpha=.2)
        mask=(x>=11.5)&(x<=13.5)
        vals=np.concatenate([truth[mask]]+[e[mask] for e in estimates.values()])
        axes[row,0].set_ylim(vals.min()-.05*np.ptp(vals),vals.max()+.05*np.ptp(vals))
        errs=np.concatenate([(estimates[name]-truth)[mask] for name in ('CONV','functional slope','Lanczos-3')])
        axes[row,1].set_ylim(errs.min()-.05*np.ptp(errs),errs.max()+.05*np.ptp(errs))
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(out/'profiles.png',dpi=160)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='/tmp/conv_functional_current')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    (plot if args.plot_only else run)(args.out)
