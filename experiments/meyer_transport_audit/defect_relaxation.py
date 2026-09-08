"""Experimental schedules for the existing coupled relaxation operations.

No line search, fitted factors, extra iterative solver, or dense operator.
The fused schedule algebraically couples the two existing Fourier responses.
The refresh schedule performs one additional pointwise memory projection.
"""
from pathlib import Path
import sys
import argparse
import json
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, State, grad, div, project_disk, solve_screened
from experiments.meyer_transport_audit.certificate import certificate
from experiments.benchmark_meyer_flow_jump import benchmark_scene


class DefectRelaxation:
    def __init__(self, model, schedule):
        self.m = model
        self.schedule = schedule
        self.k = 0
        self.previous_memory_change = None
        # Explicit feasible memory avoids reconstructing t-Dx every pass.
        # This stateful operator belongs to one sequential trajectory.
        self.memory = None
        self.hu = model.cu + model.etau * model.symbol
        self.hw = model.cw + model.etaw * model.symbol
        self.det = self.hu * self.hw + model.cu * model.cw

    def admit_defect(self, previous, extra):
        return previous[0]*extra[0] + previous[1]*extra[1] >= 0

    def step(self, z):
        self.k += 1
        m = self.m
        pu = project_disk(z.tux, z.tuy, m.ru)
        pw = project_disk(z.twx, z.twy, m.rw)
        fu = -m.etau * div(z.tux-2*pu[0], z.tuy-2*pu[1])
        fw = -m.etaw * div(z.twx-2*pw[0], z.twy-2*pw[1])
        if self.schedule.startswith('fused'):
            # Hu*u+ - cu*w+ = cu*u + fu
            # cw*u+ + Hw*w+ = cw*f + fw
            a = np.fft.fft2(m.cu*z.u + fu)
            b = np.fft.fft2(m.cw*m.image + fw)
            u = np.fft.ifft2((self.hw*a + m.cu*b)/self.det).real
            w = np.fft.ifft2((self.hu*b - m.cw*a)/self.det).real
        elif self.schedule == 'reverse' or (self.schedule == 'alternating' and self.k % 2):
            w = solve_screened(m.cw*(m.image-z.u)+fw, m.cw, m.etaw, m.symbol)
            u = solve_screened(m.cu*(z.u+w)+fu, m.cu, m.etau, m.symbol)
        else:
            u = solve_screened(m.cu*(z.u+z.w)+fu, m.cu, m.etau, m.symbol)
            w = solve_screened(m.cw*(m.image-u)+fw, m.cw, m.etaw, m.symbol)
        gu = grad(u)
        gw = grad(w)
        if self.schedule.endswith('transport') or self.schedule.endswith('reobserve'):
            oldgu = grad(z.u)
            oldgw = grad(z.w)
            baseu = (z.tux, z.tuy) if self.schedule.endswith('reobserve') else pu
            basew = (z.twx, z.twy) if self.schedule.endswith('reobserve') else pw
            pu = project_disk(baseu[0]+gu[0]-oldgu[0], baseu[1]+gu[1]-oldgu[1], m.ru)
            pw = project_disk(basew[0]+gw[0]-oldgw[0], basew[1]+gw[1]-oldgw[1], m.rw)
        if self.schedule.endswith('refresh'):
            rg, sg = (grad(z.u),grad(z.w)) if self.schedule == 'stale_refresh' else (gu,gw)
            qu = project_disk(rg[0]+pu[0], rg[1]+pu[1], m.ru)
            qw = project_disk(sg[0]+pw[0], sg[1]+pw[1], m.rw)
            if self.schedule in ['aligned_refresh', 'remembered_refresh']:
                if self.memory is None:
                    oldgu, oldgw = grad(z.u), grad(z.w)
                    self.memory = ((z.tux-oldgu[0], z.tuy-oldgu[1]),
                                   (z.twx-oldgw[0], z.twy-oldgw[1]))
                bu, bw = self.memory
                current = ((pu[0]-bu[0], pu[1]-bu[1]), (pw[0]-bw[0], pw[1]-bw[1]))
                witness = current if self.schedule == 'aligned_refresh' else self.previous_memory_change
                if witness is None:
                    qu, qw = pu, pw
                else:
                    selected = []
                    for p, q, e in zip([pu,pw], [qu,qw], witness):
                        mask = self.admit_defect(e,(q[0]-p[0],q[1]-p[1]))
                        selected.append(tuple(np.where(mask, qi, pi) for qi,pi in zip(q,p)))
                    qu, qw = selected
                if self.schedule == 'remembered_refresh':
                    self.previous_memory_change = ((qu[0]-bu[0],qu[1]-bu[1]), (qw[0]-bw[0],qw[1]-bw[1]))
                self.memory = (qu,qw)
            pu, pw = qu, qw
        return State(u, w, gu[0]+pu[0], gu[1]+pu[1], gw[0]+pw[0], gw[1]+pw[1])


def scenes(n, seed=983):
    y, x = np.mgrid[:n, :n]
    rng = np.random.default_rng(seed)
    return {'edge1d':50.+150.*(np.arange(n)[None,:]>n//2),
            'ramp':255.*x/n, 'crossing':benchmark_scene(n),
            'noise':rng.uniform(0,255,(n,n)),
            'carrier':100.+2*np.sin(2*np.pi*(x+2*y)/n)}


def run(n, steps, out, schedules, parameters, seed):
    result = {'size':n, 'steps':steps, 'seed':seed, 'cases':{},
              'note':'NumPy exploratory schedules. Per-step time excludes common checkpoint diagnostics; no convergence theorem.'}
    for lam, mu in parameters:
        for name, f in scenes(n, seed).items():
            m = ReducedMeyerMap(f, lam, mu)
            case = {}
            for schedule in schedules:
                op = DefectRelaxation(m, schedule)
                z = m.initial()
                elapsed = 0.
                rows = []
                for k in range(2, steps+1):
                    start = time.perf_counter()
                    z = op.step(z)
                    elapsed += time.perf_counter()-start
                    if k in [16,32,64,128,256,512,1024,2048] or k == steps:
                        rows.append({'pass':k, 'seconds':elapsed, **certificate(m,z)})
                case[schedule] = rows
            key = f'{name}_lambda{lam}_mu{mu}'
            result['cases'][key] = case
            Path(out).write_text(json.dumps(result, indent=2))
            print(key, {v:round(rows[-1]['gap_per_pixel'],7) for v,rows in case.items()}, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--seed', type=int, default=983)
    p.add_argument('--schedules', default='ordinary,reverse,alternating,fused,refresh,fused_refresh')
    p.add_argument('--single-parameter', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.size, a.steps, a.out, a.schedules.split(','),
        [(.05,40)] if a.single_parameter else [(.02,20),(.05,40),(.1,80)], a.seed)
