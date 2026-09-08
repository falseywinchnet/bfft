"""Matched-state convergence audit; experimental nonlinear-history acceleration.

All methods apply the existing ReducedMeyerMap. No image-dependent rules.
Times include all update work, exclude diagnostic measurements and setup.
The long-run reference is explicitly a trajectory checkpoint, not a proof.
"""
from pathlib import Path
import sys, json, time, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, State
from experiments.benchmark_meyer_flow_jump import benchmark_scene


def norm(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def finite_jump(model, z, depth=2, horizon=10):
    """Orthonormal Arnoldi; no hidden diagnostic map calls in timing."""
    r = model.pack(model.residual(z))
    beta = np.linalg.norm(r)
    if beta < 1e-25:
        return z, 1
    q = [r / beta]
    h = np.zeros((depth + 1, depth))
    for j in range(depth):
        v = model.pack(model.tangent(z, model.unpack(q[j])))
        for _ in range(2):
            for i in range(j + 1):
                c = np.dot(q[i], v)
                h[i,j] += c
                v -= c * q[i]
        h[j+1,j] = np.linalg.norm(v)
        if h[j+1,j] < 1e-13 or j == depth-1:
            break
        q.append(v / h[j+1,j])
    dim = j + 1
    p = np.zeros(dim); p[0] = 1
    total = np.zeros(dim)
    for _ in range(horizon):
        total += p
        p = h[:dim,:dim] @ p
    delta = beta * sum(total[i] * q[i] for i in range(dim))
    return model.unpack(model.pack(z) + delta), dim + 1


def history_step(model, z, history, memory=5, ridge=1e-8):
    """Type-II Anderson on the complete six-field state, with ridge LS.

    Uses one real nonlinear evaluation per update; safeguards are deliberately
    not claimed. Singular values are handled through regularized least squares.
    """
    x = model.pack(z)
    g = model.pack(model.step(z))
    r = g-x
    history.append((x, r))
    del history[:-memory-1]
    if len(history) < 2:
        return model.unpack(g)
    dx = np.stack([b[0]-a[0] for a,b in zip(history[:-1],history[1:])],axis=1)
    dr = np.stack([b[1]-a[1] for a,b in zip(history[:-1],history[1:])],axis=1)
    gram = dr.T @ dr
    scale = max(float(np.trace(gram)), 1e-30)
    coef = np.linalg.solve(gram + ridge*scale*np.eye(gram.shape[0]), dr.T @ r)
    return model.unpack(g - (dx+dr) @ coef)


def measure(model, z, refs):
    v = model.image-z.u-z.w
    return {'residual_rms': model.residual_norm(z),
            **{'texture_error_'+str(k): norm(v-ref)/max(norm(ref),1e-30)
               for k,ref in refs.items()}}


def run(image, checkpoints, reference_steps, lam=.05, mu=40):
    model=ReducedMeyerMap(image,lam,mu)
    z=model.initial(); refs={}; reference_residuals={}
    for k in range(1,reference_steps+1):
        if k>1: z=model.step(z)
        if k in [64,reference_steps//2,reference_steps]:
            refs[k]=image-z.u-z.w
            reference_residuals[k]=model.residual_norm(z)
    result={'reference_residuals':reference_residuals,
            'reference_drift':norm(refs[reference_steps]-refs[reference_steps//2])/max(norm(refs[reference_steps]),1e-30),
            'methods':{}}
    for name in ['ordinary','flow2','flow4','anderson3','anderson5','anderson8','relax15']:
        z=model.initial(); calls=1; elapsed=0.; hist=[]; rows=[]
        for step in range(2,max(checkpoints)+1):
            start=time.perf_counter()
            if name.startswith('flow') and step>4:
                depth=int(name[-1]); z,cost=finite_jump(model,z,depth,10)
                z=model.step(model.step(z)); calls+=cost+2
            elif name.startswith('anderson') and step>4:
                z=history_step(model,z,hist,int(name[-1])); calls+=1
            elif name=='relax15' and step>4:
                z=model.unpack(model.pack(z)+1.5*model.pack(model.residual(z))); calls+=1
            else:
                z=model.step(z); calls+=1
            elapsed+=time.perf_counter()-start
            if step in checkpoints or (name.startswith('flow') and step in [7,9]):
                rows.append({'iteration':step,'operator_calls':calls,'ms':elapsed*1000,**measure(model,z,refs)})
            if not np.isfinite(model.pack(z)).all(): break
        result['methods'][name]=rows
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--size',type=int,default=64)
    p.add_argument('--reference',type=int,default=2048); p.add_argument('--out',required=True)
    a=p.parse_args(); n=a.size
    y,x=np.mgrid[:n,:n]/n
    edge=92+10*x+55*((x-.3)**2+(y-.4)**2<.17**2)+40*((x>.6)&(x<.85)&(y>.55)&(y<.8))
    scenes={'edge':edge,'carrier':100+25*np.sin(2*np.pi*(9*x+5*y)),
            'crossing':benchmark_scene(n)}
    archive=np.load(ROOT/'paper/fast_meyer_bregman/results/arrays.npz')
    from scipy.ndimage import zoom
    for key in archive.files:
        if key.endswith('_source') and ('barbara' in key.lower() or 'camera' in key.lower()):
            f=archive[key]; scenes[key[:-7]]=zoom(f,(n/f.shape[0],n/f.shape[1]),order=1)
    report={'size':n,'reference_steps':a.reference,'methods_note':'NumPy prototypes with identical map backend, one run per method, no native speed claim','scenes':{}}
    for name,image in scenes.items():
        report['scenes'][name]=run(image,[8,16,24,32,48,64,96,128],a.reference)
        Path(a.out).write_text(json.dumps(report,indent=2))
        print(name,json.dumps({m:rows[-1] for m,rows in report['scenes'][name]['methods'].items()}),flush=True)

if __name__=='__main__': main()
