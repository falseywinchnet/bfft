"""Actual Chambolle stopping count and primal/dual-bracketed ROF descent.

No reference solution is used by the algorithm. The dual feasible field p
produces a lower bound at every iterate; a late primal is an upper bound.
The library's stopping-energy quantity differs from the ROF primal objective.
"""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from skimage import data
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from .adapters import Chambolle


def objective(model,p):
    d=model.divergence(p);u=model.image+d;g=model.grad(u)
    tv=float(np.linalg.norm(g,axis=0).sum());d2=float(np.sum(d*d))
    primal=.5*d2+model.weight*tv
    dual=-.5*d2-float(np.sum(model.image*d))
    library_energy=(d2+model.weight*tv)/u.size
    return u,g,primal,dual,library_energy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--size',type=int,default=512);parser.add_argument('--weight',type=float,default=.1)
    parser.add_argument('--maxiter',type=int,default=10000);a=parser.parse_args()
    image=data.camera()/255.
    if a.size!=512:image=resize(image,(a.size,a.size),anti_aliasing=True)
    noisy=image+.1*np.random.default_rng(0).standard_normal(image.shape)
    model=Chambolle(noisy,a.weight);p=model.initial.copy();rows=[];default_count=None
    best_dual=-np.inf;best_primal=np.inf;E0=previous=None
    start=perf_counter()
    for iteration in range(1,a.maxiter+1):
        u,g,P,D,E=objective(model,p)
        best_dual=max(best_dual,D);best_primal=min(best_primal,P)
        rows.append(dict(iteration=iteration,primal=P,dual=D,best_dual=best_dual,
                         gap=P-best_dual,library_energy=E))
        if iteration==1:E0=E
        elif default_count is None and iteration<=200 and abs(previous-E)<2e-4*E0:
            default_count=iteration;default_output=u.copy()
        previous=E
        length=np.linalg.norm(g,axis=0)
        p=((p.reshape(g.shape)-model.tau*g)/(1+model.tau/model.weight*length)).ravel()
        if iteration in [1,2,4,8,16,32,64,100,200,500,1000,2000,5000,10000]:
            print(rows[-1],flush=True)
    trace_seconds=perf_counter()-start
    if default_count is None:
        default_count=200
        default_output=denoise_tv_chambolle(noisy,weight=a.weight,eps=0,max_num_iter=200)
    library_times=[]
    for _ in range(5):
        start=perf_counter();actual=denoise_tv_chambolle(noisy,weight=a.weight)
        library_times.append(perf_counter()-start)
    np.testing.assert_allclose(actual,default_output,atol=1e-14,rtol=1e-14)
    initial=rows[0]['primal'];denominator=initial-best_dual
    progress={}
    for fraction in [.8,.9,.95,.99,.999,.9999,.99999]:
        hit=next((r for r in rows if (initial-r['primal'])/denominator>=fraction),None)
        progress[str(fraction)]=None if hit is None else hit['iteration']
    small_gap={}
    for tol in [1e-2,1e-3,1e-4,1e-5,1e-6]:
        hit=next((r for r in rows if r['gap'] <= tol*denominator),None)
        small_gap[str(tol)]=None if hit is None else hit['iteration']
    result=dict(protocol=dict(source='skimage.data.camera',size=a.size,noise_std=.1,seed=0,weight=a.weight,
                              maxiter=a.maxiter,default_eps=2e-4,default_cap=200,clipping=False),
                default_iterations=default_count,default_seconds=library_times,
                default_primal=rows[default_count-1]['primal'],default_gap=rows[default_count-1]['gap'],
                default_progress_lower_bound=(initial-rows[default_count-1]['primal'])/denominator,
                initial_objective=initial,floor_lower=best_dual,floor_upper=best_primal,
                floor_bracket_width=best_primal-best_dual,trace_seconds=trace_seconds,
                guaranteed_progress_iterations=progress,certified_gap_iterations=small_gap,trace=rows)
    # Distances to the late numerical solution, distinct from objective progress.
    reference=u.copy();distance0=float(np.linalg.norm(noisy-reference));q=model.initial.copy()
    image_progress={};image_rows=[]
    for iteration in range(1,201):
        current=model.output(q);distance=float(np.linalg.norm(current-reference))
        fraction=1-distance/distance0
        image_rows.append(dict(iteration=iteration,relative_distance=distance/distance0,
                               rmse=distance/np.sqrt(noisy.size)))
        for target_fraction in [.8,.9,.95,.99]:
            if str(target_fraction) not in image_progress and fraction>=target_fraction:
                image_progress[str(target_fraction)]=iteration
        q=model.step(q)
    result.update(image_progress_iterations=image_progress,image_distance_trace=image_rows,
                  reference_rmse_error_bound=float(np.sqrt(2*(rows[-1]['primal']-best_dual)/noisy.size)))
    high_count=small_gap.get('1e-05')
    if high_count is not None:
        start=perf_counter()
        denoise_tv_chambolle(noisy,weight=a.weight,eps=0,max_num_iter=high_count)
        result['high_accuracy_library_seconds_single_run']=perf_counter()-start
        result['high_accuracy_library_iterations']=high_count
    Path(a.out).write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k not in ('trace','image_distance_trace')},flush=True)

if __name__=='__main__':main()
