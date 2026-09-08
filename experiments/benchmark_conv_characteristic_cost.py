"""Matched Python/NumPy construction and sampling cost; no solvers."""
import json
import platform
import time
from pathlib import Path
import numpy as np
from experiments.conv_joint_characteristic_modes import compile_mode
from experiments.conv_joint_ridge_modes import evaluate
from experiments.probe_conv_compatible_2d import resize


def timed(fn, repeats):
    fn()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter(); result=fn(); samples.append(time.perf_counter()-start)
    return result, {'median_ms':1000*float(np.median(samples)), 'min_ms':1000*min(samples),
                    'max_ms':1000*max(samples), 'repeats':repeats}


def run(out):
    rows=[]
    for n in (9,17,33):
        y,x=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
        cases={'straight':.5*(1+np.tanh(18*(x+.7*(y-.5)-.48))),
               'radial':.5*(1+np.tanh(25*(np.hypot(x-.5,y-.5)-.23))),
               'crossing':.5+.2*np.sin(2*np.pi*(3*x+2*y))+.2*np.cos(2*np.pi*(2*x-3*y))}
        for name,source in cases.items():
            (model,diag),construction=timed(lambda:compile_mode(source),7)
            residual=None
            if model is not None:
                iy,ix=np.mgrid[:n,:n]
                residual=float(np.max(abs(evaluate(model,ix,iy)-source)))
                assert residual<1e-10
            for scale in (3,6):
                m=(n-1)*scale+1;q=np.linspace(0,n-1,m)
                yy,xx=np.meshgrid(q,q,indexing='ij')
                baseline,conv=timed(lambda:resize(source,scale,'CONV'),31)
                assert baseline.shape==(m,m) and np.isfinite(baseline).all()
                row={'n':n,'output_n':m,'scale':scale,'case':name,
                     'pairs':n*n*(n*n-1)//2,'accepted':model is not None,
                     'chart':diag.get('chart'),'construction':construction,'conv':conv,
                     'cardinality_error':residual}
                if model is not None:
                    z,ev=timed(lambda:evaluate(model,xx,yy),31)
                    row.update(evaluation=ev,knots=len(model['knots']),
                               first_use_ratio=(construction['median_ms']+ev['median_ms'])/conv['median_ms'],
                               evaluation_ratio=ev['median_ms']/conv['median_ms'])
                    assert np.isfinite(z).all()
                else:
                    row['rejection_reason']=diag['reason']
                    row['rejection_overhead_ratio']=construction['median_ms']/conv['median_ms']
                rows.append(row);print(json.dumps(row),flush=True)
                Path(out).write_text(json.dumps({'platform':platform.platform(),'numpy':np.__version__,
                    'timing':'perf_counter, one warmup, medians; compiler 7 repeats, evaluation/CONV 31; coordinate mesh excluded from both',
                    'rows':rows},indent=2)+'\n')

if __name__=='__main__':
    run('/tmp/conv_characteristic_cost.json')
