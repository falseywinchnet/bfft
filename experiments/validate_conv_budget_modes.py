"""Broader paired end-to-end cost and accuracy gate for the budgeted operator."""
import argparse,json,time
from pathlib import Path
import numpy as np
from experiments.conv_budget_modes import resize
from experiments.probe_conv_compatible_2d import resize as original
from experiments.probe_conv_budget_modes import cases


def validation_cases(n,scale):
    q=np.linspace(0,1,n);u=np.linspace(0,1,(n-1)*scale+1)
    y,x=np.meshgrid(q,q,indexing='ij');v,t=np.meshgrid(u,u,indexing='ij')
    fields=[]
    for slope in (.35,2**-.5,-.4,1.1):
        for width in (18,32):
            fields.append((f'edge {slope} {width}',lambda x,y,s=slope,w=width:.5*(1+np.tanh(w*(x+s*(y-.5)-.48)))))
    for center in ((.5,.5),(.5+.5/(n-1),.5-.5/(n-1)),(.43,.57)):
        for width in (25,45):
            fields.append((f'radial {center} {width}',lambda x,y,c=center,w=width:.5*(1+np.tanh(w*(np.hypot(x-c[0],y-c[1])-.23)))))
    fields.extend([
        ('ellipse',lambda x,y:.5*(1+np.tanh(10*(np.hypot((x-.5)/.28,(y-.5)/.19)-1)))),
        ('curved monotone',lambda x,y:.5*(1+np.tanh(20*(x+.3*y+.4*y*y-.63)))),
        ('affine',lambda x,y:.13+.37*x+.71*y),
        ('quadratic',lambda x,y:x+.03*x*x+.7*y+.06*x*y),
        ('smooth radial offcenter',lambda x,y:np.hypot(x-.43,y-.57)),
        ('constant',lambda x,y:np.full(x.shape,.37)),
        ('axis edge',lambda x,y:.5*(1+np.tanh(25*(x-.47)))),
        ('near axis edge',lambda x,y:.5*(1+np.tanh(25*(x+.001*y-.47)))),
        ('random monotone',lambda x,y:None),
    ])
    for name,f in fields:
        if name=='random monotone':
            rng=np.random.default_rng(714+n);z=np.cumsum(np.cumsum(rng.uniform(.1,1,(n,n)),axis=0),axis=1)
            yield name,z,None
        else:yield name,f(x,y),f(t,v)
    yield 'random noise',np.random.default_rng(92+n).uniform(0,1,(n,n)),None


def run(out,sizes,scales,repeats):
    rows=[]
    for n in sizes:
        for scale in scales:
            for name,src,truth in list(cases(n,scale))+list(validation_cases(n,scale)):
                a=original(src,scale,'CONV');b,d=resize(src,scale,True)
                first=[];second=[]
                for i in range(repeats):
                    for method,record in ((0,first),(1,second)) if i%2==0 else ((1,second),(0,first)):
                        start=time.perf_counter()
                        if method:b=resize(src,scale)
                        else:a=original(src,scale,'CONV')
                        record.append(1000*(time.perf_counter()-start))
                row=dict(n=n,scale=scale,case=name,**d,conv_ms=float(np.median(first)),candidate_ms=float(np.median(second)),
                         ratio=float(np.median(second)/np.median(first)),
                         conv_times=first,candidate_times=second,
                         cardinality=float(np.max(abs(b[::scale,::scale]-src))),
                         excursion=float(max(0,b.max()-src.max(),src.min()-b.min())),
                         unchanged=bool(np.array_equal(a,b)))
                if truth is not None:
                    m=truth.shape[0]-1;lo=int(np.floor(.2*m))+1;hi=int(np.ceil(.8*m));mask=np.s_[lo:hi,lo:hi]
                    ga=np.array(np.gradient(a));gb=np.array(np.gradient(b));gt=np.array(np.gradient(truth))
                    row.update(mse_conv=float(np.mean((a[mask]-truth[mask])**2)),mse_candidate=float(np.mean((b[mask]-truth[mask])**2)),
                               full_mse_conv=float(np.mean((a-truth)**2)),full_mse_candidate=float(np.mean((b-truth)**2)),
                               grad_mse_conv=float(np.mean(np.sum((ga-gt)**2,axis=0)[mask])),
                               grad_mse_candidate=float(np.mean(np.sum((gb-gt)**2,axis=0)[mask])))
                assert row['cardinality']<2e-10*max(1.,np.max(abs(src)))
                if not d['accepted']:assert row['unchanged']
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('conv_times','candidate_times')}),flush=True)
                Path(out).write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_budget_validation.json')
    p.add_argument('--sizes',default='9,13,17,33,65,129');p.add_argument('--scales',default='2,3,6');p.add_argument('--repeats',type=int,default=7)
    a=p.parse_args();run(a.out,[int(x) for x in a.sizes.split(',')],[int(x) for x in a.scales.split(',')],a.repeats)
