"""End-to-end budget/accuracy screen, including rejected-mode CONV fallback."""
import json,time
from pathlib import Path
import numpy as np
from experiments.conv_budget_modes import resize
from experiments.probe_conv_compatible_2d import resize as original


def cases(n,scale=3):
    q=np.linspace(0,1,n);u=np.linspace(0,1,(n-1)*scale+1)
    y,x=np.meshgrid(q,q,indexing='ij');v,t=np.meshgrid(u,u,indexing='ij')
    fields=[('diagonal sigmoid',lambda x,y:.5*(1+np.tanh((x+.7*y-.86)*20))),
            ('curved sigmoid',lambda x,y:.5*(1+np.tanh((np.hypot(x-.5,y-.5)-.27)*35))),
            ('diagonal step',lambda x,y:(x+.7*y>.86).astype(float)),
            ('disk',lambda x,y:(np.hypot(x-.5,y-.5)<.27).astype(float)),
            ('diagonal wave',lambda x,y:.5+.4*np.sin(2*np.pi*(5*x+3*y+.13))),
            ('crossing waves',lambda x,y:.5+.2*np.sin(2*np.pi*(5*x+2*y+.17))+.2*np.cos(2*np.pi*(2*x-5*y+.31))),
            ('chirped field',lambda x,y:.5+.4*np.sin(2*np.pi*(x+3*x*x+2*y*y+.23))),
            ('edge and texture',lambda x,y:.4*(1+np.tanh((x+.4*y-.73)*25))+.08*np.sin(2*np.pi*(5*x-3*y+.13)))]
    for name,f in fields:yield name,f(x,y),f(t,v)


def run(out='/tmp/conv_budget_modes.json',sizes=(17,33,65)):
    rows=[]
    for n in sizes:
        for name,src,truth in cases(n):
            a=original(src,3,'CONV');b,d=resize(src,3,True)
            first=[];second=[]
            for i in range(15):
                for method,record in ((0,first),(1,second)) if i%2==0 else ((1,second),(0,first)):
                    start=time.perf_counter()
                    if method:b=resize(src,3)
                    else:a=original(src,3,'CONV')
                    record.append(1000*(time.perf_counter()-start))
            m=(truth.shape[0]-1);lo=int(np.floor(.2*m))+1;hi=int(np.ceil(.8*m));mask=np.s_[lo:hi,lo:hi]
            ga=np.array(np.gradient(a));gb=np.array(np.gradient(b));gt=np.array(np.gradient(truth))
            row=dict(n=n,case=name,**d,conv_ms=float(np.median(first)),candidate_ms=float(np.median(second)),
                     ratio=float(np.median(second)/np.median(first)),
                     mse_conv=float(np.mean((a[mask]-truth[mask])**2)),mse_candidate=float(np.mean((b[mask]-truth[mask])**2)),
                     full_mse_conv=float(np.mean((a-truth)**2)),full_mse_candidate=float(np.mean((b-truth)**2)),
                     grad_mse_conv=float(np.mean(np.sum((ga-gt)**2,axis=0)[mask])),
                     grad_mse_candidate=float(np.mean(np.sum((gb-gt)**2,axis=0)[mask])),
                     excursion=float(max(0,b.max()-src.max(),src.min()-b.min())),
                     cardinality=float(np.max(abs(b[::3,::3]-src))))
            rows.append(row);print(json.dumps(row),flush=True)
            Path(out).write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':run()
