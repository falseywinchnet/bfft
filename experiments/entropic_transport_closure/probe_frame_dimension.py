import argparse,json
from pathlib import Path
import numpy as np
from .frame_transport import run

def problem(n,d,separation,seed):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,d));Y=rng.normal(size=(n,d))
    sx=rng.choice([-1.,1.],n);sy=rng.choice([-1.,1.],n)
    X[:,0]=(X[:,0]+separation*sx)/np.sqrt(1+separation**2)
    Y[:,0]=(Y[:,0]+separation*sy)/np.sqrt(1+separation**2)
    Y=Y*np.linspace(.7,1.3,d)[None,:]+.5/np.sqrt(d)
    C=((X[:,None,:]-Y[None,:,:])**2).sum(-1);C/=np.median(C)
    return C,np.full(n,1/n),np.full(n,1/n)

def geometry(K,a,b):
    v=np.ones(len(b));directions=[];changes=[];skew=[];kurt=[]
    old=None
    for t in range(128):
        kv=K@v;u=a/kv;u/=u.max();ktu=K.T@u;nxt=b/ktu;nxt/=nxt.max()
        h=np.log(nxt/v);h-=h.mean();changes.append(float(np.ptp(h)))
        norm=np.linalg.norm(h)
        if norm>1e-10:directions.append(h/norm)
        if t in [4,16,64]:
            P=K*v[None,:]/kv[:,None];mean=P@h
            centered=h[None,:]-mean[:,None]
            var=np.sum(P*centered**2,axis=1);third=np.sum(P*centered**3,axis=1);fourth=np.sum(P*centered**4,axis=1)
            w=a*var;w/=max(w.sum(),1e-300)
            skew.append(float(np.sum(w*np.abs(third)/np.maximum(var,1e-100)**1.5)))
            kurt.append(float(np.sum(w*np.abs(fourth/np.maximum(var,1e-100)**2-3))))
        v=nxt
    s=np.linalg.svd(np.stack(directions),compute_uv=False);p=s*s/(s@s)
    return dict(direction_effective_rank=float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-300))))),
        direction_count_1e6=int(np.sum(s>s[0]*1e-6)),max_single_pass_frame=max(changes),
        conditional_skew=skew,conditional_abs_excess_kurtosis=kurt)

def ordinary_moving_horizon(K,a,b,cap=256):
    v=np.ones(len(b))
    for k in range(cap):
        u=a/(K@v);u/=u.max();ktu=K.T@u
        c=v*ktu;c/=c.sum()
        if np.max(np.abs(c/b-1))<=1e-8:return max(1,k),True
        v=b/ktu;v/=v.max()
    return cap,False

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=128)
    args=p.parse_args();rows=[]
    for seed in [0,1]:
        for d in [1,2,4,8]:
            for separation in [0.,1.,3.]:
                C,a,b=problem(args.size,d,separation,seed)
                for eps in [.1,.03]:
                    K=np.exp(-C/eps);control=run(K,a,b,method='ordinary')
                    horizon,reached=ordinary_moving_horizon(K,a,b)
                    moving=run(K,a,b,method='cache',passes=horizon)
                    runout=run(K,a,b,method='cache')
                    row=dict(seed=seed,dimension=d,separation=separation,eps=eps,n=args.size,
                        products=runout['products'],reuses=runout['reuses'],resets=runout['resets'],
                        moving_horizon=horizon,reference_reached=reached,moving_products=moving['products'],moving_reuses=moving['reuses'],
                        seconds=runout['seconds'],ordinary_seconds=control['seconds'],
                        error=float(np.ptp(runout['y']-control['y'])),bound=runout['bound'],**geometry(K,a,b))
                    rows.append(row);Path(args.out).write_text(json.dumps(rows))
        print('finished seed',seed,flush=True)
if __name__=='__main__':main()
