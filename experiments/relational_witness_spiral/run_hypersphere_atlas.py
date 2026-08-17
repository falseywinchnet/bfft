#!/usr/bin/env python3
"""Structure-agnostic learned hypersphere-atlas transport on double spirals.

No coordinates, Fourier shells, graph edges, phase, or held-out samples are
given to the layer. It retains a full dense map. Learned charts generate output
directions from the current activation; transport, when enabled, acts only on
their learned spherical Gram geometry.
"""

from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64); torch.set_num_threads(8)


def spiral_points(count,lo,hi,seed,noise=.5):
    rng=np.random.default_rng(seed)
    phase=8*math.pi*np.sqrt(rng.uniform(lo*lo,hi*hi,(count,1)))
    nx=(2*rng.random((count,1))-1)*noise; ny=(2*rng.random((count,1))-1)*noise
    a=np.concatenate([.5*(np.sin(phase)*phase+nx),.5*(np.cos(phase)*phase+ny)],1)
    x=np.concatenate([a,-a]); y=np.concatenate([np.zeros(count,dtype=np.int64),np.ones(count,dtype=np.int64)])
    return torch.from_numpy(x),torch.from_numpy(y)


class HypersphereAtlasLinear(nn.Module):
    def __init__(self,n_in,n_out,mode,charts=6,rank=3):
        super().__init__(); self.mode=mode; self.charts=charts; self.rank=rank
        self.base=nn.Linear(n_in,n_out)
        self.coordinates=nn.Linear(n_in,charts*rank,bias=False)
        self.basis=nn.Parameter(torch.empty(charts,rank,n_out))
        self.coefficients=nn.Linear(n_in,charts)
        self.log_temperature=nn.Parameter(torch.tensor(-.5))
        self.atlas_scale=nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.basis,std=.08)

    def forward(self,x):
        B=len(x); q=self.coordinates(x).view(B,self.charts,self.rank)
        directions=torch.einsum('bcr,cro->bco',q,self.basis)
        directions=F.normalize(directions,dim=-1,eps=1e-8)
        coeff=torch.tanh(self.coefficients(x))
        if self.mode=='identity': transported=coeff
        elif self.mode=='isotropic': transported=coeff.mean(1,keepdim=True).expand_as(coeff)
        else:
            gram=directions@directions.transpose(1,2)
            distance=(1-gram).clamp_min(0)
            tau=self.log_temperature.exp().clamp(.05,5.)
            flow=torch.softmax(-distance/tau,dim=-1)
            if self.mode=='shuffled': flow=torch.roll(flow,1,dims=-1)
            transported=torch.einsum('bij,bj->bi',flow,coeff)
        correction=torch.einsum('bc,bco->bo',transported,directions)/math.sqrt(self.charts)
        return self.base(x)+torch.tanh(self.atlas_scale)*correction


class TinyMLP(nn.Module):
    def __init__(self,kind,width=24):
        super().__init__(); self.e=nn.Linear(2,width)
        if kind=='dense': self.u=nn.Linear(width,2*width); self.d=nn.Linear(2*width,width)
        else:
            mode=kind.removeprefix('atlas_')
            self.u=HypersphereAtlasLinear(width,2*width,mode)
            self.d=HypersphereAtlasLinear(2*width,width,mode)
        self.o=nn.Linear(width,2)
    def forward(self,x): return self.o(self.d(F.gelu(self.u(self.e(x)))))


@torch.no_grad()
def accuracy(model,x,y): return float((model(x).argmax(1)==y).double().mean())


def tail_profile(model,fraction,seed,bins=20,per_bin=250):
    vals=[]
    for j in range(bins):
        lo=fraction+(1-fraction)*j/bins; hi=fraction+(1-fraction)*(j+1)/bins
        x,y=spiral_points(per_bin,lo,hi,90000+1000*seed+j); vals.append(accuracy(model,x,y))
    survival=0
    for v in vals:
        if v<.8: break
        survival+=1
    return vals,survival,float(np.mean(vals[:5])),float(np.mean(vals))


def train(kind,seed,fraction,steps,batch,lr,width):
    torch.manual_seed(100+seed); x,y=spiral_points(1600,.015,fraction,1000+seed)
    perm=torch.randperm(len(x),generator=torch.Generator().manual_seed(2000+seed)); va,tr=perm[:len(x)//5],perm[len(x)//5:]
    model=TinyMLP(kind,width); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(3000+seed); best=None; hist=[]
    for step in range(1,steps+1):
        ix=tr[torch.randint(len(tr),(batch,),generator=gen)]
        opt.zero_grad(); loss=F.cross_entropy(model(x[ix]),y[ix]); loss.backward(); opt.step()
        if step%100==0 or step==steps:
            val=accuracy(model,x[va],y[va]); bins,_,front,_=tail_profile(model,fraction,seed,bins=5,per_bin=100)
            hist.append({'step':step,'loss':float(loss),'val':val,'frontier5':front})
            if best is None or val>best[0]: best=(val,{k:v.detach().clone() for k,v in model.state_dict().items()},float(loss))
    model.load_state_dict(best[1]); return model,best[0],best[2],sum(p.numel() for p in model.parameters()),hist


COLORS={'dense':(35,35,35),'atlas_identity':(123,78,163),'atlas_isotropic':(224,130,45),'atlas_eikonal':(22,138,173),'atlas_shuffled':(192,57,43)}


def dataset_plot(path,fraction=.5):
    train,ytr=spiral_points(600,.015,fraction,12); hold,yh=spiral_points(400,fraction,1,13)
    draw_scatter(path,None,train,ytr,hold,yh,'Double spiral: inner training and outer holdout')


def draw_scatter(path,model,train,ytr,hold,yh,title,size=700):
    allx=torch.cat([train,hold]); lo=allx.min(0).values.numpy()-1; hi=allx.max(0).values.numpy()+1
    im=Image.new('RGB',(size,size),(249,248,244)); d=ImageDraw.Draw(im)
    def xy(p): return (int(30+(p[0]-lo[0])/(hi[0]-lo[0])*(size-60)),int(size-30-(p[1]-lo[1])/(hi[1]-lo[1])*(size-60)))
    if model is not None:
        n=220; gx=np.linspace(lo[0],hi[0],n); gy=np.linspace(lo[1],hi[1],n); yy,xx=np.meshgrid(gy,gx,indexing='ij')
        grid=torch.from_numpy(np.stack([xx.ravel(),yy.ravel()],1)); probs=[]
        with torch.no_grad():
            for chunk in grid.split(4096): probs.append(torch.softmax(model(chunk),1)[:,1])
        p=torch.cat(probs).reshape(n,n).numpy()
        pix=np.empty((n,n,3),np.uint8); pix[...,0]=(65+150*p).astype(np.uint8); pix[...,1]=(105-35*np.abs(p-.5)*2).astype(np.uint8); pix[...,2]=(210-140*p).astype(np.uint8)
        # NumPy row zero is pasted at the image top, whereas the grid begins at
        # Cartesian y_min and scatter coordinates draw y_max at the top.
        pix=np.flipud(pix)
        bg=Image.fromarray(pix).resize((size-60,size-60),Image.Resampling.BILINEAR); im.paste(bg,(30,30)); d=ImageDraw.Draw(im)
    for pts,labels,outline in [(hold,yh,True),(train,ytr,False)]:
        for p,c in zip(pts.numpy(),labels.numpy()):
            x,y=xy(p); color=(30,90,210) if c==0 else (220,55,45); r=3 if outline else 2
            d.ellipse((x-r,y-r,x+r,y+r),fill=(249,248,244) if outline else color,outline=color)
    d.rectangle((30,30,size-30,size-30),outline=(80,80,80)); d.text((36,8),title,fill=(20,20,20))
    im.save(path)


def line_plot(path,series,title,ylabel,reference=None):
    W,H=1000,600; im=Image.new('RGB',(W,H),(249,248,244)); d=ImageDraw.Draw(im); L,T,R,B=80,55,25,70
    d.rectangle((L,T,W-R,H-B),outline=(80,80,80)); d.text((L,T-30),title,fill=(20,20,20)); d.text((8,T),ylabel,fill=(20,20,20))
    if reference is not None:
        y=H-B-reference*(H-B-T); d.line((L,y,W-R,y),fill=(150,150,150),width=1)
    n=max(len(v) for v in series.values())
    for name,vals in series.items():
        pts=[]
        for i,v in enumerate(vals): pts.append((L+i/(n-1)*(W-R-L),H-B-v*(H-B-T)))
        d.line(pts,fill=COLORS[name],width=4); d.text((L+180*(list(series).index(name)%5),H-40),name,fill=COLORS[name])
    im.save(path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('experiments/relational_witness_spiral/results_hypersphere'))
    ap.add_argument('--steps',type=int,default=1800); ap.add_argument('--seeds',type=int,default=5); ap.add_argument('--width',type=int,default=24)
    ap.add_argument('--batch',type=int,default=256); ap.add_argument('--lr',type=float,default=3e-3); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    kinds=['dense','atlas_identity','atlas_isotropic','atlas_eikonal','atlas_shuffled']; runs=[]; histories={}; representatives={}; start=time.time()
    for fraction in [.5,.3]:
        for seed in range(args.seeds):
            for kind in kinds:
                model,val,loss,params,hist=train(kind,seed,fraction,args.steps,args.batch,args.lr,args.width)
                bins,survival,front,tail=tail_profile(model,fraction,seed)
                row={'model':kind,'seed':seed,'train_fraction':fraction,'parameters':params,'val_accuracy':val,'loss':loss,'survival_bins_at_80pct':survival,'frontier5_accuracy':front,'tail_accuracy':tail,'tail_bins':bins}
                runs.append(row); print(json.dumps(row),flush=True)
                if fraction==.5 and seed==0: representatives[kind]=model; histories[kind]=hist
    summary=[]
    for fraction in [.5,.3]:
        for kind in kinds:
            rs=[r for r in runs if r['model']==kind and r['train_fraction']==fraction]
            summary.append({'model':kind,'train_fraction':fraction,'parameters':rs[0]['parameters'],'val_mean':float(np.mean([r['val_accuracy'] for r in rs])),'frontier5_mean':float(np.mean([r['frontier5_accuracy'] for r in rs])),'frontier5_std':float(np.std([r['frontier5_accuracy'] for r in rs])),'tail_mean':float(np.mean([r['tail_accuracy'] for r in rs])),'survival_mean':float(np.mean([r['survival_bins_at_80pct'] for r in rs])),'survival_max':int(np.max([r['survival_bins_at_80pct'] for r in rs]))})
    with (args.out/'runs.json').open('w') as f: json.dump({'runtime_seconds':time.time()-start,'runs':runs},f,indent=2)
    with (args.out/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    train_points,ytr=spiral_points(600,.015,.5,12); hold,yh=spiral_points(400,.5,1,13); dataset_plot(args.out/'dataset.png')
    for kind,model in representatives.items(): draw_scatter(args.out/f'decision_{kind}.png',model,train_points,ytr,hold,yh,f'{kind}: 50% spiral holdout')
    curves={kind:np.mean([r['tail_bins'] for r in runs if r['model']==kind and r['train_fraction']==.5],axis=0) for kind in kinds}
    line_plot(args.out/'survival_50pct.png',curves,'Accuracy by distance beyond the 50% training frontier','accuracy',.8)
    loss_curves={k:[max(0,min(1,-math.log10(max(h['loss'],1e-5))/5)) for h in v] for k,v in histories.items()}
    line_plot(args.out/'training_loss_seed0.png',loss_curves,'Training loss, seed 0 (higher means lower log-loss)','scaled -log10(loss)')
    print(json.dumps({'runtime_seconds':time.time()-start,'summary':summary},indent=2))


if __name__=='__main__': main()
