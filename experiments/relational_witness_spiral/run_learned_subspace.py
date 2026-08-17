#!/usr/bin/env python3
"""Learned-subspace continuation test (Torch, CPU only).

Unlike Fourier circles, graphs, lattices, and the fixed-view pilot, this layer
does not impose a geometry.  It strictly contains a dense linear map.  Several
learned low-rank branches act as witnesses, and either their marginal energies
or their mutual Gram matrix generates the sample-specific branch mixture.
"""

from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)


def spiral_points(count, lo, hi, seed, noise=.5):
    rng = np.random.default_rng(seed)
    phase = 8*math.pi*np.sqrt(rng.uniform(lo*lo, hi*hi, (count,1)))
    nx=(2*rng.random((count,1))-1)*noise; ny=(2*rng.random((count,1))-1)*noise
    a=np.concatenate([.5*(np.sin(phase)*phase+nx), .5*(np.cos(phase)*phase+ny)], 1)
    x=np.concatenate([a,-a]).astype(np.float64)
    y=np.concatenate([np.zeros(count,dtype=np.int64),np.ones(count,dtype=np.int64)])
    return torch.from_numpy(x), torch.from_numpy(y)


class LearnedSubspaceLinear(nn.Module):
    """Dense map plus learned low-rank witnesses; dynamic modes use no fixed basis."""
    def __init__(self, n_in, n_out, mode, slices=3, rank=4, response=16):
        super().__init__()
        self.mode, self.slices, self.rank = mode, slices, rank
        self.base = nn.Linear(n_in, n_out)
        self.down = nn.ModuleList([nn.Linear(n_in, rank, bias=False) for _ in range(slices)])
        self.up = nn.ModuleList([nn.Linear(rank, n_out, bias=False) for _ in range(slices)])
        # The zero mixture makes the realized layer start exactly dense.  Keep
        # branch outputs small but nonzero so the mixture receives a gradient.
        for up in self.up: nn.init.normal_(up.weight, std=.02)
        if mode == "static":
            self.mix = nn.Parameter(torch.zeros(slices))
        else:
            statdim = slices if mode == "marginal" else slices*(slices+1)//2
            self.response = nn.Sequential(nn.Linear(statdim,response), nn.GELU(),
                                          nn.Linear(response,slices))
            nn.init.zeros_(self.response[-1].weight)
            nn.init.zeros_(self.response[-1].bias)

    def forward(self, x):
        q=torch.stack([down(x) for down in self.down],1)  # [B,S,R]
        qn=F.normalize(q,dim=-1,eps=1e-8)
        if self.mode == "static":
            alpha=torch.tanh(self.mix)[None].expand(len(x),-1)
        elif self.mode == "marginal":
            stats=torch.log1p(q.square().mean(-1))
            alpha=torch.tanh(self.response(stats))
        else:
            gram=qn @ qn.transpose(1,2)
            tri=torch.triu_indices(self.slices,self.slices,device=x.device)
            stats=gram[:,tri[0],tri[1]]
            alpha=torch.tanh(self.response(stats))
        branch=torch.stack([up(q[:,i]) for i,up in enumerate(self.up)],1)
        return self.base(x)+(alpha[:,:,None]*branch).sum(1)/math.sqrt(self.slices)


class TinyMLP(nn.Module):
    def __init__(self, kind, width):
        super().__init__()
        self.e=nn.Linear(2,width)
        if kind == "reference_linear":
            self.u=nn.Linear(width,2*width); self.d=nn.Linear(2*width,width)
        else:
            mode=kind.removeprefix("subspace_")
            self.u=LearnedSubspaceLinear(width,2*width,mode)
            self.d=LearnedSubspaceLinear(2*width,width,mode)
        self.o=nn.Linear(width,2)
    def forward(self,x): return self.o(self.d(F.gelu(self.u(self.e(x)))))


@torch.no_grad()
def accuracy(model,x,y): return float((model(x).argmax(1)==y).double().mean())


def train(kind,width,seed,fraction,steps,batch,lr):
    torch.manual_seed(100+seed)
    x,y=spiral_points(1600,.015,fraction,1000+seed)
    perm=torch.randperm(len(x),generator=torch.Generator().manual_seed(2000+seed))
    va,tr=perm[:len(x)//5],perm[len(x)//5:]
    model=TinyMLP(kind,width).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(3000+seed); best=None
    for step in range(1,steps+1):
        ix=tr[torch.randint(len(tr),(batch,),generator=gen)]
        opt.zero_grad(); loss=F.cross_entropy(model(x[ix]),y[ix]); loss.backward(); opt.step()
        if step%100==0 or step==steps:
            val=accuracy(model,x[va],y[va])
            if best is None or val>best[0]:
                best=(val,{k:v.detach().clone() for k,v in model.state_dict().items()},float(loss))
    model.load_state_dict(best[1])
    return model,best[0],best[2],sum(p.numel() for p in model.parameters())


def tail_profile(model,fraction,seed,bins=20,per_bin=250):
    values=[]
    for j in range(bins):
        lo=fraction+(1-fraction)*j/bins; hi=fraction+(1-fraction)*(j+1)/bins
        x,y=spiral_points(per_bin,lo,hi,90000+1000*seed+j)
        values.append(accuracy(model,x,y))
    survival=0
    for value in values:
        if value<.8: break
        survival+=1
    return values,survival,float(np.mean(values[:5])),float(np.mean(values))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('experiments/relational_witness_spiral/results_learned'))
    ap.add_argument('--steps',type=int,default=1800); ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--width',type=int,default=24); ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--lr',type=float,default=3e-3)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); start=time.time()
    kinds=['reference_linear','subspace_static','subspace_marginal','subspace_gram']; runs=[]
    for fraction in [.5,.3]:
        for seed in range(args.seeds):
            for kind in kinds:
                model,val,loss,parameters=train(kind,args.width,seed,fraction,args.steps,args.batch,args.lr)
                bins,survival,frontier5,tail=tail_profile(model,fraction,seed)
                row={'model':kind,'seed':seed,'train_fraction':fraction,'parameters':parameters,
                     'val_accuracy':val,'loss':loss,'survival_bins_at_80pct':survival,
                     'frontier5_accuracy':frontier5,'tail_accuracy':tail,'tail_bins':bins}
                runs.append(row); print(json.dumps(row),flush=True)
    fields=['model','train_fraction','parameters','val_mean','val_std','frontier5_mean',
            'frontier5_std','tail_mean','survival_mean','survival_max']
    summary=[]
    for fraction in [.5,.3]:
        for kind in kinds:
            rs=[r for r in runs if r['model']==kind and r['train_fraction']==fraction]
            summary.append({'model':kind,'train_fraction':fraction,'parameters':rs[0]['parameters'],
                'val_mean':float(np.mean([r['val_accuracy'] for r in rs])),
                'val_std':float(np.std([r['val_accuracy'] for r in rs])),
                'frontier5_mean':float(np.mean([r['frontier5_accuracy'] for r in rs])),
                'frontier5_std':float(np.std([r['frontier5_accuracy'] for r in rs])),
                'tail_mean':float(np.mean([r['tail_accuracy'] for r in rs])),
                'survival_mean':float(np.mean([r['survival_bins_at_80pct'] for r in rs])),
                'survival_max':int(np.max([r['survival_bins_at_80pct'] for r in rs]))})
    runtime=time.time()-start
    with (args.out/'runs.json').open('w') as f: json.dump({'runtime_seconds':runtime,'runs':runs},f,indent=2)
    with (args.out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    print(json.dumps({'runtime_seconds':runtime,'summary':summary},indent=2))


if __name__=='__main__': main()
