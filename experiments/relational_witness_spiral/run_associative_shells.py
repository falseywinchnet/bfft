#!/usr/bin/env python3
"""A shared associative medium with implicit Newton relation shells.

Each context observation contributes one rank-one key/value update. Queries
never enter the context that predicts them. Pair and higher relations arise
only through repeated application of the summed memory operator.
"""

from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_hypersphere_atlas import accuracy, draw_scatter, spiral_points, tail_profile
import run_jet_transport as jet_plots

torch.set_default_dtype(torch.float64); torch.set_num_threads(8)


class AssociativeShellNet(nn.Module):
    def __init__(self, width=24, shells=2):
        super().__init__(); self.width=width; self.shells=shells
        self.key = nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))
        self.value = nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width))
        self.query = nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))
        self.base = nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,2))
        self.output = nn.Linear(width,2)
        self.shell_scale = nn.Parameter(torch.full((shells,),-1.0))

    def memory(self,x,y,shuffle=False,generator=None):
        signed=2*y.to(x.dtype)-1
        k=self.key(x); v=self.value(torch.cat([x,signed[:,None]],1))
        if shuffle:
            # Matched null: retain key and value marginals but destroy which
            # rank-one write belongs to which observation.
            v=v[torch.randperm(len(v),generator=generator)]
        # A simple shared medium: one rank-one write per observation.
        return torch.einsum('ni,nj->ij',v,k)/len(x)

    def forward(self,x,memory=None):
        if self.shells==0 or memory is None: return self.base(x)
        h=self.query(x)
        for index in range(self.shells):
            h=h+F.softplus(self.shell_scale[index])*F.silu(h@memory.T)
        return self.output(h)


class BoundMemory(nn.Module):
    def __init__(self,model,memory): super().__init__(); self.model=model; self.register_buffer('bound_memory',memory)
    def forward(self,x): return self.model(x,self.bound_memory)


def sample_episode(train_indices,context_size,query_size,generator):
    order=train_indices[torch.randperm(len(train_indices),generator=generator)]
    context=order[:context_size]; query=order[context_size:context_size+query_size]
    return context,query


@torch.no_grad()
def bound_accuracy(model,memory,x,y): return accuracy(BoundMemory(model,memory),x,y)


def train(kind,seed,fraction,steps,batch,lr,width):
    torch.manual_seed(100+seed); x,y=spiral_points(1600,.015,fraction,1000+seed)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(2000+seed)); va,tr=order[:len(x)//5],order[len(x)//5:]
    shells=0 if kind=='query_only' else int(kind.removeprefix('memory').removesuffix('_shuffled'))
    shuffled=kind.endswith('_shuffled'); model=AssociativeShellNet(width,shells)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(3000+seed); best=None; history=[]
    context_choices=[128,256,512,1024]
    for step in range(1,steps+1):
        context_size=context_choices[step%len(context_choices)]
        context,query=sample_episode(tr,context_size,batch,generator)
        optimizer.zero_grad()
        memory=None if shells==0 else model.memory(x[context],y[context],shuffled,generator)
        loss=F.cross_entropy(model(x[query],memory),y[query]); loss.backward(); optimizer.step()
        if step%100==0 or step==steps:
            full_memory=None if shells==0 else model.memory(x[tr],y[tr],shuffled,torch.Generator().manual_seed(8000+seed))
            val=bound_accuracy(model,full_memory,x[va],y[va])
            context_gain=val-bound_accuracy(model,torch.zeros_like(full_memory) if full_memory is not None else None,x[va],y[va])
            history.append({'step':step,'loss':float(loss),'validation':val,'context_gain':context_gain})
            score=val+.05*context_gain-.001*float(loss)
            if best is None or score>best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},float(loss))
    model.load_state_dict(best[1])
    full_memory=None if shells==0 else model.memory(x[tr],y[tr],shuffled,torch.Generator().manual_seed(8000+seed))
    return model,full_memory,bound_accuracy(model,full_memory,x[va],y[va]),best[2],sum(p.numel() for p in model.parameters()),history


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=Path('experiments/relational_witness_spiral/results_associative_shells'))
    parser.add_argument('--steps',type=int,default=1800); parser.add_argument('--seeds',type=int,default=5); parser.add_argument('--width',type=int,default=24)
    parser.add_argument('--batch',type=int,default=192); parser.add_argument('--lr',type=float,default=3e-3); args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    kinds=['query_only','memory1','memory2','memory3','memory3_shuffled']; runs=[]; representatives={}; started=time.time()
    for fraction in [.5,.3]:
        for seed in range(args.seeds):
            for kind in kinds:
                model,memory,val,loss,parameters,history=train(kind,seed,fraction,args.steps,args.batch,args.lr,args.width)
                bound=BoundMemory(model,memory) if memory is not None else model
                bins,survival,frontier,tail=tail_profile(bound,fraction,seed)
                zero_val=val if memory is None else bound_accuracy(model,torch.zeros_like(memory),*spiral_points(400,.015,fraction,7000+seed))
                row={'model':kind,'seed':seed,'train_fraction':fraction,'parameters':parameters,'val_accuracy':val,'loss':loss,
                     'memory_frobenius':0 if memory is None else float(memory.norm()),'zero_memory_accuracy':zero_val,
                     'survival_bins_at_80pct':survival,'frontier5_accuracy':frontier,'first_bin_accuracy':bins[0],'tail_accuracy':tail,'tail_bins':bins}
                runs.append(row); print(json.dumps(row),flush=True)
                with (args.out/'runs.partial.json').open('w') as handle: json.dump({'runs':runs},handle,indent=2)
                if fraction==.5 and seed==0: representatives[kind]=bound
    summary=[]
    for fraction in [.5,.3]:
        for kind in kinds:
            selected=[r for r in runs if r['model']==kind and r['train_fraction']==fraction]
            summary.append({'model':kind,'train_fraction':fraction,'parameters':selected[0]['parameters'],'val_mean':float(np.mean([r['val_accuracy'] for r in selected])),
                            'first_bin_mean':float(np.mean([r['first_bin_accuracy'] for r in selected])),'frontier5_mean':float(np.mean([r['frontier5_accuracy'] for r in selected])),
                            'frontier5_std':float(np.std([r['frontier5_accuracy'] for r in selected])),'tail_mean':float(np.mean([r['tail_accuracy'] for r in selected])),
                            'survival_mean':float(np.mean([r['survival_bins_at_80pct'] for r in selected])),'survival_max':int(np.max([r['survival_bins_at_80pct'] for r in selected]))})
    with (args.out/'runs.json').open('w') as handle: json.dump({'runtime_seconds':time.time()-started,'configuration':vars(args),'runs':runs},handle,indent=2,default=str)
    with (args.out/'summary.csv').open('w',newline='') as handle: writer=csv.DictWriter(handle,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    train_x,train_y=spiral_points(600,.015,.5,12); hold_x,hold_y=spiral_points(400,.5,1,13)
    for kind,model in representatives.items(): draw_scatter(args.out/f'decision_{kind}.png',model,train_x,train_y,hold_x,hold_y,f'{kind}: shared memory, 50% holdout')
    curves={kind:np.mean([r['tail_bins'] for r in runs if r['model']==kind and r['train_fraction']==.5],axis=0) for kind in kinds}
    jet_plots.COLORS.update({'query_only':(35,35,35),'memory1':(125,82,164),'memory2':(226,131,38),
                             'memory3':(8,132,160),'memory3_shuffled':(196,55,46)})
    jet_plots.survival_plot(args.out/'survival_50pct.png',curves)
    print(json.dumps({'runtime_seconds':time.time()-started,'summary':summary},indent=2))


if __name__=='__main__': main()
