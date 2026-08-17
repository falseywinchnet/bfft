#!/usr/bin/env python3
"""Projective structural roles with context-fitted affine laws.

There is no coordinate-to-class head. The complete inner context produces a
covariance operator which moves learned subspaces. Projection energy into the
subspaces supplies non-positional labels; every label pools an affine law from
the observations assigned to it. Queries must use those same roles and laws.
"""

from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_hypersphere_atlas import accuracy, draw_scatter, spiral_points, tail_profile
import run_jet_transport as jet_plots

torch.set_default_dtype(torch.float64); torch.set_num_threads(8)


class ProjectiveQuotient(nn.Module):
    def __init__(self, width=20, dimension=12, labels=8, rank=2,
                 mode="projective", ridge=2e-2):
        super().__init__(); self.dimension=dimension; self.labels=labels; self.rank=rank
        self.mode=mode; self.ridge=ridge
        self.feature=nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,dimension))
        self.generators=nn.Parameter(torch.randn(labels,dimension,rank)/math.sqrt(dimension))
        self.label_bias=nn.Parameter(torch.zeros(labels))
        self.log_temperature=nn.Parameter(torch.tensor(-.4))
        if mode=="positional":
            self.centers=nn.Parameter(torch.randn(labels,2)*2)
            self.log_radius=nn.Parameter(torch.tensor(0.0))

    def encode(self,x):
        # Raw affine information is an unbounded residual. The nonlinear path
        # is generic and receives no radius, phase, Fourier, or ordering cue.
        return torch.cat((self.feature(x),x,torch.ones_like(x[:,:1])),1)

    def scene_basis(self,z):
        d=z.shape[1]; centered=z-z.mean(0,keepdim=True)
        covariance=centered.T@centered/max(1,len(z)-1)
        covariance=covariance+1e-4*torch.eye(d,dtype=z.dtype,device=z.device)
        if self.mode=="diagonal": covariance=torch.diag(torch.diagonal(covariance))
        padding=torch.zeros(self.labels,d-self.dimension,self.rank,dtype=z.dtype,device=z.device)
        generators=torch.cat((self.generators,padding),1)
        moved=torch.einsum("de,ker->kdr",covariance,generators)
        return torch.linalg.qr(moved,mode="reduced").Q

    def assignments(self,x,z,basis):
        if self.mode=="global": return torch.ones(len(x),1,dtype=z.dtype,device=z.device)
        if self.mode=="positional":
            radius=self.log_radius.exp().clamp(.1,20.)
            logits=-torch.cdist(x,self.centers).square()/radius.square()
        else:
            projected=torch.einsum("nd,kdr->nkr",z,basis).square().sum(2)
            logits=torch.log(projected/z.square().sum(1,keepdim=True).clamp_min(1e-9)+1e-8)
            logits=logits+self.label_bias
        return torch.softmax(logits/self.log_temperature.exp().clamp(.08,5.),1)

    def fit_scene(self,x,y,destroy_relations=False,generator=None):
        z=self.encode(x); basis=None if self.mode in {"global","positional"} else self.scene_basis(z)
        weights=self.assignments(x,z,basis)
        if destroy_relations and weights.shape[1]>1:
            weights=weights[torch.randperm(len(weights),generator=generator)]
        targets=2*y.to(z.dtype)-1; eye=torch.eye(z.shape[1],dtype=z.dtype,device=z.device); laws=[]
        for k in range(weights.shape[1]):
            w=weights[:,k]
            lhs=z.T@(w[:,None]*z)+self.ridge*eye; rhs=z.T@(w*targets)
            laws.append(torch.linalg.solve(lhs,rhs))
        return torch.stack(laws),basis

    def predict_with_scene(self,x,laws,basis):
        z=self.encode(x); weights=self.assignments(x,z,basis); values=z@laws.T
        score=(weights*values).sum(1)
        return torch.stack((-score,score),1)

    def forward_episode(self,context_x,context_y,query_x,destroy_relations=False,generator=None):
        laws,basis=self.fit_scene(context_x,context_y,destroy_relations,generator)
        return self.predict_with_scene(query_x,laws,basis)

    def regularization(self,x):
        if self.mode in {"global","positional"}: return torch.zeros((),dtype=x.dtype,device=x.device)
        z=self.encode(x); basis=self.scene_basis(z); weights=self.assignments(x,z,basis)
        mass=weights.mean(0)
        balance=(mass*torch.log(mass*self.labels+1e-9)).sum()
        projectors=basis@basis.transpose(1,2); overlap=projectors.flatten(1)@projectors.flatten(1).T
        off=overlap-torch.diag(torch.diagonal(overlap))
        return .02*balance+.001*off.square().mean()


class BoundScene(nn.Module):
    def __init__(self,model,context_x,context_y,destroy_relations=False,seed=0):
        super().__init__(); self.model=model
        with torch.no_grad():
            laws,basis=model.fit_scene(context_x,context_y,destroy_relations,
                                       torch.Generator().manual_seed(9000+seed))
        self.register_buffer("laws",laws)
        if basis is None: self.basis=None
        else: self.register_buffer("basis",basis)
    def forward(self,x): return self.model.predict_with_scene(x,self.laws,self.basis)


def sample_episode(indices,context_size,query_size,generator):
    order=indices[torch.randperm(len(indices),generator=generator)]
    return order[:context_size],order[context_size:context_size+query_size]


@torch.no_grad()
def label_diagnostics(model,x):
    z=model.encode(x); basis=None if model.mode in {"global","positional"} else model.scene_basis(z)
    weights=model.assignments(x,z,basis); mass=weights.mean(0)
    return float(torch.exp(-(mass*torch.log(mass+1e-12)).sum())),float(weights.max(1).values.mean())


def train(kind,seed,fraction,steps,batch,lr,width,dimension,labels,rank):
    torch.manual_seed(100+seed); x,y=spiral_points(1600,.015,fraction,1000+seed)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(2000+seed)); va,tr=order[:len(x)//5],order[len(x)//5:]
    mode="projective"; destroy=False
    if kind=="projective_diagonal": mode="diagonal"
    elif kind=="projective_shuffled": destroy=True
    elif kind=="positional": mode="positional"
    elif kind=="global_affine": mode="global"; labels=rank=1
    model=ProjectiveQuotient(width,dimension,labels,rank,mode)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=2e-4)
    generator=torch.Generator().manual_seed(3000+seed); best=None; history=[]
    sizes=[192,384,768]
    for step in range(1,steps+1):
        context,query=sample_episode(tr,sizes[step%3],batch,generator)
        optimizer.zero_grad(); logits=model.forward_episode(x[context],y[context],x[query],destroy,generator)
        loss=F.cross_entropy(logits,y[query])+model.regularization(x[context]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        if step%100==0 or step==steps:
            bound=BoundScene(model,x[tr],y[tr],destroy,seed); val=accuracy(bound,x[va],y[va])
            history.append({"step":step,"loss":float(loss),"validation":val}); score=val-.001*float(loss)
            if best is None or score>best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},float(loss))
    model.load_state_dict(best[1]); bound=BoundScene(model,x[tr],y[tr],destroy,seed)
    effective,certainty=label_diagnostics(model,x[tr])
    return model,bound,accuracy(bound,x[va],y[va]),best[2],sum(p.numel() for p in model.parameters()),history,effective,certainty


def label_plot(path,bound,fraction=.5,seed=0,size=700):
    x,_=spiral_points(1000,.015,fraction,1000+seed)
    with torch.no_grad():
        z=bound.model.encode(x); basis=None if bound.model.mode in {"global","positional"} else bound.model.scene_basis(z)
        labels=bound.model.assignments(x,z,basis).argmax(1).numpy()
    lo=x.min(0).values.numpy()-1; hi=x.max(0).values.numpy()+1
    palette=[(32,116,172),(238,124,48),(94,170,83),(196,62,61),(141,103,171),(148,103,86),(218,106,169),(120,120,120)]
    image=Image.new("RGB",(size,size),(249,248,244)); draw=ImageDraw.Draw(image)
    for point,label in zip(x.numpy(),labels):
        px=int(25+(point[0]-lo[0])/(hi[0]-lo[0])*(size-50)); py=int(size-25-(point[1]-lo[1])/(hi[1]-lo[1])*(size-50))
        draw.ellipse((px-2,py-2,px+2,py+2),fill=palette[label%len(palette)])
    draw.text((30,8),"Learned projective structural roles (inner scene only)",fill=(20,20,20)); image.save(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("experiments/relational_witness_spiral/results_projective_quotient"))
    parser.add_argument("--steps",type=int,default=1500); parser.add_argument("--seeds",type=int,default=5)
    parser.add_argument("--width",type=int,default=20); parser.add_argument("--dimension",type=int,default=12)
    parser.add_argument("--labels",type=int,default=8); parser.add_argument("--rank",type=int,default=2)
    parser.add_argument("--batch",type=int,default=192); parser.add_argument("--lr",type=float,default=2e-3)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    kinds=["global_affine","positional","projective_diagonal","projective_shuffled","projective"]
    runs=[]; representatives={}; started=time.time()
    for fraction in [.5,.3]:
        for seed in range(args.seeds):
            for kind in kinds:
                model,bound,val,loss,parameters,history,effective,certainty=train(kind,seed,fraction,args.steps,args.batch,args.lr,args.width,args.dimension,args.labels,args.rank)
                bins,survival,frontier,tail=tail_profile(bound,fraction,seed)
                row={"model":kind,"seed":seed,"train_fraction":fraction,"parameters":parameters,"val_accuracy":val,"loss":loss,
                     "effective_labels":effective,"assignment_certainty":certainty,"survival_bins_at_80pct":survival,
                     "first_bin_accuracy":bins[0],"frontier5_accuracy":frontier,"tail_accuracy":tail,"tail_bins":bins}
                runs.append(row); print(json.dumps(row),flush=True)
                with (args.out/"runs.partial.json").open("w") as handle: json.dump({"runs":runs},handle,indent=2)
                if fraction==.5 and seed==0: representatives[kind]=bound
    summary=[]
    for fraction in [.5,.3]:
        for kind in kinds:
            selected=[r for r in runs if r["model"]==kind and r["train_fraction"]==fraction]
            summary.append({"model":kind,"train_fraction":fraction,"parameters":selected[0]["parameters"],
                "val_mean":float(np.mean([r["val_accuracy"] for r in selected])),"first_bin_mean":float(np.mean([r["first_bin_accuracy"] for r in selected])),
                "frontier5_mean":float(np.mean([r["frontier5_accuracy"] for r in selected])),"frontier5_std":float(np.std([r["frontier5_accuracy"] for r in selected])),
                "tail_mean":float(np.mean([r["tail_accuracy"] for r in selected])),"survival_mean":float(np.mean([r["survival_bins_at_80pct"] for r in selected])),
                "survival_max":int(np.max([r["survival_bins_at_80pct"] for r in selected])),"effective_labels":float(np.mean([r["effective_labels"] for r in selected]))})
    with (args.out/"runs.json").open("w") as handle: json.dump({"runtime_seconds":time.time()-started,"configuration":vars(args),"runs":runs},handle,indent=2,default=str)
    with (args.out/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    train_x,train_y=spiral_points(600,.015,.5,12); hold_x,hold_y=spiral_points(400,.5,1,13)
    for kind,bound in representatives.items(): draw_scatter(args.out/f"decision_{kind}.png",bound,train_x,train_y,hold_x,hold_y,f"{kind}: projective quotient, 50% holdout")
    label_plot(args.out/"labels_projective.png",representatives["projective"])
    curves={kind:np.mean([r["tail_bins"] for r in runs if r["model"]==kind and r["train_fraction"]==.5],axis=0) for kind in kinds}
    jet_plots.COLORS.update({"global_affine":(35,35,35),"positional":(125,82,164),"projective_diagonal":(226,131,38),"projective_shuffled":(196,55,46),"projective":(8,132,160)})
    jet_plots.survival_plot(args.out/"survival_50pct.png",curves)
    print(json.dumps({"runtime_seconds":time.time()-started,"summary":summary},indent=2))


if __name__=="__main__": main()
