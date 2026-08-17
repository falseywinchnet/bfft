#!/usr/bin/env python3
"""Affine transport between projective roles learned from an inner scene.

Random half-space occlusions create generic continuation episodes entirely
inside the observed scene. Context adjacency induces a directed transition on
projective roles. A query anchors to the visible context and must traverse
powers of that transition before the context-fitted affine laws are mixed.
"""

from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_hypersphere_atlas import accuracy, draw_scatter, spiral_points, tail_profile
from run_projective_quotient import ProjectiveQuotient
import run_jet_transport as jet_plots

torch.set_default_dtype(torch.float64); torch.set_num_threads(8)


class ProjectiveTransition(ProjectiveQuotient):
    def __init__(self,width=20,dimension=12,labels=8,rank=2,steps=4,mode="transition",ridge=2e-2,crossings_only=False):
        super().__init__(width,dimension,labels,rank,"projective",ridge)
        d=dimension+3; self.steps=steps; self.transition_mode=mode; self.crossings_only=crossings_only
        self.clock=nn.Linear(d,1,bias=False)
        self.step_selector=nn.Sequential(nn.Linear(2,12),nn.SiLU(),nn.Linear(12,steps+1))
        self.log_edge_temperature=nn.Parameter(torch.tensor(-.3))
        self.log_anchor_temperature=nn.Parameter(torch.tensor(0.0))
        self.self_loop=nn.Parameter(torch.tensor(0.0))

    def scene_basis(self,z):
        # The quotient must retain the same coordinates across different
        # occlusions. Only the transport is inferred from the visible scene.
        d=z.shape[1]
        padding=torch.zeros(self.labels,d-self.dimension,self.rank,dtype=z.dtype,device=z.device)
        return torch.linalg.qr(torch.cat((self.generators,padding),1),mode="reduced").Q

    def context_state(self,x,y,destroy_edges=False,generator=None):
        z=self.encode(x); basis=self.scene_basis(z); roles=self.assignments(x,z,basis)
        targets=2*y.to(z.dtype)-1; eye=torch.eye(z.shape[1],dtype=z.dtype,device=z.device); laws=[]
        for k in range(self.labels):
            w=roles[:,k]; laws.append(torch.linalg.solve(z.T@(w[:,None]*z)+self.ridge*eye,z.T@(w*targets)))
        # Adjacency uses only the observed geometry. Direction is learned from
        # a scalar structural clock, not supplied radius or path ordering.
        distances=torch.cdist(x,x)
        distances=distances.masked_fill(torch.eye(len(x),dtype=torch.bool,device=x.device),float("inf"))
        neighbors=distances.topk(min(6,len(x)-1),largest=False).indices
        source=torch.arange(len(x),device=x.device)[:,None].expand_as(neighbors).reshape(-1)
        destination=neighbors.reshape(-1)
        source_roles=roles[source]; destination_roles=roles[destination]
        if destroy_edges:
            destination_roles=destination_roles[torch.randperm(len(destination_roles),generator=generator)]
        clock=self.clock(z).squeeze(1); delta=clock[destination]-clock[source]
        edge_weight=torch.sigmoid(delta/self.log_edge_temperature.exp().clamp(.08,5.))
        transition=torch.einsum("e,ei,ej->ij",edge_weight,source_roles,destination_roles)
        identity=torch.eye(self.labels,dtype=z.dtype,device=z.device)
        if self.crossings_only:
            # Quotient adjacency retains crossings between roles. Edges
            # internal to one role are represented by the zero-step selector.
            transition=transition*(1-identity)+1e-6*(1-identity)
        else:
            transition=transition+F.softplus(self.self_loop)*identity
        transition=transition/transition.sum(1,keepdim=True).clamp_min(1e-9)
        reverse=transition.T; reverse=reverse/reverse.sum(1,keepdim=True).clamp_min(1e-9)
        return {"z":z,"basis":basis,"roles":roles,"laws":torch.stack(laws),
                "transition":transition,"reverse":reverse,"clock":clock}

    def transported_roles(self,query_x,state):
        qz=self.encode(query_x)
        distance=torch.cdist(qz,state["z"])
        nearest=distance.min(1).values
        context_nn=torch.cdist(state["z"],state["z"])
        context_nn=context_nn.masked_fill(torch.eye(len(state["z"]),dtype=torch.bool,device=qz.device),float("inf"))
        scale=context_nn.min(1).values.median().clamp_min(1e-6)
        anchor_temperature=self.log_anchor_temperature.exp().clamp(.05,20.)*scale
        anchor_weight=torch.softmax(-distance.square()/anchor_temperature.square(),1)
        anchor=anchor_weight@state["roles"]
        anchor_clock=anchor_weight@state["clock"]; query_clock=self.clock(qz).squeeze(1)
        delta=(query_clock-anchor_clock)/self.log_edge_temperature.exp().clamp(.08,5.)
        selector_input=torch.stack((torch.log1p(nearest/scale),delta.abs()),1)
        step_weight=torch.softmax(self.step_selector(selector_input),1)
        forward=[anchor]; backward=[anchor]
        for _ in range(self.steps):
            forward.append(forward[-1]@state["transition"])
            backward.append(backward[-1]@state["reverse"])
        direction=torch.sigmoid(delta)[:,None,None]
        paths=direction*torch.stack(forward,1)+(1-direction)*torch.stack(backward,1)
        transported=(step_weight[:,:,None]*paths).sum(1)
        return transported/transported.sum(1,keepdim=True).clamp_min(1e-9),qz,step_weight

    def predict_episode(self,context_x,context_y,query_x,destroy_edges=False,generator=None):
        state=self.context_state(context_x,context_y,destroy_edges,generator); qz=self.encode(query_x)
        if self.transition_mode=="direct": roles=self.assignments(query_x,qz,state["basis"]); step_weight=None
        elif self.transition_mode=="identity":
            original_transition=state["transition"]; original_reverse=state["reverse"]
            identity=torch.eye(self.labels,dtype=qz.dtype,device=qz.device)
            state["transition"]=identity; state["reverse"]=identity
            roles,qz,step_weight=self.transported_roles(query_x,state)
            state["transition"]=original_transition; state["reverse"]=original_reverse
        else: roles,qz,step_weight=self.transported_roles(query_x,state)
        values=qz@state["laws"].T; score=(roles*values).sum(1)
        return torch.stack((-score,score),1),state,step_weight

    def regularization(self,x,y):
        state=self.context_state(x,y); mass=state["roles"].mean(0)
        balance=(mass*torch.log(mass*self.labels+1e-9)).sum()
        role_entropy=-(state["roles"]*torch.log(state["roles"]+1e-9)).sum(1).mean()
        transition_entropy=-(state["transition"]*torch.log(state["transition"]+1e-9)).sum(1).mean()
        projectors=state["basis"]@state["basis"].transpose(1,2)
        overlap=projectors.flatten(1)@projectors.flatten(1).T
        off=overlap-torch.diag(torch.diagonal(overlap))
        return .015*balance+.015*role_entropy+.001*transition_entropy+.001*off.square().mean()


class BoundTransition(nn.Module):
    def __init__(self,model,x,y,destroy_edges=False,seed=0):
        super().__init__(); self.model=model; self.destroy_edges=destroy_edges; self.seed=seed
        self.register_buffer("context_x",x); self.register_buffer("context_y",y)
    def forward(self,x):
        return self.model.predict_episode(self.context_x,self.context_y,x,self.destroy_edges,
            torch.Generator().manual_seed(9000+self.seed))[0]


def occlusion_episode(x,indices,batch,generator):
    # A random oriented cut is generic missing-region supervision. The query is
    # a narrow band just across the cut; all points come from the inner scene.
    direction=torch.randn(2,generator=generator,dtype=x.dtype); direction=direction/direction.norm()
    projection=x[indices]@direction
    quantile=.30+.40*float(torch.rand((),generator=generator))
    threshold=torch.quantile(projection,quantile)
    flip=bool(torch.randint(2,(),generator=generator))
    context_mask=projection<=threshold if flip else projection>=threshold
    query_mask=~context_mask
    context=indices[context_mask]; candidates=indices[query_mask]
    gap=(x[candidates]@direction-threshold).abs(); query=candidates[gap.argsort()[:batch]]
    if len(context)>768: context=context[torch.randperm(len(context),generator=generator)[:768]]
    return context,query


@torch.no_grad()
def diagnostics(model,x,y):
    state=model.context_state(x,y); transition=state["transition"]
    entropy=float(-(transition*torch.log(transition+1e-9)).sum(1).mean())
    return entropy,float(torch.diagonal(transition).mean()),float(state["roles"].max(1).values.mean())


def train(kind,seed,fraction,steps,batch,lr,width,dimension,labels,rank,transport_steps,crossings_only=False):
    torch.manual_seed(100+seed); x,y=spiral_points(1600,.015,fraction,1000+seed)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(2000+seed)); va,tr=order[:len(x)//5],order[len(x)//5:]
    mode="transition"; destroy=False
    if kind=="direct_projective": mode="direct"
    elif kind=="anchor_identity": mode="identity"
    elif kind=="transition_shuffled": destroy=True
    model=ProjectiveTransition(width,dimension,labels,rank,transport_steps,mode,crossings_only=crossings_only)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=2e-4)
    generator=torch.Generator().manual_seed(3000+seed); best=None; history=[]
    for step in range(1,steps+1):
        if step%5:
            context,query=occlusion_episode(x,tr,batch,generator)
        else:
            shuffled=tr[torch.randperm(len(tr),generator=generator)]
            context,query=shuffled[:768],shuffled[768:768+batch]
        optimizer.zero_grad(); logits,_,step_weight=model.predict_episode(x[context],y[context],x[query],destroy,generator)
        loss=F.cross_entropy(logits,y[query])+model.regularization(x[context],y[context]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        if step%100==0 or step==steps:
            bound=BoundTransition(model,x[tr],y[tr],destroy,seed); val=accuracy(bound,x[va],y[va])
            history.append({"step":step,"loss":float(loss),"validation":val}); score=val-.001*float(loss)
            if best is None or score>best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},float(loss))
    model.load_state_dict(best[1]); bound=BoundTransition(model,x[tr],y[tr],destroy,seed)
    entropy,self_loop,certainty=diagnostics(model,x[tr],y[tr])
    return model,bound,accuracy(bound,x[va],y[va]),best[2],sum(p.numel() for p in model.parameters()),history,entropy,self_loop,certainty


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("experiments/relational_witness_spiral/results_projective_transition"))
    parser.add_argument("--steps",type=int,default=1500); parser.add_argument("--seeds",type=int,default=5)
    parser.add_argument("--width",type=int,default=20); parser.add_argument("--dimension",type=int,default=12); parser.add_argument("--labels",type=int,default=8)
    parser.add_argument("--rank",type=int,default=2); parser.add_argument("--transport-steps",type=int,default=4)
    parser.add_argument("--batch",type=int,default=192); parser.add_argument("--lr",type=float,default=2e-3); parser.add_argument("--fractions",default=".5,.3")
    parser.add_argument("--crossings-only",action="store_true",help="discard within-role adjacency and let zero steps encode dwell time")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True); fractions=[float(v) for v in args.fractions.split(",")]
    kinds=["direct_projective","anchor_identity","transition_shuffled","transition"]
    runs=[]; representatives={}; started=time.time()
    for fraction in fractions:
        for seed in range(args.seeds):
            for kind in kinds:
                model,bound,val,loss,parameters,history,entropy,self_loop,certainty=train(kind,seed,fraction,args.steps,args.batch,args.lr,args.width,args.dimension,args.labels,args.rank,args.transport_steps,args.crossings_only)
                bins,survival,frontier,tail=tail_profile(bound,fraction,seed)
                row={"model":kind,"seed":seed,"train_fraction":fraction,"parameters":parameters,"val_accuracy":val,"loss":loss,
                     "transition_entropy":entropy,"transition_self_loop":self_loop,"assignment_certainty":certainty,
                     "survival_bins_at_80pct":survival,"first_bin_accuracy":bins[0],"frontier5_accuracy":frontier,"tail_accuracy":tail,"tail_bins":bins}
                runs.append(row); print(json.dumps(row),flush=True)
                with (args.out/"runs.partial.json").open("w") as handle: json.dump({"runs":runs},handle,indent=2)
                if fraction==fractions[0] and seed==0: representatives[kind]=bound
    summary=[]
    for fraction in fractions:
        for kind in kinds:
            selected=[r for r in runs if r["model"]==kind and r["train_fraction"]==fraction]
            summary.append({"model":kind,"train_fraction":fraction,"parameters":selected[0]["parameters"],"val_mean":float(np.mean([r["val_accuracy"] for r in selected])),
                "first_bin_mean":float(np.mean([r["first_bin_accuracy"] for r in selected])),"frontier5_mean":float(np.mean([r["frontier5_accuracy"] for r in selected])),
                "frontier5_std":float(np.std([r["frontier5_accuracy"] for r in selected])),"tail_mean":float(np.mean([r["tail_accuracy"] for r in selected])),
                "survival_mean":float(np.mean([r["survival_bins_at_80pct"] for r in selected])),"survival_max":int(np.max([r["survival_bins_at_80pct"] for r in selected])),
                "transition_entropy":float(np.mean([r["transition_entropy"] for r in selected])),"transition_self_loop":float(np.mean([r["transition_self_loop"] for r in selected]))})
    with (args.out/"runs.json").open("w") as handle: json.dump({"runtime_seconds":time.time()-started,"configuration":vars(args),"runs":runs},handle,indent=2,default=str)
    with (args.out/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    train_x,train_y=spiral_points(600,.015,fractions[0],12); hold_x,hold_y=spiral_points(400,fractions[0],1,13)
    for kind,bound in representatives.items(): draw_scatter(args.out/f"decision_{kind}.png",bound,train_x,train_y,hold_x,hold_y,f"{kind}: role-transition transport")
    curves={kind:np.mean([r["tail_bins"] for r in runs if r["model"]==kind and r["train_fraction"]==fractions[0]],axis=0) for kind in kinds}
    jet_plots.COLORS.update({"direct_projective":(35,35,35),"anchor_identity":(125,82,164),"transition_shuffled":(196,55,46),"transition":(8,132,160)})
    jet_plots.survival_plot(args.out/f"survival_{int(fractions[0]*100)}pct.png",curves)
    print(json.dumps({"runtime_seconds":time.time()-started,"summary":summary},indent=2))


if __name__=="__main__": main()
