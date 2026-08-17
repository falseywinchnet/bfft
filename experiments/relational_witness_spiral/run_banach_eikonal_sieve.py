#!/usr/bin/env python3
"""Continuous Banach-sieve induction by eikonal transport on a circle.

Finite views are quadrature particles sampling one periodic operator curve.
Observed context creates a smooth tanh-bounded potential on S^1. Its analytic
derivative transports the particles; their empirical density selects the
hidden operator. Queries never enter the evidence state used for transport.
"""

from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image,ImageDraw

from run_hypersphere_atlas import accuracy, draw_scatter, spiral_points, tail_profile
from run_projective_transition import occlusion_episode
import run_jet_transport as jet_plots

torch.set_default_dtype(torch.float64); torch.set_num_threads(8)


def circular_basis(theta,harmonics):
    values=[torch.ones_like(theta)]
    for order in range(1,harmonics+1):
        values.extend((torch.cos(order*theta),torch.sin(order*theta)))
    return torch.stack(values,-1)


def circular_basis_derivative(theta,harmonics):
    values=[torch.zeros_like(theta)]
    for order in range(1,harmonics+1):
        values.extend((-order*torch.sin(order*theta),order*torch.cos(order*theta)))
    return torch.stack(values,-1)


class BanachEikonalSieve(nn.Module):
    def __init__(self,width=18,layers=3,views=16,harmonics=3,flow_steps=6,mode="transport",eikonal_epsilon=.05):
        super().__init__(); self.width=width; self.layers=layers; self.views=views
        self.harmonics=harmonics; self.flow_steps=flow_steps; self.mode=mode; self.eikonal_epsilon=eikonal_epsilon
        modes=1+2*harmonics; context_width=18
        self.embed=nn.Linear(2,width)
        self.context_point=nn.Sequential(nn.Linear(3,context_width),nn.SiLU(),nn.Linear(context_width,context_width))
        self.potential=nn.Linear(2*context_width,layers*modes)
        self.potential_bias=nn.Parameter(torch.zeros(layers,modes))
        with torch.no_grad():
            for layer in range(layers):
                phase=2*math.pi*layer/layers
                self.potential_bias[layer,1]=.35*math.cos(phase)
                self.potential_bias[layer,2]=.35*math.sin(phase)
                if harmonics>=2:
                    self.potential_bias[layer,3]=.15*math.cos(2*phase)
                    self.potential_bias[layer,4]=.15*math.sin(2*phase)
        self.radial=nn.Linear(2*context_width,layers)
        self.speed=nn.Linear(2*context_width,layers)
        self.operator_coeff=nn.Parameter(torch.randn(layers,modes,width,width+1)/math.sqrt(width*modes))
        self.layer_scale=nn.Parameter(torch.full((layers,),-1.0))
        self.output=nn.Linear(width,2)
        phase=2*math.pi*(torch.arange(views,dtype=torch.get_default_dtype())+.5)/views
        self.register_buffer("initial_phase",phase)
        permutation=torch.arange(views); permutation=(7*permutation+3)%views
        self.register_buffer("broken_permutation",permutation)

    def context_summary(self,x,y):
        signed=2*y.to(x.dtype)-1
        points=self.context_point(torch.cat((x,signed[:,None]),1))
        return torch.cat((points.mean(0),points.std(0,unbiased=False)),0)

    def potential_coefficients(self,summary):
        modes=1+2*self.harmonics
        learned=self.potential(summary).view(self.layers,modes)+self.potential_bias
        # A smooth biased field: biases are coordinates of the same periodic
        # potential, not independent categorical gates.
        return learned

    def transport_phases(self,summary,layer):
        coefficients=self.potential_coefficients(summary)[layer]
        radial=torch.sigmoid(self.radial(summary)[layer])
        speed=.15+1.10*torch.sigmoid(self.speed(summary)[layer])
        theta=self.initial_phase
        if self.mode=="frozen": return theta,torch.full_like(theta,1/self.views),radial
        if self.mode=="reweight":
            potential=torch.tanh(circular_basis(theta,self.harmonics)@coefficients)
            weights=torch.softmax(3*potential,0)
            return theta,weights,radial
        # Conservative Eulerian transport on the circular quadrature mesh.
        # Nodes are samples, not categories; increasing `views` refines this
        # finite-volume approximation to the same continuity equation.
        density=torch.full_like(theta,1/self.views); spacing=2*math.pi/self.views
        edge_theta=theta+.5*spacing
        for _ in range(self.flow_steps):
            basis=circular_basis(edge_theta,self.harmonics)
            derivative=circular_basis_derivative(edge_theta,self.harmonics)
            raw=basis@coefficients; d_raw=derivative@coefficients
            d_potential=(1-torch.tanh(raw).square())*d_raw
            velocity=-speed*d_potential/torch.sqrt(d_potential.square()+self.eikonal_epsilon)
            if self.mode=="broken": velocity=velocity[self.broken_permutation]
            right=torch.roll(density,-1)
            flux=F.relu(velocity)*density-F.relu(-velocity)*right
            divergence=flux-torch.roll(flux,1)
            density=density-divergence/(self.flow_steps*spacing)
            density=density.clamp_min(1e-8); density=density/density.sum()
        return theta,density,radial

    def operator_at(self,layer,theta):
        return torch.einsum("vm,mij->vij",circular_basis(theta,self.harmonics),self.operator_coeff[layer])

    def induced_operator(self,summary,layer):
        theta,weights,radial=self.transport_phases(summary,layer)
        operator=torch.einsum("v,vij->ij",weights,self.operator_at(layer,theta))
        return radial*operator,theta,weights,radial

    def forward_episode(self,context_x,context_y,query_x,return_state=False):
        summary=self.context_summary(context_x,context_y); h=self.embed(query_x); states=[]
        for layer in range(self.layers):
            operator,theta,weights,radial=self.induced_operator(summary,layer)
            homogeneous=torch.cat((h,torch.ones_like(h[:,:1])),1)
            update=homogeneous@operator.T
            h=h+F.softplus(self.layer_scale[layer])*F.silu(update)
            states.append((theta,weights,radial))
        output=self.output(h)
        return (output,states) if return_state else output


class BoundSieve(nn.Module):
    def __init__(self,model,x,y):
        super().__init__(); self.model=model; self.register_buffer("context_x",x); self.register_buffer("context_y",y)
    def forward(self,x): return self.model.forward_episode(self.context_x,self.context_y,x)


@torch.no_grad()
def density_diagnostics(model,x,y):
    _,states=model.forward_episode(x,y,x[:4],True); concentrations=[]; radial=[]
    for theta,weights,rho in states:
        phasor=(weights*torch.exp(1j*theta)).sum(); concentrations.append(float(phasor.abs())); radial.append(float(rho))
    median=x[:,0].median(); left=x[:,0]<=median; right=~left
    _,left_states=model.forward_episode(x[left],y[left],x[:4],True)
    _,right_states=model.forward_episode(x[right],y[right],x[:4],True)
    response=[]
    for (_,left_weights,_),(_,right_weights,_) in zip(left_states,right_states):
        response.append(.5*float((left_weights-right_weights).abs().sum()))
    return float(np.mean(concentrations)),float(np.mean(radial)),float(np.mean(response))


def train(kind,seed,fraction,steps,batch,lr,width,layers,views,harmonics,flow_steps,eikonal_epsilon):
    torch.manual_seed(100+seed); x,y=spiral_points(1600,.015,fraction,1000+seed)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(2000+seed)); va,tr=order[:len(x)//5],order[len(x)//5:]
    model=BanachEikonalSieve(width,layers,views,harmonics,flow_steps,kind,eikonal_epsilon)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=2e-4)
    generator=torch.Generator().manual_seed(3000+seed); best=None; history=[]
    for step in range(1,steps+1):
        if step%2:
            context,query=occlusion_episode(x,tr,batch,generator)
        else:
            shuffled=tr[torch.randperm(len(tr),generator=generator)]; context,query=shuffled[:768],shuffled[768:768+batch]
        optimizer.zero_grad(); logits=model.forward_episode(x[context],y[context],x[query])
        loss=F.cross_entropy(logits,y[query]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        if step%100==0 or step==steps:
            bound=BoundSieve(model,x[tr],y[tr]); val=accuracy(bound,x[va],y[va])
            history.append({"step":step,"loss":float(loss),"validation":val}); score=val-.001*float(loss)
            if best is None or score>best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},float(loss))
    model.load_state_dict(best[1]); bound=BoundSieve(model,x[tr],y[tr]); concentration,radial,response=density_diagnostics(model,x[tr],y[tr])
    return model,bound,accuracy(bound,x[va],y[va]),best[2],sum(p.numel() for p in model.parameters()),history,concentration,radial,response


def density_plot(path,bound,size=(960,330)):
    model=bound.model
    with torch.no_grad(): _,states=model.forward_episode(bound.context_x,bound.context_y,bound.context_x[:4],True)
    image=Image.new("RGB",size,(249,248,244)); draw=ImageDraw.Draw(image); panel=size[0]//model.layers
    for layer,(theta,weights,rho) in enumerate(states):
        cx=panel*layer+panel//2; cy=170; radius=105
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=(175,175,175),width=2)
        for index,(phase,weight) in enumerate(zip(theta.tolist(),weights.tolist())):
            x=cx+radius*math.cos(phase); y=cy-radius*math.sin(phase)
            point_radius=max(3,min(13,int(3+30*weight)))
            draw.ellipse((x-point_radius,y-point_radius,x+point_radius,y+point_radius),fill=(8,132,160))
        phasor=(weights*torch.exp(1j*theta)).sum(); ex=cx+radius*float(phasor.real); ey=cy-radius*float(phasor.imag)
        draw.line((cx,cy,ex,ey),fill=(196,55,46),width=3)
        draw.text((cx-radius,35),f"layer {layer+1}: R={float(phasor.abs()):.3f}, radius={float(rho):.3f}",fill=(20,20,20))
    draw.text((25,305),"Particles are quadrature mass on the interpretation circle; red is the circular resultant.",fill=(20,20,20))
    image.save(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("experiments/relational_witness_spiral/results_banach_eikonal_sieve"))
    parser.add_argument("--steps",type=int,default=1600); parser.add_argument("--seeds",type=int,default=5); parser.add_argument("--batch",type=int,default=192)
    parser.add_argument("--width",type=int,default=18); parser.add_argument("--layers",type=int,default=3); parser.add_argument("--views",type=int,default=16)
    parser.add_argument("--harmonics",type=int,default=3); parser.add_argument("--flow-steps",type=int,default=6); parser.add_argument("--lr",type=float,default=2e-3)
    parser.add_argument("--eikonal-epsilon",type=float,default=.05)
    parser.add_argument("--fractions",default=".5,.3"); args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    fractions=[float(v) for v in args.fractions.split(",")]; kinds=["frozen","reweight","broken","transport"]
    runs=[]; representatives={}; started=time.time()
    for fraction in fractions:
        for seed in range(args.seeds):
            for kind in kinds:
                model,bound,val,loss,parameters,history,concentration,radial,response=train(kind,seed,fraction,args.steps,args.batch,args.lr,args.width,args.layers,args.views,args.harmonics,args.flow_steps,args.eikonal_epsilon)
                bins,survival,frontier,tail=tail_profile(bound,fraction,seed)
                row={"model":kind,"seed":seed,"train_fraction":fraction,"parameters":parameters,"val_accuracy":val,"loss":loss,
                     "density_concentration":concentration,"radial_inducement":radial,"context_response_tv":response,"survival_bins_at_80pct":survival,
                     "first_bin_accuracy":bins[0],"frontier5_accuracy":frontier,"tail_accuracy":tail,"tail_bins":bins}
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
                "density_concentration":float(np.mean([r["density_concentration"] for r in selected])),"radial_inducement":float(np.mean([r["radial_inducement"] for r in selected])),
                "context_response_tv":float(np.mean([r["context_response_tv"] for r in selected]))})
    with (args.out/"runs.json").open("w") as handle: json.dump({"runtime_seconds":time.time()-started,"configuration":vars(args),"runs":runs},handle,indent=2,default=str)
    with (args.out/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    train_x,train_y=spiral_points(600,.015,fractions[0],12); hold_x,hold_y=spiral_points(400,fractions[0],1,13)
    for kind,bound in representatives.items(): draw_scatter(args.out/f"decision_{kind}.png",bound,train_x,train_y,hold_x,hold_y,f"{kind}: continuous Banach-eikonal sieve")
    for kind,bound in representatives.items(): density_plot(args.out/f"density_{kind}.png",bound)
    curves={kind:np.mean([r["tail_bins"] for r in runs if r["model"]==kind and r["train_fraction"]==fractions[0]],axis=0) for kind in kinds}
    jet_plots.COLORS.update({"frozen":(35,35,35),"reweight":(125,82,164),"broken":(196,55,46),"transport":(8,132,160)})
    jet_plots.survival_plot(args.out/f"survival_{int(fractions[0]*100)}pct.png",curves)
    print(json.dumps({"runtime_seconds":time.time()-started,"summary":summary},indent=2))


if __name__=="__main__": main()
