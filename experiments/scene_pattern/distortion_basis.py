"""Region-balanced distortion model search, with held-out region acceptance.

No object names, scene coordinates, or supplied correspondences enter this solver.
All maps carry reference atlas coordinates to source atlas coordinates.
"""
import numpy as np
from scipy import ndimage,optimize
from .core import srgb_to_lab

DIMENSIONS={'affine':6,'projective':8,'quadratic':12}

def map_basis(points,p,center,model):
    q=(np.asarray(points)-center)/50
    p=np.asarray(p);v=q*50+p[:2]+q@p[2:6].reshape(2,2).T
    if model=='projective':
        denominator=1+q@p[6:8]/50
        v=v/np.maximum(denominator[...,None],.08)
    elif model=='quadratic':
        terms=np.stack((q[...,0]**2,q[...,0]*q[...,1],q[...,1]**2),-1)
        v+=terms@p[6:12].reshape(2,3).T
    return center+v

def features(rgb,weight):
    lab=srgb_to_lab(rgb);valid=(weight>.01).astype(float)
    # Normalized convolution avoids manufacturing edges at missing-image borders.
    def smooth(x,sigma):
        den=ndimage.gaussian_filter(valid,sigma)
        return ndimage.gaussian_filter(x*valid,sigma)/np.maximum(den,1e-6)
    l=lab[...,0]
    return np.stack((smooth(l,.6)-smooth(l,3),smooth(l,1.5),smooth(l,5),smooth(lab[...,1],2),smooth(lab[...,2],2)),-1)

def geometry(points,p,center,model):
    q=map_basis(points,p,center,model)
    dx=(map_basis(points+[.1,0],p,center,model)-map_basis(points-[.1,0],p,center,model))/.2
    dy=(map_basis(points+[0,.1],p,center,model)-map_basis(points-[0,.1],p,center,model))/.2
    det=dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
    denom=1+((points-center)/50)@p[6:8]/50 if model=='projective' else np.ones(len(points))
    return dict(min_jacobian=float(det.min()),max_jacobian=float(det.max()),min_denominator=float(denom.min()),max_displacement=float(np.linalg.norm(q-points,axis=-1).max()))

def fit_models(reference,moving,reference_weight,moving_weight,regions,allow_visibility_change=False):
    masks=[np.asarray(r,bool)&(reference_weight>.2)&(moving_weight>.2) for r in regions]
    active=[k for k,m in enumerate(masks) if m.sum()>=150]
    if not active:return dict(status='insufficient_observed_support',trials=[],active_regions=[])
    union=np.logical_or.reduce([masks[k] for k in active]);yy,xx=np.nonzero(union);points=np.c_[xx,yy].astype(float);center=points.mean(0)
    labels=np.stack([masks[k][yy,xx] for k in active]);split=(xx//4+2*(yy//4))%3
    fa=features(reference,reference_weight);fb=features(moving,moving_weight)
    for f in [fa,fb]:
        mean=np.mean(f[yy,xx],0);std=np.maximum(np.std(f[yy,xx],0),[.008,.025,.025,.008,.008]);f-=mean;f/=std
    fw=np.array([.7,.7,1.,.6,.6]);target=fa[yy,xx]*fw
    # Each region receives equal total training mass, independent of area.
    mass=np.zeros(len(points))
    for label in labels:mass+=label/max(int(np.sum(label&(split==0))),1)
    mass=np.sqrt(mass*max(int(np.sum(split==0)),1)/len(labels))
    def residual(p,model):
        q=map_basis(points,p,center,model)
        value=np.stack([ndimage.map_coordinates(fb[...,k],[q[:,1],q[:,0]],order=1,mode='nearest') for k in range(5)],-1)*fw
        valid=ndimage.map_coordinates(moving_weight,[q[:,1],q[:,0]],order=1,mode='constant',cval=0)
        return value-target,valid
    def scores(p,model,part):
        r,v=residual(p,model);result=[]
        for label in labels:
            use=label&(split==part)
            if allow_visibility_change:
                confidence=np.clip(v[use]/.2,0,1);coverage=float(confidence.mean())
                loss=float(np.sum(np.mean(np.minimum(r[use]**2,9),axis=1)*confidence)/max(confidence.sum(),1e-8))
                result.append(loss+.25*(1-coverage)+20*max(.75-coverage,0))
            else:result.append(float(np.mean(np.minimum(r[use]**2,9)))+10*float(np.mean(v[use]<.05)))
        return result
    zero=np.zeros(6);baseline=scores(zero,'affine',1);test_before=scores(zero,'affine',2)
    grid=[]
    for dy in range(-48,49,4):
        for dx in range(-48,49,4):
            p=np.array([dx,dy,0,0,0,0.]);grid.append((np.mean(scores(p,'affine',0)),p))
    seeds=[]
    for _,p in sorted(grid,key=lambda z:z[0]):
        if all(np.linalg.norm(p[:2]-s[:2])>=8 for s in seeds):seeds.append(p)
        if len(seeds)==6:break
    seeds.append(zero);trials=[]
    for model in ['affine','projective','quadratic']:
        dimension=DIMENSIONS[model]
        if model!='affine':
            prior=sorted([t for t in trials if t['model']=='affine'],key=lambda t:t['selection_mean'])[:3]
            seeds=[np.r_[t['parameters'],np.zeros(dimension-6)] for t in prior]+[np.zeros(dimension)]
        for seed in seeds:
            def objective(p):
                r,v=residual(p,model)
                if allow_visibility_change:
                    confidence=np.clip(v/.2,0,1)
                    penalties=[max(.75-float(np.mean(confidence[label&(split==0)])),0)*40 for label in labels]
                    return np.r_[(r[split==0]*mass[split==0,None]*np.sqrt(confidence[split==0,None])).ravel(),penalties,p[2:6]*.015,p[6:]*.025]
                return np.r_[(r[split==0]*mass[split==0,None]).ravel(),np.maximum(.08-v[split==0],0)*8,p[2:6]*.015,p[6:]*.025]
            def jac(p):return np.column_stack([(objective(p+d)-objective(p-d))/.06 for d in np.eye(dimension)*.03])
            bound=np.r_[[64,64,45,45,45,45],np.full(dimension-6,12.)]
            result=optimize.least_squares(objective,np.clip(seed,-bound+.001,bound-.001),jac=jac,bounds=(-bound,bound),loss='soft_l1',f_scale=.5,max_nfev=100)
            p=result.x;g=geometry(points[::max(1,len(points)//1200)],p,center,model);validation=scores(p,model,1)
            # Region validation is an actual acceptance condition, not a report.
            admissible=g['min_jacobian']>.2 and g['max_jacobian']<5 and g['min_denominator']>.25
            coverage=[float(np.mean(np.clip(residual(p,model)[1][label]/.2,0,1))) for label in labels]
            passes=admissible and (not allow_visibility_change or min(coverage)>=.75) and all(a<=b*1.02 for a,b in zip(validation,baseline)) and np.mean(validation)<np.mean(baseline)*.95
            trials.append(dict(model=model,parameters=p.tolist(),selection=validation,selection_mean=float(np.mean(validation)),geometry=g,admissible=bool(admissible),coverage=coverage,accepted=bool(passes),reporting=scores(p,model,2)))
    eligible=[t for t in trials if t['accepted']]
    # Require the added degrees of freedom to earn their complexity on selection blocks.
    best=None
    for model in ['affine','projective','quadratic']:
        group=[t for t in eligible if t['model']==model]
        if group:
            candidate=min(group,key=lambda t:t['selection_mean'])
            if best is None or candidate['selection_mean']<best['selection_mean']*.97:best=candidate
    return dict(status='accepted' if best else 'unresolved_with_current_basis',visibility_change=allow_visibility_change,center=center.tolist(),active_regions=active,region_pixels=[int(masks[k].sum()) for k in active],before_selection=baseline,before_reporting=test_before,trials=trials,selected=best)
