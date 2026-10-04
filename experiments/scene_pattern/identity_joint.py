"""Joint three-camera / local-depth fit with revisable image-map alternatives."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from .identity_pose_graph import ray
from .identity_geometry import roles
from .affine_camera_geometry import measurements,epi_error,epi_residual,skew
from .identity_exclusion import freeze_split
from .multiview_geometry import triangulate,unit

DELTA=np.array([(x,y) for y in (-3.,0.,3.) for x in (-3.,0.,3.)])
BASIS=np.c_[np.ones(9),DELTA]


def project(points,camera,R,t):
    q=points@R.T+t;f=camera['focal'];fx=camera.get('focal_x',f);fy=camera.get('focal_y',f);h,w=camera['shape'];cx,cy=camera.get('principal_point',[(w-1)/2,(h-1)/2])
    if camera['projection']=='perspective':
        z=np.where(np.abs(q[...,2])>1e-8,q[...,2],1e-8)
        return np.stack([fx*q[...,0]/z+cx,fy*q[...,1]/z+cy],axis=-1)
    if camera['projection']=='cylindrical':return np.stack([fx*np.arctan2(q[...,0],q[...,2])+cx,fy*q[...,1]/np.maximum(np.hypot(q[...,0],q[...,2]),1e-8)+cy],axis=-1)
    raise ValueError('Unsupported camera projection')


def groups(rows_ab,rows_ac):
    a={};c={}
    for r in rows_ab:a.setdefault(tuple(r['p']),[]).append(r)
    for r in rows_ac:c.setdefault(tuple(r['p']),[]).append(r)
    return [dict(p=np.array(p),b=a[p],c=c[p]) for p in sorted(a.keys()&c.keys())]


def targets(options):return np.array([np.asarray(r['q'])+DELTA@np.asarray(r['A']).T for r in options])


def depth_from_ab(group,cameras,ab):
    a=ray(group['p']+DELTA,cameras[0]);options=[]
    for k,expected in enumerate(targets(group['b'])):
        b=ray(expected,cameras[1]);X,da,db,miss,angles=triangulate(a,b,ab['R'],ab['t'])
        if np.any(da<=0) or np.any(db<=0) or np.median(angles)<np.deg2rad(.3):continue
        parameters=np.linalg.lstsq(BASIS,np.log(da),rcond=None)[0]
        points=a*np.exp(BASIS@parameters)[:,None];predicted=project(points,cameras[1],ab['R'],ab['t']);error=float(np.mean((predicted-expected)**2))
        options.append((parameters,k,error,points))
    return a,options


def choose(predicted,options):
    expected=targets(options);cost=np.mean(np.minimum((expected-predicted)**2,100),axis=(1,2))+.02*np.array([1-r['training'] for r in options]);return int(np.argmin(cost)),float(np.min(cost))


def fit(rows_ab,rows_ac,cameras,ab,ac,capture=0,maximum=100,rounds=3,third_rows=None,third_capture=None,frozen_split=None,use_third_factors=True):
    allgroups=groups(rows_ab,rows_ac);training,held,blocks=roles([dict(p=g['p']) for g in allgroups],capture) if allgroups else ([],[],[])
    third_measurements=None;third_groups={};partition=None
    if third_rows is not None:
        partition=frozen_split if frozen_split is not None else freeze_split(allgroups,third_rows,capture,third_capture)
        training=np.array(partition['training_a'],bool);held=np.array(partition['held_a'],bool);blocks=partition['blocks_a']
        third_measurements=measurements(third_rows,cameras[1:]) if third_rows else None
        for i in (np.flatnonzero(partition['training_b']) if use_third_factors else []):third_groups.setdefault(third_rows[i]['site'],[]).append(int(i))
    train=[];initial=[];rays=[];initial_choices=[]
    for index,g in enumerate(allgroups):
        if not training[index]:continue
        rr,options=depth_from_ab(g,cameras,ab)
        if not options:continue
        choices=[]
        for parameters,k,error,points in options:
            kc,cost=choose(project(points,cameras[2],ac['R'],ac['t']),g['c']);choices.append((error+cost,parameters,k,kc))
        _,parameters,k,kc=min(choices,key=lambda r:r[0]);train.append(index);initial.append(parameters);rays.append(rr);initial_choices.append((k,kc))
    if len(train)<8:return dict(accepted=False,reason='insufficient_training_depth_support',training_regions=len(train),partition=partition)
    n=len(train);rr=np.array(rays);x=np.r_[Rotation.from_matrix(ab['R']).as_rotvec(),ab['t'],Rotation.from_matrix(ac['R']).as_rotvec(),ac['t'],np.array(initial).ravel()]
    third_sites=sorted(third_groups);nb=len(third_sites)
    sparsity=lil_matrix((n*36+nb*3+1,len(x)),dtype=int)
    for view in range(2):
        for i in range(n):
            sl=slice(view*n*18+i*18,view*n*18+(i+1)*18);sparsity[sl,view*6:(view+1)*6]=1;sparsity[sl,12+3*i:15+3*i]=1
    if nb:sparsity[n*36:n*36+nb*3,:12]=1
    sparsity[-1,3:6]=1;sparsity=sparsity.tocsr();history=[]
    def decode(v):return Rotation.from_rotvec(v[:3]).as_matrix(),unit(v[3:6]),Rotation.from_rotvec(v[6:9]).as_matrix(),v[9:12]
    def predict(v):
        Rb,tb,Rc,tc=decode(v);depth=np.exp(np.clip(v[12:].reshape(n,3)@BASIS.T,-20,20));points=rr*depth[...,None]
        return project(points,cameras[1],Rb,tb),project(points,cameras[2],Rc,tc)
    def third_matrix(v):
        Rb,tb,Rc,tc=decode(v);R=Rc@Rb.T;t=tc-R@tb
        return skew(t)@R
    def third_choices(v):
        if not nb:return []
        errors=epi_error(third_measurements,third_matrix(v))
        return [min(third_groups[site],key=lambda k:float(errors[k])+.15*(1-third_rows[k]['training'])) for site in third_sites]
    selected=initial_choices;selected_third=third_choices(x);initial_third=selected_third[:]
    def subset(ids):return {k:v[ids] if isinstance(v,np.ndarray) else v for k,v in third_measurements.items()}
    for iteration in range(rounds):
        pb,pc=predict(x);new=[(choose(pb[i],allgroups[k]['b'])[0],choose(pc[i],allgroups[k]['c'])[0]) for i,k in enumerate(train)]
        before=sum(a!=b for a,b in zip(selected,new));selected=new
        new_third=third_choices(x);third_changed=sum(a!=b for a,b in zip(selected_third,new_third));selected_third=new_third
        third_data=subset(selected_third) if nb else None
        expected_b=np.array([targets(allgroups[k]['b'])[s[0]] for k,s in zip(train,selected)]);expected_c=np.array([targets(allgroups[k]['c'])[s[1]] for k,s in zip(train,selected)])
        def residual(v):
            pb,pc=predict(v);extra=(epi_residual(third_data,third_matrix(v))/3).ravel() if nb else np.array([])
            return np.r_[((pb-expected_b)/3).ravel(),((pc-expected_c)/3).ravel(),extra,10*(np.linalg.norm(v[3:6])-1)]
        opt=least_squares(residual,x,jac_sparsity=sparsity,loss='soft_l1',f_scale=.5,max_nfev=maximum,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        x=opt.x;history.append(dict(round=iteration,assignments_changed=before,third_assignments_changed=third_changed,evaluations=opt.nfev,cost=float(opt.cost),status=int(opt.status),optimality=float(opt.optimality)))
    Rb,tb,Rc,tc=decode(x);new_ab=dict(R=Rb,t=tb);new_ac=dict(R=Rc,t=tc);checks=[]
    for k,g in enumerate(allgroups):
        if not held[k]:continue
        _,options=depth_from_ab(g,cameras,new_ab)
        if not options:checks.append(dict(p=g['p'].tolist(),resolved=False,passed=False));continue
        # Camera C is not consulted when selecting the held region's depth.
        params,index,error,points=min(options,key=lambda o:o[2]);predicted=project(points,cameras[2],Rc,tc);errors=np.linalg.norm(targets(g['c'])-predicted,axis=2).max(1);best=int(np.argmin(errors));positive=bool(np.all(np.sum((points@Rc.T+tc)*ray(targets(g['c'])[best],cameras[2]),axis=1)>0))
        checks.append(dict(p=g['p'].tolist(),resolved=True,maximum_pixels=float(errors[best]),passed=bool(errors[best]<=1.5 and positive),source_b_alternative=index,target_c_alternative=best))
    passed=[c for c in checks if c['passed']];tiles=len({tuple(np.floor(np.array(c['p'])/48).astype(int)) for c in passed});training_tiles=len({blocks[k] for k in train});accepted=len(passed)>=4 and tiles>=2 and len(passed)>=.6*len(checks) and training_tiles>=3
    pb,pc=predict(x);final_choices=[(choose(pb[i],allgroups[k]['b'])[0],choose(pc[i],allgroups[k]['c'])[0]) for i,k in enumerate(train)]
    depths=np.exp(np.clip(x[12:].reshape(n,3)@BASIS.T,-20,20));world=rr*depths[...,None];positive=True
    for pose,camera,key,which in [(new_ab,cameras[1],'b',0),(new_ac,cameras[2],'c',1)]:
        expected=np.array([targets(allgroups[k][key])[ss[which]] for k,ss in zip(train,final_choices)])
        rays_target=ray(expected.reshape(-1,2),camera).reshape(n,9,3)
        positive=positive and bool(np.all(np.sum((world@pose['R'].T+pose['t'])*rays_target,axis=2)>0))
    accepted=accepted and positive
    return dict(accepted=bool(accepted),partition=partition,third_pair_training_regions=nb,third_pair_identity_changes=sum(a!=b for a,b in zip(initial_third,third_choices(x))),third_pair_selected_training=third_choices(x),training_positive=positive,training_regions=n,training_tiles=training_tiles,check_count=len(passed),check_tiles=tiles,check_total=len(checks),checks=checks,history=history,
        assignment_changes_from_initial=sum(a!=b for a,b in zip(initial_choices,final_choices)),
        poses=[dict(rotation=np.eye(3).tolist(),translation=[0,0,0],center=[0,0,0]),dict(rotation=Rb.tolist(),translation=tb.tolist(),center=(-Rb.T@tb).tolist()),dict(rotation=Rc.tolist(),translation=tc.tolist(),center=(-Rc.T@tc).tolist())],
        surfaces=[dict(p=allgroups[k]['p'].tolist(),depth_parameters=x[12+3*i:15+3*i].tolist(),selected_b=s[0],selected_c=s[1]) for i,(k,s) in enumerate(zip(train,final_choices))],
        limitations='Joint local-depth and camera fit with reversible regional choices. Held source regions predict C from A/B only after camera fitting. Correspondence discovery and initial pair hypotheses used these photographs; this is conditional validation, not independent scene truth.')
