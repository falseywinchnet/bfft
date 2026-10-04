"""Conditional multi-origin reconstruction from cross-capture regional tracks."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .multiview_geometry import bearings,unit,triangulate,hypotheses


def skew(b):
    x,y,z=b;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def pose_dlt(points,directions):
    center=points.mean(0);extent=max(float(np.sqrt(np.mean((points-center)**2))),1e-8);x=(points-center)/extent
    matrix=np.concatenate([np.kron(skew(b),np.r_[p,1][None,:]) for p,b in zip(x,directions)])
    _,_,v=np.linalg.svd(matrix,full_matrices=False);p=v[-1].reshape(3,4)
    if np.linalg.det(p[:,:3])<0:p=-p
    u,s,v=np.linalg.svd(p[:,:3]);r=u@np.diag([1,1,np.linalg.det(u@v)])@v;t=p[:,3]/np.mean(s)*extent-r@center
    return r,t


def planar_pose(points,directions):
    """Plane-coordinate pose initializer; evaluate it against actual 3-D points.

The ordinary 3-D DLT is rank deficient on a wall or ground plane. This branch
fits a bearing homography on the measured point plane and returns both signs.
It proposes a camera pose; it does not force the scene onto that plane.
"""
    center=points.mean(0);_,singular,v=np.linalg.svd(points-center,full_matrices=False)
    if len(singular)<2 or singular[1]<1e-8:return []
    basis=np.c_[v[0],v[1],np.cross(v[0],v[1])];xy=(points-center)@basis[:,:2]
    matrix=np.concatenate([np.kron(skew(b),np.r_[q,1][None,:]) for q,b in zip(xy,directions)])
    _,_,vh=np.linalg.svd(matrix,full_matrices=False);h=vh[-1].reshape(3,3);scale=(np.linalg.norm(h[:,0])+np.linalg.norm(h[:,1]))/2
    if scale<1e-12:return []
    candidates=[]
    for sign in (1,-1):
        first=sign*h[:,:2]/scale;raw=np.c_[first,np.cross(first[:,0],first[:,1])];u,_,vv=np.linalg.svd(raw);r=u@np.diag([1,1,np.linalg.det(u@vv)])@vv@basis.T;t=sign*h[:,2]/scale-r@center;candidates.append((r,t))
    return candidates


def pose_errors(points,directions,r,t):
    predicted=unit(points@r.T+t)
    return np.arctan2(np.linalg.norm(np.cross(predicted,directions),axis=1),np.sum(predicted*directions,axis=1))


def fit_pose(points,directions,track_ids,trials=200,threshold=np.deg2rad(1.2),initializations=()):
    hold=np.asarray(track_ids)%7==0;_,first=np.unique(track_ids,return_index=True);eligible=np.sort(first);train=eligible[~hold[eligible]];test=eligible[hold[eligible]];rng=np.random.default_rng(92)
    if len(train)<10 or len(test)<4:return None
    best=None
    # Relative pair poses propose orientation and a baseline direction only.
    # Solve its unknown scale using training rays, then compete with DLT under
    # the identical training score. No proposed camera is accepted from its edge.
    for r,offset,direction in initializations:
        q=points@r.T+offset;u=np.cross(directions,direction);v=np.cross(directions,q)
        scale=-np.sum(u*v,axis=1)/np.maximum(np.sum(u*u,axis=1),1e-12)
        for value in np.quantile(scale[train],np.linspace(.05,.95,31)):
            t=offset+value*direction;error=pose_errors(points,directions,r,t);good=train[error[train]<threshold]
            score=(len(good),-np.median(error[good]) if len(good) else -100)
            if best is None or score>best[0]:best=(score,r,t,good)
    for trial in range(trials):
        if trial%2==0:
            subset4=rng.choice(train,4,replace=False)
            for rp,tp in planar_pose(points[subset4],directions[subset4]):
                pe=pose_errors(points,directions,rp,tp);pg=train[pe[train]<threshold];ps=(len(pg),-np.median(pe[pg]) if len(pg) else -100)
                if best is None or ps>best[0]:best=(ps,rp,tp,pg)
        subset=rng.choice(train,6,replace=False)
        try:r,t=pose_dlt(points[subset],directions[subset]);error=pose_errors(points,directions,r,t)
        except np.linalg.LinAlgError:continue
        good=train[error[train]<threshold];score=(len(good),-np.median(error[good]) if len(good) else -100)
        if best is None or score>best[0]:best=(score,r,t,good)
    if best is None or len(best[3])<10:return None
    _,r,t,good=best;initial=np.r_[Rotation.from_matrix(r).as_rotvec(),t]
    def residual(p):
        rot=Rotation.from_rotvec(p[:3]).as_matrix();return (unit(points[good]@rot.T+p[3:])-directions[good]).ravel()
    opt=least_squares(residual,initial,loss='soft_l1',f_scale=.01,max_nfev=80)
    r=Rotation.from_rotvec(opt.x[:3]).as_matrix();t=opt.x[3:];errors=pose_errors(points,directions,r,t);good=train[errors[train]<threshold];checked=test[errors[test]<threshold]
    # Single final gate: no trial is refitted or selected using this holdout.
    accepted=len(good)>=10 and len(checked)>=4 and len(checked)>=.6*len(test)
    return dict(accepted=bool(accepted),rotation=r,translation=t,center=-r.T@t,train=len(good),holdout=len(checked),holdout_total=len(test),
        holdout_median_degrees=float(np.rad2deg(np.median(errors[test]))),median_degrees=float(np.rad2deg(np.median(errors))),errors=errors)


def track_bearing(observation,cameras):
    c=cameras[observation['capture']];return bearings([observation['xy']],c['shape'],c['focal'],c['projection'],focal_x=c.get('focal_x'),focal_y=c.get('focal_y'),principal_point=c.get('principal_point'))[0]


def reconstruct(evidence,maximum_seed_trials=6,seed_pair=None,allow_photo_seeds=False):
    cameras=evidence['cameras'];tracks=evidence['tracks'];site_lookup={(o['capture'],o['site']):o['xy'] for track in tracks for o in track['observations']};init_cache={};lookup={t['id']:{o['capture']:o for o in t['observations']} for t in tracks};pairs=[]
    for ei,e in enumerate(evidence['edges']):
        if seed_pair is not None and (e['a'],e['b'])!=tuple(seed_pair):continue
        if e['status']!='translation_supported':continue
        if not allow_photo_seeds and not all(evidence['records'][k]['panoramic'] for k in (e['a'],e['b'])):continue
        common=[t for t in tracks if e['a'] in lookup[t['id']] and e['b'] in lookup[t['id']] and any(e['a'] in c and e['b'] in c for c in t['cycle_components'])]
        if len(common)<18:continue
        m=e['models']['essential'];pairs.append((len(common),m['heldout_count'],ei,common))
    pairs.sort(key=lambda row:(-row[0],-row[1],row[2]));audit=[];best=None
    for _,_,ei,cycle_common in pairs[:maximum_seed_trials]:
        e=evidence['edges'][ei];a,b=e['a'],e['b']
        common=[];used_a=set();used_b=set()
        for track in tracks:
            observations=lookup[track['id']]
            if a not in observations or b not in observations:continue
            sa,sb=observations[a]['site'],observations[b]['site']
            if sa in used_a or sb in used_b:continue
            used_a.add(sa);used_b.add(sb);common.append(track)
        pose=e['models']['essential']['poses'][0];r=np.array(pose['rotation']);t=np.array(pose['translation'])
        aa=np.array([track_bearing(lookup[x['id']][a],cameras) for x in common]);bb=np.array([track_bearing(lookup[x['id']][b],cameras) for x in common]);xyz,da,db,miss,angles=triangulate(aa,bb,r,t)
        valid=(da>0)&(db>0)&(angles>np.deg2rad(.7))&(miss/np.maximum(np.minimum(da,db),1e-8)<.025)
        cloud={common[k]['id']:xyz[k] for k in np.flatnonzero(valid)};poses={a:dict(rotation=np.eye(3),translation=np.zeros(3),center=np.zeros(3),role='seed_gauge'),b:dict(rotation=r,translation=t,center=-r.T@t,role='unit_baseline_seed')};checks=[];support_audit=[]
        # Each candidate camera is tested against already reconstructed tracks.
        # A rejected camera does not cause a refit against its held-out points.
        remaining=set(range(len(cameras)))-set(poses)
        for round_index in range(4):
            proposals=[]
            for i in sorted(remaining):
                ids=[tid for tid in cloud if i in lookup[tid]]
                if round_index==0:support_audit.append(dict(capture=i,available_tracks=len(ids),holdout_tracks=sum(lookup[tid][i]['site']%7==0 for tid in ids)))
                if len(ids)<16:continue
                directions=np.array([track_bearing(lookup[tid][i],cameras) for tid in ids]);p=np.array([cloud[tid] for tid in ids]);initial=[]
                for edge_index,edge in enumerate(evidence['edges']):
                    if i not in (edge['a'],edge['b']) or edge['selected_model'] is None:continue
                    other=edge['b'] if edge['a']==i else edge['a']
                    if other not in poses or 'essential' not in edge['models']:continue
                    # Exclude every candidate touching a held-out target site
                    # before estimating even the initialization. Previously saved
                    # pair rotations have seen those sites and are not a holdout.
                    excluded={lookup[tid][i]['site'] for tid in ids if lookup[tid][i]['site']%7==0}
                    key=(edge_index,i,tuple(sorted(excluded)))
                    if key not in init_cache:
                        column=0 if edge['a']==i else 1
                        cc=[[*c,k] for k,c in enumerate(edge['candidates']) if c[column] not in excluded and (edge['a'],c[0]) in site_lookup and (edge['b'],c[1]) in site_lookup]
                        models={}
                        if len(cc)>=20:
                            ca,cb=cameras[edge['a']],cameras[edge['b']]
                            ba=bearings([edge['a_points'][c[4]] if 'a_points' in edge else site_lookup[(edge['a'],c[0])] for c in cc],ca['shape'],ca['focal'],ca['projection'])
                            bb=bearings([edge['b_points'][c[4]] if 'b_points' in edge else site_lookup[(edge['b'],c[1])] for c in cc],cb['shape'],cb['focal'],cb['projection'])
                            models=hypotheses(ba,bb,np.array([c[0] for c in cc]),np.array([c[1] for c in cc]),trials=400)
                        init_cache[key]=models.get('essential',{}).get('poses',[])
                    for ep in init_cache[key]:
                        re=np.asarray(ep['rotation']);te=np.asarray(ep['translation'])
                        if edge['a']==i:te=-re.T@te;re=re.T
                        initial.append((re@poses[other]['rotation'],re@poses[other]['translation'],te))
                result=fit_pose(p,directions,np.array([lookup[tid][i]['site'] for tid in ids]),initializations=initial)
                if result is None:continue
                checks.append(dict(capture=i,round=round_index,**{k:v for k,v in result.items() if k not in ('rotation','translation','center','errors')}))
                if result['accepted']:proposals.append((i,result))
            if not proposals:break
            for i,result in proposals:
                poses[i]={k:result[k] for k in ('rotation','translation','center','train','holdout','holdout_total','holdout_median_degrees')};poses[i]['role']='heldout_pose';remaining.remove(i)
            # Add a track only if its rays intersect consistently in >=3 solved
            # captures; two-camera ties alone cannot expand the accepted cloud.
            for tid,obs in lookup.items():
                if tid in cloud:continue
                available=[i for i in obs if i in poses]
                if len(available)<3:continue
                centers=np.array([poses[i]['center'] for i in available]);rays=np.array([track_bearing(obs[i],cameras)@poses[i]['rotation'] for i in available]);projector=np.eye(3)[None,:,:]-rays[:,:,None]*rays[:,None,:]
                lhs=projector.sum(0)
                if np.linalg.cond(lhs)>1e5:continue
                x=np.linalg.solve(lhs,np.einsum('nij,nj->i',projector,centers));d=x-centers;depth=np.sum(d*rays,axis=1);errors=np.arctan2(np.linalg.norm(np.cross(unit(d),rays),axis=1),np.sum(unit(d)*rays,axis=1))
                if np.all(depth>0) and errors.max()<np.deg2rad(1.2):cloud[tid]=x
        check=dict(seed=[a,b],cycle_seed_tracks=len(cycle_common),seed_tracks=int(valid.sum()),third_view_support=sorted(support_audit,key=lambda row:-row['available_tracks'])[:10],poses=len(poses),points=len(cloud),pose_checks=checks);audit.append(check)
        if best is None or (len(poses),len(cloud))>(len(best['poses']),len(best['cloud'])):best=dict(seed=[a,b],poses=poses,cloud=cloud)
    if best is None:return dict(accepted=False,reason='no_eligible_seed_with_sufficient_cycle_supported_tracks',audit=audit)
    # The seed alone is not validated reconstruction. Require a third view
    # passing an untouched pose holdout before presenting a conditional cloud.
    return dict(accepted=len(best['poses'])>=3,seed=best['seed'],poses={str(i):{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()} for i,p in best['poses'].items()},
        points=[dict(track=tid,xyz=x.tolist()) for tid,x in best['cloud'].items()],audit=audit,
        limitations='Conditional partial reconstruction with independent capture origins and nominal intrinsics. Scale is the arbitrary seed baseline; positions are not georeferenced. The pose holdout uses physical target-site IDs, stable across track graph changes. Target holdout sites are excluded from pose initialization and refinement. Track discovery used the collection, and seed selection uses these predictive checks, so the resulting winner needs a further independent audit. No fusion or visibility claim follows from sparse points alone.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();start=time.perf_counter();e=json.loads(args.evidence.read_text());result=reconstruct(e);result['seconds']=time.perf_counter()-start;result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('points','poses','audit')}));print(json.dumps({'poses':len(result.get('poses',{})),'points':len(result.get('points',[]))}))
if __name__=='__main__':main()
