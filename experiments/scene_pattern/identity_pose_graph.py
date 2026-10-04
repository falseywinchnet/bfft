"""Keep relative-camera alternatives; test three-view rotation and baseline loops."""
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from .multiview_geometry import triangulate,bearings,unit
from .affine_camera_geometry import measurements,epi_error
from .identity_geometry import roles


def angle(R):return float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))


def collect(folder):
    pairs={}
    for file in sorted(Path(folder).glob('[0-9][0-9][0-9]-[0-9][0-9][0-9].json')):
        row=json.loads(file.read_text());a,b=row['a'],row['b'];key=tuple(sorted([a,b]))
        for k,m in enumerate(row['models']):
            if not m['accepted'] or m['kind']!='essential' or not m.get('translation_observed'):continue
            p=m['poses'][0];R=np.array(p['rotation']);t=np.array(p['translation'])
            if a>b:t=-R.T@t;R=R.T
            candidate=dict(R=R,t=t,source=file.name,model=k,train=m['train_count'],check=m['check_count'])
            if any(angle(R@old['R'].T)<.002 and np.linalg.norm(t-old['t'])<.02 for old in pairs.get(key,[])):continue
            pairs.setdefault(key,[]).append(candidate)
    return pairs


def loops(pairs):
    captures=sorted({i for pair in pairs for i in pair});receipts=[]
    for a,b,c in itertools.combinations(captures,3):
        if not all(key in pairs for key in ((a,b),(b,c),(a,c))):continue
        for i,j,k in itertools.product(range(len(pairs[a,b])),range(len(pairs[b,c])),range(len(pairs[a,c]))):
            ab,bc,ac=pairs[a,b][i],pairs[b,c][j],pairs[a,c][k]
            error=angle(bc['R']@ab['R']@ac['R'].T)
            if error>np.deg2rad(1.2):continue
            dab=-ab['R'].T@ab['t'];dbc=-ab['R'].T@bc['R'].T@bc['t'];dac=-ac['R'].T@ac['t'];matrix=np.column_stack((dab,dbc));scales,residual=nnls(matrix,dac-matrix@np.full(2,.001));scales+=.001
            receipts.append(dict(captures=[a,b,c],models=[i,j,k],rotation_closure_degrees=float(np.rad2deg(error)),direction_residual=float(residual),positive_direction_scales=scales.tolist(),direction_rank=int(np.linalg.matrix_rank(matrix,tol=.01)),
                direction_compatible=bool(residual<=.1 and np.all(scales>=.001) and np.all(scales<50))))
    return receipts


def ray(xy,c):return bearings(np.asarray(xy),c['shape'],c['focal'],c['projection'],c.get('focal_x'),c.get('focal_y'),c.get('principal_point'))


def scale_candidates(rows_ab,rows_ac,cameras,ab,ac,capture):
    """Shared source pixels create alternatives, never an irreversible track union."""
    if not rows_ab or not rows_ac:return [],0
    aa={};cc={}
    from .affine_camera_geometry import skew
    for rows,cam,p,groups in [(rows_ab,cameras[:2],ab,aa),(rows_ac,[cameras[0],cameras[2]],ac,cc)]:
        err=epi_error(measurements(rows,cam),skew(p['t'])@p['R'])
        for i in np.flatnonzero(err<=1.5):groups.setdefault(tuple(rows[i]['p']),[]).append(rows[i])
    result=[];shared=sorted(aa.keys()&cc.keys())
    for site,pixel in enumerate(shared):
        ba=ray([pixel],cameras[0])[0]
        training,held,_=roles([dict(p=pixel)],capture)
        for rab,rac in itertools.product(aa[pixel],cc[pixel]):
            bb=ray([rab['q']],cameras[1]);X,da,db,miss,parallax=triangulate(ba[None],bb,ab['R'],ab['t'])
            if da[0]<=0 or db[0]<=0 or parallax[0]<np.deg2rad(.7):continue
            bc=ray([rac['q']],cameras[2])[0];base=ac['R']@X[0];u=np.cross(bc,ac['t']);v=np.cross(bc,base);den=u@u
            if den<1e-8:continue
            scale=float(-(u@v)/den)
            if not .01<scale<100:continue
            result.append(dict(site=site,p=list(pixel),X=X[0],ray=bc,scale=scale,training=bool(training[0]),held=bool(held[0]),
                qa=rab['q'],qc=rac['q'],Aab=rab['A'],Aac=rac['A'],appearance=min(rab['training'],rac['training'])))
    return result,len(shared)


def fit_scale(candidates,ac,camera,maximum=3):
    """Select scale modes from training identities; predict reserved source tiles."""
    train=[i for i,r in enumerate(candidates) if r['training']];held=[i for i,r in enumerate(candidates) if r['held']]
    if len({candidates[i]['site'] for i in train})<8:return []
    X=np.array([r['X'] for r in candidates]);rays=np.array([r['ray'] for r in candidates]);scales=np.array([r['scale'] for r in candidates]);base=X@ac['R'].T;focal=camera['focal']
    def errors(scale):
        v=base+scale*ac['t'];dot=np.sum(v*rays,axis=1);unit_v=unit(v);err=np.arctan2(np.linalg.norm(np.cross(unit_v,rays),axis=1),np.sum(unit_v*rays,axis=1))*focal;err[dot<=0]=1e6;return err
    def select(err,ids):
        used=set();good=[]
        for i in sorted(ids,key=lambda i:(err[i],-candidates[i]['appearance'])):
            if err[i]<=1.5 and candidates[i]['site'] not in used:used.add(candidates[i]['site']);good.append(i)
        return good
    proposals=[]
    # Fixed log-scale bins propose multiple modes; dense variants cannot add votes.
    for key in sorted(set(np.round(np.log(scales[train])/.08).astype(int))):
        bucket=[i for i in train if int(np.round(np.log(scales[i])/.08))==key]
        s=float(np.median(scales[bucket]));good=select(errors(s),train)
        if len(good)<8:continue
        for _ in range(3):
            if not good:break
            s=float(np.median(scales[good]));good=select(errors(s),train)
        if len(good)<8:continue
        err=errors(s);good=select(err,train);checked=select(err,held);score=(len(good),-float(np.median(err[good])) if good else -1e10)
        proposals.append(dict(scale=s,good=good,check=checked,score=score,errors=err))
    kept=[]
    for p in sorted(proposals,key=lambda r:r['score'],reverse=True):
        if any(abs(np.log(p['scale']/r['scale']))<.08 for r in kept):continue
        kept.append(p)
        if len(kept)>=maximum:break
    output=[];nheld=len({candidates[i]['site'] for i in held})
    for p in kept:
        def blocks(ids):return len({tuple(np.floor(np.array(candidates[i]['p'])/48).astype(int)) for i in ids})
        accepted=len(p['good'])>=8 and blocks(p['good'])>=3 and len(p['check'])>=4 and blocks(p['check'])>=2 and len(p['check'])>=.6*nheld
        output.append(dict(scale=p['scale'],accepted=bool(accepted),train_count=len(p['good']),train_tiles=blocks(p['good']),check_count=len(p['check']),check_tiles=blocks(p['check']),check_total=nheld,
            check_median=float(np.median(p['errors'][p['check']])) if p['check'] else None,
            selected_training=p['good'],selected_check=p['check']))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--fields',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();start=time.perf_counter();pairs=collect(args.geometry);cyclic=loops(pairs);policy=json.loads((args.geometry/'run-policy.json').read_text());cameras=policy['cameras'];results=[];field_cache={}
    for loop in cyclic:
        if not loop['direction_compatible']:continue
        a,b,c=loop['captures'];i,j,k=loop['models'];ab,bc,ac=pairs[a,b][i],pairs[b,c][j],pairs[a,c][k]
        for key in ((a,b),(a,c)):
            if key not in field_cache:field_cache[key]=json.loads((args.fields/f'{key[0]:03d}-{key[1]:03d}.json').read_text())['rows']
        candidates,shared=scale_candidates(field_cache[a,b],field_cache[a,c],[cameras[a],cameras[b],cameras[c]],ab,ac,a);fits=fit_scale(candidates,ac,cameras[c])
        for fit in fits:
            ratio=fit['scale'];predicted_t=ratio*ac['t']-bc['R']@ab['t'];direction=float(np.rad2deg(np.arccos(np.clip(unit(predicted_t)@bc['t'],-1,1))))
            fit['third_baseline_direction_degrees']=direction;fit['accepted']=bool(fit['accepted'] and direction<=5.)
        results.append(dict(**loop,shared_regions=shared,candidates=len(candidates),scale_hypotheses=fits))
    result=dict(pair_hypotheses={f'{a}-{b}':[{**r,'R':r['R'].tolist(),'t':r['t'].tolist()} for r in rows] for (a,b),rows in pairs.items()},orientation_loops=cyclic,triad_fits=results,
        summary=dict(relative_pairs=len(pairs),captures=len({i for key in pairs for i in key}),orientation_loops=len(cyclic),direction_compatible_loops=sum(r['direction_compatible'] for r in cyclic),triad_scale_hypotheses=sum(len(r['scale_hypotheses']) for r in results),accepted_triad_hypotheses=sum(s['accepted'] for r in results for s in r['scale_hypotheses']),seconds=time.perf_counter()-start),
        scene_promoted=False,source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('identity_pose_graph.py','identity_geometry.py','affine_camera_geometry.py')},
        limitations='Conditional camera and scale alternatives. Source-tile scale checks are conditioned on prior image matching and pair-model discovery. No global scene or fixed station labels are promoted.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']))
if __name__=='__main__':main()
