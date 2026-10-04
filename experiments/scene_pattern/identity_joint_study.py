"""Run all distinct three-view joint hypotheses from an automatically found graph."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .identity_joint import fit
from .affine_camera_geometry import measurements,epi_error,skew
from .identity_geometry import roles,select


def work(spec):
    job,cameras,fields,out=spec;a,b,c=job['captures'];ab,ac=job['ab'],job['ac']
    for p in (ab,ac):p['R']=np.array(p['R']);p['t']=np.array(p['t'])
    rows_ab=json.loads((Path(fields)/f'{a:03d}-{b:03d}.json').read_text())['rows'];rows_ac=json.loads((Path(fields)/f'{a:03d}-{c:03d}.json').read_text())['rows'];start=time.perf_counter()
    result=fit(rows_ab,rows_ac,[cameras[i] for i in (a,b,c)],ab,ac,a,maximum=120,rounds=3)
    if 'poses' in result:
        rb,tb=np.array(result['poses'][1]['rotation']),np.array(result['poses'][1]['translation']);rc,tc=np.array(result['poses'][2]['rotation']),np.array(result['poses'][2]['translation']);R=rc@rb.T;t=tc-R@tb
        rows_bc=json.loads((Path(fields)/f'{b:03d}-{c:03d}.json').read_text())['rows'];err=epi_error(measurements(rows_bc,[cameras[b],cameras[c]]),skew(t)@R) if rows_bc else np.array([])
        train,held,blocks=roles(rows_bc,b);check=select(err,np.flatnonzero(held),rows_bc);total=len({r['site'] for i,r in enumerate(rows_bc) if held[i]});tiles=len({blocks[i] for i in check});passed=len(check)>=4 and tiles>=2 and len(check)>=.6*total
        result['third_pair_check']=dict(passed=bool(passed),count=len(check),total=total,tiles=tiles,median=float(np.median(err[check])) if len(check) else None)
        result['scale_prediction_passed']=result['accepted'];result['accepted']=bool(result['accepted'] and passed)
    result.update(job=job['id'],captures=job['captures'],seconds=time.perf_counter()-start,initial_scale=job['scale']);(Path(out)/f"{job['id']:03d}.json").write_text(json.dumps(result))
    return {k:result.get(k) for k in ('job','captures','accepted','training_regions','check_count','check_total','assignment_changes_from_initial','third_pair_check','seconds')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--fields',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if list(args.out.glob('*.json')):raise ValueError('Use a fresh output directory')
    graph=json.loads(args.graph.read_text());geometry=json.loads((args.geometry/'run-policy.json').read_text());cameras=geometry['cameras'];jobs=[];seen=set()
    for trial in graph['triad_fits']:
        a,b,c=trial['captures'];i,j,k=trial['models'];ab=graph['pair_hypotheses'][f'{a}-{b}'][i];ac=graph['pair_hypotheses'][f'{a}-{c}'][k]
        for s in trial['scale_hypotheses']:
            if s['train_count']<8 or s['train_tiles']<3:continue
            key=(a,b,c,i,k,round(s['scale'],8))
            if key in seen:continue
            seen.add(key);jobs.append(dict(id=len(jobs),captures=[a,b,c],scale=s['scale'],ab=dict(R=ab['R'],t=ab['t']),ac=dict(R=ac['R'],t=(np.array(ac['t'])*s['scale']).tolist())))
    sources=['identity_joint_study.py','identity_joint.py','identity_pose_graph.py','identity_geometry.py','affine_camera_geometry.py'];receipt=dict(jobs=jobs,source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources},graph_sha256=hashlib.sha256(args.graph.read_bytes()).hexdigest(),field_policy_sha256=hashlib.sha256((args.fields/'run-policy.json').read_bytes()).hexdigest())
    (args.out/'run-policy.json').write_text(json.dumps(receipt,indent=2));start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work,[(j,cameras,str(args.fields),str(args.out)) for j in jobs]):rows.append(row);print(json.dumps(row),flush=True)
    summary=dict(jobs=rows,accepted_hypotheses=sum(bool(r['accepted']) for r in rows),seconds=time.perf_counter()-start,scene_promoted=False);(args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}),flush=True)
if __name__=='__main__':main()
