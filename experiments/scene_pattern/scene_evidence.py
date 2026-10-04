"""Multi-view regional hypotheses, angular geometry and ambiguity-aware tracks.

This runner does not load any failed atlas coordinates or mesh fields.
"""
import argparse,hashlib,json,time,platform
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from .multiview_geometry import bearings,hypotheses,essential_errors,rotation_errors,essential_poses
from .sweep_graph import load_features,components

POLICY=dict(version=2,alternative_matches=2,maximum_descriptor_distance=.27,
    angular_threshold_degrees=1.2,ransac_trials=400,holdout='physical reference-site ID modulo seven',
    min_train=12,min_holdout=4,max_holdout_permutation_p=.05,min_cheirality=.8,min_parallax_degrees=.7,
    panorama_projection='cylindrical',intrinsics='28mm-equivalent EXIF with nominal uncropped sensor diagonal; panorama uses native frame height, not stitched width',
    capture_centers='independent, not forced into one station or straight line',seed=71)


def canonical_features(path):
    a=load_features(path);points,ids=np.unique(np.round(a['points'],4),axis=0,return_inverse=True)
    a['sites']=points;a['site_ids']=ids;return a


def candidates(a,b):
    da,db=a['descriptors'],b['descriptors'];oa=np.argsort(a['site_ids']);ob=np.argsort(b['site_ids']);ia=a['site_ids'][oa];ib=b['site_ids'][ob]
    sa=np.r_[0,np.flatnonzero(np.diff(ia))+1];sb=np.r_[0,np.flatnonzero(np.diff(ib))+1]
    d=cdist(da[oa],db[ob],'sqeuclidean');d=np.minimum.reduceat(np.minimum.reduceat(d,sb,axis=1),sa,axis=0)
    top=np.argsort(d,axis=1)[:,:2];back=np.argsort(d,axis=0)[:2,:];rows=[]
    for i,js in enumerate(top):
        for rank,j in enumerate(js):
            if d[i,j]>.27 or i not in back[:,j]:continue
            if rank and d[i,j]>d[i,js[0]]+.04:continue
            rows.append((i,int(j),float(d[i,j]),rank))
    return rows


def camera_models(records,metadata,features):
    meta={r['name']:r for r in metadata};ordinary=[r['original_size'] for r in records if not r['panoramic']];sensor=np.median(ordinary,axis=0);diagonal=np.linalg.norm(sensor)
    result=[]
    for row,f in zip(records,features):
        focal35=meta[row['name']]['focal_35mm'];h,w=f['shape']
        focal=focal35/np.hypot(36,24)*diagonal*h/row['original_size'][1] if row['panoramic'] else focal35/np.hypot(36,24)*np.hypot(h,w)
        result.append(dict(shape=[int(h),int(w)],focal=float(focal),projection='cylindrical' if row['panoramic'] else 'perspective',calibration_status='nominal_unvalidated'))
    return result


def model_support(models):
    eligible={}
    for name,m in models.items():
        if m['train_count']>=POLICY['min_train'] and m['heldout_count']>=POLICY['min_holdout'] and m['heldout_permutation_p']<=.05:
            eligible[name]=m
    if not eligible:return 'unresolved',None
    rot=eligible.get('rotation');epi=eligible.get('essential')
    if epi:
        pose=epi['poses'][0];parallax=np.rad2deg(pose['median_parallax'])
        if pose['positive_fraction']>=.8 and parallax>=.7 and (rot is None or epi['heldout_count']>=rot['heldout_count']+3):
            return 'translation_supported','essential'
    if rot:return 'rotation_compatible','rotation'
    return 'ambiguous_geometry','essential'



def cycle_components(vertices,edges):
    adj={v:set() for v in vertices}
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    bridges=set();seen={};low={};counter=[0]
    def visit(v,parent=None):
        seen[v]=low[v]=counter[0];counter[0]+=1
        for w in adj[v]:
            if w==parent:continue
            if w not in seen:
                visit(w,v);low[v]=min(low[v],low[w])
                if low[w]>seen[v]:bridges.add(tuple(sorted((v,w))))
            else:low[v]=min(low[v],seen[w])
    for v in vertices:
        if v not in seen:visit(v)
    unseen=set(vertices);groups=[]
    while unseen:
        todo=[min(unseen)];unseen.remove(todo[0]);group=[]
        while todo:
            v=todo.pop();group.append(v)
            for w in adj[v]&unseen:
                if tuple(sorted((v,w))) not in bridges:todo.append(w);unseen.remove(w)
        if len(group)>=3:groups.append(sorted(group))
    return groups

def build_tracks(edges,features):
    """Union only unambiguous endpoints; record capture-identity conflicts."""
    links=[];alternatives=0
    for ei,e in enumerate(edges):
        selected=e['selected_model']
        if selected is None:continue
        m=e['models'][selected];ids=m['train_inliers']+m['heldout_inliers'];multiplicity={}
        for k in ids:
            row=e['candidates'][k];key=row[0];multiplicity[key]=multiplicity.get(key,0)+1
        for k in ids:
            sa,sb,distance,rank=e['candidates'][k]
            links.append((e['status']=='translation_supported',m['heldout_count'], -distance,(e['a'],sa),(e['b'],sb),ei,k))
        alternatives+=sum(row[3]>0 for row in e['candidates'])
    parent={};members={};conflicts=[];retained=[]
    def root(n):
        if n not in parent:parent[n]=n;members[n]={n[0]:n[1]}
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    for *_,a,b,ei,k in sorted(links,reverse=True):
        ra,rb=root(a),root(b)
        if ra==rb:retained.append((a,b,ei,k));continue
        overlap=members[ra].keys()&members[rb].keys()
        if any(members[ra][i]!=members[rb][i] for i in overlap):
            conflicts.append(dict(a=a,b=b,edge=ei,candidate=k,reason='would_merge_distinct_sites_in_one_capture'));continue
        if len(members[ra])<len(members[rb]):ra,rb=rb,ra
        parent[rb]=ra;members[ra].update(members.pop(rb));retained.append((a,b,ei,k))
    groups={}
    for node in parent:groups.setdefault(root(node),[]).append(node)
    tracks=[]
    for nodes in sorted(groups.values(),key=lambda ns:(-len(ns),sorted(ns))):
        if len(nodes)<3:continue
        nodes=sorted(nodes);keyset=set(nodes);ls=[r for r in retained if r[0] in keyset and r[1] in keyset]
        captures={n[0] for n in nodes};edge_pairs={tuple(sorted((a[0],b[0]))) for a,b,_,_ in ls};cycles=max(0,len(edge_pairs)-len(captures)+1)
        cyclic=cycle_components(captures,edge_pairs)
        tracks.append(dict(cycle_components=cyclic,id=len(tracks),observations=[dict(capture=i,site=s,xy=features[i]['sites'][s].tolist()) for i,s in nodes],links=len(edge_pairs),independent_cycles=cycles))
    return tracks,conflicts,alternatives


def associations(records,tracks):
    rows=[]
    for i,row in enumerate(records):
        if row['panoramic']:continue
        scores={}
        for t in tracks:
            ids={o['capture'] for o in t['observations']}
            if i not in ids:continue
            for j in ids:
                if records[j]['panoramic']:
                    s=scores.setdefault(j,dict(panorama=j,tracks=0,cycle_supported_tracks=0));s['tracks']+=1;s['cycle_supported_tracks']+=any(i in group and j in group for group in t['cycle_components'])
        rows.append(dict(capture=i,candidates=sorted(scores.values(),key=lambda s:(-s['cycle_supported_tracks'],-s['tracks'],s['panorama']))[:8],assignment='uncommitted_overlap_candidates'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--candidate-graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',type=int,default=0);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    records=json.loads((args.data/'manifest.json').read_text());metadata=json.loads((args.data/'camera-metadata.json').read_text());features=[canonical_features(args.features/(Path(r['name']).stem+'.npz')) for r in records];cameras=camera_models(records,metadata,features)
    # Old graph supplies only the candidate-pair inventory, never its matrices,
    # acceptance decisions, mesh fields, fused pixels or selected point subset.
    inventory=json.loads(args.candidate_graph.read_text());pairs=[(e['a'],e['b']) for e in inventory['edges']]
    if args.probe:pairs=pairs[:args.probe]
    edges=[];ledger=args.out/'pairs.jsonl';done=set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():e=json.loads(line);edges.append(e);done.add((e['a'],e['b']))
    policy_hash=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()+Path(__file__).read_bytes()+Path(__file__).with_name('multiview_geometry.py').read_bytes()).hexdigest()
    receipt_path=args.out/'policy.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text())['sha256']!=policy_hash:raise RuntimeError('policy changed: use a new output directory')
    receipt_path.write_text(json.dumps(dict(policy=POLICY,sha256=policy_hash),indent=2))
    with ledger.open('a') as stream:
        for a,b in pairs:
            if (a,b) in done:continue
            cc=candidates(features[a],features[b]);models={}
            if len(cc)>=20:
                sa=np.array([c[0] for c in cc]);sb=np.array([c[1] for c in cc]);ca,cb=cameras[a],cameras[b]
                aa=bearings(features[a]['sites'][sa],ca['shape'],ca['focal'],ca['projection']);bb=bearings(features[b]['sites'][sb],cb['shape'],cb['focal'],cb['projection'])
                models=hypotheses(aa,bb,sa,sb,trials=400)
            status,selected=model_support(models);e=dict(a=a,b=b,candidates=cc,models=models,status=status,selected_model=selected);edges.append(e);stream.write(json.dumps(e)+'\n');stream.flush()
            if len(edges)%20==0:print(json.dumps({'pairs':len(edges),'total':len(pairs),'supported':sum(e['selected_model'] is not None for e in edges),'seconds':time.perf_counter()-start}),flush=True)
    tracks,conflicts,alternatives=build_tracks(edges,features);affinity=associations(records,tracks)
    receipt=dict(policy=POLICY,policy_sha256=policy_hash,records=records,cameras=cameras,edges=edges,tracks=tracks,conflicts=conflicts,alternative_candidates=alternatives,associations=affinity,
        seconds=time.perf_counter()-start,runtime=dict(python=platform.python_version(),numpy=np.__version__),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('multiview_geometry.py')]},
        limitations='Multi-view evidence under nominal intrinsics. Associations are overlapping-scene candidates, not station assignments. Rotation compatibility does not prove a shared center; essential support does not establish metric depth. No failed atlas coordinates are used.')
    (args.out/'evidence.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'tracks':len(tracks),'cycle_tracks':sum(t['independent_cycles']>0 for t in tracks),'conflicts':len(conflicts),'supported_pairs':sum(e['selected_model'] is not None for e in edges),'seconds':receipt['seconds']}),flush=True)
if __name__=='__main__':main()
