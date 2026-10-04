"""Close independently fitted affine region loops and synchronize their points."""
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from .multiview_geometry import bearings,hypotheses
from .scene_evidence import model_support,cycle_components,associations
from .scene_pose import reconstruct
from .scene_promotion import audit


def close_triangle(ab,bc,ac):
    predicted=bc['q']+bc['A']@(ab['q']-bc['p'])
    delta=predicted-ac['q'];scale=max(np.sqrt(abs(np.linalg.det(ac['A']))),.25)
    closure=float(np.linalg.norm(delta)/scale)
    jac=float(np.linalg.norm(bc['A']@ab['A']-ac['A'])/max(np.linalg.norm(ac['A']),.25))
    return closure,jac


def synchronize(nodes,links):
    nodes=sorted(nodes);anchor=nodes[0];unknown={n:k for k,n in enumerate(nodes[1:])};centers={}
    for l in links:centers[l['a']]=l['p'];centers[l['b']]=l['site_q']
    matrix=[];rhs=[]
    for l in links:
        m=np.zeros((2,len(unknown)*2));v=l['q']-l['A']@l['p']
        for n,a in ((l['a'],-l['A']),(l['b'],np.eye(2))):
            if n==anchor:v=v-a@centers[anchor]
            else:m[:,2*unknown[n]:2*unknown[n]+2]+=a
        weight=max(l['correlation'],.1);matrix.append(m*weight);rhs.append(v*weight)
    x=np.linalg.lstsq(np.concatenate(matrix),np.concatenate(rhs),rcond=None)[0];positions={anchor:centers[anchor]}
    positions.update({n:x[2*k:2*k+2] for n,k in unknown.items()})
    residuals=[float(np.linalg.norm(positions[l['b']]-(l['q']+l['A']@(positions[l['a']]-l['p'])))) for l in links]
    shift=max(float(np.linalg.norm(positions[n]-centers[n])) for n in nodes)
    return positions,dict(maximum_residual=max(residuals),median_residual=float(np.median(residuals)),maximum_center_shift=shift)


def build(folder):
    meta=json.loads((folder/'transport.json').read_text());records=meta['records'];links={};pair_index={};site_positions={}
    for row in meta['pairs']:
        a,b=row['a'],row['b'];file=folder/f'{a:03d}-{b:03d}.npz'
        if not file.exists():continue
        with np.load(file) as f:
            for k in np.flatnonzero(f['well_localized']):
                sa,sb=map(int,f['site_ids'][k]);key=(a,b,sa,sb)
                l=dict(a=(a,sa),b=(b,sb),p=f['source'][k],site_q=f['target_site'][k],q=f['center'][k],A=f['affine'][k],correlation=float(f['validation_correlation'][k]),reverse_q=f['reverse_center'][k],reverse_A=f['reverse_affine'][k])
                links[key]=l;pair_index.setdefault((a,b),{}).setdefault(sa,{})[sb]=key;site_positions[a,sa]=l['p'];site_positions[b,sb]=l['site_q']
    triangles=[];triangle_attempts=0
    captures=sorted({i for p in pair_index for i in p})
    for a,b,c in itertools.combinations(captures,3):
        ab=pair_index.get((a,b));bc=pair_index.get((b,c));ac=pair_index.get((a,c))
        if not ab or not bc or not ac:continue
        for sa in ab.keys()&ac.keys():
            for sb,kab in ab[sa].items():
                for sc in bc.get(sb,{}).keys()&ac[sa].keys():
                    keys=(kab,bc[sb][sc],ac[sa][sc]);ls=[links[k] for k in keys];error,jac=close_triangle(*ls);triangle_attempts+=1
                    if error<=1.5 and jac<=.45:triangles.append(dict(nodes=((a,sa),(b,sb),(c,sc)),keys=keys,closure=error,jacobian=jac,quality=min(l['correlation'] for l in ls)))
    parent={};members={};retained=[];conflicts=[]
    def root(n):
        if n not in parent:parent[n]=n;members[n]={n[0]:n[1]}
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    for tri in sorted(triangles,key=lambda t:(-t['quality'],t['closure'],t['nodes'])):
        roots={root(n) for n in tri['nodes']};merged={};bad=False
        for r in roots:
            for capture,site in members[r].items():
                if capture in merged and merged[capture]!=site:bad=True
                merged[capture]=site
        if bad:conflicts.append(dict(nodes=tri['nodes'],reason='alternative_regions_in_same_capture'));continue
        chosen=min(roots)
        for r in roots:
            if r!=chosen:parent[r]=chosen;members.pop(r)
        members[chosen]=merged;retained.append(tri)
    groups={}
    for n in parent:groups.setdefault(root(n),set()).add(n)
    group_links={r:set() for r in groups}
    for tri in retained:group_links[root(tri['nodes'][0])].update(tri['keys'])
    tracks=[];failed=[]
    for r,nodes in sorted(groups.items(),key=lambda row:(-len(row[1]),sorted(row[1]))):
        if len(nodes)<3 or not group_links[r]:continue
        keys=sorted(group_links[r]);ls=[links[k] for k in keys]
        # Both directions were fitted independently. Keep both in synchronization.
        reverse=[dict(a=l['b'],b=l['a'],p=l['site_q'],site_q=l['p'],q=l['reverse_q'],A=l['reverse_A'],correlation=l['correlation']) for l in ls]
        xy,diagnostics=synchronize(nodes,ls+reverse)
        if diagnostics['maximum_residual']>2.5 or diagnostics['maximum_center_shift']>12:
            failed.append(dict(nodes=sorted(nodes),**diagnostics));continue
        edge_pairs={(l['a'][0],l['b'][0]) for l in ls};cyclic=cycle_components({n[0] for n in nodes},edge_pairs)
        tracks.append(dict(id=len(tracks),observations=[dict(capture=n[0],site=n[1],xy=xy[n].tolist(),centroid=site_positions[n].tolist()) for n in sorted(nodes)],
            cycle_components=cyclic,independent_cycles=len(edge_pairs)-len(nodes)+1,links=len(edge_pairs),transport_residual=diagnostics,transport_links=[dict(a=l['a'],b=l['b'],p=l['p'].tolist(),q=l['q'].tolist(),affine=l['A'].tolist(),correlation=l['correlation']) for l in ls]))
    # Fit bearing geometry from synchronized transported points, not centroids.
    bypair={}
    for track in tracks:
        for x,y in itertools.combinations(track['observations'],2):bypair.setdefault((x['capture'],y['capture']),[]).append((x,y))
    edges=[];cameras=meta['cameras']
    for (a,b),observations in sorted(bypair.items()):
        ca,cb=cameras[a],cameras[b];sa=np.array([x['site'] for x,y in observations]);sb=np.array([y['site'] for x,y in observations]);aa=bearings([x['xy'] for x,y in observations],ca['shape'],ca['focal'],ca['projection']);bb=bearings([y['xy'] for x,y in observations],cb['shape'],cb['focal'],cb['projection']);models=hypotheses(aa,bb,sa,sb,trials=400) if len(observations)>=20 else {};status,selected=model_support(models)
        edges.append(dict(a=a,b=b,candidates=[[int(x),int(y),0.,0] for x,y in zip(sa,sb)],models=models,status=status,selected_model=selected))
    summary=dict(direct_localized_maps=len(links),triangle_attempts=triangle_attempts,closed_triangles=len(triangles),retained_triangles=len(retained),conflicting_triangles=len(conflicts),synchronized_tracks=len(tracks),failed_synchronizations=len(failed),observed_captures=len({o['capture'] for t in tracks for o in t['observations']}))
    return dict(records=records,cameras=cameras,tracks=tracks,edges=edges,conflicts=conflicts,failed_synchronizations=failed,summary=summary,associations=associations(records,tracks),policy=meta['policy'],limitations='Regional affine transport and continuous cycle closure. Local maps synchronize physical pattern coordinates before camera fitting. No manual region or inherited atlas. Sparse tracks and nominal camera fits do not certify dense geometry or fusion.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--transport',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();e=build(args.transport);e['seconds']=time.perf_counter()-start;e['source_hashes']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('transport_tracks.py','region_transport.py','multiview_geometry.py','scene_pose.py')};(args.out/'evidence.json').write_text(json.dumps(e,indent=2));print(json.dumps(e['summary']),flush=True)
    pose=audit(e,reconstruct(e));(args.out/'pose.json').write_text(json.dumps(pose,indent=2));print(json.dumps(dict(accepted=pose['accepted'],cameras=len(pose.get('poses',{})),points=len(pose.get('points',[])),promotion=pose.get('promotion_counts',{}),seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
