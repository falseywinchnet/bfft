"""Start joint fitting from shared image evidence before demanding pose closure.

Two relative-pose hypotheses and common source regions propose a three-view
fit. The third pair remains a final prediction check, not a prerequisite that
independently fitted poses already agree. No failed pose loop is promoted.
"""
import argparse, hashlib, itertools, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .identity_pose_graph import collect, scale_candidates, fit_scale


def directed(pairs):
    result = {}
    for (a,b), rows in pairs.items():
        result[a,b] = rows
        result[b,a] = [dict(**{k:v for k,v in r.items() if k not in ('R','t')},
                            R=r['R'].T, t=-r['R'].T@r['t']) for r in rows]
    return result


def wedges(pairs, available):
    """All automatic source anchors; no imposed capture ordering or stations."""
    result=[]
    for a in sorted({a for a,b in pairs}):
        neighbors=sorted(b for x,b in pairs if x==a and (a,b) in available)
        for b,c in itertools.combinations(neighbors,2):
            # A third image map can evaluate the prediction even when no
            # independently fitted third-pair camera branch was accepted.
            if (b,c) in available: result.append((a,b,c))
    return result


def work(spec):
    ids, pairs, cameras, folder = spec; a,b,c=ids
    rows_ab=json.loads((Path(folder)/f'{a:03d}-{b:03d}.json').read_text())['rows']
    rows_ac=json.loads((Path(folder)/f'{a:03d}-{c:03d}.json').read_text())['rows']
    result=[]
    for i,ab in enumerate(pairs[a,b]):
        for k,ac in enumerate(pairs[a,c]):
            candidates,shared=scale_candidates(rows_ab,rows_ac,[cameras[j] for j in ids],ab,ac,a)
            fits=fit_scale(candidates,ac,cameras[c])
            # fit_scale selects modes on training regions only. Its check
            # outcomes do not decide which modes may enter the joint fit.
            if fits: result.append(dict(captures=list(ids),models=[i,-1,k],
                                        shared_regions=shared,candidates=len(candidates),
                                        scale_hypotheses=fits,initial_closure_required=False))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('geometry','fields','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);args=p.parse_args();start=time.perf_counter()
    base=collect(args.geometry);pairs=directed(base)
    policy=json.loads((args.geometry/'run-policy.json').read_text());cameras=policy['cameras']
    available={tuple(map(int,f.stem.split('-'))) for f in args.fields.glob('[0-9][0-9][0-9]-[0-9][0-9][0-9].json')}
    jobs=wedges(pairs,available);results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,rows in enumerate(pool.map(work,[(ids,pairs,cameras,str(args.fields)) for ids in jobs])):
            results.extend(rows)
            if (i+1)%10==0:print(json.dumps(dict(wedges_done=i+1,total=len(jobs),seed_branches=len(results),seconds=time.perf_counter()-start)),flush=True)
    result=dict(pair_hypotheses={f'{a}-{b}':[{**r,'R':r['R'].tolist(),'t':r['t'].tolist()} for r in rows] for (a,b),rows in pairs.items()},
                orientation_loops=[],triad_fits=results,
                summary=dict(relative_pairs=len(base),captures=len({i for pair in base for i in pair}),
                             orientation_loops=0,proposal_wedges=len(jobs),
                             triad_scale_hypotheses=sum(len(r['scale_hypotheses']) for r in results),
                             seconds=time.perf_counter()-start),scene_promoted=False,
                source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('identity_wedges.py','identity_pose_graph.py','identity_geometry.py','affine_camera_geometry.py')},
                geometry_policy_sha256=hashlib.sha256((args.geometry/'run-policy.json').read_bytes()).hexdigest(),
                field_policy_sha256=hashlib.sha256((args.fields/'run-policy.json').read_bytes()).hexdigest(),
                limitations='Training-selected shared-image initializations, including those without an already-closed pose loop. Joint image prediction and separate third-pair checks remain mandatory. A seed is neither an accepted triad nor a reconstructed scene.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
