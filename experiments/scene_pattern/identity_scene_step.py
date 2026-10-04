"""One reproducible identity-to-camera iteration over a completed field collection."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--fields',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--prior-geometry',type=Path);p.add_argument('--scope',choices=['all','panoramas'],default='all');p.add_argument('--initialization',choices=['closed-loops','image-wedges'],default='closed-loops');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    geometry=a.out/'geometry';graph=a.out/'pose-graph.json';joint=a.out/'joint';start=time.perf_counter()
    def run(module,*args):subprocess.run([sys.executable,'-m','experiments.scene_pattern.'+module,*map(str,args)],check=True)
    run('identity_geometry','--fields',a.fields,'--data',a.data,'--features',a.features,'--out',geometry,'--workers',a.workers,'--scope',a.scope,*(['--prior-geometry',a.prior_geometry] if a.prior_geometry else []))
    run('identity_wedges' if a.initialization=='image-wedges' else 'identity_pose_graph','--geometry',geometry,'--fields',a.fields,'--out',graph,*(['--workers',a.workers] if a.initialization=='image-wedges' else []))
    run('identity_joint_study','--geometry',geometry,'--graph',graph,'--fields',a.fields,'--out',joint,'--workers',a.workers)
    g=json.loads((geometry/'summary.json').read_text());pg=json.loads(graph.read_text());j=json.loads((joint/'summary.json').read_text())
    result=dict(scope=a.scope,initialization=a.initialization,proposal_wedges=pg['summary'].get('proposal_wedges',0),accepted_directed_pairs=g['accepted_directed_pairs'],relative_pairs=pg['summary']['relative_pairs'],captures_in_relative_pairs=pg['summary']['captures'],geometric_identity_updates=g['identity_changes'],orientation_loops=pg['summary']['orientation_loops'],joint_candidates=len(j['jobs']),accepted_joint_hypotheses=j['accepted_hypotheses'],seconds=time.perf_counter()-start,scene_promoted=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='A full collection association/camera iteration, retaining local alternatives. Accepted local hypotheses still require global reconciliation and coverage; this packet is not the fused scene.')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
