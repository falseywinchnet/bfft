"""Fixed-policy, panorama-first correspondence graph. No image IDs are special."""
import argparse,json,time,hashlib,platform
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from .sweep_features import match

POLICY=dict(version=1,panorama_aspect=2.4,panorama_feature_side=768,photo_feature_side=384,
    radii=[8,16,24],descriptor='DAISY gradient distributions plus Oklab cartoon color at causal-density region centers',
    retrieval_neighbors=6,sequence_neighbors=2,ratio=.82,ransac_pixels=4,min_inliers=10,
    min_spatial_cells=4,min_axis_span=.08,seed=29,maximum_descriptors=1500)


def load_features(path):
    with np.load(path) as f:a={k:f[k] for k in ('points','descriptors','shape')}
    if len(a['points'])>POLICY['maximum_descriptors']:
        ix=np.linspace(0,len(a['points'])-1,POLICY['maximum_descriptors']).astype(int)
        a['points']=a['points'][ix];a['descriptors']=a['descriptors'][ix]
    return a


def retrieval(features):
    ds=[f['descriptors'][::max(1,len(f['points'])//100)] for f in features]
    scores=np.full((len(ds),len(ds)),np.inf)
    for i in range(len(ds)):
        for j in range(i):
            d=cdist(ds[i],ds[j],'sqeuclidean')
            scores[i,j]=np.mean(np.sort(d.min(1))[:max(8,len(d)//5)])
            scores[j,i]=np.mean(np.sort(d.min(0))[:max(8,d.shape[1]//5)])
    return scores


def components(n,edges):
    adj=[set() for _ in range(n)]
    for e in edges:
        if e['accepted']:adj[e['a']].add(e['b']);adj[e['b']].add(e['a'])
    unseen=set(range(n));result=[]
    while unseen:
        todo=[min(unseen)];component=[];unseen.remove(todo[0])
        while todo:
            i=todo.pop();component.append(i)
            for j in adj[i]&unseen:unseen.remove(j);todo.append(j)
        result.append(sorted(component))
    return sorted(result,key=lambda c:(-len(c),c[0]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=json.loads((args.data/'manifest.json').read_text());features=[load_features(args.features/(Path(r['name']).stem+'.npz')) for r in records]
    panos=[i for i,r in enumerate(records) if r['panoramic']];photos=[i for i,r in enumerate(records) if not r['panoramic']];start=time.perf_counter()
    scores=retrieval(features);np.save(args.out/'retrieval.npy',scores)
    pairs=[]
    for ii,i in enumerate(panos):
        for j in panos[ii+1:]:pairs.append((i,j,1))
    second=set()
    for i in photos:
        for pool in (panos,photos):
            for j in sorted((j for j in pool if j!=i),key=lambda j:scores[i,j])[:POLICY['retrieval_neighbors']]:second.add((min(i,j),max(i,j),2))
        pos=photos.index(i)
        for j in photos[max(0,pos-2):pos+3]:
            if j!=i:second.add((min(i,j),max(i,j),2))
    pairs+=sorted(second)
    cache=args.out/'edges.jsonl';edges=[];done=set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            e=json.loads(line);edges.append(e);done.add((e['a'],e['b']))
    with cache.open('a') as f:
        for a,b,stage in pairs:
            if (a,b) in done:continue
            e=match(features[a],features[b]);e.update(a=a,b=b,stage=stage);edges.append(e);f.write(json.dumps(e)+'\n');f.flush()
            if len(edges)%10==0 or e['accepted']:
                print(json.dumps({'pairs':len(edges),'planned':len(pairs),'stage':stage,'a':a,'b':b,'accepted':e['accepted'],'inliers':e.get('inliers',0),'seconds':time.perf_counter()-start}),flush=True)
    receipt=dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("sweep*.py")},runtime=dict(python=platform.python_version(),numpy=np.__version__),policy=POLICY,records=records,edges=edges,components=components(len(records),edges),seconds=time.perf_counter()-start)
    (args.out/'graph.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'complete':len(edges),'accepted':sum(e['accepted'] for e in edges),'components':receipt['components']}),flush=True)
if __name__=='__main__':main()
