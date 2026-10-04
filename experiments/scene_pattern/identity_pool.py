"""Reverify image-map proposals into a revisable, physically keyed identity pool.

The dense parent supplies image-local proposals only. Its camera estimates,
cycle votes and track unions never enter the output. Every proposal is fitted
again to the current original-image Lab cache, including reverse transport.
"""
import argparse, hashlib, json, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .identity_study import feature
from .dense_transport import from_prior

POLICY = dict(source_quantum_pixels=2, distinct_target_pixels=2,
              distinct_jacobian=.15, alternatives_per_site=4, batch=256,
              selection='training correlation; never average conflicting targets',
              inherited_geometry=False, inherited_cycle_votes=False)


def physical_site(p):
    """Stable within-image key independent of proposal file, order or lattice."""
    x, y = np.rint(np.asarray(p) / 2).astype(int)
    if x < 0 or y < 0:
        raise ValueError('Negative source location')
    return int((x+y)*(x+y+1)//2+y)


def canonicalize(row):
    p = np.asarray(row['p'], float)
    anchor = np.rint(p / 2) * 2
    A = np.asarray(row['A'], float)
    return dict(p=anchor, q=np.asarray(row['q'])+A@(anchor-p), A=A,
                site=physical_site(anchor), origins=list(row['origins']))


def retain(rows):
    """One count per physical source site, distinct hypotheses kept separately."""
    bank = {}
    for row in sorted(rows, key=lambda r: -r['training']):
        alternatives = bank.setdefault(row['site'], [])
        same = next((r for r in alternatives
                     if np.linalg.norm(row['q']-r['q']) < 2
                     and np.linalg.norm(row['A']-r['A']) < .15), None)
        if same is not None:
            same['origins'] = sorted(set(same['origins']+row['origins']))
        elif len(alternatives) < POLICY['alternatives_per_site']:
            alternatives.append(row)
    return [r for site in sorted(bank) for r in bank[site]]


def reverify(a, b, proposals):
    proposals = [canonicalize(r) for r in proposals]
    checked = []
    for start in range(0, len(proposals), POLICY['batch']):
        batch = proposals[start:start+POLICY['batch']]
        result = from_prior(a, b, np.array([r['p'] for r in batch]),
                            np.array([r['q'] for r in batch]),
                            np.array([r['A'] for r in batch]))
        for i, proposal in enumerate(batch):
            checked.append(dict(**proposal, prior=proposal['q'],
                                prior_affine=proposal['A'],
                                target=result['target'][i],
                                fitted_affine=result['affine'][i],
                                training=float(result['training'][i]),
                                validation=float(result['correlation'][i]),
                                passed=bool(result['valid'][i])))
    accepted = [dict(p=r['p'], q=r['target'], A=r['fitted_affine'],
                     site=r['site'], origins=r['origins'],
                     training=r['training'], validation=r['validation'])
                for r in checked if r['passed']]
    return retain(accepted), checked


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def work(spec):
    a, b, files, dense, fields, seeds, out = spec
    start = time.perf_counter(); proposals = list(seeds)
    dense_path = Path(dense)/f'{a:03d}-{b:03d}.npz'
    if dense_path.exists():
        with np.load(dense_path) as f:
            p = f['grid'].reshape(-1, 2)
            for i in np.flatnonzero(f['valid']):
                proposals.append(dict(p=p[i], q=f['target'][i], A=f['affine'][i],
                                      origins=[f'dense:{i}']))
    field_path = Path(fields)/f'{a:03d}-{b:03d}.json'
    if field_path.exists():
        for i, r in enumerate(json.loads(field_path.read_text())['rows']):
            proposals.append(dict(p=r['p'], q=r['q'], A=r['A'],
                                  origins=[f'ringing_field:{i}']))
    rows, checks = reverify(feature(files[a])['lab'], feature(files[b])['lab'], proposals)
    packet = dict(a=a, b=b, rows=rows, checks=checks, seconds=time.perf_counter()-start)
    (Path(out)/f'{a:03d}-{b:03d}.json').write_text(json.dumps(packet, default=serial))
    return dict(a=a, b=b, proposals=len(proposals), passing=sum(c['passed'] for c in checks),
                hypotheses=len(rows), regions=len({r['site'] for r in rows}),
                competing_sites=sum(n>1 for n in Counter(r['site'] for r in rows).values()),
                seconds=packet['seconds'])


def main():
    parser = argparse.ArgumentParser()
    for name in ('dense', 'fields', 'identities', 'data', 'features', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--scope', choices=['panoramas', 'all'], default='panoramas')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.iterdir()): raise ValueError('Use a fresh output directory')
    start = time.perf_counter()
    records = json.loads((args.data/'manifest.json').read_text())
    dense_meta = json.loads((args.dense/'dense.json').read_text())
    if dense_meta['records'] != records: raise ValueError('Dense capture identities differ')
    active = {i for i,r in enumerate(records) if args.scope=='all' or r['panoramic']}
    files = [str(args.features/(Path(r['name']).stem+'.npz')) for r in records]
    pairs = set(); inputs = {}
    for folder, suffix in ((args.dense, 'npz'), (args.fields, 'json')):
        for path in sorted(folder.glob('[0-9][0-9][0-9]-[0-9][0-9][0-9].'+suffix)):
            a,b = map(int, path.stem.split('-'))
            if {a,b} <= active:
                pairs.add((a,b)); inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    seeds = {}
    for i,e in enumerate(json.loads(args.identities.read_text())):
        a,b = e['a'][0],e['b'][0]
        if not e['accepted'] or not {a,b} <= active: continue
        pairs.update(((a,b),(b,a)))
        seeds.setdefault((a,b), []).append(dict(p=e['p'], q=e['q'], A=e['A'], origins=[f'ringing_seed:{i}:forward']))
        seeds.setdefault((b,a), []).append(dict(p=e['q'], q=e['p'], A=np.linalg.inv(e['A']), origins=[f'ringing_seed:{i}:reverse_proposal']))
    names = ['identity_pool.py','identity_study.py','dense_transport.py','region_transport.py']
    sources = args.out/'sources'; sources.mkdir()
    for name in names: (sources/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    policy = dict(policy=POLICY, scope=args.scope, records=records,
                  source_hashes={n:hashlib.sha256((sources/n).read_bytes()).hexdigest() for n in names},
                  feature_hashes={records[i]['name']:hashlib.sha256(Path(files[i]).read_bytes()).hexdigest() for i in sorted(active)},
                  proposal_hashes=inputs, identities_sha256=hashlib.sha256(args.identities.read_bytes()).hexdigest(),
                  dense_metadata_sha256=hashlib.sha256((args.dense/'dense.json').read_bytes()).hexdigest())
    (args.out/'run-policy.json').write_text(json.dumps(policy, indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        specs=[(a,b,files,str(args.dense),str(args.fields),seeds.get((a,b),[]),str(args.out)) for a,b in sorted(pairs)]
        for row in pool.map(work, specs):
            results.append(row)
            if len(results)%30==0:
                print(json.dumps(dict(done=len(results),total=len(pairs),hypotheses=sum(r['hypotheses'] for r in results),seconds=time.perf_counter()-start)), flush=True)
    summary=dict(scope=args.scope, pairs=results, proposals=sum(r['proposals'] for r in results),
                 passing=sum(r['passing'] for r in results), hypotheses=sum(r['hypotheses'] for r in results),
                 regions=sum(r['regions'] for r in results), competing_sites=sum(r['competing_sites'] for r in results),
                 captures=len({i for r in results if r['regions'] for i in (r['a'],r['b'])}),
                 seconds=time.perf_counter()-start, scene_promoted=False,
                 limitations='Reverified image-only hypotheses from two proposal inventories. Training and reserved-pixel checks are conditioned on image discovery. Correlated neighboring regions are not independent identity or geometric proof.')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='pairs'}), flush=True)


if __name__ == '__main__': main()
