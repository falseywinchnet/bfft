"""Summarize sampled certificate acquisition; missing targets stay explicit."""
import argparse
import json
import statistics
from pathlib import Path


def analyze(path):
    data = json.loads(Path(path).read_text())
    methods = sorted(set(next(iter(data['cases'].values()))) - {'ordinary'})
    out = {'input':str(path),'size':data['size'],'seed':data['seed'],'methods':{}}
    for method in methods:
        rows = []
        for target in [.01,.001,.0001]:
            pairs, candidate_only, ordinary_only, neither = [],[],[],[]
            for name,case in data['cases'].items():
                base = next((r for r in case['ordinary'] if r['gap']<=target),None)
                trial = next((r for r in case[method] if r['gap']<=target),None)
                if base and trial:
                    pairs.append({'case':name,'ordinary_pass':base['pass'],'candidate_pass':trial['pass'],
                                  'ordinary_seconds':base['seconds'],'candidate_seconds':trial['seconds'],
                                  'speedup':base['seconds']/trial['seconds']})
                elif trial: candidate_only.append(name)
                elif base: ordinary_only.append(name)
                else: neither.append(name)
            rows.append({'target':target,'paired':len(pairs),
                         'median_speedup':statistics.median(p['speedup'] for p in pairs) if pairs else None,
                         'time_wins':sum(p['speedup']>1 for p in pairs),
                         'candidate_only':candidate_only,'ordinary_only':ordinary_only,'neither':neither,'pairs':pairs})
        out['methods'][method] = rows
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--out',required=True)
    a = p.parse_args()
    result = analyze(a.input)
    Path(a.out).write_text(json.dumps(result,indent=2))
    for method,rows in result['methods'].items():
        for row in rows:
            print(method,row['target'], 'median speedup',row['median_speedup'],
                  'wins',row['time_wins'],'of',row['paired'],
                  'candidate-only',len(row['candidate_only']),'ordinary-only',len(row['ordinary_only']),
                  'neither',len(row['neither']))
