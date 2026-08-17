#!/usr/bin/env python3
"""Map when coherence-derived Eikonal shell transport helps."""

import csv, json, time
from pathlib import Path
import numpy as np
import run_experiment as core

OUT=Path('experiments/fourier_eikonal_transport/results_regimes')


def main():
    OUT.mkdir(parents=True,exist_ok=True); start=time.time(); rows=[]
    for views in [3,5,8]:
        for noise in [.05,.10,.20]:
            for seed in range(20):
                _,_,_,trial=core.run_trial(seed,24,96,views,noise)
                iso=trial['isotropic_atlas']; eik=trial['eikonal_atlas']; shuf=trial['shuffled_metric']
                rows.append({'views':views,'noise':noise,'seed':seed,
                    'iso_nmse':iso['nmse'],'eik_nmse':eik['nmse'],'shuffled_nmse':shuf['nmse'],
                    'iso_missing':iso['missing_nmse'],'eik_missing':eik['missing_nmse'],
                    'shuffled_missing':shuf['missing_nmse']})
            print(json.dumps({'views':views,'noise':noise,'done':True}),flush=True)
    summary=[]
    for views in [3,5,8]:
        for noise in [.05,.10,.20]:
            rs=[r for r in rows if r['views']==views and r['noise']==noise]
            def a(name): return np.array([r[name] for r in rs])
            summary.append({'views':views,'noise':noise,
                'whole_win_rate':float(np.mean(a('eik_nmse')<a('iso_nmse'))),
                'missing_win_rate':float(np.mean(a('eik_missing')<a('iso_missing'))),
                'whole_median_ratio':float(np.median(a('eik_nmse')/a('iso_nmse'))),
                'missing_median_ratio':float(np.median(a('eik_missing')/a('iso_missing'))),
                'eik_whole_median':float(np.median(a('eik_nmse'))),
                'iso_whole_median':float(np.median(a('iso_nmse'))),
                'eik_missing_median':float(np.median(a('eik_missing'))),
                'iso_missing_median':float(np.median(a('iso_missing'))),
                'eik_whole_p90':float(np.quantile(a('eik_nmse'),.9)),
                'iso_whole_p90':float(np.quantile(a('iso_nmse'),.9)),
                'shuffled_whole_median':float(np.median(a('shuffled_nmse'))),
                'shuffled_missing_median':float(np.median(a('shuffled_missing')))})
    with (OUT/'runs.json').open('w') as f: json.dump({'runtime_seconds':time.time()-start,'runs':rows},f,indent=2)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print(json.dumps({'runtime_seconds':time.time()-start,'summary':summary},indent=2))


if __name__=='__main__': main()
