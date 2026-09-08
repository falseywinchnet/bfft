"""Stationary-object control: does noisy position alone induce false motion?"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import make_case
from .study import METHODS, evaluate, save, summarize


def run(results, out):
    configurations = json.loads(Path(results).read_text())['metadata']['configs']
    rows = []
    for corrupt in (False, True):
        for i in range(8):
            case = make_case('line', 30000+100*int(corrupt)+i, corrupt)
            case['truth'][:] = [.75, 0., 8.]
            available = np.isfinite(case['observations']).all(axis=1)
            case['observations'] = case['truth']+case['bias']+case['noise']
            case['observations'][~available] = np.nan
            for method in METHODS:
                metrics, _ = evaluate(case, method, configurations[method], particles=6144)
                rows.append({'method': method, 'family': 'stationary', 'corrupt': corrupt,
                             'seed': case['seed'], **metrics})
        print('completed stationary control', corrupt, flush=True)
    save(out, {'scope': 'independent stationary-object/null-motion control; no retuning',
               'overall': summarize(rows), 'runs': rows})
    print(json.dumps(summarize(rows), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run(args.results, args.out)
