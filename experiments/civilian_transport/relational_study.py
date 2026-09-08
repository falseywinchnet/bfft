"""Frozen exploratory comparison of the explicit relational current prior."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .data import FAMILIES, make_case
from .relational_kalman import forecast, btb_proximal_check, LENGTHS, RelationalKalman
from .baselines import ConventionalFilter
from .study import save


def run(out, seeds=6):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((Path(__file__).parent/'results/development.json').read_text())['chosen']
    rows, traces, audits = [], [], []
    t = np.r_[np.linspace(0, 6, 65), np.linspace(6.1, 8, 20)]
    for family_index, family in enumerate(FAMILIES):
        for repeat in range(seeds):
            seed = 80000+family_index*1000+repeat
            case = make_case(family, seed, samples=len(t), observation_times=t)
            noise = np.random.default_rng(seed+900000).normal(size=case['truth'].shape)
            for sigma, count in [(.35, 65), (1.4, 65), (.35, 9)]:
                idx = np.arange(0, 65, 64//(count-1))
                y = case['truth'][idx]+sigma*noise[idx]
                query = t[65:]
                for method in ['relational', 'severed', 'single_length', 'cv', 'ca', 'imm', 'robust_imm', 'turn_ukf']:
                    start = time.perf_counter()
                    if method in ('relational', 'severed', 'single_length'):
                        model = RelationalKalman(y[0], sigma,
                                          lengths=(1.5,) if method=='single_length' else LENGTHS,
                                          sever_at=6. if method=='severed' else None)
                        for k in range(1, len(idx)):
                            model.update(t[idx[k]], y[k])
                        result = model.forecast(query)
                        mean, cov = result['mean'], result['covariance']
                    else:
                        model = ConventionalFilter(y[0], method=method, **dict(chosen[method], sigma=sigma))
                        for k in range(1, len(idx)):
                            model.predict(t[idx[k]]-t[idx[k-1]])
                            model.update(y[k])
                        draws = model.forecast_paths(np.diff(t[64:]), samples=2048, seed=seed+73417)
                        # Matching pre-existing baseline forecast implementation.
                        mean = draws[:, 1:].mean(axis=0)
                        cov = np.array([np.cov(draws[:, j].T) for j in range(1, len(query)+1)])
                    elapsed = time.perf_counter()-start
                    error = mean[-1]-case['truth'][-1]
                    coverage = float(error @ np.linalg.solve(cov[-1], error)) <= 7.814727903
                    rows.append(dict(family=family, seed=seed, sigma=sigma, count=count, method=method,
                                     sq=float(error @ error), path_sq=float(np.mean(np.sum((mean-case['truth'][65:])**2, axis=1))),
                                     covered=bool(coverage), radius=float(7.814727903**.5*np.linalg.det(cov[-1])**(1/6)),
                                     seconds=elapsed))
                    if repeat==0 and sigma==.35 and count==65:
                        traces.append(dict(family=family, method=method, times=t, truth=case['truth'], observations=y,
                                           forecast_times=query, mean=mean, covariance=cov))
                if repeat==0 and sigma==.35 and count==65:
                    base = forecast(t[idx], y, sigma, query)
                    refined = forecast(t[idx], y, sigma, query, cells=128)
                    tempered = forecast(t[idx], y, sigma, query, beta_steps=4)
                    real = forecast(t[idx], y, sigma, query, coordinates='current')
                    audits.append(dict(family=family,
                        resolution_endpoint_difference=float(np.linalg.norm(base['mean'][-1]-refined['mean'][-1])),
                        resolution_cov_difference=float(np.linalg.norm(base['covariance'][-1]-refined['covariance'][-1])),
                        zak_difference=float(np.max(np.abs(base['mean']-real['mean']))),
                        tempering_mean_difference=float(np.max(np.abs(base['mean']-tempered['mean']))),
                        tempering_cov_difference=float(np.max(np.abs(base['covariance']-tempered['covariance']))),
                        weights=base['weights'], proximal=btb_proximal_check(t[idx], y, sigma)))
                save(out/'partial.json', {'runs': rows, 'audits': audits})
            print(f'{family} seed {repeat+1}/{seeds}', flush=True)
    hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
              for name in ['relational_kalman.py', 'relational_study.py', 'baselines.py', 'data.py']}
    save(out/'results.json', dict(runs=rows, audits=audits, source_hashes=hashes,
         metadata=dict(seeds=seeds, seed_base=80000, cells=64, lengths=LENGTHS,
                       sensor='independent Gaussian', horizon=2., history=6., baseline_draws=2048,
                       tuned_on_evaluation=False, coverage='moment ellipsoid diagnostic, not exact mixture 95% set')))
    save(out/'traces.json', traces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, default=6)
    args = parser.parse_args()
    run(args.out, args.seeds)
