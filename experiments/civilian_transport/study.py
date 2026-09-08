"""Development-only tuning, frozen held-out evaluation and paired uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .baselines import ConventionalFilter
from .geometric_filter import GeometricFilter
from .data import FAMILIES, make_case


METHODS = ['cv', 'ca', 'imm', 'robust_imm', 'turn_ukf', 'geometric']


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(str(type(x)))


def save(path, value):
    Path(path).write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False))


def build(method, observation, config, seed, particles):
    if method == 'geometric':
        return GeometricFilter(observation, seed=seed, particles=particles, **config)
    return ConventionalFilter(observation, method=method, **config)


def energy_score(draws, truth):
    # Proper score, using distinct independent-index pair approximation for E|X-X'|.
    n = len(draws)
    return float(np.linalg.norm(draws-truth, axis=1).mean()
                 -.5*np.linalg.norm(draws[:n//2]-draws[n//2:2*(n//2)], axis=1).mean())


def evaluate(case, method, config, particles=1024, forecast_samples=256,
             stride=10, keep_trace=False, ablations=False):
    observations, times, truth = case['observations'], case['times'], case['truth']
    algorithm_seed = case['seed']+73417
    model = build(method, observations[0], config, algorithm_seed, particles)
    metrics = {name: [] for name in ('track_sq', 'noise_sq', 'bias_sq', 'track_95', 'observation_nll',
                                    'forecast1_sq', 'forecast2_sq', 'energy2', 'forecast2_95',
                                    'boundary_endpoint_brier', 'boundary_crossing_brier')}
    alternatives = {a: {'forecast2_sq': [], 'energy2': [], 'boundary_crossing_brier': []}
                    for a in ('uncoupled', 'straight')} if ablations else {}
    trace = []
    update_seconds, forecast_seconds = 0., 0.
    for k in range(len(times)):
        started = time.perf_counter()
        if k:
            model.predict(times[k]-times[k-1])
            y = observations[k] if np.isfinite(observations[k]).all() else None
            evidence = model.update(y)
        else:
            evidence = None
        center, covariance = model.state()
        diagnostic = model.diagnostics()
        update_seconds += time.perf_counter()-started
        if k >= 20:
            error = center-truth[k]
            metrics['track_sq'].append(float(error @ error))
            metrics['track_95'].append(float(error @ np.linalg.solve(covariance, error) <= 7.814727903))
            diff = diagnostic['bias']-case['bias'][k]
            metrics['bias_sq'].append(float(diff @ diff))
            if evidence is not None:
                metrics['observation_nll'].append(-evidence)
                diff = diagnostic['noise']-case['noise'][k]
                metrics['noise_sq'].append(float(diff @ diff))
        saved_forecast = None
        if k >= 20 and k+20 < len(times) and k % stride == 0:
            intervals = np.diff(times[k:k+21])
            # Only the future query timestamps are passed; no future values.
            started = time.perf_counter()
            draws = model.forecast_paths(intervals, forecast_samples, seed=algorithm_seed+k)
            forecast_seconds += time.perf_counter()-started
            for h, name in [(10, 'forecast1_sq'), (20, 'forecast2_sq')]:
                error = draws[:, h].mean(axis=0)-truth[k+h]
                metrics[name].append(float(error @ error))
            endpoint = draws[:, -1]
            metrics['energy2'].append(energy_score(endpoint, truth[k+20]))
            covariance = np.cov(endpoint.T)+np.eye(3)*1e-8
            error = endpoint.mean(axis=0)-truth[k+20]
            metrics['forecast2_95'].append(float(error @ np.linalg.solve(covariance, error) <= 7.814727903))
            projected = draws @ case['boundary_normal']-case['boundary_offset']
            actual = truth[k:k+21] @ case['boundary_normal']-case['boundary_offset']
            endpoint_prob = np.mean(projected[:, -1] >= 0)
            crossing_prob = np.mean((projected.min(axis=1) <= 0)&(projected.max(axis=1) >= 0))
            actual_crossing = float(actual.min() <= 0 <= actual.max())
            metrics['boundary_endpoint_brier'].append(float((endpoint_prob-float(actual[-1] >= 0))**2))
            metrics['boundary_crossing_brier'].append(float((crossing_prob-actual_crossing)**2))
            for ablation in alternatives:
                other = model.forecast_paths(intervals, forecast_samples, seed=algorithm_seed+k, ablation=ablation)
                delta = other[:, -1].mean(axis=0)-truth[k+20]
                alternatives[ablation]['forecast2_sq'].append(float(delta @ delta))
                alternatives[ablation]['energy2'].append(energy_score(other[:, -1], truth[k+20]))
                proj = other @ case['boundary_normal']-case['boundary_offset']
                prob = np.mean((proj.min(axis=1) <= 0)&(proj.max(axis=1) >= 0))
                alternatives[ablation]['boundary_crossing_brier'].append(float((prob-actual_crossing)**2))
            if keep_trace:
                saved_forecast = {'times': times[k:k+21], 'mean': draws.mean(axis=0),
                                  'paths': draws[:48], 'crossing_probability': crossing_prob,
                                  'crossing_truth': actual_crossing}
        if keep_trace:
            trace.append({'time': times[k], 'mean': center, 'diagnostic': diagnostic,
                          'forecast': saved_forecast})
    result = {key: float(np.mean(values)) if values else None for key, values in metrics.items()}
    result['counts'] = {key: len(values) for key, values in metrics.items()}
    result['update_ms'] = update_seconds*1000/len(times)
    result['forecast_ms'] = forecast_seconds*1000/max(1, len(metrics['forecast2_sq']))
    result['ablations'] = {a: {key: float(np.mean(v)) for key, v in vals.items()} for a, vals in alternatives.items()}
    return result, trace


def candidates(method):
    if method in ('geometric', 'turn_ukf'):
        return [{'sigma': .35, 'turn_diffusion': w, 'speed_diffusion': s}
                for w, s in [(.18, .10), (.4, .18), (.75, .30)]]
    return [{'sigma': .35, 'q': q} for q in [.1, 1., 10.]]


def tune(directory):
    cases = [make_case(family, 100+10*i+int(corrupt), corrupt)
             for i, family in enumerate(FAMILIES) for corrupt in (False, True)]
    trials, chosen = [], {}
    for method in METHODS:
        options = []
        for config in candidates(method):
            results = [evaluate(case, method, config, particles=384, forecast_samples=128, stride=20)[0]
                       for case in cases]
            # Fixed objective gives both tracking and two-second prediction weight.
            objective = float(np.mean([r['track_sq']+.25*r['forecast2_sq'] for r in results]))
            options.append((objective, config))
            trials.append({'method': method, 'config': config, 'objective': objective, 'cases': results})
        chosen[method] = min(options, key=lambda x: x[0])[1]
        print('selected', method, chosen[method], flush=True)
        save(directory/'development.json', {'chosen': chosen, 'trials': trials})
    return chosen


def summarize(rows):
    result = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        summary = {}
        for key in selected[0]:
            if key in ('method', 'family', 'corrupt', 'seed', 'counts', 'ablations'):
                continue
            values = [r[key] for r in selected if r[key] is not None]
            mean = float(np.mean(values)) if values else None
            summary[key.replace('_sq', '_rmse')] = np.sqrt(mean) if mean is not None and key.endswith('_sq') else mean
        result[method] = summary
    return result


def run(out, seeds=8, particles=1536, stage='all', test_seed_base=10000):
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    if stage in ('all', 'tune'):
        chosen = tune(directory)
    else:
        chosen = json.loads((directory/'development.json').read_text())['chosen']
    if stage == 'tune':
        return
    rows, traces = [], {}
    for i, family in enumerate(FAMILIES):
        for corrupt in (False, True):
            for j in range(seeds):
                seed = test_seed_base+i*1000+int(corrupt)*100+j
                case = make_case(family, seed, corrupt)
                keep = j == 0 and corrupt and family in ('helix', 'switching')
                for method in METHODS:
                    metrics, trace = evaluate(case, method, chosen[method], particles,
                                              keep_trace=keep, ablations=method == 'geometric')
                    rows.append({'method': method, 'family': family, 'corrupt': corrupt, 'seed': seed, **metrics})
                    if keep:
                        traces.setdefault(family, {})[method] = trace
                if keep:
                    traces[family]['truth'] = case['truth']
                    traces[family]['observations'] = [[float(v) for v in row] if np.isfinite(row).all() else None for row in case['observations']]
                    traces[family]['times'] = case['times']
                save(directory/'partial.json', rows)
            print('completed', family, 'corrupt='+str(corrupt), flush=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    metadata = {'seeds_per_family_noise': seeds, 'particles': particles, 'forecast_samples': 256,
                'development_seeds': '100 + 10*family_index + corruption',
                'test_seeds': f'{test_seed_base} + 1000*family_index + 100*corruption + repeat',
                'query_horizons': '10/20 future timestamp intervals, approximately 1/2 seconds',
                'observations': 'timestamp and noisy 3-D Cartesian position only',
                'tuning_objective': 'mean(track MSE + .25 * two-second forecast MSE)',
                'source_sha256': source_hashes, 'configs': chosen}
    payload = {'metadata': metadata, 'overall': summarize(rows),
               'by_case': {f+'_'+str(c): summarize([r for r in rows if r['family'] == f and r['corrupt'] == c])
                           for f in FAMILIES for c in (False, True)}, 'runs': rows}
    save(directory/'results.json', payload)
    save(directory/'traces.json', traces)
    print(json.dumps(payload['overall'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--particles', type=int, default=1536)
    parser.add_argument('--stage', choices=['all', 'tune', 'test'], default='all')
    parser.add_argument('--test-seed-base', type=int, default=10000)
    args = parser.parse_args()
    run(args.out, args.seeds, args.particles, args.stage, args.test_seed_base)
