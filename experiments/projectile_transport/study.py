"""Fixed-configuration causal tracking and observation-free forecast screen."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .core import GRAVITY, make_filter


METHODS = ['ballistic_kf', 'fixed_joint_ekf', 'motion_only', 'noise_only', 'joint_transport']
SCENARIOS = ['ballistic', 'drag', 'changing_force', 'sensor_change', 'combined', 'unseen_drag']


def forcing(t, scenario):
    if scenario not in ('changing_force', 'combined', 'unseen_drag'):
        return np.zeros(2)
    return np.array([2.8*np.exp(-((t-1.45)/.38)**2),
                     1.2*np.sin(2.4*t)*np.exp(-((t-1.8)/.9)**2)])


def simulate(seed, scenario, dt=.05, count=71):
    """Independent fine-step RK4 truth; no filter code generates the dynamics."""
    rng = np.random.default_rng(seed)
    beta = {'ballistic': 0., 'drag': .012, 'changing_force': .012,
            'sensor_change': .012, 'combined': .012, 'unseen_drag': .018}[scenario]
    state = np.array([0., 30., 12., 18.])+rng.normal(size=4)*[.3, .3, 1., 1.]
    bias = np.zeros(2)
    records = []
    def derivative(t, x):
        v = x[2:4]
        return np.r_[v, GRAVITY-beta*np.linalg.norm(v)*v+forcing(t, scenario)]
    for k in range(count):
        t = k*dt
        sensor = scenario in ('sensor_change', 'combined', 'unseen_drag')
        if sensor:
            bias = np.exp(-dt/3)*bias + .12*np.sqrt(dt)*rng.normal(size=2)
            # An intentionally unmodeled camera drift event.
            bias += dt*np.array([.5, -.25])*(1.2 < t < 2.2)
        sigma = 2. if sensor and 1.1 < t < 2.1 else .4
        epsilon = sigma*rng.normal(size=2)
        spike = sensor and k in (46, 57)
        if spike:
            epsilon += rng.normal(size=2)*5.
        missing = sensor and (35 <= k <= 39 or k == 52)
        records.append({'t': t, 'state': state.copy(), 'bias': bias.copy(),
                        'drag': -beta*np.linalg.norm(state[2:4])*state[2:4],
                        'force': forcing(t, scenario), 'epsilon': epsilon,
                        'sigma': sigma, 'spike': spike,
                        'observation': None if missing else state[:2]+bias+epsilon})
        h = dt/10
        for sub in range(10):
            u = t+sub*h
            k1 = derivative(u, state)
            k2 = derivative(u+h/2, state+h*k1/2)
            k3 = derivative(u+h/2, state+h*k2/2)
            k4 = derivative(u+h, state+h*k3)
            state += h*(k1+2*k2+2*k3+k4)/6
    return records


def assess(records, name, dt=.05, keep_trace=False):
    estimator = make_filter(name, dt)
    metrics = {k: [] for k in ['position_sq', 'velocity_sq', 'bias_sq', 'drag_sq',
                               'force_sq', 'total_acceleration_sq', 'noise_sq',
                               'position_95_coverage', 'observation_nll',
                               'forecast_0.5s_sq', 'forecast_1.0s_sq',
                               'forecast_1.0s_95_coverage', 'forecast_1.0s_nll']}
    trace = []
    started = time.perf_counter()
    for k, row in enumerate(records):
        if k:
            estimator.predict()
        evidence = estimator.update(row['observation'])
        mean, cov = estimator.state()
        cm, cp = estimator.components()
        future_points = {}
        # One copy, nested horizons: no future measurements or truth passed in.
        future = estimator.forecast(10)
        forecasts = {10: future}
        forecasts[20] = future.forecast(10)
        if k >= 10:  # fixed 0.5 s warm-up, not chosen from any fit outcome
            for key, diff in [('position', mean[:2]-row['state'][:2]),
                              ('velocity', mean[2:4]-row['state'][2:4]),
                              ('bias', mean[8:10]-row['bias']),
                              ('drag', cm[:2]-row['drag']),
                              ('force', cm[2:4]-row['force']),
                              ('total_acceleration', cm[:2]+cm[2:4]-row['drag']-row['force'])]:
                metrics[key+'_sq'].append(float(diff @ diff))
            error = mean[:2]-row['state'][:2]
            metrics['position_95_coverage'].append(float(error @ np.linalg.solve(cov[:2, :2], error) <= 5.991464547))
            if evidence is not None:
                metrics['observation_nll'].append(-evidence)
                diff = estimator.last_noise[0][-2:]-row['epsilon']
                metrics['noise_sq'].append(float(diff @ diff))
        for horizon, forecast in forecasts.items():
            fm, fc = forecast.state()
            future_points[str(horizon)] = {'mean': fm[:2], 'cov': fc[:2, :2]}
            if k >= 10 and k+horizon < len(records):
                error = fm[:2]-records[k+horizon]['state'][:2]
                metrics['forecast_%.1fs_sq' % (horizon*dt)].append(float(error @ error))
                if horizon == 20:
                    metrics['forecast_1.0s_95_coverage'].append(float(error @ np.linalg.solve(fc[:2, :2], error) <= 5.991464547))
                    metrics['forecast_1.0s_nll'].append(-forecast.position_logpdf(records[k+horizon]['state'][:2]))
        if keep_trace:
            trace.append({'t': row['t'], 'mean': mean, 'cov': cov,
                          'components_mean': cm, 'components_cov': cp,
                          'weights': estimator.weights.copy(),
                          'expected_sigma2': sum(w*m.sigma**2 for w, m in zip(estimator.weights, estimator.modes)),
                          'noise_joint': estimator.last_noise,
                          'forecasts': future_points})
    elapsed = time.perf_counter()-started
    result = {}
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else None
        result[key+'_count'] = len(values)
    result['seconds_including_forecasts'] = elapsed
    return result, trace


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(out, seeds, start_seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs, traces = [], {}
    for scenario in SCENARIOS:
        for seed in range(start_seed, start_seed+seeds):
            records = simulate(seed, scenario)
            for method in METHODS:
                keep = seed == start_seed and scenario == 'combined'
                metrics, trace = assess(records, method, keep_trace=keep)
                runs.append({'scenario': scenario, 'seed': seed, 'method': method, **metrics})
                if keep:
                    traces[method] = trace
            if seed == start_seed and scenario == 'combined':
                traces['truth'] = records
        print('completed', scenario, flush=True)
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for method in METHODS:
            rows = [r for r in runs if r['scenario'] == scenario and r['method'] == method]
            result = {}
            for key in rows[0]:
                if key in ('scenario', 'seed', 'method') or key.endswith('_count'):
                    continue
                values = [r[key] for r in rows if r[key] is not None]
                avg = float(np.mean(values)) if values else None
                result[key.replace('_sq', '_rmse')] = np.sqrt(avg) if key.endswith('_sq') and avg is not None else avg
            summary[scenario][method] = result
    metadata = {'seeds': seeds, 'start_seed': start_seed, 'dt': .05, 'samples': 71,
                'warmup_samples': 10, 'units': 'meters, seconds',
                'coverage': '95% moment-Gaussian 2D ellipse; empirical, not mixture credible region',
                'model': 'IMM Gaussian-mixture EKF; fixed catalog, full joint covariances',
                'methods': {name: [m.__dict__ for m in make_filter(name).modes] for name in METHODS}}
    (out/'results.json').write_text(json.dumps({'metadata': metadata, 'summary': summary, 'runs': runs}, indent=2, default=serial))
    (out/'trace.json').write_text(json.dumps(traces, default=serial))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--start-seed', type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    run(args.out, args.seeds, args.start_seed)
