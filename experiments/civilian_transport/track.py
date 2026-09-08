"""Track user-supplied timestamp,x,y,z CSV using no other observation fields."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .study import METHODS, build, save


def track(input_path, output_path, method, config, particles=1536, horizon=2., query_dt=.1):
    with open(input_path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'timestamp', 'x', 'y', 'z'}.issubset(rows[0]):
        raise ValueError('CSV requires timestamp,x,y,z columns and at least one row')
    if horizon <= 0 or query_dt <= 0:
        raise ValueError('horizon and query_dt must be positive')
    count = int(np.ceil(horizon/query_dt))
    intervals = np.full(count, horizon/count)
    model, previous = None, None
    estimates = []
    for index, row in enumerate(rows):
        t = float(row['timestamp'])
        if not np.isfinite(t) or (previous is not None and t <= previous):
            raise ValueError('timestamps must be finite and strictly increasing')
        values = np.array([float(row[key]) if row[key].strip() else np.nan for key in ('x', 'y', 'z')])
        observation = values if np.isfinite(values).all() else None
        if model is None:
            if observation is None:
                previous = t
                continue
            model = build(method, observation, config, seed=9381, particles=particles)
        else:
            model.predict(t-previous)
            model.update(observation)
        previous = t
        mean, covariance = model.state()
        paths = model.forecast_paths(intervals, samples=256, seed=49381+index)
        # Civilian x=0 boundary is a report query, not an estimator input.
        crossing = np.mean((paths[:, :, 0].min(axis=1) <= 0)&(paths[:, :, 0].max(axis=1) >= 0))
        estimates.append({'timestamp': t, 'position_mean': mean, 'position_covariance': covariance,
                          'diagnostics': model.diagnostics(), 'forecast_times': t+np.r_[0., np.cumsum(intervals)],
                          'forecast_position_mean': paths.mean(axis=0),
                          'forecast_position_covariance': np.array([np.cov(paths[:, j].T) for j in range(count+1)]),
                          'x0_boundary_crossing_probability_at_query_samples': float(crossing)})
    if not estimates:
        raise ValueError('CSV contains no complete finite position observation')
    save(output_path, {'method': method, 'config': config,
                       'scope': 'experimental estimates; empirical uncertainty, no operational guarantee',
                       'estimates': estimates})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('output_json')
    parser.add_argument('--method', choices=METHODS, default='geometric')
    parser.add_argument('--config-results', default=str(Path(__file__).parent/'results'/'results.json'))
    parser.add_argument('--horizon', type=float, default=2.)
    parser.add_argument('--query-dt', type=float, default=.1)
    args = parser.parse_args()
    configs = json.loads(Path(args.config_results).read_text())['metadata']['configs']
    track(args.input_csv, args.output_json, args.method, configs[args.method], horizon=args.horizon, query_dt=args.query_dt)
