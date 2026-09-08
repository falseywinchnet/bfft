"""Paired noise/acquisition sweep; no parameter selection on these cases."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .certificates import pair_bound, Ball
from .data import FAMILIES, make_case
from .study import METHODS, build, save

SIGMAS = [.1, .35, .7, 1.4, 2.8]
COUNTS = [65, 33, 17, 9, 5]
TIMES = np.r_[np.linspace(0, 6, 65), np.linspace(6.1, 8, 20)]
ALPHA = .05
# A single bound over every possible parameter of every family, never selected
# from the realized truth, family label, observations or held-out performance.
FAMILY_A = {
    'line': math.hypot(.08*.6**2, .05*.9**2),
    'helix': 3.5*.7,
    'figure8': 3.5*.7*math.sqrt(1 + 1.8**2 + (.35*.7**2)**2),
    'stop_go': math.hypot(.432, .3) + math.sqrt(3.5**2+.72**2+.3**2)*8/(9*math.sqrt(3)),
    'switching': 5.,
}
A = max(FAMILY_A.values())


def acquisition_indices(count):
    if count not in COUNTS:
        raise ValueError('Unsupported nested acquisition count')
    return np.arange(0, 65, 64//(count-1))


def fixture(family, seed, sigma, count, biased):
    case = make_case(family, seed, samples=len(TIMES), observation_times=TIMES)
    rng = np.random.default_rng(seed+900000)
    standard_noise = rng.normal(size=case['truth'].shape)
    direction = rng.normal(size=3)
    bias = .75*direction/np.linalg.norm(direction) if biased else np.zeros(3)
    # Identical standard noise, bias direction, truth and filter seed across
    # every noise/density condition for a case. Dense acquisitions are nested.
    case['observations'] = case['truth'] + sigma*standard_noise + bias
    case['bias_vector'] = bias
    case['indices'] = acquisition_indices(count)
    return case


def pair_region(case, sigma, biased):
    # Joint 95% observation event over the fixed maximum 65 timestamps. Reusing
    # this event preserves simultaneous validity across every thinned subset.
    epsilon = sigma*math.sqrt(6*math.log(6*65/ALPHA))
    indices = case['indices']
    balls = []
    for jj, j in enumerate(indices[1:], 1):
        for i in indices[:jj]:
            ball = pair_bound(TIMES[i], tuple(case['observations'][i]), epsilon,
                              TIMES[j], tuple(case['observations'][j]), epsilon, 8., A)
            # Shared bias enters affine extrapolation with coefficient one.
            balls.append(Ball(ball.center, ball.radius + (.75 if biased else 0)))
    centers = np.array([b.center for b in balls])
    radii = np.array([b.radius for b in balls])
    best = int(np.argmin(radii))
    target = case['truth'][-1]
    return centers, radii, best, bool(np.all(np.linalg.norm(centers-target, axis=1) <= radii))


def run(out, seeds=6, particles=6144, draws=1024):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((Path(__file__).parent/'results/development.json').read_text())['chosen']
    conditions = [(s, 65) for s in SIGMAS] + [(.35, n) for n in COUNTS[1:]]
    rows, certificate_rows = [], []
    for biased in (False, True):
        for family_index, family in enumerate(FAMILIES):
            for repeat in range(seeds):
                seed = 40000+1000*family_index+repeat
                for sigma, count in conditions:
                    case = fixture(family, seed, sigma, count, biased)
                    centers, radii, best, joint_covered = pair_region(case, sigma, biased)
                    target = case['truth'][-1]
                    certificate_rows.append(dict(family=family, seed=seed, biased=biased,
                        sigma=sigma, count=count, radius=float(radii[best]),
                        center_sq=float(np.sum((centers[best]-target)**2)),
                        intersection_covered=joint_covered))
                    available = set(case['indices'])
                    for method in METHODS:
                        config = dict(chosen[method], sigma=sigma)
                        model = build(method, case['observations'][0], config, seed+73417, particles)
                        started = time.perf_counter()
                        for k in range(1, 65):
                            model.predict(TIMES[k]-TIMES[k-1])
                            model.update(case['observations'][k] if k in available else None)
                        center, _ = model.state()
                        track_sq = float(np.sum((center-case['truth'][64])**2))
                        paths = model.forecast_paths(np.diff(TIMES[64:]), draws, seed=seed+73500)
                        seconds = time.perf_counter()-started
                        endpoint = paths[:, -1]
                        mean = endpoint.mean(axis=0)
                        error = mean-target
                        cov = np.cov(endpoint.T)+np.eye(3)*1e-8
                        eig = np.linalg.eigvalsh(cov)
                        # 95% Gaussian ellipsoid, a descriptive region for
                        # mixtures; empirical coverage is measured, not assumed.
                        covered = bool(error @ np.linalg.solve(cov, error) <= 7.814727903)
                        volume_radius = float(math.sqrt(7.814727903)*np.prod(eig)**(1/6))
                        certified_radius = float(np.min(radii+np.linalg.norm(centers-mean, axis=1)))
                        rows.append(dict(family=family, seed=seed, biased=biased, sigma=sigma,
                            count=count, method=method, track_sq=track_sq,
                            forecast_sq=float(error @ error), gaussian_95_covered=covered,
                            gaussian_equiv_radius=volume_radius, certified_radius=certified_radius,
                            certified_covered=bool(np.linalg.norm(error) <= certified_radius), seconds=seconds))
                save(out/'partial.json', {'runs': rows, 'certificates': certificate_rows})
            print(f'completed biased={biased} family={family}', flush=True)
    metadata = dict(sigmas=SIGMAS, counts=COUNTS, seeds_per_family=seeds,
        trajectories=5*seeds, particles=particles, forecast_draws=draws,
        history_seconds=6, horizon_seconds=2, alpha=ALPHA,
        uniform_acceleration_bound=A, family_proof_bounds=FAMILY_A,
        configs=chosen, bias_bound=.75, seed_base=40000,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).parent.glob('*.py')})
    save(out/'results.json', {'metadata':metadata, 'runs':rows, 'certificates':certificate_rows})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, default=6)
    p.add_argument('--particles', type=int, default=6144)
    p.add_argument('--draws', type=int, default=1024)
    a = p.parse_args()
    run(a.out, a.seeds, a.particles, a.draws)
