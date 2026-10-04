"""Reproducible matched-accuracy application screen; no oracle in timed solvers."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np

from experiments.entropic_transport_closure.sinkhorn import solve, solve_blocks


def cases(size=256, seed=0):
    rng = np.random.default_rng(seed)
    uniform = np.ones(size)/size
    yield 'positive_dense', np.exp(rng.normal(size=(size, size))), uniform, uniform
    x = np.linspace(0, 1, size)
    a = .1+np.exp(-((x-(.28+.04*seed))/(.18+.01*seed))**2)
    b = .1+np.exp(-((x-(.68-.03*seed))/.2)**2)
    a /= a.sum(); b /= b.sum()
    for epsilon in [.02, .003]:
        yield 'line_eps_' + str(epsilon), np.exp(-(x[:, None]-x[None, :])**2/epsilon), a, b
    X, Y = rng.uniform(.1, .9, (size, 2)), rng.uniform(.1, .9, (size, 2))
    dist = ((X[:, None, :]-Y[None, :, :])**2).sum(axis=2)
    a = .1+np.exp(-((X-np.array([.3, .4]))**2).sum(axis=1)/.03)
    b = .1+np.exp(-((Y-np.array([.7, .6]))**2).sum(axis=1)/.025)
    a /= a.sum(); b /= b.sum()
    for epsilon in [.03, .006, .003]:
        yield 'plane_eps_' + str(epsilon), np.exp(-dist/epsilon), a, b


def serializable(result):
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def run(size, repeats, seeds, tolerance, seed_start=0, methods=None):
    methods = methods or ['ordinary', 'linear', 'quadratic', 'quadratic2']
    rng = np.random.default_rng(991)
    records = []
    for seed in range(seed_start, seed_start+seeds):
        for name, K, a, b in cases(size, seed):
            # Warm each path before shuffled recorded runs.
            for method in methods:
                solve(K, a, b, method=method, tolerance=tolerance)
            jobs = [(method, rep) for rep in range(repeats)
                    for method in methods]
            rng.shuffle(jobs)
            for method, rep in jobs:
                result = solve(K, a, b, method=method, tolerance=tolerance)
                record = dict(case=name, size=size, seed=seed, repeat=rep,
                              method=method, **serializable(result))
                records.append(record)
            medians = {method: float(np.median([r['seconds'] for r in records
                if r['case'] == name and r['seed'] == seed and r['method'] == method]))
                for method in methods}
            print(json.dumps(dict(case=name, seed=seed, medians=medians)), flush=True)
    blocks = []
    for size_block in [size, 4*size]:
        g = np.arange(size_block)%2
        h = np.arange(size_block)%2
        a, b = np.ones(size_block)/size_block, np.ones(size_block)/size_block
        rf = rng.uniform(.5, 1.5, size_block)
        sf = rng.uniform(.5, 1.5, size_block)*(1+2*(h == 0))
        for coupling in [.03, .001]:
            C = np.array([[1., coupling], [coupling, 1.]])
            for method in ['ordinary', 'power']:
                solve_blocks(C, g, h, rf, sf, a, b, method=method, tolerance=tolerance)
            jobs = [(method, rep) for rep in range(repeats) for method in ['ordinary', 'power']]
            rng.shuffle(jobs)
            for method, rep in jobs:
                result = solve_blocks(C, g, h, rf, sf, a, b,
                                      method=method, tolerance=tolerance)
                blocks.append(dict(size=size_block, coupling=coupling, method=method,
                                   repeat=rep, **serializable(result)))
            print(json.dumps(dict(block_size=size_block, coupling=coupling)), flush=True)
    return dict(metadata=dict(platform=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, tolerance=tolerance, size=size, seeds=seeds,
                repeats=repeats, seed_start=seed_start, methods=methods,
                depth=4, quadratic2_depth=2, max_horizon=128, relative_budget=.05,
                cooldown=8, input_kernel_construction_timed=False,
                setup_checks_rejections_reconstruction_timed=True),
                general=records, blocks=blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--methods', default='ordinary,linear,quadratic,quadratic2')
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    result = run(args.size, args.repeats, args.seeds, args.tolerance,
                 args.seed_start, args.methods.split(','))
    result['metadata']['complete_run_seconds'] = time.perf_counter()-start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
