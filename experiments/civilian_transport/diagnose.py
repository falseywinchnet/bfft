"""Mechanism diagnostics on development cases only; no test retuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import FAMILIES, make_case
from .study import evaluate, save


def run(results, out):
    config = json.loads(Path(results).read_text())['metadata']['configs']['geometric']
    rows = []
    for particles, noise in [(384, True), (1536, True), (6144, True), (1536, False)]:
        for i, family in enumerate(FAMILIES):
            for corrupt in (False, True):
                case = make_case(family, 100+10*i+int(corrupt), corrupt)
                parameters = dict(config, adaptive_noise=noise)
                result, _ = evaluate(case, 'geometric', parameters, particles=particles,
                                     forecast_samples=256, stride=20)
                rows.append({'particles': particles, 'adaptive_noise': noise, 'family': family,
                             'corrupt': corrupt, **result})
        print('completed diagnostic', particles, noise, flush=True)
    summary = []
    for particles, noise in [(384, True), (1536, True), (6144, True), (1536, False)]:
        values = [r for r in rows if r['particles'] == particles and r['adaptive_noise'] == noise]
        summary.append({'particles': particles, 'adaptive_noise': noise,
                        'track_rmse': float(np.sqrt(np.mean([r['track_sq'] for r in values]))),
                        'forecast2_rmse': float(np.sqrt(np.mean([r['forecast2_sq'] for r in values]))),
                        'noise_rmse': float(np.sqrt(np.mean([r['noise_sq'] for r in values]))),
                        'bias_rmse': float(np.sqrt(np.mean([r['bias_sq'] for r in values]))),
                        'track_95': float(np.mean([r['track_95'] for r in values]))})
    save(out, {'scope': 'development-only mechanism diagnostic; no altered held-out claim',
               'frozen_base_config': config, 'summary': summary, 'runs': rows})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.results, args.out)
