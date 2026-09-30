"""Untimed checks of state/rule prediction on the SAME retained subspace."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from experiments.entropic_transport_closure.run_sinkhorn import cases
from experiments.entropic_transport_closure.sinkhorn import (
    Kernel, evaluate, conditional, prepare, rollout, qnorm,
)


def run(size=128):
    records = []
    for seed in [0, 1]:
        for name, K, a, b in cases(size, seed):
            kernel = Kernel(K)
            state = evaluate(kernel, a, b, np.zeros(len(b)))
            for step in range(129):
                if step in [8, 32, 128] and state.residual > 1e-8:
                    quadratic = prepare(kernel, state, 4, True)
                    if quadratic is None:
                        continue
                    linear = replace(quadratic, quadratic=False)
                    projection = quadratic.U.T*(state.c/state.c.sum())[None, :]
                    for horizon in [16, 32]:
                        truth = state
                        for _ in range(horizon):
                            truth = evaluate(kernel, a, b, truth.f)
                        trueJ = conditional(kernel, truth,
                            conditional(kernel, truth, quadratic.U), reverse=True)
                        trueH = projection@trueJ
                        change = float(np.linalg.norm(trueH-quadratic.H))
                        for method, model in [('linear', linear), ('quadratic', quadratic)]:
                            proposal = rollout(model, horizon, relative_budget=1e100)
                            if proposal is None or proposal['horizon'] != horizon:
                                records.append(dict(case=name, seed=seed, anchor=step,
                                    horizon=horizon, method=method, finite_rollout=False))
                                continue
                            error = qnorm(truth.y-state.y-proposal['displacement'])
                            predictedH = model.reduced_derivative(proposal['coords'])
                            records.append(dict(case=name, seed=seed, anchor=step,
                                horizon=horizon, method=method, finite_rollout=True,
                                state_error=error, actual_motion=qnorm(truth.y-state.y),
                                bound=proposal['bound'], bound_holds=error <= proposal['bound']+1e-12,
                                projection_bound=proposal['projection_bound'],
                                remainder_bound=proposal['remainder_bound'],
                                rule_error=float(np.linalg.norm(predictedH-trueH)),
                                actual_rule_change=change))
                state = evaluate(kernel, a, b, state.f)
    return dict(size=size, seeds=[0, 1], anchors=[8, 32, 128], horizons=[16, 32],
                note='Untimed diagnostic; no true trajectory is used by solve().', records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.size)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(records=len(result['records']),
          failed_bounds=sum(not r.get('bound_holds', True) for r in result['records']))))
