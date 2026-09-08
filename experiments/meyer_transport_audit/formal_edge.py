"""Dense diagnostic of discrete transport jets; no acceleration timing claim."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap, rms
from experiments.meyer_transport_audit.intermediate_chart import pack, unpack, labels


def pattern(m, x):
    return np.concatenate([labels(x[128:192], m.ru), labels(x[192:], m.rw)])


def run():
    f = (50 + 150 * (np.arange(64) > 32))[None, :]
    m = ReducedMeyerMap(f, .05, 40)
    z = m.initial()
    states = [pack(z)]
    for _ in range(399):
        z = m.step(z)
        states.append(pack(z))
    z64 = unpack(m, states[63])
    eye = np.eye(256)
    A = np.column_stack([pack(m.tangent(z64, unpack(m, e))) for e in eye])
    c = pack(m.step(z64)) - A @ states[63]
    M = np.eye(256) - A
    U, s, VT = np.linalg.svd(M)
    null = s < 1e-10
    left, right = U[:, null], VT[null].T
    pairing = left.T @ right
    rows = []
    for start in [64, 128, 192]:
        x = states[start-1]
        h = states[start] - x
        a = (A - eye) @ h
        b = (A - eye) @ a
        drift = right @ np.linalg.solve(pairing, left.T @ h)
        p0 = pattern(m, x)
        event = next(j for j in range(1, 201)
                     if np.any(pattern(m, states[start+j-1]) != p0))
        frozen = x.copy()
        v = h.copy()
        remainder = np.zeros_like(x)
        max_affine_error = 0.
        max_remainder_error = 0.
        first = {str(p): None for p in [1, 2, 3]}
        probes = []
        jets = [h, a, b]
        for j in range(1, 201):
            frozen += v
            v = A @ v
            # Exact error of the linear-in-pass predictor, carried recursively.
            remainder = A @ remainder + (j-1) * a
            max_remainder_error = max(max_remainder_error,
                                      rms(frozen - x - j*h - remainder))
            if j <= event:
                max_affine_error = max(max_affine_error,
                                       rms(frozen - states[start+j-1]))
            for p in [1, 2, 3]:
                candidate = x + sum(math.comb(j, q+1) * jets[q]
                                    for q in range(p) if q+1 <= j)
                if first[str(p)] is None and np.any(pattern(m, candidate) != p0):
                    first[str(p)] = start+j
            if j in sorted(set([1, 8, 32, event-1, event])):
                probes.append({'horizon':j,
                    'linear_error_rms':rms(frozen-x-j*h),
                    'quadratic_error_rms':rms(frozen-x-j*h-math.comb(j,2)*a),
                    'cubic_error_rms':rms(frozen-x-j*h-math.comb(j,2)*a-math.comb(j,3)*b)})
        assert max_affine_error < 1e-10, max_affine_error
        assert max_remainder_error < 1e-10, max_remainder_error
        assert rms(A @ drift - drift) < 1e-12
        assert np.linalg.norm(left.T @ h) > 1e-3
        rows.append({'start':start, 'actual_event_pass':start+event,
                     'polynomial_first_event_pass':first,
                     'h_rms':rms(h), 'a_over_h':float(np.linalg.norm(a)/np.linalg.norm(h)),
                     'drift_rms':rms(drift), 'drift_tu33':float(drift[128+33]),
                     'drift_invariance_rms':rms(A@drift-drift),
                     'left_drift_norm':float(np.linalg.norm(left.T@h)),
                     'affine_match_through_first_event_rms':max_affine_error,
                     'linear_remainder_identity_rms':max_remainder_error,
                     'probes':probes})
    return {'dimension':256, 'svd_nullity':int(null.sum()),
            'left_right_pairing_condition':float(np.linalg.cond(pairing)),
            'affine_offset_rms':rms(c),
            'note':'Dense floating-point structural diagnostics, not rigorous interval certificates or speed measurements.',
            'rows':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run()
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
