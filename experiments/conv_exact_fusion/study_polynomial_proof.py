"""Small exact-algebra coverage probe; no production timing claim."""
import argparse
import json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from experiments.conv_exact_fusion.polynomial_proof import (
    poly, first_cell, compile_line, certified, restrict, evaluate,
    scalar_profile, raw_bank, nonnegative,
)
from experiments.convstar import raw_current_jet_bank, ordered_sign_ledger, project_signed_fibres


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(905)
    # Independent existing float64 implementation checks the rational oracle.
    max_error=0.0
    for n in (5,7,11):
        for _ in range(10):
            values=rng.integers(-5,6,size=n)
            a,d=raw_current_jet_bank(values[:,None])
            expected=project_signed_fibres(a,ordered_sign_ledger(a,d),d)[:,:,0]
            exact=scalar_profile(list(map(int,values)))
            actual=np.array([[float(p[0]) for p in c] for c in exact])
            max_error=max(max_error,float(np.max(np.abs(actual-expected))))
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)
    coarse=np.load('output/support_geometry/conv_admission_band/conv_admission_band_full/arrays.npz')['coarse']
    scenes={
        'integer_nonseparable':rng.integers(-5,6,size=(7,7)).tolist(),
        'smooth_nonseparable':[[i*i+3*j*j+i*j for i in range(7)] for j in range(7)],
        'camera_coarse_crop_24_31':[[F(float(x)) for x in row] for row in coarse[24:31,24:31]],
    }
    report={'scope':'7x7 line certificates, two selected source cells in each order; crop uses its own boundary closure, not a full-image acceptance estimate',
            'independent_float64_profiles':30,'maximum_current_error':max_error,'scenes':{}}
    for name,rows in scenes.items():
        records=[]
        for direction,data in [('xy',rows),('yx',list(map(list,zip(*rows))))]:
            for cell in (1,3):
                y=first_cell(data,cell)
                for subdivisions in (1,8):
                    successes=0
                    nonlinear=0
                    failed={}
                    for part in range(subdivisions):
                        z=[restrict(p,F(part,subdivisions),F(part+1,subdivisions)) for p in y]
                        c,checks=compile_line(z)
                        ok=certified(checks)
                        for p,strict,kind in checks:
                            if not nonnegative(p,strict):failed[kind]=failed.get(kind,0)+1
                        if not ok:continue
                        successes+=1
                        if c!=raw_bank(z)[0]:nonlinear+=1
                        # Off-midpoint rational verification against fresh admission.
                        for u in (F(0),F(1,7),F(2,3),F(1)):
                            direct=scalar_profile([evaluate(p,u) for p in z])
                            assert [[evaluate(p,u) for p in a] for a in c] == [[p[0] for p in a] for a in direct]
                    records.append({'order':direction,'cell':cell,'parts':subdivisions,
                                    'certified':successes,'certified_with_nonlinear_correction':nonlinear,
                                    'failed_inequalities_by_kind':failed})
        report['scenes'][name]=records
        print(name,[(r['parts'],r['certified']) for r in records],flush=True)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
