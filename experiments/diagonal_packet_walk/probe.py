"""Reproducible size sweep. Run with M4 Python; write JSON outside mirror."""
import argparse
import json
import platform
import numpy as np
from packet import Packet, direct_matrices, components


def probe(r):
    p = Packet(r)
    Q = p.matrix()
    F, G = direct_matrices(p.m)
    col, phase = p.monomial()
    P = np.zeros_like(Q)
    P[np.arange(p.n), col] = phase
    Cp = (Q@F@Q.conj().T)@P.conj().T
    mask = np.abs(Cp)>1e-10
    blocks = components(mask)
    rng = np.random.default_rng(1729+r)
    x = rng.standard_normal(p.n)+1j*rng.standard_normal(p.n)
    summary = {
        "r": r, "m": p.m, "N": p.n, "a": p.a, "b": p.b,
        "monomial_error": float(np.max(np.abs(Q@G@Q.conj().T-P))),
        "carry_action_error": float(np.max(np.abs(p.carry(x)-Cp@x))),
        "fft_error": float(np.max(np.abs(p.transform(x)-np.fft.fft(x,norm="ortho")))),
        "carry_nonzeros": int(mask.sum()),
        "row_support_sizes": sorted(set(mask.sum(axis=1).tolist())),
        "component_sizes": [len(c) for c in blocks],
        "component_nonzeros": [int(mask[np.ix_(c,c)].sum()) for c in blocks],
    }
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-r",type=int,default=5)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result={"host":platform.platform(),"numpy":np.__version__,
            "threshold":1e-10,"results":[probe(r) for r in range(1,args.max_r+1)]}
    with open(args.out,"w") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
