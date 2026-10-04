"""Matched reparameterization screen, keeping the same depth/horizon/cost rules."""
import argparse
from pathlib import Path
from .study import run


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for rho in [.001,.01,.1,1.,10.]:
        run(argparse.Namespace(n=256,seeds=3,repeats=3,budget=16384,
                               target=1e-6,families='admm_lasso,admm_shadow',condition=1000.,
                               admm_rho=rho,skip_finite=False,
                               out=str(out/(f'rho_{rho:g}.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    main(p.parse_args().out)
