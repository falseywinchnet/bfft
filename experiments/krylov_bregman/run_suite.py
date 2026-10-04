"""Retained v2 comparison: primary scales plus an untuned ADMM penalty screen."""
import argparse
from pathlib import Path
from .study import run


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    all_small='quadratic_mirror,entropy_dual_affine,entropy_simplex,admm_lasso,entropy_primal_ablation'
    large='quadratic_mirror,entropy_simplex,admm_lasso,entropy_boundary_drift'
    jobs=[('primary_n96',96,all_small,.1,False,16384,5),
          ('primary_n512',512,large,.1,True,32768,5)]
    jobs += [(f'admm_rho_{rho:g}',256,'admm_lasso',rho,True,16384,3)
             for rho in [.001,.01,.1,1.,10.]]
    for name,n,families,rho,skip,budget,repeats in jobs:
        print('START',name,flush=True)
        run(argparse.Namespace(n=n,seeds=3,repeats=repeats,budget=budget,
                               target=1e-6,families=families,condition=1000.,
                               admm_rho=rho,skip_finite=skip,out=str(out/(name+'.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    main(p.parse_args().out)
