#!/usr/bin/env python3
from __future__ import annotations
import csv, math, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from nystrom_operator import uniform_hankel, fredholm_summary

DATA,PLOTS=ROOT/"data",ROOT/"plots"
def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
    omegas=(2,1.5,1.25,1.1,1.05,1.01,.95,.8,.65,.55)
    avals=(1.05,1.1,1.25,1.5,2,3,4); rows=[]
    for omega in omegas:
        for a in avals:
            _,_,_,K=uniform_hankel(omega,a,128)
            s=fredholm_summary(K); rows.append({"omega":omega,"a":a,**s,
                "quadrature":"midpoint_log_grid_split-aware-kernel-sampling",
                "ordinary_fredholm_claim":bool(omega>.5 and s["determinant_arguments_positive"])})
    DATA.mkdir(exist_ok=True);write(DATA/"omega_a_spectral_sweep.csv",rows)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    for key,name in [("spectral_radius","spectral_radius_heatmap.png"),("defect_margin","defect_margin_heatmap.png")]:
        M=np.array([[next(r[key] for r in rows if r["omega"]==o and r["a"]==a) for a in avals] for o in omegas])
        plt.imshow(M,aspect="auto",origin="lower",extent=[0,len(avals)-1,0,len(omegas)-1]);plt.colorbar(label=key)
        plt.xticks(range(len(avals)),avals);plt.yticks(range(len(omegas)),omegas);plt.xlabel("a");plt.ylabel("omega");plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()
if __name__=="__main__":main()
