#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from xi_validation import validate_mellin
from nystrom_operator import nystrom_matrix, fredholm_summary, uniform_hankel
from dip_hankel import compare_dense_dip, build_binding
from unitary_dilation import halmos_dilation
from tree_csd import recursive_csd
from coherence_analysis import (refinement_metrics, effective_rank,
                                description_length, control_contractions)
from dip_packet_observer import observe_forward, write_observations

DATA, PLOTS, BUILD = ROOT / "data", ROOT / "plots", ROOT / "build"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_cell_parameters(rows):
    csv_path = DATA / "cell_parameters.csv"
    write_csv(csv_path, rows)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), DATA / "cell_parameters.parquet")
        return "parquet"
    except Exception as exc:
        # Never disguise CSV as Parquet. Preserve data and a machine-readable
        # explanation until optional pyarrow is installed.
        (DATA / "cell_parameters.parquet.unavailable.json").write_text(
            json.dumps({"reason": str(exc), "csv_fallback": str(csv_path)}, indent=2))
        return "csv_fallback"


def plots(fredholm, cells, refine, ranks, descriptions):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    for omega in sorted({r["omega"] for r in fredholm}):
        rr = [r for r in fredholm if r["omega"] == omega and r["N"] == max(q["N"] for q in fredholm if q["omega"] == omega)]
        plt.plot([r["a"] for r in rr], [r["m"] for r in rr], "o-", label=str(omega))
    plt.yscale("log"); plt.xlabel("a"); plt.ylabel("m_omega(a)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "m_omega_vs_a.png", dpi=160); plt.close()
    for omega in sorted({r["omega"] for r in cells if r.get("control", "zeta") == "zeta"}):
        rr=[r for r in cells if r["omega"]==omega and r.get("control","zeta")=="zeta"]
        plt.scatter([r["depth"] for r in rr],[r["theta_mean"] for r in rr],s=12,label=str(omega))
    plt.xlabel("depth"); plt.ylabel("mean principal angle"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "cell_angles_by_depth.png", dpi=160); plt.close()
    if refine:
        for omega in sorted({r["omega"] for r in refine}):
            rr=[r for r in refine if r["omega"]==omega]
            plt.plot([r["coarse_N"] for r in rr],[r["max_drift"] for r in rr],"o-",label=str(omega))
        plt.yscale("log"); plt.xlabel("coarse N"); plt.ylabel("maximum drift"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "refinement_error_vs_N.png", dpi=160); plt.close()
    for name in sorted({r["control"] for r in ranks}):
        rr=[r for r in ranks if r["control"]==name]
        plt.plot([r["N"] for r in rr],[r["rank_99.9pct"] for r in rr],"o-",label=name)
    plt.xlabel("N"); plt.ylabel("99.9% effective rank"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(PLOTS / "effective_rank_vs_N.png", dpi=160); plt.close()
    for name in sorted({r["control"] for r in descriptions}):
        rr=[r for r in descriptions if r["control"]==name and r["N"]==max(q["N"] for q in descriptions if q["control"]==name)]
        plt.plot([r["parameters"] for r in rr],[r["relative_error"] for r in rr],"o-",label=name)
    plt.xscale("log"); plt.yscale("symlog",linthresh=1e-14); plt.xlabel("parameters"); plt.ylabel("relative error"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(PLOTS / "description_length_curve.png", dpi=160); plt.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--full",action="store_true"); ap.add_argument("--skip-mellin",action="store_true")
    args=ap.parse_args(); DATA.mkdir(exist_ok=True); BUILD.mkdir(exist_ok=True)
    build_binding()
    rng=np.random.default_rng(20260711)
    _,packet_rows=observe_forward(rng.standard_normal(256),transform_id=0)
    write_observations(DATA/"dip_packet_observables.csv",packet_rows)
    write_csv(DATA / "fft_validation.csv", [{"contract":"DIP forward/inverse is trusted", "fft_revalidation_performed":False,
               "replacement_check":"reflection+zero-padding+DIP convolution+cropping vs dense"}])
    if not args.skip_mellin: validate_mellin(DATA / "mellin_validation.csv")
    omegas=(2,1.5,1.25,1.1); avals=(1.05,1.1,1.25,1.5,2,3,4)
    ns=(64,128,256,512,1024) if args.full else (64,128)
    fred=[]
    for omega in omegas:
        for a in avals:
            previous=None
            for n in ns:
                _,_,_,K=uniform_hankel(omega,a,n)
                row={"omega":omega,"a":a,**fredholm_summary(K)}
                row["discretization"]="piecewise-constant Galerkin; triangular cell averages split at log integers"
                row["log_m_delta_from_previous"] = abs(row["log_m"]-previous) if previous is not None else float("nan")
                row["converged_1e-8"] = bool(previous is not None and row["log_m_delta_from_previous"]<1e-8)
                previous=row["log_m"]; fred.append(row)
    write_csv(DATA / "fredholm_baseline.csv",fred)
    dip=[]
    for omega in omegas:
        for a in (1.25,2,4):
            _,du,k,_=uniform_hankel(omega,a,128)
            for r in compare_dense_dip(k,du,sizes=(16,32,64,128)):
                dip.append({"omega":omega,"a":a,**r})
    write_csv(DATA / "dip_dense_comparison.csv",dip)
    cells=[]; refine=[]; ranks=[]; desc=[]
    tree_ns=(32,64,128) if args.full else (16,32,64)
    for omega in omegas:
        by_n={}
        for n in tree_ns:
            *_,K=uniform_hankel(omega,2,n)
            if np.linalg.norm(K,2)>=1: continue
            U,_,res=halmos_dilation(K)
            rr=recursive_csd(U,omega,2,n)
            for x in rr: x.update(res); x["control"]="zeta"
            cells.extend(rr); by_n[n]=rr
            er=effective_rank(rr); er.update({"omega":omega,"a":2,"N":n,"control":"zeta"}); ranks.append(er)
            for x in description_length(rr): x.update({"omega":omega,"a":2,"N":n,"control":"zeta"}); desc.append(x)
        for lo,hi in zip(tree_ns[:-1],tree_ns[1:]):
            if lo in by_n and hi in by_n:
                x=refinement_metrics(by_n[lo],by_n[hi]); x.update({"omega":omega,"a":2,"coarse_N":lo,"fine_N":hi}); refine.append(x)
    for n in tree_ns:
        for name,K in control_contractions(n).items():
            U,_,res=halmos_dilation(K); rr=recursive_csd(U,float("nan"),2,n)
            for x in rr: x.update(res); x["control"]=name
            cells.extend(rr)
            er=effective_rank(rr); er.update({"omega":float("nan"),"a":2,"N":n,"control":name}); ranks.append(er)
            for x in description_length(rr): x.update({"omega":float("nan"),"a":2,"N":n,"control":name}); desc.append(x)
    fmt=write_cell_parameters(cells)
    write_csv(DATA / "refinement_coherence.csv",refine); write_csv(DATA / "compression_rank.csv",ranks); write_csv(DATA / "description_length.csv",desc)
    plots(fred,cells,refine,ranks,desc)
    (DATA / "baseline_manifest.json").write_text(json.dumps({"cell_format":fmt,"full":args.full},indent=2))
    print(json.dumps({"max_dip_error":max(r["relative_error"] for r in dip),"cell_format":fmt,"rows":len(cells)},indent=2))

if __name__=="__main__": main()
