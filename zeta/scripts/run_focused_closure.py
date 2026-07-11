#!/usr/bin/env python3
"""Decisive closure experiment at omega=2, a=1.5 only."""
from __future__ import annotations
import csv,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky,solve_triangular,subspace_angles
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from suzuki_kernel import k_omega
from cusp_galerkin import cusp_fitted_matrices
from nystrom_operator import uniform_hankel
from unitary_dilation import halmos_dilation
from tree_csd import recursive_csd
from coherence_analysis import refinement_metrics,effective_rank,description_length,control_contractions

DATA,PLOTS=ROOT/"data",ROOT/"plots";OMEGA=2.;AA=1.5
def write(name,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with (DATA/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def standard_operator(A,M):
    L=cholesky(M,lower=True)
    B=solve_triangular(L,A,lower=True)
    B=solve_triangular(L,B.T,lower=True).T
    return (B+B.T)/2,L
def prolongation(n):
    P=np.zeros((2*n+1,n+1))
    for i in range(n+1):P[2*i,i]=1
    for i in range(n):P[2*i+1,i:i+2]=.5
    return P
def main():
    DATA.mkdir(exist_ok=True);PLOTS.mkdir(exist_ok=True)
    A0=math.log(AA);trace=.5*quad(lambda r:k_omega(OMEGA,r),-2*A0,2*A0,
        points=[0,math.log(2)],epsabs=2e-13,epsrel=2e-13,limit=300)[0]
    levels=(64,128,256,512,1024);ops={};conv=[]
    previous=None
    for n in levels:
        _,A,M=cusp_fitted_matrices(OMEGA,AA,n,order=6);B,L=standard_operator(A,M)
        vals,V=np.linalg.eigh(B);raw=float(np.sum(np.log((1+vals)/(1-vals))))
        corrected=raw+2*(trace-float(vals.sum()))
        row={"omega":OMEGA,"a":AA,"N_cells":n,"dofs":n+1,"exact_adaptive_trace":trace,
             "raw_log_m":raw,"trace_corrected_log_m":corrected,
             "corrected_delta":abs(corrected-previous) if previous is not None else float('nan'),
             "spectral_radius":float(np.max(abs(vals))),"lambda_max":float(vals[-1]),"lambda_min":float(vals[0])}
        conv.append(row);previous=corrected;ops[n]=(B,L,vals,V,M)
    write("focused_fredholm_convergence.csv",conv)
    stability=[]
    for n in levels[:-1]:
        Bc,Lc,vc,Vc,Mc=ops[n];Bf,Lf,vf,Vf,Mf=ops[2*n];P=prolongation(n)
        Q=Lf.T@P@np.linalg.inv(Lc.T)
        # Nested hat spaces make Q orthonormal to roundoff. Do not QR it:
        # that would rotate the coarse eigenvector coordinates unless the
        # corresponding R factor were also applied.
        diff=np.linalg.norm(Bf-Q@Bc@Q.T,2)
        row={"omega":OMEGA,"a":AA,"N":n,"2N":2*n,"extended_operator_norm_difference":float(diff)}
        for k in (1,2,4,8):
            ic=np.argsort(np.abs(vc))[-k:];iff=np.argsort(np.abs(vf))[-k:]
            angles=subspace_angles(Q@Vc[:,ic],Vf[:,iff])
            row[f"leading_{k}_max_angle_rad"]=float(np.max(angles))
            row[f"leading_{k}_projector_distance"]=float(np.sin(np.max(angles)))
        stability.append(row)
    write("focused_operator_stability.csv",stability)
    cells=[];ranks=[];descs=[];by={}
    for n in (32,64,128):
        *_,K=uniform_hankel(OMEGA,AA,n);U,_,_=halmos_dilation(K);rr=recursive_csd(U,OMEGA,AA,n);by[n]=rr
        for x in rr:x["control"]="zeta"
        cells.extend(rr);er=effective_rank(rr);er.update({"N":n,"control":"zeta"});ranks.append(er)
        for x in description_length(rr):x.update({"N":n,"control":"zeta"});descs.append(x)
        for name,C in control_contractions(n).items():
            U,_,_=halmos_dilation(C);cr=recursive_csd(U,float('nan'),AA,n)
            for x in cr:x["control"]=name
            cells.extend(cr);er=effective_rank(cr);er.update({"N":n,"control":name});ranks.append(er)
            for x in description_length(cr):x.update({"N":n,"control":name});descs.append(x)
    ref=[]
    for n in (32,64):
        r=refinement_metrics(by[n],by[2*n]);r.update({"omega":OMEGA,"a":AA,"N":n,"2N":2*n});ref.append(r)
    write("focused_refinement.csv",ref);write("focused_compression_rank.csv",ranks);write("focused_description_length.csv",descs)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.plot([r['N_cells'] for r in conv[1:]],[r['corrected_delta'] for r in conv[1:]],'o-');plt.axhline(1e-8,color='r',ls='--');plt.xscale('log',base=2);plt.yscale('log');plt.xlabel('N cells');plt.ylabel('|corrected log m_N-log m_(N/2)|');plt.tight_layout();plt.savefig(PLOTS/'focused_fredholm_convergence.png',dpi=160);plt.close()
    print('final corrected delta',conv[-1]['corrected_delta'])
    print('refinement',ref)
    print('N=128 ranks',[r for r in ranks if r['N']==128])
if __name__=='__main__':main()
