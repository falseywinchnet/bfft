"""Stage observer for the shipped real Bruun/DIP cell algebra.

The production transform remains untouched. This observer mirrors its documented
cell equations to expose the experimental packet observables unavailable through
the kernel's public forward/inverse API.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def _stage(src, t, N):
    e, q = 1 << t, N >> t; q2 = q >> 1
    dst = np.empty_like(src); records=[]
    for j in range(q2):
        ev,od=src[j],src[q2+j];dst[j]=ev+od;dst[q2+j]=ev-od
    if e>=2:
        d=e>>1
        dst[(2*d)*q2:(2*d+1)*q2]=src[q:q+q2]
        dst[(2*d+1)*q2:(2*d+2)*q2]=src[q+q2:q+q]
    for d in range(1,e>>1):
        theta=np.pi*d/e;c=np.cos(theta);s=np.sin(theta)
        sa,sb=(2*d)*q,(2*d+1)*q;la,lb=(2*d)*q2,(2*d+1)*q2
        ha,hb=(2*(e-d))*q2,(2*(e-d)+1)*q2
        ea,oa=src[sa:sa+q2],src[sa+q2:sa+q]
        eb,ob=src[sb:sb+q2],src[sb+q2:sb+q]
        rr=c*oa-s*ob;ii=s*oa+c*ob
        dst[la:la+q2]=ea+rr;dst[lb:lb+q2]=eb+ii
        dst[ha:ha+q2]=ea-rr;dst[hb:hb+q2]=ii-eb
        # Apply the exact local inverse to make residual a packet observable.
        rea=(dst[la:la+q2]+dst[ha:ha+q2])/2
        rrr=(dst[la:la+q2]-dst[ha:ha+q2])/2
        rii=(dst[lb:lb+q2]+dst[hb:hb+q2])/2
        reb=(dst[lb:lb+q2]-dst[hb:hb+q2])/2
        roa=c*rrr+s*rii;rob=-s*rrr+c*rii
        denom=max(np.linalg.norm(np.r_[ea,oa,eb,ob]),np.finfo(float).tiny)
        residual=np.linalg.norm(np.r_[rea-ea,roa-oa,reb-eb,rob-ob])/denom
        records.append({"depth":t,"d":d,"e":e,"span_width":q,
            "phase_table_index":d*N//(2*e),"theta":theta,
            "input_energy":float(np.sum(ea*ea+oa*oa+eb*eb+ob*ob)),
            "low_child_energy":float(np.sum(dst[la:la+q2]**2+dst[lb:lb+q2]**2)),
            "high_child_energy":float(np.sum(dst[ha:ha+q2]**2+dst[hb:hb+q2]**2)),
            "roundtrip_residual":float(residual)})
    return dst,records


def observe_forward(x, transform_id=0, direction="forward"):
    x=np.asarray(x,dtype=float);N=x.size
    if N<4 or N&(N-1):raise ValueError("length must be a power of two >=4")
    state=x.copy();rows=[]
    for t in range(int(np.log2(N))):
        state,rr=_stage(state,t,N)
        for r in rr:r.update({"transform_id":transform_id,"direction":direction})
        rows.extend(rr)
    return state,rows


def write_observations(path:Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=("transform_id","direction","depth","d","e","span_width",
          "phase_table_index","theta","input_energy","low_child_energy",
          "high_child_energy","roundtrip_residual")
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
