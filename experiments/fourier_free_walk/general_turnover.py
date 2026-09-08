"""Three-wire GL(3) chart change, permitting genuinely oblique local frames.

Write M=[[m00,v],[u,L]]. Choose a so K=L-a*u*v has rank one.
Then B_left * A * B_right = M, where A=[[m00,1],[1,a]],
B_left=[u,w], B_right=[v;z], and K=w*z. All are two-wire cells.
This chart is admitted only when its explicit denominator and pivots are nonzero.
"""
import numpy as np
from walk import Gate
from turnover import product


def general(word,wires):
    m=product(word,wires);u=m[1:,0];v=m[0,1:];L=m[1:,1:]
    adj=np.array([[L[1,1],-L[0,1]],[-L[1,0],L[0,0]]])
    denominator=float(v@adj@u)
    if abs(denominator)<1e-11:return None
    a=float(np.linalg.det(L)/denominator);K=L-a*np.outer(u,v)
    p,q=np.unravel_index(np.argmax(abs(K)),K.shape)
    if abs(K[p,q])<1e-11:return None
    w=K[:,q];z=K[p,:]/K[p,q]
    left=np.column_stack([u,w]);middle=np.array([[m[0,0],1.],[1.,a]]);right=np.row_stack([v,z])
    # A numerical admission guard, not a claim these omitted charts cannot exist.
    for cell in (left,middle,right):
        if abs(np.linalg.det(cell))<1e-11 or np.max(abs(cell))>1e5:return None
    i,j,k=wires;result=[Gate(j,k,right),Gate(i,j,middle),Gate(j,k,left)]
    if np.max(abs(product(result,wires)-m))>1e-9:return None
    return result
