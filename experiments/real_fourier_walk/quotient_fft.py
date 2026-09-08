"""Real-to-packed-DFT via balanced real polynomial quotient packets.

normalized=True is an explicitly derived Bruun-type control, not a novelty
claim. No complex array or FFT call is used in either computational route.
"""
import math
import numpy as np


def transform(x,normalized=True,order="depth"):
    x=np.asarray(x,dtype=float)
    n=len(x)
    if n<2 or n&(n-1):raise ValueError("N must be a power of two >=2")
    # Output [DC, Re1, -Im1, ..., Re(N/2-1), -Im(N/2-1), Nyquist].
    out=np.empty(n)
    counts={"real_additions":0,"real_multiplications":0,"packet_steps":0,
            "peak_frontier_coordinates":n}
    # A task holds either f mod (t^L-1), or a quadratic-in-t^w residue.
    tasks=[("ridge",x.copy(),0.)]
    frontier=n
    while tasks:
        index=0 if order=="breadth" else -1
        if order=="turn":index=(len(tasks)*7//11)
        kind,v,theta=tasks.pop(index)
        frontier-=len(v);counts["packet_steps"]+=1
        if kind=="ridge":
            if len(v)==2:
                out[0]=v[0]+v[1];out[-1]=v[0]-v[1]
                counts["real_additions"]+=2
            else:
                h=len(v)//2
                tasks.extend([("ridge",v[:h]+v[h:],0.),("quad",v[:h]-v[h:],math.pi/2)])
                frontier+=len(v);counts["real_additions"]+=len(v)
        elif len(v)==2:
            k=int(round(theta*n/(2*math.pi)))
            assert 0<k<n//2
            if normalized:
                out[2*k-1:2*k+1]=v
            else:
                out[2*k-1]=v[0]+math.cos(theta)*v[1]
                out[2*k]=math.sin(theta)*v[1]
                counts["real_multiplications"]+=2;counts["real_additions"]+=1
        else:
            h=len(v)//4
            C=math.cos(theta/2)
            if normalized:
                S=math.sin(theta/2)
                A0,A1,B0,B1=(v[i*h:(i+1)*h] for i in range(4))
                r=C*A1-S*B1;im=S*A1+C*B1
                low=np.concatenate((A0+r,B0+im))
                high=np.concatenate((A0-r,im-B0))
                counts["real_multiplications"]+=4*h
            else:
                r0,r1,r2,r3=(v[i*h:(i+1)*h] for i in range(4))
                u=r0-r2;w=r1+(2*math.cos(theta)+1)*r3
                p=2*C*r3;q=2*C*r2
                low=np.concatenate((u-p,w+q));high=np.concatenate((u+p,w-q))
                counts["real_multiplications"]+=3*h
            counts["real_additions"]+=6*h
            tasks.extend([("quad",low,theta/2),("quad",high,math.pi-theta/2)])
            frontier+=len(v)
        counts["peak_frontier_coordinates"]=max(counts["peak_frontier_coordinates"],frontier)
    return out,counts
