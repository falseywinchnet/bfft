import numpy as np
from math import cos,pi
def rb2(x):
    N=len(x)
    if N==1: return [(x[0],0.0)]
    if N==2: return [(x[0]+x[1],0.0),(x[0]-x[1],0.0)]
    E=rb2(x[0::2]);O=rb2(x[1::2]);res=[None]*(N//2+1)
    res[0]=(E[0][0]+O[0][0],0.0);res[N//2]=(E[0][0]-O[0][0],0.0);res[N//4]=(E[N//4][0],O[N//4][0])
    for k in range(1,N//4):
        c=cos(2*pi*k/N);ea,eb=E[k];oa,ob=O[k]
        A=ea-eb;Q=2*c*eb;P=2*c*ob;B=oa-ob+2*c*P
        res[k]=(A-P,B+Q);res[N//2-k]=(A+P,B-Q)
    return res
def extract(res,N):
    X=np.zeros(N//2+1,dtype=complex);X[0]=res[0][0];X[N//2]=res[N//2][0]
    for k in range(1,N//2):
        a,b=res[k];th=2*pi*k/N;X[k]=a+b*np.exp(-1j*th)
    return X
def rb4(x):
    N=len(x);M=N//4
    Q=[rb2(x[j::4]) for j in range(4)]
    res=[None]*(N//2+1)
    dc=[Q[j][0][0] for j in range(4)]
    res[0]=(dc[0]+dc[1]+dc[2]+dc[3],0.0)          # DC
    res[N//2]=(dc[0]-dc[1]+dc[2]-dc[3],0.0)        # Nyquist
    def qres(j,idx):
        i=idx%M
        return Q[j][i] if i<=M//2 else Q[j][M-i]
    for k in range(1,N//2):
        c=cos(2*pi*k/N); a4=1-4*c*c; b4=8*c**3-4*c
        zc=[(1.0,0.0),(0.0,1.0),(-1.0,2*c),(-2*c,4*c*c-1)]
        ra=rb=0.0
        for j in range(4):
            Ra,Rb=qres(j,k); la=Ra+Rb*a4; lb=Rb*b4; za,zb=zc[j]
            ra+=za*la-zb*lb; rb+=za*lb+zb*la+2*c*zb*lb
        res[k]=(ra,rb)
    return res
rng=np.random.default_rng(0)
print("=== radix-4 real Bruun DIT, full verification ===")
for L in [4,5,6,7,8,10,12,14]:
    N=1<<L; x=list(rng.standard_normal(N))
    got=extract(rb4(x),N); ref=np.fft.rfft(x)
    err=np.abs(got-ref).max()/max(np.abs(ref).max(),1)
    print(f"  N={N:6d} err={err:.2e} {'OK' if err<1e-9 else 'FAIL'}")
