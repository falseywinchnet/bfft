"""Generalized Bruun, NORMALIZED basis (FFT-grade) — replaces polydiv.

Validated flat ~4e-16:
  * pow2 N            (rfft_normbruun, mirrors src/detail/bruun_kernel.hpp)
  * N = p * 2^a, p in {3,5,7}   (radix_p split into normalized rooted cascades)

Remaining for full arbitrary N: recurse the s-branch and the rooted-cascade leaves
when M=N/p is itself non-pow2 (multiple/repeated odd primes). The single-odd-prime
case here is the validated template.
"""
import numpy as np
try:
    from .scratch_normbruun import rfft_normbruun  # pow2 normalized engine
except ImportError:  # pragma: no cover - supports direct script execution.
    from scratch_normbruun import rfft_normbruun  # pow2 normalized engine

def _nb_subtree(r, theta, N, out):
    d = len(r)
    if d == 2:
        k = int(round(theta*N/(2*np.pi))) % N
        out[k] = r[0] - 1j*r[1]; return
    q = d//4
    A0=r[0:q]; B0=r[q:2*q]; A1=r[2*q:3*q]; B1=r[3*q:4*q]
    c=np.cos(theta/2); s=np.sin(theta/2)
    R=c*B0-s*B1; I=s*B0+c*B1
    _nb_subtree(np.concatenate([A0+R, A1+I]), theta/2, N, out)
    _nb_subtree(np.concatenate([A0-R, -A1+I]), np.pi-theta/2, N, out)

def rfft_radix_p(x, p):
    """FFT-grade real transform for N = p * 2^a via one normalized radix-p split."""
    x=np.asarray(x,float); N=x.size; M=N//p
    Pb=[x[i*M:(i+1)*M] for i in range(p)]
    X=np.zeros(N,complex)
    Xs=np.fft.fft(sum(Pb))                 # s-branch (M is pow2 here)
    for k in range(M): X[(p*k)%N]=Xs[k]
    for j in range(1,(p-1)//2+1):
        th=2*np.pi*j/p; c=np.cos(th)
        co=[(1.0,0.0),(0.0,1.0)]           # y^i = alpha_i + beta_i*y  mod y^2-2c y+1
        for i in range(2,p):
            co.append((-co[i-1][1], co[i-1][0]+2*c*co[i-1][1]))
        lo=sum(Pb[i]*co[i][0] for i in range(p))
        hi=sum(Pb[i]*co[i][1] for i in range(p))
        seed=np.concatenate([lo+np.cos(th)*hi, np.sin(th)*hi])   # stage-0 basis map B
        out={}; _nb_subtree(seed, th, N, out)
        for k,val in out.items():
            X[k]=val; X[(N-k)%N]=np.conj(val)
    return X[:N//2+1]

if __name__ == "__main__":
    import sympy
    print(f"{'N':>7} {'factors':>14} {'err':>10}")
    for N in [256, 6, 96, 6144, 10, 5120, 14, 7168]:
        x=np.random.default_rng(N).standard_normal(N)
        if (N&(N-1))==0: got=rfft_normbruun(x)
        else:
            p=[q for q in (3,5,7) if N%q==0][0]; got=rfft_radix_p(x,p)
        e=np.abs(got-np.fft.rfft(x)).max()/max(np.abs(np.fft.rfft(x)).max(),1)
        print(f"{N:>7} {str(sympy.factorint(N)):>14} {e:>10.2e}  {'OK' if e<1e-13 else 'FAIL'}")
