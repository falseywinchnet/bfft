"""Four-real-coordinate exchanges in an ordered quadratic Newton state."""
import sympy as sp
import math

T=sp.Symbol("t")


def swap(a,b,p,q):
    """a+p*b = a_new+q*b_new; monic quadratics p=(p0,p1), q=(q0,q1).

    No differences of roots are divided by. Works at repeated/coincident roots.
    """
    delta0=p[0]-q[0];delta1=p[1]-q[1]
    carry=delta1*b[1]
    return ((a[0]+delta0*b[0]-q[0]*carry,
             a[1]+delta1*b[0]+delta0*b[1]-q[1]*carry),
            (b[0]+carry,b[1]))


class Walk:
    def __init__(self,coefficients,factors):
        self.values=[tuple(x) for x in coefficients]
        self.factors=[tuple(x) for x in factors]
        assert len(self.values)==len(self.factors)
        self.steps=0

    @classmethod
    def samples(cls,x):
        assert len(x)%2==0
        # f=a0+t²*a1+t^4*a2+...: sample coordinates already are this chart.
        return cls(list(zip(x[::2],x[1::2])),[(0,0)]*(len(x)//2))

    def exchange(self,i):
        self.values[i],self.values[i+1]=swap(self.values[i],self.values[i+1],
                                          self.factors[i],self.factors[i+1])
        self.factors[i],self.factors[i+1]=self.factors[i+1],self.factors[i]
        self.steps+=1

    def replace(self,i,factor):
        """Move factor to the end, change it, and move back: local exchanges only.

        The last factor does not enter the degree-bounded Newton expansion.
        Replacing it changes no values; all route changes are explicit exchanges.
        """
        for j in range(i,len(self.factors)-1):self.exchange(j)
        self.factors[-1]=tuple(factor)
        for j in reversed(range(i,len(self.factors)-1)):self.exchange(j)

    def polynomial_oracle(self):
        """For validation only. Exchange/replace never call this."""
        value=sp.S(0)
        for (a,b),(p0,p1) in reversed(list(zip(self.values,self.factors))):
            value=sp.expand(a+b*T+(T*T+p1*T+p0)*value)
        return value

    def expose(self,i):
        """Move a factor to the first position; first pair is its full residue."""
        for j in reversed(range(i)):self.exchange(j)
        return self.values[0]


def walk_rfft(x,order="natural"):
    """Actual r2c endpoint by adjacent real exchanges. Quadratic work, not FFT."""
    n=len(x);m=n//2
    if n<4 or n&(n-1):raise ValueError("N must be a power of two >=4")
    indices=list(range(m))
    if order=="bitreverse":
        bits=m.bit_length()-1
        indices.sort(key=lambda k:int(format(k,'0%db'%bits)[::-1],2))
    factors={k:((-1.,0.) if k==0 else (1.,-2*math.cos(2*math.pi*k/n))) for k in indices}
    w=Walk.samples(x)
    for i,k in enumerate(indices):w.replace(i,factors[k])
    out=[0.]*n
    max_coordinate=max(abs(v) for pair in w.values for v in pair)
    for k in indices:
        u,v=w.expose(w.factors.index(factors[k]))
        max_coordinate=max(max_coordinate,*(abs(v) for pair in w.values for v in pair))
        if k==0:out[0]=u+v;out[-1]=u-v
        else:
            theta=2*math.pi*k/n
            out[2*k-1]=u+math.cos(theta)*v;out[2*k]=math.sin(theta)*v
    assert w.steps==3*m*(m-1)//2
    return out,{"exchanges":w.steps,"largest_sampled_coordinate":max_coordinate,
                "state_real_coordinates":n}
