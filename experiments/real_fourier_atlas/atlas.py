"""Real polynomial-residue atlas. No FFT, dense inverse, or complex state.

Exact algebraic coefficients are used so path closure is an algebraic test,
not a floating-point threshold. This is a traversal oracle, not a fast kernel.
"""
from functools import lru_cache
from itertools import combinations
import sympy as sp

T=sp.Symbol("t")


def canonical(blocks):
    return tuple(sorted(tuple(sorted(b)) for b in blocks))


class Atlas:
    def __init__(self,n):
        if n<4 or n&(n-1):raise ValueError("Use a power of two N>=4")
        self.n=n
        self.c=sp.cos(2*sp.pi/n)
        self.domain=sp.QQ.algebraic_field(self.c)
        self.atoms=tuple(range(n//2+1))
        self.start=(self.atoms,)
        self.end=tuple((k,) for k in self.atoms)
        self.factors={0:self.poly(T-1),n//2:self.poly(T+1)}
        for k in range(1,n//2):
            self.factors[k]=self.poly(T*T-2*sp.chebyshevt(k,self.c)*T+1)
        assert self.modulus(self.atoms)==self.poly(T**n-1)

    def poly(self,expr):return sp.Poly(expr,T,domain=self.domain)

    @lru_cache(None)
    def modulus(self,block):
        p=self.poly(1)
        for k in block:p*=self.factors[k]
        return p

    def validate(self,partition):
        p=canonical(partition)
        assert all(p) and sorted(k for b in p for k in b)==list(self.atoms)
        return p

    def input(self,x):
        assert len(x)==self.n
        return {self.atoms:self.poly(sum(v*T**j for j,v in enumerate(x)))}

    @lru_cache(None)
    def inverse_mod(self,left,right):
        # Extended polynomial Euclid, on geometry only, not a numerical solve.
        return sp.invert(self.modulus(left),self.modulus(right))

    def merge(self,pieces):
        """CRT by r_new = r + A*((s-r)*A^-1 mod B)."""
        block,r=pieces[0]
        for other,s in pieces[1:]:
            correction=((s-r)*self.inverse_mod(block,other)).rem(self.modulus(other))
            r=r+self.modulus(block)*correction
            block=tuple(sorted(block+other))
            assert r.degree()<self.modulus(block).degree()
        return r

    def move(self,state,target):
        """Change partitions by intersections; never reconstruct a global input.

        No hidden source vector or walk history is available to this method.
        A full polynomial is formed only if the requested target is that packet.
        """
        self.validate(state)
        target=self.validate(target)
        output={}
        for b in target:
            if b in state:
                output[b]=state[b]
                continue
            pieces=[]
            for old,r in sorted(state.items()):
                intersection=tuple(sorted(set(b)&set(old)))
                if intersection:
                    pieces.append((intersection,r.rem(self.modulus(intersection))))
            output[b]=self.merge(pieces)
        assert sum(self.modulus(b).degree() for b in output)==self.n
        return output

    def oracle(self,x,partition):
        """Independent direct remainder from input; never called by move."""
        full=self.input(x)[self.atoms]
        return {b:full.rem(self.modulus(b)) for b in self.validate(partition)}

    def readout(self,state):
        """N real coordinates [DC,Re1,-Im1,...,Nyquist], no complex evaluation."""
        assert canonical(state)==self.end
        out=[state[(0,)].nth(0)]
        for k in range(1,self.n//2):
            r=state[(k,)]
            c=sp.chebyshevt(k,self.c)
            s=sp.chebyshevt(abs(self.n//4-k),self.c)
            out.extend([sp.expand(r.nth(0)+c*r.nth(1)),sp.expand(s*r.nth(1))])
        out.append(state[(self.n//2,)].nth(0))
        return out


def partitions(items):
    """All set partitions, with canonical output and no duplicates."""
    if not items:
        yield ()
        return
    first,*rest=items
    for p in partitions(rest):
        yield canonical(((first,),)+p)
        for i,b in enumerate(p):
            yield canonical(p[:i]+(tuple(sorted((first,)+b)),)+p[i+1:])


def neighbors(partition):
    """Edges split one packet in two, or merge two packets; no fixed tree."""
    result=set()
    for i,j in combinations(range(len(partition)),2):
        result.add(canonical([b for k,b in enumerate(partition) if k not in (i,j)]
                             +[partition[i]+partition[j]]))
    for i,b in enumerate(partition):
        for mask in range(1,2**(len(b)-1)):
            left=(b[0],)+tuple(b[k+1] for k in range(len(b)-1) if mask&(1<<k))
            right=tuple(k for k in b if k not in left)
            if right:result.add(canonical(partition[:i]+partition[i+1:]+(left,right)))
        if len(b)>1:
            result.add(canonical(partition[:i]+partition[i+1:]+((b[0],),b[1:])))
    return sorted(result)
