"""Arbitrary real polynomial charts, including moving spectral locations.

Chart factors may change, not just be regrouped. Factors must be pairwise
coprime; individual factors may have repeated roots and then retain jets.
"""
from functools import lru_cache
import sympy as sp
from atlas import T


class MovingAtlas:
    def __init__(self,n):
        self.n=n
        self.c=sp.cos(sp.pi/n)
        self.domain=sp.QQ.algebraic_field(self.c)

    def poly(self,x):return sp.Poly(x,T,domain=self.domain)

    def chart(self,factors):
        p=tuple(self.poly(f).monic() for f in factors)
        if any(f.degree()<1 for f in p) or sum(f.degree() for f in p)!=self.n:
            raise ValueError("A chart must carry exactly N real coordinates")
        for i,f in enumerate(p):
            for g in p[:i]:
                if sp.gcd(f,g).degree()!=0:
                    raise ValueError("Separate packets share a root; merge to retain its jet")
        return p

    @lru_cache(None)
    def weights(self,chart):
        weights=[]
        for i,p in enumerate(chart):
            complement=self.poly(1)
            for j,q in enumerate(chart):
                if i!=j:complement*=q
            weights.append((complement,sp.invert(complement,p)))
        return tuple(weights)

    def input(self,x):
        assert len(x)==self.n
        chart=self.chart([T**self.n-1])
        return chart,(self.poly(sum(v*T**i for i,v in enumerate(x))),)

    def transport(self,state,target):
        source,residues=state
        target=self.chart(target)
        assert len(source)==len(residues)
        assert all(r.degree()<p.degree() for r,p in zip(residues,source))
        weights=self.weights(source)
        # Each corrected source residue has only its packet's degree.
        corrected=[(r*inv).rem(p) for r,p,(_,inv) in zip(residues,source,weights)]
        output=[]
        for q in target:
            if q in source:
                output.append(residues[source.index(q)])
            else:
                value=self.poly(0)
                for (complement,_),r in zip(weights,corrected):
                    value+=(complement.rem(q)*r.rem(q)).rem(q)
                output.append(value.rem(q))
        return target,tuple(output)

    def fourier_chart(self,odd=False):
        if odd:
            return self.chart([T*T-2*sp.chebyshevt(2*k+1,self.c)*T+1 for k in range(self.n//2)])
        return self.chart([T-1]+[T*T-2*sp.chebyshevt(2*k,self.c)*T+1 for k in range(1,self.n//2)]+[T+1])

    def oracle(self,x,chart):
        p=self.input(x)[1][0]
        return tuple(p.rem(q) for q in chart)

    def readout(self,state,odd=False):
        chart,residues=state
        assert chart==self.fourier_chart(odd)
        out=[]
        for i,r in enumerate(residues):
            if not odd and i in (0,len(residues)-1):
                out.append(r.nth(0));continue
            k=2*i+1 if odd else 2*i
            c=sp.chebyshevt(k,self.c)
            s=sp.chebyshevt(abs(self.n//2-k),self.c)
            out.extend([r.nth(0)+c*r.nth(1),s*r.nth(1)])
        return out
