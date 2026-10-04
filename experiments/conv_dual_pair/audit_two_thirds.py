"""Independent rational and sampled audits of the 2/3 mutual-pair derivation."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import sympy as s

from experiments.conv_dual_pair.derive_two_thirds import (
    families,matrices,basin_moment,z,
)


def original_analysis():
    h,g,p=families()
    substitution={p[0]:s.Rational(603,163840),p[1]:s.Rational(2307019,4423680)}
    return matrices(h,g)[0].subs(substitution)


def canonical_polynomial_defects(A,order=4):
    # Formal Taylor arithmetic at exp(t), not a finite-difference estimate.
    aa=[A.applyfunc(lambda v:sum(s.expand(v).coeff(z,l)*(3*l)**k/s.factorial(k)
                               for l in range(-2,3))) for k in range(order+1)]
    adjoint=[(-1)**k*v.T for k,v in enumerate(aa)]
    gram=[sum((aa[j]*adjoint[k-j] for j in range(k+1)),s.zeros(2)) for k in range(order+1)]
    inv=[gram[0].inv()]
    for k in range(1,order+1):
        inv.append(-inv[0]*sum((gram[j]*inv[k-j] for j in range(1,k+1)),s.zeros(2)))
    grow=[sum((adjoint[j]*inv[k-j] for j in range(k+1)),s.zeros(3,2)) for k in range(order+1)]
    measured=[s.Matrix([basin_moment(s.Rational(3,2)*j,k)/s.factorial(k) for j in range(2)]) for k in range(order+1)]
    return [[s.factor(v*s.factorial(k)) for v in
             sum((grow[j]*measured[k-j] for j in range(k+1)),s.zeros(3,1))-
             s.Matrix([s.Rational(j)**k/s.factorial(k) for j in range(3)])]
            for k in range(order+1)]


def apply_bank(rows,inputs,source_block):
    """Periodic scalar polyphase bank, rows are index->coefficient dictionaries."""
    values=np.asarray(inputs,float)
    blocks=len(values)//source_block
    out=np.empty(blocks*len(rows))
    for block in range(blocks):
        for phase,row in enumerate(rows):
            out[block*len(rows)+phase]=sum(v*values[(source_block*block+int(k))%len(values)] for k,v in row.items())
    return out


def main():
    root=Path(__file__).parent
    packet=json.loads((root/'two-thirds-feasibility.json').read_text())
    A=original_analysis()
    gram=A*A.subs(z,1/z).T
    determinant=s.factor(gram.det())
    numerator,denominator=s.fraction(determinant)
    poles=s.nroots(numerator,maxsteps=200)
    defects=canonical_polynomial_defects(A)
    # Verify that the original raw analysis conserves the physical line sum.
    assert all(s.factor(sum(A[:,j]).subs(z,1)-s.Rational(2,3))==0 for j in range(3))
    assert defects[1]==[0,s.Rational(29973,655360),-s.Rational(29973,655360)]
    h,g,p=families()
    familyA,_=matrices(h,g)
    kernel=familyA.row(0).cross(familyA.row(1))
    orthogonality=[s.factor(sum(s.expand(kernel[j]).coeff(z,l)*s.Rational(j-3*l)**k
                     for j in range(3) for l in range(-2,3))) for k in (1,3,4)]
    mass=sum(familyA[:,0]).subs(z,1)-s.Rational(2,3)
    impossibility=s.groebner(orthogonality+[mass],*p[:2])
    assert len(impossibility.polys)==1 and impossibility.polys[0].as_expr()==1
    rng=np.random.default_rng(4421)
    candidates=[]
    for candidate in packet['candidates']:
        low=rng.normal(size=128)
        high=apply_bank(candidate['grow'],low,2)
        recovered=apply_bank(candidate['shrink'],high,3)
        step=np.r_[np.zeros(64),np.ones(64)]
        response=apply_bank(candidate['grow'],step,2)[80:112]
        hh=[{int(k):v for k,v in row.items()} for row in candidate['shrink']]
        physical_column_sums=[sum(v for row in hh for k,v in row.items() if k%3==j)*1.5 for j in range(3)]
        candidates.append({
            'parameter':candidate['parameter'],
            'coarseCycleMaxError':float(np.max(np.abs(low-recovered))),
            'unitStepMinimum':float(response.min()),
            'unitStepMaximum':float(response.max()),
            'physicalColumnMasses':physical_column_sums,
            'validConservativeAdmittedPair':False,
        })
    # Independently sample canonical reciprocity / projection loss and rank.
    function=s.lambdify(z,A,'numpy')
    reciprocal=0.;loss_error=0.;smallest=1.
    for omega in np.linspace(-np.pi,np.pi,1025):
        a=np.asarray(function(np.exp(1j*omega)),complex)
        gg=a.conj().T@np.linalg.inv(a@a.conj().T)
        reciprocal=max(reciprocal,float(np.max(np.abs(a@gg-np.eye(2)))))
        loss_error=max(loss_error,abs(float(np.linalg.norm(np.eye(3)-gg@a)**2)-1))
        smallest=min(smallest,float(np.linalg.svd(a,compute_uv=False).min()))
    result={
        'runtime':platform.python_version(),
        'sympy':s.__version__,
        'originalAnalysis':[[str(v) for v in A.row(j)] for j in range(2)],
        'gramDeterminant':str(determinant),
        'stablePoles':[[float(s.re(v)),float(s.im(v))] for v in poles if abs(complex(v))<1],
        'canonicalPolynomialDefects':[[str(v) for v in row] for row in defects],
        'conservativeQuarticOrthogonalProjectionBasis':list(map(str,impossibility.polys)),
        'candidateChecks':candidates,
        'canonicalSampled':{'frequencyCount':1025,'coarseCycleMaxError':reciprocal,
                            'projectionLossErrorFromOne':loss_error,'minimumSingularValue':smallest},
        'sourceHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                        for name in ('derive_two_thirds.py','audit_two_thirds.py')},
    }
    (root/'two-thirds-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
