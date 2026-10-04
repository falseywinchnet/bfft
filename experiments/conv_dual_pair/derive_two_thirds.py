"""Exact 2/3 compact-pair feasibility, using the physical basin moments."""
import json
from pathlib import Path
import sympy as s

z = s.Symbol('z', nonzero=True)
t = s.Symbol('x')
L = s.Rational(3, 2)


def basin_moment(center, degree):
    return s.expand(((center+L/2)**(degree+1)-(center-L/2)**(degree+1))/(L*(degree+1)))


def linear_family(weights, equations, variables):
    solution = list(s.linsolve(equations, variables))[0]
    return {k:s.factor(v.subs(dict(zip(variables,solution)), simultaneous=True)) for k,v in weights.items()}


def families(grow_radius=6):
    a=s.symbols('a0:4');b=s.symbols('b0:4')
    # Radius measured in fine-grid physical coordinates. Reflection relates
    # grow phases 1 and 2. Include every low-grid site inside this radius.
    cmax=int(s.floor(s.Rational(grow_radius)/L))
    jmin=int(s.ceiling((1-s.Rational(grow_radius))/L))
    jmax=int(s.floor((1+s.Rational(grow_radius))/L))
    c=s.symbols('c0:'+str(cmax+1));d=s.symbols('d0:'+str(jmax-jmin+1))
    h0={k:a[abs(k)] for k in range(-3,4)}
    h1={k:b[min(k+2,5-k)] for k in range(-2,6)}
    h0=linear_family(h0,[sum(v*k**p for k,v in h0.items())-basin_moment(0,p) for p in (0,2,4)],a)
    h1=linear_family(h1,[sum(v*(s.Rational(k)-L)**p for k,v in h1.items())-basin_moment(0,p) for p in (0,2,4)],b)
    g0={k:c[abs(k)] for k in range(-cmax,cmax+1)}
    g1={k:d[k-jmin] for k in range(jmin,jmax+1)}
    g0=linear_family(g0,[sum(v*basin_moment(L*k,p) for k,v in g0.items())-int(p==0) for p in (0,2,4)],c)
    g1=linear_family(g1,[sum(v*basin_moment(L*k,p) for k,v in g1.items())-1 for p in range(5)],d)
    g2={2-k:v for k,v in g1.items()}
    return (h0,h1),(g0,g1,g2),(a[3],b[3],*c[3:],*d[5:])


def matrices(h,g):
    A=s.Matrix(2,3,lambda p,q:sum(v*z**(k//3) for k,v in h[p].items() if k%3==q))
    G=s.Matrix(3,2,lambda p,q:sum(v*z**(k//2) for k,v in g[p].items() if k%2==q))
    return A,G


def equations(A,G):
    answer=[]
    for value in A*G-s.eye(2):
        polynomial=s.Poly(s.expand(value*z**5),z)
        answer.extend(s.factor(v) for v in polynomial.all_coeffs() if v!=0)
    return list(dict.fromkeys(answer))


def main():
    infeasible=[]
    for radius in (s.Rational(9,2),5,s.Rational(11,2)):
        hh,gg,pp=families(radius)
        aa,ggg=matrices(hh,gg)
        bb=s.groebner(equations(aa,ggg),*pp)
        infeasible.append({'radius':str(radius),'growTapCounts':list(map(len,gg)),
                           'basis':list(map(str,bb.polys))})
        assert len(bb.polys)==1 and bb.polys[0].as_expr()==1
    h,g,parameters=families(6)
    A,G=matrices(h,g)
    eq=equations(A,G)
    basis=s.groebner(eq,*parameters)
    parameter=parameters[-1]
    quadratic=basis.polys[-1].as_expr()
    roots=s.solve(quadratic,parameter)
    substitutions=s.solve([p.as_expr() for p in basis.polys[:-1]],parameters[:-1])
    hfamily=[{k:s.factor(v.subs(substitutions)) for k,v in row.items()} for row in h]
    gfamily=[{k:s.factor(v.subs(substitutions)) for k,v in row.items()} for row in g]
    AA,GG=matrices(hfamily,gfamily)
    for residual in equations(AA,GG):
        assert s.rem(residual,quadratic,parameter)==0
    error= s.eye(3)-GG*AA
    objective=s.expand(sum(v*v.subs(z,1/z) for v in error)).coeff(z,0)
    objective=s.rem(objective,s.Poly(quadratic,parameter).as_expr(),parameter)
    candidates=[]
    for root in roots:
        candidate={'parameter':str(root),'parameterFloat':float(root),
                   'projectionLoss':float(objective.subs(parameter,root)),
                   'shrink':[{str(k):float(v.subs(parameter,root)) for k,v in row.items()} for row in hfamily],
                   'grow':[{str(k):float(v.subs(parameter,root)) for k,v in row.items()} for row in gfamily]}
        candidates.append(candidate)
    result={
        'ratio':'2/3',
        'shrinkSupports':[list(row) for row in h],
        'growSupports':[sorted(row) for row in g],
        'constraint':'reflection symmetry, physical 3/2 basin moments and inverse polynomial reproduction through degree 4, AG=I',
        'parameters':list(map(str,parameters)),
        'families':[{str(k):str(v) for k,v in row.items()} for row in (*h,*g)],
        'reciprocityEquations':list(map(str,eq)),
        'groebnerBasis':list(map(str,basis.polys)),
        'feasible':not (len(basis.polys)==1 and basis.polys[0].as_expr()==1),
        'smallerSupports':infeasible,
        'minimalRadius':6,
        'algebraicParameter':str(parameter),
        'parameterEquation':str(quadratic),
        'shrinkFamily':[{str(k):str(v) for k,v in row.items()} for row in hfamily],
        'growFamily':[{str(k):str(v) for k,v in row.items()} for row in gfamily],
        'lossModuloParameterEquation':str(s.factor(objective)),
        'candidates':candidates,
        'bestCandidate':min(range(len(candidates)),key=lambda i:candidates[i]['projectionLoss']),
    }
    out=Path(__file__).with_name('two-thirds-feasibility.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
