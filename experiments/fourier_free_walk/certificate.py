"""Exact symbolic local identities and optional exact recovery of a saved walk."""
import argparse,json,pathlib
import sympy as s


def identities():
    m00,u0,u1,v0,v1,a,b,c,d=s.symbols('m00 u0 u1 v0 v1 a b c d')
    u=s.Matrix([u0,u1]);v=s.Matrix([[v0,v1]]);L=s.Matrix([[a,b],[c,d]])
    denominator=(v*L.adjugate()*u)[0];alpha=L.det()/denominator
    K=L-alpha*u*v
    assert s.factor(K.det())==0
    w=K[:,0];z=K[0,:]/K[0,0]
    left=s.eye(3);right=s.eye(3);middle=s.eye(3)
    left[1:,1:]=u.row_join(w);right[1:,1:]=v.col_join(z)
    middle[:2,:2]=s.Matrix([[m00,1],[1,alpha]])
    target=s.Matrix([[m00,v0,v1],[u0,a,b],[u1,c,d]])
    assert all(s.factor(x)==0 for x in left*middle*right-target)
    # The four-edge invariant is unchanged by arbitrary nonzero row/column scales.
    A,B,C,D,r0,r1,q0,q1=s.symbols('A B C D r0 r1 q0 q1',nonzero=True)
    scaled=s.Matrix([[r0*A/q0,r0*B/q1],[r1*C/q0,r1*D/q1]])
    assert s.factor(scaled[0,0]*scaled[1,1]/(scaled[0,1]*scaled[1,0])-A*D/(B*C))==0
    # Nonorthogonal shear turnover, exact for the indicated nonzero chart pivot.
    x,y,z0=s.symbols('x y z')
    def shear(i,j,t):
        m=s.eye(3);m[i,j]=t;return m
    original=shear(0,1,z0)*shear(1,2,y)*shear(0,1,x)
    alternative=shear(1,2,x*y/(x+z0))*shear(0,1,x+z0)*shear(1,2,z0*y/(x+z0))
    assert all(s.factor(t)==0 for t in original-alternative)
    return {'general_GL3_reconstruction_entries':9,'rank_one_determinant_identity':True,
            'wire_gauge_cycle_invariant':True,'shear_turnover_entries':9,
            'conditions':['v adj(L) u != 0','chosen rank-one pivot != 0','shear x+z != 0']}


def recover(path):
    record=json.loads(pathlib.Path(path).read_text());n=record.get('n',record.get('N'))
    constants=[s.sqrt(2),s.sqrt(3)]
    result=s.eye(n);gates=[]
    for cell in record['physical_gates']:
        entries=[]
        for row in cell['matrix']:
            entries.append([s.S.Zero if abs(x)<1e-10 else s.nsimplify(x,constants,tolerance=1e-10,full=False) for x in row])
        g=s.Matrix(entries);i,j=cell['i'],cell['j'];a,b=result[i,:],result[j,:]
        result[i,:]=(g[0,0]*a+g[0,1]*b).applyfunc(s.simplify)
        result[j,:]=(g[1,0]*a+g[1,1]*b).applyfunc(s.simplify)
        gates.append({'i':i,'j':j,'matrix':[[str(g[r,c]) for c in range(2)] for r in range(2)]})
    rows=[[s.S.One]*n]
    for k in range(1,n//2):
        rows.extend([[s.cos(2*s.pi*k*j/n) for j in range(n)],[-s.sin(2*s.pi*k*j/n) for j in range(n)]])
    rows.append([s.Integer(-1)**j for j in range(n)])
    target=s.Matrix(rows)
    certified=all(s.simplify(x)==0 for x in result-target)
    return {'source':str(path),'N':n,'certified':certified,'exact_matrix_entries_checked':n*n,
            'physical_gates':gates if certified else [],
            'note':'Recovered algebraic coefficients are accepted only after exact full-matrix equality.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--out',required=True);args=p.parse_args()
    result={'local_identities':identities()}
    if args.plan:result['whole_walk']=recover(args.plan)
    pathlib.Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v if k!='whole_walk' else {x:y for x,y in v.items() if x!='physical_gates'} for k,v in result.items()}))
