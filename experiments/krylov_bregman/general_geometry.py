"""Geometry-compatible nonlinear mirror maps and observable recurrence discovery.

Identification is numerical, not a global closure certificate. The rational
identity used by the control family is proved separately in the paper.
"""
import numpy as np


def mirror_inverse(s, kind):
    if kind == 'euclidean':
        return s.copy()
    if kind == 'quartic':
        # Unique real root of q + q**3 = s, without an iterative solve.
        return 2 / np.sqrt(3) * np.sinh(np.arcsinh(1.5*np.sqrt(3)*s)/3)
    if kind == 'entropy':
        return np.exp(s)
    if kind == 'hypentropy':
        return np.sinh(s)
    raise ValueError(kind)


def mirror_gradient(q, kind):
    if kind == 'euclidean': return q.copy()
    if kind == 'quartic': return q + q**3
    if kind == 'entropy': return np.log(q)
    if kind == 'hypentropy': return np.arcsinh(q)
    raise ValueError(kind)


def mirror_hessian(q, kind):
    if kind == 'euclidean': return np.ones_like(q)
    if kind == 'quartic': return 1 + 3*q*q
    if kind == 'entropy': return 1/q
    if kind == 'hypentropy': return 1/np.sqrt(1+q*q)
    raise ValueError(kind)


class RationalMirror:
    """h(x)=sum h_i((Cx)_i), f'_i(q)=s-a*s/(1+b*s), step size 1.

    f is convex on s>0 for 0<a<=1,b>=0. C is supplied geometry, not learned
    from future states. The full primal step executes both mirror transforms.
    """
    def __init__(self, kind='quartic', n=24, seed=0):
        self.kind = kind
        rng = np.random.default_rng(seed)
        self.rotation = np.linalg.qr(rng.normal(size=(n,n)))[0]
        self.scale = np.geomspace(.7,1.4,n)
        self.C = self.scale[:,None]*self.rotation
        self.Cinv = self.rotation.T/self.scale
        self.a = np.resize([.92,.96,.98],n)
        self.b = rng.uniform(.01,.05,n)
        self.initial_s = rng.uniform(.5,2,n)
        self.initial = self.decode(self.initial_s)

    def encode(self, x):
        return mirror_gradient(self.C@x, self.kind)

    def decode(self, s):
        return self.Cinv@mirror_inverse(s,self.kind)

    def dual_step(self, s):
        return self.a*s/(1+self.b*s)

    def gradient(self, x):
        s = self.encode(x)
        return self.C.T@(s-self.dual_step(s))

    def step(self, x):
        # Written as the mirror update, not as a conjugacy shortcut.
        y = self.C.T@self.encode(x)
        next_s = self.Cinv.T@(y-self.gradient(x))
        return self.decode(next_s)

    def hessian(self, x):
        return self.C.T@(mirror_hessian(self.C@x,self.kind)[:,None]*self.C)


def observable(s, name):
    s = np.asarray(s)
    if name == 'dual': v = s
    elif name == 'reciprocal': v = 1/s
    elif name == 'log': v = np.log(s)
    else: raise ValueError(name)
    return np.r_[1.,v]


def fit_recurrence(samples, relative_tolerance=1e-10):
    """SVD snapshot compression and least-squares identification of T.

    Full-state samples only; no objective optimum, known a/b, or future
    reference enters fitting. The training defect is returned, not hidden.
    """
    Z = np.column_stack(samples)
    U, singular, _ = np.linalg.svd(Z, full_matrices=False)
    rank = int(np.sum(singular > relative_tolerance*singular[0]))
    Q = U[:,:rank]
    X, Y = Q.T@Z[:,:-1], Q.T@Z[:,1:]
    T = Y@np.linalg.pinv(X, rcond=relative_tolerance)
    defect = np.linalg.norm(Z[:,1:]-Q@T@X)/np.linalg.norm(Z[:,1:])
    return Q,T,{'rank':rank,'training_defect':float(defect),
                'singular_values':singular.tolist()}


def predict(Q,T,initial,horizon):
    return Q@np.linalg.matrix_power(T,horizon)@(Q.T@initial)


def compatibility_defect(hessian_at_image, jacobian):
    A = hessian_at_image@jacobian
    return np.linalg.norm(A-A.T)


def discover_mobius(samples):
    """Identify degree-(1,1) relations and derive cross-ratio observables.

    Each coordinate fits [s,1,-s*s_next,-s_next] @ [a,b,c,d] = 0.
    Fixed points then synthesize the chart, rather than looking up reciprocal
    features. This finite rational grammar is an explicit prior. Distinct real
    finite fixed points are required; repeated/infinite roots are rejected.
    """
    Z=np.asarray(samples)
    coefficients=[]; roots=[]; rates=[]; defects=[]
    for i in range(Z.shape[1]):
        s,t=Z[:-1,i],Z[1:,i]
        A=np.column_stack([s,np.ones_like(s),-s*t,-t])
        _,singular,Vt=np.linalg.svd(A,full_matrices=False)
        if singular[-2] < 1e-12*singular[0]:
            raise ValueError('unidentified rational relation')
        v=Vt[-1]; a,b,c,d=v
        if abs(c) < 1e-10: raise ValueError('infinite fixed point: unsupported chart')
        rr=np.roots([c,d-a,-b])
        if np.max(np.abs(rr.imag))>1e-10 or abs(rr[0]-rr[1])<1e-8:
            raise ValueError('nonreal or repeated fixed point')
        rr=rr.real
        # Pick the attracting root from the inferred derivative only.
        multipliers=(a*d-b*c)/(c*rr+d)**2
        j=np.argmin(np.abs(multipliers)); r1,r2=rr[j],rr[1-j]
        rate=(a-r1*c)/(a-r2*c)
        coefficients.append(v); roots.append([r1,r2]); rates.append(rate)
        defects.append(float(np.linalg.norm(A@v)/np.linalg.norm(A)))
    return np.asarray(roots),np.asarray(rates),{'maximum_relation_defect':max(defects),
                                              'coefficients':np.asarray(coefficients).tolist()}


def mobius_transport(initial,roots,rates,horizon):
    r1,r2=roots.T
    t=(initial-r1)/(initial-r2)
    t=t*rates**horizon
    return (t*r2-r1)/(t-1)
