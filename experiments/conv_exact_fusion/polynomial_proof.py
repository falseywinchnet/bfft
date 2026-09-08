"""Exact-rational proof oracle for frozen CONV admission paths.

Diagnostic algebra, not a proposed production rational-arithmetic backend.
A single midpoint execution supplies the existing ledger/projection path.
Bernstein inequalities certify that path on [0,1], without root finding.
"""
from fractions import Fraction as F
from math import comb


def poly(a):
    a = tuple(F(x) for x in a)
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return a or (F(0),)


def add(a, b):
    return poly([(a[i] if i < len(a) else 0) +
                 (b[i] if i < len(b) else 0)
                 for i in range(max(len(a), len(b)))])


def scale(a, s):
    return poly([x*s for x in a])


def sub(a, b):
    return add(a, scale(b, -1))


def mul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return poly(out)


def total(items):
    out = poly([0])
    for a in items:
        out = add(out, a)
    return out


def evaluate(a, u):
    out = F(0)
    for x in reversed(a):
        out = out*u+x
    return out


def bernstein(a):
    n = len(a)-1
    return tuple(sum(a[k]*F(comb(j, k), comb(n, k))
                     for k in range(j+1)) for j in range(n+1))


def nonnegative(a, strict=False):
    b = bernstein(a)
    return min(b) > 0 if strict else min(b) >= 0


def tails():
    basis = [scale(mul(poly([0]*j+[1]),
                       _power(poly([1, -1]), 5-j)), comb(5, j))
             for j in range(6)]
    return [total(basis[k+1:]) for k in range(5)]


def _power(a, n):
    out = poly([1])
    for _ in range(n):
        out = mul(out, a)
    return out


def raw_bank(y):
    """Exact existing jet bank, with the existing endpoint closures."""
    n = len(y)
    if n < 5:
        raise ValueError('at least five nodes')
    def lin(ids, weights, divisor=1):
        return scale(total(scale(y[i], w) for i, w in zip(ids, weights)), F(1, divisor))
    m = [poly([0])]*n
    q = list(m)
    m[0] = lin([0,1,2], [-3,4,-1], 2)
    m[1] = lin([0,2], [-1,1], 2)
    m[-2] = lin([n-3,n-1], [-1,1], 2)
    m[-1] = lin([n-3,n-2,n-1], [1,-4,3], 2)
    q[0] = lin([0,1,2,3], [2,-5,4,-1])
    q[1] = lin([0,1,2], [1,-2,1])
    q[-2] = lin([n-3,n-2,n-1], [1,-2,1])
    q[-1] = lin([n-4,n-3,n-2,n-1], [-1,4,-5,2])
    for i in range(2, n-2):
        m[i] = lin([i-2,i-1,i+1,i+2], [1,-8,8,-1], 12)
        q[i] = lin([i-2,i-1,i,i+1,i+2], [-1,16,-30,16,-1], 12)
    d = [sub(y[i+1], y[i]) for i in range(n-1)]
    a = []
    for i in range(n-1):
        a.append([scale(m[i], F(1,5)),
                  add(scale(m[i], F(1,5)), scale(q[i], F(1,20))),
                  add(sub(d[i], scale(add(m[i],m[i+1]), F(2,5))),
                      scale(sub(q[i+1],q[i]), F(1,20))),
                  sub(scale(m[i+1],F(1,5)),scale(q[i+1],F(1,20))),
                  scale(m[i+1],F(1,5))])
    return a, d


def compile_line(y):
    """Return polynomial currents and a sufficient whole-interval certificate.

    Each recorded inequality belongs to the ORIGINAL finite admission rule.
    Failure means 'not certified'; it does not mean inadmissible input.
    """
    a, d = raw_bank(y)
    at = F(1,2)
    checks = []
    def sign(p):
        v = evaluate(p, at)
        s = (v > 0)-(v < 0)
        if s:
            checks.append((scale(p,s), True, 'sign'))
        else:
            # Isolated roots must not be treated as an identically zero branch.
            checks.append((poly([0]) if p == poly([0]) else poly([-1]), False, 'zero'))
        return s
    coarse = [sign(p) for p in d]
    for i in range(1,len(coarse)):
        if not coarse[i]: coarse[i] = coarse[i-1]
    for i in range(len(coarse)-2,-1,-1):
        if not coarse[i]: coarse[i] = coarse[i+1]
    flat = [p for cell in a for p in cell]
    rawsign = [sign(p) for p in flat]
    ledger = [coarse[0]]*len(flat)
    previous = 0
    for knot in range(1,len(coarse)):
        if coarse[knot] == coarse[knot-1]: continue
        centre = 5*knot
        candidates = list(range(max(previous+1,centre-4),min(len(flat),centre+5)))
        costs = []
        for b in candidates:
            costs.append(total(mul(flat[j],flat[j])
                         for j in range(centre-5,centre+5)
                         if (coarse[knot-1] if j < b else coarse[knot])*rawsign[j] < 0))
        selected = min(range(len(costs)), key=lambda k: evaluate(costs[k],at))
        for k,cost in enumerate(costs):
            if k != selected:
                checks.append((sub(cost,costs[selected]), k < selected, 'ledger'))
        previous = candidates[selected]
        ledger[previous:] = [coarse[knot]]*(len(flat)-previous)
    currents = []
    for i, cell in enumerate(a):
        signs = ledger[5*i:5*i+5]
        if not any(signs):
            currents.append([poly([0])]*5)
            continue
        av = [evaluate(p,at) for p in cell]
        dv = evaluate(d[i],at)
        cuts = sorted(set(av))
        samples = [cuts[0]-1]+[(x+y)/2 for x,y in zip(cuts,cuts[1:])]+[cuts[-1]+1]
        chosen = None
        for threshold in samples:
            active = [j for j in range(5) if signs[j]*(av[j]-threshold) > 0]
            if not active: continue
            lam = (sum(av[j] for j in active)-dv)/len(active)
            if all(signs[j]*(av[j]-lam) >= 0 if j in active
                   else signs[j]*(av[j]-lam) <= 0 for j in range(5)):
                chosen = active
                break
        if chosen is None:
            if dv != 0: raise AssertionError('no finite projection face')
            # Zero solution: pick a feasible threshold interval directly.
            lo = max((av[j] for j in range(5) if signs[j]>0), default=None)
            hi = min((av[j] for j in range(5) if signs[j]<0), default=None)
            lam = lo if lo is not None else hi
            # This degenerate branch is usable at one point; conservatively
            # refuse interval certification rather than invent a varying lambda.
            checks.append((poly([-1]),False,'degenerate_face'))
            currents.append([poly([0])]*5)
            continue
        lam = scale(sub(total(cell[j] for j in chosen),d[i]), F(1,len(chosen)))
        c = []
        for j in range(5):
            gap = scale(sub(cell[j],lam), signs[j])
            checks.append((gap if j in chosen else scale(gap,-1),False,'KKT'))
            c.append(sub(cell[j],lam) if j in chosen else poly([0]))
        currents.append(c)
    return currents, checks


def certified(checks):
    return all(nonnegative(p, strict) for p,strict,_ in checks)


def scalar_profile(values):
    return compile_line([poly([x]) for x in values])[0]


def first_cell(rows, cell):
    """Actual admitted horizontal reconstruction as vertical polynomial data."""
    t = tails()
    out = []
    for row in rows:
        c = scalar_profile(row)[cell]
        out.append(add(poly([row[cell]]), total(mul(t[k],c[k]) for k in range(5))))
    return out


def restrict(a, lo, hi):
    """Map a fixed interval [lo,hi] to [0,1] by exact substitution."""
    out = poly([0])
    for x in reversed(a):
        out = add(mul(out,poly([lo,hi-lo])),poly([x]))
    return out
