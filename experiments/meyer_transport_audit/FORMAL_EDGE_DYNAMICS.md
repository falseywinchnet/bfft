# Discrete transport, its evolving increments, and the first release

September 5, 2026. An exact mathematical formulation of the synthetic edge,
with a small numerical diagnostic. This uses the investigative perspective
of numerical integration, without claiming to reconstruct Runge's or Kutta's
personal reasoning or to identify this recurrence with an ODE.

## 1. A completely specified problem

Use the periodic lattice Z/64Z, with unscaled forward difference

    (Dx)_i = x_(i+1) - x_i,    L = D*D.

Take f_i = 50 for 0 <= i <= 32 and f_i = 200 otherwise. The wrap supplies
a second edge. Let lambda = 1/20 and mu = 40. The variational problem is

    minimize TV(u) + (lambda/2)||f-u-v||^2
    over u and v in G_mu = {D*g : ||g||_infinity <= mu},
    TV(u) = sum_i |(Du)_i|.

The complete one-dimensional recurrence has 256 real coordinates
z = (u,w,t_u,t_w). The two transverse fields in the six-field implementation
vanish identically. Define

    c_u = 1/20, eta_u = 1/10, a_u = 10,
    c_w = 1/40, eta_w = 1/4, a_w = 4,
    S_u = (c_u I + eta_u L)^(-1),
    S_w = (c_w I + eta_w L)^(-1),
    p_u = clip(t_u,-a_u,a_u), p_w = clip(t_w,-a_w,a_w).

Then the implemented map T is exactly

    u+   = S_u[c_u(u+w) + eta_u D*(t_u-2p_u)],
    w+   = S_w[c_w(f-u+) + eta_w D*(t_w-2p_w)],
    t_u+ = Du+ + p_u,
    t_w+ = Dw+ + p_w.

The source-driven first state is

    u_1 = S_u c_u f, w_1 = S_w c_w(f-u_1),
    t_u,1 = Du_1, t_w,1 = Dw_1.

Both screened matrices are positive definite. This defines the trajectory
uniquely. With these rational data all exact-arithmetic states are rational;
the numerical implementation evaluates them with floating-point FFTs.
The changing u/w driving fields are included in T itself.

The bottom has an independent meaning: a zero primal-dual gap. At any state,
q = eta_u D*p_u is a feasible TV dual, and
v_F = (eta_w/c_w)D*p_w belongs to G_mu. Thus

    P = TV(u) + (lambda/2)||f-u-v_F||^2,
    Q = <f,q> - ||q||^2/(2lambda) - mu TV(q),
    P-Q >= 0.

Equality certifies optimality of the recovered feasible primal/dual pair.
The emitted f-u-w need not equal v_F. A projection event alone does not
certify that the bottom has been reached.

## 2. The local law is exactly affine

Fix the interior/positive-saturated/negative-saturated choice at all 128
projection sites. On the resulting polyhedral cell C,

    p_b = J_b t_b + beta_b,

where J_b is diagonal with entries 1 in the interior and 0 outside, and
beta_b is respectively 0, +a_b, or -a_b. Substitution into the recurrence
gives T(z) = Az+c on C. At a boundary, neighboring formulas agree in value.

For any perturbation e, A acts through the entire coupled pass:

    e_u+  = S_u[c_u(e_u+e_w) + eta_u D*(I-2J_u)e_tu],
    e_w+  = S_w[-c_w e_u+ + eta_w D*(I-2J_w)e_tw],
    e_tu+ = De_u+ + J_u e_tu,
    e_tw+ = De_w+ + J_w e_tw.

This is exact inside the cell, including both updated driving fields.
It is not an assumption that one isolated inner problem stays fixed.
In two dimensions, exterior disk projection has curvature; a fixed
inside/outside pattern there does not make T affine.

## 3. Transport of transport is a discrete differential hierarchy

At a state z define h = T(z)-z and K = A-I. While the cell remains valid,

    z_(j+1) = z_j + h_j,
    h_(j+1) = A h_j,
    Delta h_j = K h_j.

Consequently the exact m-pass advance of the frozen affine law is

    z_m = z + sum_(j=0)^(m-1) A^j h
        = z + sum_(r=0)^(m-1) binom(m,r+1) K^r h.       (1)

Proof: A = I+K; expand its integer powers by the binomial theorem and use
sum_(j=r)^(m-1) binom(j,r) = binom(m,r+1).

The first three carried quantities are h, a=Kh, and b=K^2h. Their successive
predictions are z+mh, z+mh+binom(m,2)a, and that expression plus binom(m,3)b.
The coefficients express discrete repeated transport; they are not adjustable
relaxation factors. Truncation must still be justified for the chosen horizon.

An exact remainder identity for the linear predictor is

    R_m = z_m-z-mh
        = sum_(i=0)^(m-2) (m-1-i) A^i a.               (2)

Equivalently R_0=0 and R_(j+1)=A R_j+j a. Therefore, in any induced norm,

    ||R_m|| <= ||a|| sum_(i=0)^(m-2) (m-1-i)||A^i||.   (3)

Even if ||A^i|| <= C throughout the horizon, this bound grows as
C binom(m,2)||a||. Small instantaneous change in transport does not certify
a long advance. One cannot assume C=1 from the observed decay of one vector;
the coupled matrix need not be a contraction in the chosen norm.

## 4. Why the discrete clock matters

Suppose one invents the residual ODE dz/ds = T(z)-z. Its vector field on C
is F(z)=Kz+c. Its smooth flow begins

    z(s) = z + s h + (s^2/2)Kh + ... .

Equation (1) instead has m(m-1)/2 multiplying Kh. For example, two ordinary
passes give exactly z+2h+Kh, whereas an explicit midpoint RK step of length
2 applied to that affine ODE gives z+2h+2Kh. More accurate integration of
the invented ODE does not establish faithful advancement of this discrete
trajectory. Both have the same equilibrium equation, but different paths.

The useful numerical-integration question is: what is the evolution law,
what information do intermediate stages carry, and how is the entire
advance's error controlled? It is not sufficient to choose an RK tableau.
Hairer, Norsett and Wanner's *Solving Ordinary Differential Equations I*
provides the standard integration context: https://www.unige.ch/~hairer/books.html.

## 5. A projection regime can mathematically forbid settling

Let ell satisfy A*ell=ell. Then, throughout this affine regime,

    <ell,h_j> = <ell,h>,
    <ell,z_m-z> = m <ell,h>.                            (4)

Also <ell,h>=<ell,c>. If it is nonzero, (I-A)z*=c has no solution.
Thus the cell's own equations forbid a fixed point, independently of how
small other transient changes have become. This alone does not guarantee
a finite exit from an unbounded cell; the observed edge does exit.

There is a stronger vector decomposition when eigenvalue 1 is semisimple.
Let V and U be bases of ker(I-A) and ker((I-A)*). If U*V is invertible,

    P_1 = V(U*V)^(-1)U*,  d=P_1h,  r=(I-P_1)h.

P_1 commutes with A, Ad=d, and

    h_j = d+A^j r,
    z_m = z+m d+sum_(j=0)^(m-1) A^j r.                 (5)

This separates persistent transport from transient transport without
discarding either. Decay of the transient requires additional spectral
conditions; (5) itself does not assume decay. Building P_1 densely here is
an explanatory diagnostic, not a proposed cheap runtime operation.

## 6. The intermediate path has to admit the advance

Write the closed cell as Hx<=b. Define its frozen affine orbit by (1).
For this orbit to equal the real recurrence through z_m, it suffices that
z_j belongs to C for every integer j=0,...,m-1. The last state z_m may
be the first state outside C. Proof is induction using T(z_j)=Az_j+c.
That first outside state can be computed by the old law; its outgoing
step must use the new projection geometry.

Endpoint membership by itself does not prove all preceding memberships.
For a proposed path xhat_j with proven coordinate error bounds E_j,

    H xhat_j + |H|E_j <= b                             (6)

certifies membership at that stage. For the frozen affine law an error tube
can be propagated from E_0=0 using the path defect

    delta_j = xhat_(j+1) - (A xhat_j+c),
    E_(j+1) = |A|E_j + |delta_j|.                      (7)

Equations (6)-(7) give a sufficient admission rule by induction. It may be
very conservative and expensive. A practical cheap version is unresolved;
floating-point comparisons without roundoff enclosures are not rigorous
interval certificates. Nevertheless this states exactly what a faithful
multi-pass advance must establish, beyond predicting a plausible endpoint.

## 7. What the formal synthetic probe measured

`formal_edge.py` builds the 256-dimensional A from the implemented tangent
map at pass 64. The numerical nullity of I-A is 4 at tolerance 1e-10;
the U*V pairing condition number is 1.2733. The recovered invariant drift
has t_u[33] component -0.1932110654 per pass, independently recovered at
passes 64, 128, and 192. Its invariance residual RMS is below 8e-17.
Nonzero left-invariant transport obstructs an affine fixed point.
These are numerical rank/invariance measurements, not exact rank proofs.

The frozen affine orbit agrees with ordinary transport through the first
event, pass 214, to at most 5.81e-13 RMS. The remainder identity (2) agrees
to below 5e-14 RMS over the 200-pass frozen-law diagnostic horizons.
Beyond the first event the frozen orbit is only a diagnostic extension.

At pass 128, eight-pass prediction RMS errors are 0.005775 (linear),
0.001778 (quadratic), and 0.0002881 (cubic). At the actual event 86 passes
away, the errors become 0.12177, 0.84657, and 2.92740. Their first predicted
integer projection changes are respectively passes 214, 212, and 201.
At pass 64, the predicted events are 194, 93, and 86 instead of 214.
Higher local order helps short advances but fails to justify these long
advances. All RMS measurements here use four nonzero-field blocks, unlike
the earlier six-block RMS; ratios of norms are unchanged by zero padding.

The supported interpretation is persistent drift plus evolving transients
within a constraint regime that cannot contain a fixed point. The useful
next mathematical object is a controlled advance of that complete evolution
up to its first release. No fast general implementation is established.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/formal_edge.py \
  --out /tmp/meyer_formal_edge.json
```

Copy the JSON back immediately with the host selected by m4host. The script
asserts affine-orbit agreement, the exact finite remainder identity, invariant
drift, and nonzero left-invariant transport. Results are preserved in
`meyer_formal_edge.json`. Dense construction costs are additional diagnostic
work, not an acceleration benchmark. Production code is unchanged.
