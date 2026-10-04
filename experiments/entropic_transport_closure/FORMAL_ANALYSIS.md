# Entropic transport and the evolution of its transport rule

September 30, 2026. Formal analysis of the user's question: can the changing
transport rule itself be carried and advanced, rather than only extrapolating
the state? The existing coupled Meyer construction is the fixed reference.
No changes to it or to its numerical schedules are proposed here.

## Result and scope

For strictly positive finite Sinkhorn kernels:

1. The state displacement and its changing conditional geometry have an
   exact, finite, closed nonlinear recurrence.
2. The local transport is positive semidefinite and self-adjoint in a
   changing metric. The current displacement determines that metric.
3. Two explicit log-moment remainders account exactly for the error of a
   frozen local operator. They give a finite-horizon polynomial error bound.
4. A nonconstant log-moment primitive cannot be an exact finite-degree
   polynomial in displacement amplitude. This is a precise obstruction to
   one representation, not a theorem against other coordinates or lifts.
5. Every positive 2x2 problem has an exact projective linear lift. Matrix
   powers advance the state and its changing derivative together. An exact
   proportional-block family extends this to arbitrarily large kernels.

Thus universal nontransportability of the evolving entropic rule is false.
General inexpensive closure in a small representation remains unproved.
The exact full recurrence costs the same order as ordinary dense Sinkhorn;
no general speedup or new algorithmic novelty is claimed.

## 1. Assumptions, state, and phase

Let K be an m by n strictly positive matrix, and a,b strictly positive
marginals with equal total mass. All exponentials, logarithms, quotients,
and products between vectors below are componentwise. Work in log scaling
coordinates y=log(v). One complete pass is

    x(y) = log(a) - log(K exp(y)),
    F(y) = log(b) - log(K^T exp(x(y))).

The phase is after a column scaling, with the next row scaling x(y)
determined by y. It is legitimate to eliminate x here because it has no
independent memory in this ordinary Sinkhorn recurrence. This is not an
argument for dropping independent Meyer Bregman fields.

Gauge: F(y+c 1)=F(y)+c 1. Comparisons of solutions should quotient this
constant direction. A unit eigenvalue in this direction is not failure to
converge to the coupling. Define osc(d)=max(d)-min(d).

The standard scaling formulation, its gauge, and its fixed-point spectral
analysis are established; see Lehmann et al. (2022), equations (1),(3),(7):
https://doi.org/10.1007/s11590-021-01830-0 . General context:
Peyre and Cuturi, Computational Optimal Transport,
https://arxiv.org/abs/1803.00567 . The proofs below specialize these structures
to the transport-of-transport question; no priority claim is intended.

## 2. Exact conditional geometry and its moving metric

Set u=exp(x), v=exp(y), and define the row-normalized coupling

    Pi = diag(u) K diag(v),  Pi 1 = a,
    c = Pi^T 1,
    P = diag(1/a) Pi,       Q = diag(1/c) Pi^T.

Both P (m by n) and Q (n by m) are row stochastic. Direct differentiation
gives Dx[d]=-P d and

    J(y) = DF(y) = Q P.                              (1)

The two conditional operators are adjoints under the a- and c-weighted
inner products. In particular,

    diag(c) J = Pi^T diag(1/a) Pi,
    S = diag(sqrt(c)) J diag(1/sqrt(c))
      = [diag(1/sqrt(a)) Pi diag(1/sqrt(c))]^T
        [diag(1/sqrt(a)) Pi diag(1/sqrt(c))].          (2)

Consequently J is self-adjoint and positive semidefinite in the c metric,
at every state, not only at a solution. Jensen's inequality for P and Q
shows that their weighted norms are at most one; hence 0 <= S <= I.
Strict positivity makes the constant Perron direction simple.

For the actual displacement d=F(y)-y,

    c = b exp(-d).                                   (3)

Proof: exp(F(y))=b/(K^T u), while c=v (K^T u).
The changing metric is therefore already encoded in the displacement.
It does not require a separate learned metric model. However c alone does
not specify Pi or J: conditional relationships must still be represented.

An instantaneous symmetric chart does not make consecutive transports
equal. Changing the whitening W=diag(sqrt(c)) introduces W_next J W^-1
when a tangent is carried between successive charts. Freezing W discards
part of the geometry evolution.

## 3. An exact closed recurrence for displacement AND geometry

For any displacement d, define

    e = -log(P exp(d)),
    g = -log(Q exp(e)).                              (4)

Then x(y+d)-x(y)=e and F(y+d)-F(y)=g exactly. The next geometry is

    P_next = diag(1/(P exp(d))) P diag(exp(d)),
    Q_next = diag(1/(Q exp(e))) Q diag(exp(e)).        (5)

Proof: substitute exp(y+d)=v exp(d) and exp(x+e)=u exp(e) into the two
conditional normalizations. Every row remains stochastic.

Initialize d=F(y)-y. Equations (4),(5), together with

    y_next=y+d,   d_next=g,                          (6)

are an exact autonomous recurrence on (y,d,P,Q). Equations (4),(5) alone
close the transport state (d,P,Q); y is accumulated when reconstruction is
needed. Algebraic consistency with the original problem is required at
initialization. Arbitrary independent stochastic P,Q need not describe a
given K,a,b. Exact evolution preserves that consistency.

There is also a single-coupling form:

    Pi_next = diag(exp(e)) Pi diag(exp(d)),
    c_next = b exp(-d_next).                         (7)

For actual d, exp(d)=b/c: this is precisely column correction followed by
row normalization, expressed at the phase where row sums equal a.
Every cross ratio Pi_ij Pi_lk/(Pi_ik Pi_lj) is invariant and equals the
corresponding cross ratio of K. Geometry changes inside a fixed diagonal
scaling orbit; it is not an arbitrary changing matrix.

Explicitly storing/updating these dense conditionals costs O(mn), versus
the two O(mn) kernel applications in an ordinary dense pass. Keeping their
diagonal factors avoids extra dense storage but still needs the kernel
applications for normalization. Closure is not yet amortization. Structured
kernel applications may be much cheaper than dense arithmetic; comparison
must use the same kernel representation on both sides.

## 4. Differential and finite transport of transport

For a row stochastic matrix R define its directional reweighting derivative

    V_R(h)_ij = R_ij [h_j-(R h)_i].

Then

    DP(y)[h] = V_P(h),
    DQ(y)[h] = V_Q(-P h),
    DJ(y)[h] = V_Q(-P h) P + Q V_P(h).               (8)

This is an explicit law for the local rule's variation, including both
conditional changes. Its application may still be expensive; it is not
free higher-order information.

An exact stochastic secant representation is also available. Define
R_bar(h)=integral_0^1 diag(1/(R exp(th))) R diag(exp(th)) dt.
The fundamental theorem of calculus gives log(R exp(h))=R_bar(h) h.
Thus for (4),

    g = A(y,d) d,   A(y,d)=Q_bar(e) P_bar(d).        (9)

Both factors, and A, are row stochastic. The integral is a proof
representation; (4) evaluates the increment without quadrature. It is not
claimed to be a unique secant operator or an inexpensive explicit matrix.

Along the actual trajectory let A_k=A(y_k,d_k) and alpha_k=d_(k+1)-d_k.
Exactly as for the Meyer finite lift,

    alpha_(k+1) = A_(k+1) alpha_k
                  +(A_(k+1)-A_k) d_k.              (10)

Equation (5) gives the finite geometry update omitted by freezing A. This
establishes the coupled principle in this geometry without equating its
mechanism or its cost with the Meyer construction.

## 5. Exact nonlinear sources and a polynomial error certificate

For any stochastic R define its log-moment remainder

    L_R(h)=log(R exp(h))-R h.

With e from (4), subtraction of Jd gives the exact identity

    F(y+d)-F(y)-J(y)d = Q L_P(d)-L_Q(e).            (11)

This is the entropic counterpart of two primitive nonlinear sources
propagating through a coupled map. It does not assume either source small.

For each row, let ell(t)=log(sum_j R_ij exp(t h_j)). Then ell''(t) is the
variance of h under the tilted row probabilities. It lies between zero and
osc(h)^2/4. Integrating the second derivative twice from zero to one proves

    0 <= L_R(h)_i <= osc(h)^2/8.                    (12)

Also e_i lies between -max(d) and -min(d), so osc(e)<=osc(d).
The two nonnegative terms on the right of (11) each lie in
[0,osc(d)^2/8], not merely in a symmetric interval. Therefore

    ||F(y+d)-F(y)-Jd||_infinity <= osc(d)^2/8.       (13)

This is a global finite-displacement bound in exact arithmetic. Constants
are gauge invariant. It may be very conservative; it is not a performance
claim or a bound on objective error.

Freeze J at anchor y and let d0=F(y)-y. The polynomial trajectory is

    s_0=0,  s_(j+1)=d0+J s_j,
    s_H=(I+J+...+J^(H-1))d0.

Since DF is stochastic everywhere, F is globally 1-Lipschitz in infinity
norm. Applying (13) at every model displacement s_j and accumulating the
one-step discrepancies proves

    ||F^H(y)-(y+s_H)||_infinity
       <= (1/8) sum_(j=0)^(H-1) osc(s_j)^2.        (14)

This certifies state-trajectory error for the EXACT polynomial action.
An approximate Krylov action, rounded arithmetic, or a compressed kernel
needs its own additional error accounting. It does not certify objective
accuracy or convergence to the optimizer. Testing its sharpness and the
cost of obtaining s_j is separate work.

## 6. What a finite polynomial cannot represent

Assume a row R_i has positive weights and h is nonconstant on its support.
The function ell(t) above is real analytic on all real t, with ell''(0)>0.
It grows at most linearly as t tends to either infinity (bounded finite h).
If ell agreed with a finite-degree polynomial on any open interval,
analytic continuation on the real line would give equality everywhere.
Linear growth forces degree at most one, contradicting ell''(0)>0.

Thus no exact finite Taylor polynomial in displacement amplitude represents
this primitive on an open interval, except the constant-on-support case.
For positive K this exception is the global gauge direction.

This does NOT prove impossibility for the composed pass (cancellations may
occur), polynomial approximation, a different coordinate system, a nonlinear
finite lift, or a particular discrete trajectory. Nor does an infinite
cumulant expansion prove infinite independent information: a finite row
distribution already represents all its moments. The issue is economical
evaluation and invariant representation, not a lack of finite state.

## 7. A genuine changing-geometry case with exact polynomial closure

Take any positive 2x2 kernel K=[[p,q],[r,s]], positive a,b, and quotient
the scale gauge using t=v1/v2. One ordinary pass gives

    t_next = (A t+B)/(C t+D),
    A=(b1/b2)(q a1 r+s a2 p),
    B=(b1/b2) q s (a1+a2),
    C=p r (a1+a2),
    D=p a1 s+r a2 q.                               (15)

Proof: substitute u1=a1/(p t+q), u2=a2/(r t+s) into
t_next=(b1/b2)(q u1+s u2)/(p u1+r u2), and clear denominators.

Let M=[[A,B],[C,D]]. Then the H-pass quotient is exactly

    t_H = ((M^H)_11 t0+(M^H)_12)
          /((M^H)_21 t0+(M^H)_22).                (16)

For ps != qr, det(M)=(b1/b2)a1 a2(ps-qr)^2>0. The derivative of (15) is
det(M)/(Ct+D)^2, which varies with t. This is NOT the constant-Jacobian
entropy example from the preceding conversation. Its evolving tangent is
advanced exactly alongside the state:

    dt_H/dt0 = det(M)^H / ((M^H)_21 t0+(M^H)_22)^2. (17)

The current one-step derivative at t_H is likewise recovered from (15).
In log quotient coordinates its value is t_H f'(t_H)/f(t_H).

By Cayley-Hamilton, M^H=alpha_H M+beta_H I, with
alpha_0=0,beta_0=1 and

    alpha_(H+1)=tr(M)alpha_H+beta_H,
    beta_(H+1)=-det(M)alpha_H.

Binary powering obtains the same matrix using O(log H) fixed-size matrix
products. This is exact algebraic polynomial transport in a homogeneous
lift, with nonlinear decoding. It advances an ordinary finite-time orbit;
it does not solve for a fixed point or fit a frozen derivative.

Arithmetic-operation savings do not imply constant bit complexity: exact
rational numerator lengths grow, and floating powers need normalization
and conditioning analysis. Equations (16),(17) concern the quotient state
and its tangent. Reconstructing a full coupling costs the corresponding
output work and needs the correct row/column phase; no free dense output
or recovery of the absolute scaling gauge is claimed.
For the coupling at the ordinary post-column phase after H passes, advance
the quotient through H-1 passes and perform the final ordinary row/column
pair in any consistent gauge. This recovers the correct staggered pair
(u_H,v_H); using x(y_H) instead would select the following row phase.

## 8. Exact closure in an arbitrarily large structured family

Suppose known partitions g(i),h(j) and positive factors r_i,s_j give

    K_ij = r_i C_(g(i),h(j)) s_j.

Let A_g=sum_(i:g(i)=g) a_i and B_h=sum_(j:h(j)=h) b_j. After one ordinary
full pass, every initial positive v has the form v_j=(b_j/s_j)t_(h(j)).
Indeed K^T u has the form s_j times a function of h(j).
Set w_h=B_h t_h. Substitution then proves the invariant reduced recurrence

    w_next = B / [C^T (A/(C w))],
    v_j = (b_j/s_j) w_(h(j))/B_(h(j)).              (18)

Thus arbitrary matrix dimensions with two row groups and two column groups
reduce EXACTLY to (15)-(17) after the entry pass. Within-manifold tangent
evolution also follows the lift; arbitrary perturbations before entering
the manifold require differentiating the entry map as well.

If this structure is supplied, aggregation is linear work in the marginal
sizes and subsequent quotient jumps take O(log H) fixed-size arithmetic.
If the representation must be discovered or verified from an arbitrary
dense K, that work must be charged. Approximate block structure does not
justify exact claims. Rank two alone is insufficient for this argument:
the proportional-block identity is used when reciprocals and row/column
normalizations are taken. General structured kernel baselines can use the
same reduction; savings must be compared against that reduced ordinary map.

## 9. What this establishes about adaptive self-acceleration

The evolving entropic rule is transportable, with no missing infinite
hierarchy: (4)-(7) close it exactly. What is not supplied is a cheap generic
way to compose many such changes. Freezing J loses the explicit source
(11); the proof quantifies that omission rather than treating entropy as
flat. The large proportional-block family shows what a successful reusable
representation looks like: the geometry stays in a family closed under the
ACTUAL nonlinear update, allowing a fixed lift to advance both motion and
its changing derivative.

This is a constructive closure result, not evidence that an adaptive
learner has discovered this structure on general kernels. The appropriate
next empirical question is whether the current general kernels have an
accurate, economical approximately invariant representation, and whether
its acquisition, nonlinear sources, and verification amortize. The present
proof neither assumes that answer nor substitutes a 2x2 timing for it.

## Verification

`test_closure.py` independently checks the finite geometry trajectory,
Jacobian and its derivative, gauge and cross ratios, moving weighted metric,
variance source, exact nonlinear source identity and global bound, finite
polynomial error bound, exact rational 2x2 state AND tangent jumps, and
arbitrary-initial-state entry and evolution in a 7x6 block example.
Rational assertions use Python Fraction and compare against independent
ordinary row/column scaling. Floating checks use deterministic seeds.
These checks support, but do not replace, the derivations above.

All nine tests passed on the M4 Mini system Python on September 30, 2026
(0.010 seconds reported by unittest). The authoritative test file was copied
to `/tmp/entropic_transport_closure_20260930`; no source was edited on the
Mini. The copied-back console record is `verification.txt`. These are
identity checks, not timings of an accelerated production implementation.

Run on the Mini from the repository using m4build, or copy this isolated
test into a temporary Mini directory and run `python3 -m unittest
test_closure -v`. No production runtime or benchmark is changed.
