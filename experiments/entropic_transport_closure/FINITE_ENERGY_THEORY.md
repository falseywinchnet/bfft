# Finite energy, resonance, and acquisition of the changing rule

The scalar acquisition experiment failed its held-out drift predictions. That
does not settle whether the changing rule can be transported. It separates two
questions: how much nonlinear forcing accumulates, and which information must
be retained to predict that forcing. This note derives bounds for the first and
an observability obstruction for the second. It also audits the retained six
acquisition cases. These are theory results and scoring diagnostics, not a new
accelerator or a claim of priority over the literature.

## 1. Geometry at any anchor

Let K have strictly positive entries, and let a,b be strictly positive probability
marginals. One full log-Sinkhorn step is

    F(y) = log b - log(K^T (a / (K exp y))).

All logarithms, divisions, and exponentials here act componentwise. At an anchor y,
write P for the row conditional of K diag(exp y), and Q for the reverse
conditional after imposing a. Set c = a^T P and J = QP. Then

    c_j Q_ji = a_i P_ij,
    J 1 = 1,   c^T J = c^T,
    J is positive semidefinite and self-adjoint in <u,v>_c.

These identities hold away from the fixed point too; c need not equal b.
Write B(u,v) = D²F(y)[u,v]/2, and define

    E_y(h) = sum_i a_i Var_{P_i}(h)
           = <h,(I-J)h>_c.

The curvature identity is

    2 B(u,v) = Q Cov_P(u,v) - Cov_Q(Pu,Pv).

Weighted conditional Cauchy–Schwarz, followed by scalar Cauchy–Schwarz, gives

    ||B(u,v)||_{c,1} <= sqrt(E_y(u) E_y(v)).                 (1)

For the second covariance use
sum_j c_j Var_{Q_j}(Pu) = <u,(J-J²)u>_c <= E_y(u).
This is the useful geometric fact: slow directions have small conditional
variance, hence small nonlinear forcing. Small eigenvalue gaps alone overstate
the danger when the forcing and the gap come from the same conditionals.

## 2. A finite remainder bound, not just a Taylor coefficient

Let R = osc(h) = max h - min h. Along y_t = y+t h, 0 <= t <= 1,

    exp(-tR) <= (P_t)_ij/P_ij <= exp(tR),
    exp(-tR) <= (c_t)_j/c_j <= exp(tR),
    E_{y_t}(h) <= exp(tR) E_y(h).

The first inequality follows from exponential tilting; the last follows by
evaluating each tilted variance around the original conditional mean rather
than its minimizing mean.

Differentiate the conditional covariance identity. With kappa_3 the third
central moment, the exact third derivative is

    D³F[h,h,h] = Q kappa_{3,P}(h)
                - 3 Cov_Q(Ph, Var_P(h)) + kappa_{3,Q}(Ph).

Since |kappa_3(X)| <= range(X) Var(X), and
|Cov(A,V)| <= range(A) E[V] for V >= 0, its weighted L1 norm at the current
anchor is at most 5 R E_y(h). Comparing weights along the tilt gives

    ||D³F(y_t)[h,h,h]||_{c,1} <= 5 R exp(2tR) E_y(h).

Similarly ||D²F(y_t)[h,h]||_{c,1} <= 2 exp(2tR) E_y(h).
Define phi_m(z) = (exp(z)-sum_{j=0}^{m-1} z^j/j!)/z^m, with
phi_m(0)=1/m!. Taylor's integral remainder therefore proves

    ||F(y+h)-F(y)-Jh||_{c,1} <= C2(R) E_y(h),
    ||F(y+h)-F(y)-Jh-B(h,h)||_{c,1} <= C3(R) E_y(h),       (2)

where C2(R)=2 phi_2(2R), and C3(R)=5 R phi_3(2R).
In particular C2(0)=1 and C3(R)=5R/6+O(R²). This supplies the
finite-amplitude control that a small quadratic coefficient by itself cannot.

For an arbitrary predicted path s_j about this same anchor, put r=F(y)-y and

    eta_j = osc(s_{j+1} - [r+Js_j+B(s_j,s_j)]).

Order preservation and translation equivariance imply that F is nonexpansive
in oscillation. With m=min c_j, the actual trajectory starting at y+s_0 obeys

    osc(F^H(y+s_0)-y-s_H)
      <= sum_{j=0}^{H-1} [eta_j + C3(osc s_j) E_y(s_j)/m]. (3)

This statement needs no fixed point. Projection errors and discrepancies
between lifted tensor coordinates and s_j tensor s_j belong in eta_j; they
cannot be omitted. Evaluating every term sequentially is a valid proof oracle,
but would defeat the purpose of a fast horizon skip.

## 3. A horizon-uniform theorem at a fixed point

Now assume F(y*)=y*. Here c=b. For c-centered h, set
A=||h||_{c,2}, R=osc h, and m=min c. The linear path ell_j=J^j h satisfies

    sum_{j=0}^{H-1} E(ell_j)
      = sum_i alpha_i² (1-lambda_i^(2H))/(1+lambda_i)
      <= A².                                             (4)

The gauge eigenvalue 1 contributes zero, interpreted by continuity in the
displayed expression. The inequality also follows directly from
E(u) <= ||u||²_c-||Ju||²_c. Thus slow relaxation spends a bounded total
amount of Dirichlet energy. Combining (2), (4), and nonexpansivity proves

    sup_H osc(F^H(y*+h)-y*-J^H h) <= C2(R) A²/m.           (5)

There is no inverse spectral gap in this bound. Uniformity concerns horizon
and gap at fixed amplitude and marginal geometry, not all dimensions or all
ways of approaching the boundary of the probability simplex.

The second jet retains the accumulated quadratic forcing:

    ell_0=h, q_0=0,
    ell_{j+1}=J ell_j,
    q_{j+1}=J q_j+B(ell_j,ell_j),   z_j=ell_j+q_j.

It is powered by the finite linear matrix

    L = [ J    B_flat  ]
        [ 0    J ⊗ J   ]

on (h,h tensor h). This is the second jet of the iterated map. It is not
repeated application of the quadratic Taylor polynomial.

To bound it uniformly, set f_j=B(ell_j,ell_j). Equation (1) and (4) give
sum_j ||f_j||_{c,1} <= A². Markov contraction gives osc q_j <= A²/m.
For a single impulse, sum_i E(J^i f) <= ||f||²_{c,2}. Applying Minkowski in
the space of energy sequences to the impulse expansion of q gives

    sqrt(sum_j E(q_j)) <= sum_j ||f_j||_{c,2}
                       <= A²/sqrt(m) = S.

Consequently osc z_j <= Rbar=R+A²/m,
sum_j E(z_j) <= (A+S)², and by (1)

    sum_j ||B(z_j,z_j)-B(ell_j,ell_j)||_{c,1} <= 2AS+S².

Insert these bounds into (3). For every H,

    osc(F^H(y*+h)-y*-z_H)
      <= [2AS+S²+C3(Rbar)(A+S)²]/m.                       (6)

For fixed positive c this is O(A³) as A tends to zero, uniformly in H and
without dividing by any resonance denominator. Resonant interactions remain
in L. No coordinate change is needed to remove them.

Two practical restrictions are substantial. The theorem is centered at an
unknown fixed point. Also Rbar includes A²/m inside an exponential in C3;
the constants can become vacuous for small marginal masses. Reconstructing h
from a residual by (I-J)^(-1) can reintroduce inverse-gap costs. Equation (6)
does not solve that acquisition problem. The arbitrary-anchor certificate (3)
avoids an unknown h, but requires a compressed, certified proposed path.

## 4. Accumulated energy itself has a finite powered representation

If a proposed displacement has a finite representation s_j=C L^j xi, define
D=diag(c)(I-J) and W=C^T D C. Its cumulative energy is

    sum_{j=0}^{H-1} E(s_j) = xi^T G_H xi,
    G_H = sum_{j=0}^{H-1} (L^j)^T W L^j.

For consecutive blocks a,b,

    P_{a+b}=P_b P_a,   G_{a+b}=G_a+P_a^T G_b P_a.

Binary composition computes both L^H and G_H in O(d³ log H) dense arithmetic,
where d is the retained lift dimension. It uses no inverse of I-L, tolerates
unit eigenvalues and Jordan blocks, and does not expand each skipped step.
If osc(s_j)<=Rhat is independently certified, the remainder part of (3)
is at most C3(Rhat) xi^T G_H xi/m. The model-defect sum still needs its own
bound. This is a component of a certificate, not a complete compressed
certificate or evidence that acquisition plus d³ work is economical.

## 5. Why state compression need not close the changing rule

At first order let h'=Jh. Let O collect the outputs to retain: selected state
coordinates and the changing-rule entries

    G_ij[h] = 2 <v_i,B(h,v_j)>_c.

For an exact linear Markov representation xi=R h, xi'=T xi, Oh=M xi for
every h, one must have RJ=TR and O=MR. The smallest possible dimension is

    dim span rows {O, OJ, ..., OJ^(n-1)}.                  (7)

Necessity follows because the row space of R is J-invariant and contains O.
Sufficiency follows by choosing a basis of the displayed invariant space;
Cayley–Hamilton closes it. This is an exact linear observability statement,
not a lower bound on every approximate nonlinear representation.

A state coordinate can span one eigenmode while its rule drift depends on
many modes. For scalar drift q_k=sum_i gamma_i alpha_i lambda_i^k, the minimal
exact recurrence order is the number of distinct eigenvalues with nonzero
combined coefficients. Near coincidences may permit approximation; a small
observed drift does not establish it. As a concrete ambiguity, with
J=diag(.9,.8,.7), output ell=(1,1,1), and d=(1,-2,1),

    ell d = ell Jd = 0,   ell J²d = .02.

Two exact observations do not determine the third in this class. Eliminating
hidden coordinates converts their effect into memory: for x'=Ax+Bz,
z'=Cx+Dz, the eliminated equation contains BD^k z_0 and the convolution
sum_{j<k} BD^(k-1-j)Cx_j. Delays can represent this information; they do not
make it disappear. This connects the work to the established
[Wiener–Koopman–Mori–Zwanzig framework of Lin and Lu](https://arxiv.org/abs/1908.07725)
and [delay observables studied by Kamb et al.](https://arxiv.org/abs/1810.01479).
The polynomial lift is in the Carleman tradition; existing continuous-time
finite-section error work includes [Amini et al.](https://arxiv.org/abs/2207.07755).
Those connections do not substitute for the discrete conditional-energy
proof above, or establish its novelty.

## 6. Retained evidence and falsifiers

`results/theory/entropic_energy_theory.json` retains 50 finite-remainder probes,
28 fixed-point horizon/amplitude probes, and 18 acquisition-anchor audits.
The source acquisition JSON is copied unchanged into this directory with
SHA256 `374a90885faf4770d6ba12a287973c25dee2c3fdbc4e38a51831330afb518c66`.
The generator reproduces the original RNG order at n=128. Dense spectra and
converged reference trajectories are scoring oracles, not algorithm inputs.

All 50 finite bounds held. Maximum observed/bound ratios were 0.480 for the
linear remainder and 0.0523 for the quadratic remainder. At amplitude 0.1,
the controlled gap shrank from 0.386 to 0.00145 while H=ceil(3/gap) grew from
8 to 2070. Maximum second-jet error changed from 1.13e-4 to 1.17e-4 rather
than blowing up like inverse gap. At the smallest gap:

| Amplitude | Maximum linear error | Maximum second-jet error |
|---:|---:|---:|
| 0.100 | 5.231e-4 | 1.1675e-4 |
| 0.050 | 1.328e-4 | 1.4583e-5 |
| 0.025 | 3.345e-5 | 1.8219e-6 |

Halving amplitude reduced the second-jet error by approximately eight. The
all-horizon bound at amplitude .1 was 0.00704; the pathwise bound at H=2070
was 0.000724. Thus the theorem is valid but conservative even in dimension
three. The exact-resonance case also passed. Powered and sequential energy
sums differed by at most 9.55e-16. These are floating-point regressions, not
outward-rounded machine certificates.

The acquisition audit sharpens three failures:

* At seed 1, epsilon=.01, first anchor, the slow-only first-order drift is
  +0.02639, the all-mode derivative is -0.002041, and the actual leading
  eigenvalue drift is -0.002567. Hidden-mode contributions reverse the sign.
  Even at the late anchor, the leading-mode squared overlap is .999967 while
  scalar drift is .001156 versus actual .000740. Accurate mode direction
  alone does not imply accurate rule closure.
* At the middle epsilon=.001 anchors, squared overlap of the fixed-point
  leading mode with the current leading mode is .00452 and .00105. The
  two-dimensional current leading subspace captures .9943 and .9586 of that
  reference direction. A cluster representation is better motivated here
  than a permanently labeled scalar mode, but two directions are not proved
  sufficient for complete rule closure.
* At seed 1, epsilon=.001, first anchor, actual leading Ritz error is
  .0011587 while the reported `kato_temple` quantity is .00009826: an
  underestimate by about 11.8. A gap between two Ritz values is not an
  independently certified spectral separation. The quantity cannot be used
  as an upper-bound certificate merely because it often overestimates error.

Endpoint misalignment is not proof of an eigenvalue crossing. On sampled
straight interpolation from the fixed point to the middle anchor, seed 0
retains a continuously rotating leading direction: minimum adjacent overlap
improves from .820 to .942 on refining 32 to 64 subintervals, with positive
sampled top gap. Seed 1 continuation remains resolution-sensitive. Neither
sampled path is the actual time trajectory, and neither excludes a smaller
unsampled gap. The defensible observation is failure of a fixed scalar label.

## 7. What this makes the research question

The sharpened conjecture is that some kernels admit a small jointly observable
state-and-rule subspace whose projected second jet, cumulative energy, and
model defects can be advanced and bounded by finite block composition. The
candidate should retain interacting clusters rather than divide out resonant
terms. Its dimension must be chosen for the outputs of the changing rule,
not solely the current state residual.

The gap-uniform theorem removes one apparent obstruction to this conjecture.
Equation (7) supplies another obstruction that cannot be wished away. The
remaining work is a non-oracle acquisition rule, certified control of omitted
rule-observable directions and path radius, and a complete cost comparison
including failures. No present result establishes that this can beat ordinary
iteration on the retained general-kernel cases.

Claude's independent critique of this derivation found no error in the
conditional-energy proof and emphasized the exponential dependence on the
radius bound. In the follow-up exchange, he identified a useful refinement:
generic exact observability rank can be n, so practical compression needs a
tolerance-dependent dimension. That is a distinct question from exact closure.
In whitened coordinates S=diag(sqrt(c)) J diag(1/sqrt(c)), let Otilde be the
correspondingly transformed output map, with explicit output scaling. Define

    Wobs,H = sum_{j=0}^{H-1} (S^j)^T Otilde^T Otilde S^j.

This is another powered Gramian. If Pi keeps its r leading eigenvectors and
its next eigenvalue is at most tau, then the omitted initial component obeys

    sum_{j<H} ||Otilde S^j (I-Pi) x||² <= tau ||(I-Pi)x||².

This follows directly from the spectral theorem. It measures output loss over
a specified horizon for a frozen linear map. It does not prove that Pi is
S-invariant, close the projected dynamics, or control nonlinear generation of
discarded modes. Those errors still require the model-defect term in (3).
The observability and energy Gramians use the same composition algorithm but
different weights; one cannot replace the other without a comparison bound.
Building the full Gramian would also be an oracle-cost operation here.

The cost question should therefore use r(tolerance,H,kernel), not presume a
fixed cluster size. Relevant mixed rule actions can cost quadratically in r,
and the full second-jet lift is quadratic in r before further structure is
used. Cluster separation and excitation affect acquisition; neither follows
from a large overlap with one reference vector. A decaying trajectory may
lose visibility of some amplitudes before acquisition, while nonlinear
interactions can later generate components absent from a first-order fit.
These observations sharpen the conjecture without establishing a numerical
rank bound or an acquisition theorem. In particular r=O(1) would be helpful,
but is not a logically necessary condition for speedup: the actual test is
total acquisition, transport, certification, and rejection cost versus the
matched control at the same achieved accuracy.

Proof code: `energy_theory.py`. Focused tests: `test_energy_theory.py`.
Reproduction: `study_energy_theory.py`; Mini commands are in root `AGENTS.md`.
Render the retained figure locally with
`python -m experiments.entropic_transport_closure.report_energy_theory` in an
environment with Matplotlib. No new numerical study is performed by rendering.
