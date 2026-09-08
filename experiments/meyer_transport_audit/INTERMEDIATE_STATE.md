# What the intermediate Meyer transport state knows

September 4, 2026. This investigation follows the request to examine the
intermediate state closely. It uses ordinary, unmodified trajectories. No
factor, momentum rule, terminal capacity shortcut, or production change is
introduced. Dense linear algebra below is a structural probe, not a speed claim.

## 1. Recovering the two objects hidden in each stored vector field

Use the repository's periodic forward gradient D, adjoint D*=-div, and
L=D*D. For either branch, x is its primal image (u or w), a=1/eta, and
the stored field is

    t = Dx + b.

The incoming Bregman field is therefore available exactly as b=t-Dx;
it need not be stored separately. The first source-driven initialization
has b=0. At every later ordinary state, b is the previous disk projection
and obeys |b_i|<=a, up to floating-point roundoff.

The current projection and shrinkage are

    p = Pi_a(t),       d = t-p,       delta b = p-b.

They satisfy the **exact intermediate identity**

    Dx-d = delta b.                                      (1)

The new Bregman field is precisely the discrepancy between the current
integrable image gradient and the locally shrunk field. This is a vector
constraint residual. Calling it simply momentum or an accumulated image
correction misses its algebraic role.

The local proximal relation is already exact:

    (p_i/a) dot d_i = |d_i|,       |p_i/a|<=1.             (2)

But d need not be the gradient of any scalar image. Projection solves a
pointwise norm relation; the global solve must reconcile that relation with
spatial integrability and the current driving image.

This shrink/Bregman interpretation is established mathematics, not a novelty
claim. See Goldstein–Osher, *The Split Bregman Method for L1-Regularized
Problems*, https://epubs.siam.org/doi/10.1137/080725891 . The useful work here
is recovering and testing its precise role in this reduced, coupled recurrence.

## 2. The full state increment is a second, staggered constraint residual

After the global solve produces x+, the stored state becomes

    t+ = Dx+ + p.

Consequently

    delta t = D(delta x) + delta b = Dx+ - d.             (3)

Equations (1) and (3) describe two different stages. Before the solve,
delta b measures the mismatch with the current image. After the solve,
delta t measures the mismatch with the new image. The same locally
admissible d is being reconciled with two successive globally integrable
gradients. Their differences can cancel in delta t; neither individual
norm should be silently identified with the full-state update.

For the 64-square crossing at (.05,40), pass 4 has u-branch memory-change
RMS 1.0579 and gradient-change RMS .6115. The norm of their sum is only
.6458 times the sum of their norms. At pass 64, the w-branch memory-change
RMS is .02454 versus gradient-change RMS .00717. The stored field is still
reorganizing appreciably while its primal gradient changes much less.

## 3. Exact propagation through the changing coupled solves

For the u branch define q_old=eta_u D*b_u and q_new=eta_u D*p_u. For the
w branch define v_old=(eta_w/c_w)D*b_w and v_new=(eta_w/c_w)D*p_w. Write
v_out=f-u-w. Substitution of t=Dx+b into the implemented recurrence gives

    (I + eta_u/c_u L) delta u
        = w - (2 q_new-q_old)/c_u,                      (4)

    (I + eta_w/c_w L) delta w
        = v_out-delta u-(2 v_new-v_old).                (5)

The reflected term 2p-b here is an exact consequence of the existing
shrink algebra, not an added relaxation factor. The second equation uses
the newly computed delta u. Thus these identities retain the changing
driving image rather than treating successive inner problems as identical.

The exact global solve responds to a discrepancy between a primal field
and a reflected dual/flux estimate. This supplies a more specific account
of progress than saying that the method takes a preconditioned step.

Across 15 source/parameter cases and eight checkpoints at size 64, identities
(4) and (5) held to maximum RMS discrepancies 1.82e-13 and 4.31e-13.

## 4. Intermediate complementarity has explicit bounds

The usual cartoon complementarity term at the current projection is

    A = TV(u) - <Du, eta_u p_u>.

Since eta_u p_u supports the norm at d_u and Du=d_u+delta b_u,

    0 <= A <= 2 sum_i |delta b_u,i|.                    (6)

For q=eta_u D*p_u and g=(eta_w/c_w)p_w, |g_i|<=mu. The texture
complementarity term obeys

    B = mu TV(q) - <Dq,g>,
    0 <= B <= 2 mu sum_i |D(q-lambda w)_i
                              + lambda delta b_w,i|.   (7)

To prove (7), substitute Dq=lambda d_w + D(q-lambda w)+lambda delta b_w
and use that g/mu supports the norm at d_w. These bounds isolate the
relevant within-pass mismatches. They are generally loose and do not by
themselves give a new convergence rate or an accelerated algorithm.

## 5. Information hidden from divergence can affect a later solve

Let P_L=D L^dagger D* be the periodic longitudinal projection and
P_T=I-P_L. P_T includes both circulation and constant harmonic fields.
It is not correct to call every such component a local curl.

Since P_T Dx=0, equation (1) gives

    P_T d = -P_T(delta b).                              (8)

The retained update carries exactly the nonintegrable part of the locally
shrunk field. In particular,

    min_y ||Dy-d||_2^2 = ||P_T d||_2^2.                 (8a)

No change to a scalar image alone can remove this part of the current
gradient mismatch. In the pass-64 crossing, it accounts for 45.89% and
64.02% of squared mismatch in the u and w branches; in noise the shares
are 87.00% and 78.79%. The local projected/shrunk fields must also change
to reconcile that portion. This is a direct geometric limitation on what
one isolated primal solve can accomplish, not a bound on the performance
of a complete coupled algorithm.

Meanwhile D*P_T p=0, so the current divergence cannot reveal
P_T p. It would be a mistake to conclude that it can be deleted: the
pointwise projection does not commute with the Hodge decomposition.

For a divergence-free perturbation h, away from a projection boundary,

    delta [D* R(t)] = -2 D* J_Pi(t) h,                 (9)

where R=I-2Pi. This can be nonzero although D*h=0. In an entirely interior
projection region J_Pi=I, so that mechanism vanishes. Outside a disk,
J_Pi depends on the local capacity direction and radius; spatial variation
allows a previously invisible field to change the projected divergence.

At pass 64 in the crossing example, 14.49% of the u projection's squared
norm and 18.09% of the w projection's squared norm are in P_T. At pass 128
in noise, the corresponding fractions are 26.03% and 19.02%. These are
field-energy fractions, not percentages of useful work or acceleration.
Their divergences are at rounding scale, while their linearized responses
through the next projection are nonzero. A separate finite-difference test
checks (9). This demonstrates information loss from a divergence-only
representation; it does not demonstrate that more transverse energy is better.

## 6. Every whole-pass nonlinear error has two local sources

Let z+=T(z), h=z+-z, and A=T'(z), using the implemented projection
derivative (interior choice at an exact boundary). For each branch define

    e_u = Pi(t_u+h_tu)-Pi(t_u)-J_u h_tu,
    e_w = Pi(t_w+h_tw)-Pi(t_w)-J_w h_tw.

The exact nonlinear remainder

    E = T(z+h)-T(z)-A h

is then

    E_u = -2 eta_u S_u D*e_u,
    E_w = S_w[-c_w E_u - 2 eta_w D*e_w],
    E_tu = D E_u + e_u,
    E_tw = D E_w + e_w.                                (10)

There is no Taylor approximation in (10); e contains the actual remainder.
All Fourier solves and triangular coupling propagate those two projection
errors linearly. The complete identity held to maximum RMS discrepancy
3.23e-14 across the screen.

This gives a possible computational structure: a global linear response
with nonlinear sources localized by the current projection geometry. It
does not yet justify ignoring any source or predict future crossings.
Computing these diagnostic remainders used additional evaluations; they are
not claimed to be available for free in production.

For the crossing at pass 64, only .244% of u sites and .610% of w sites
change projection branch over the next step. Those crossings account for
about 98.2% and 99.9% of the respective projection-remainder energies.
This is a promising measured concentration, not a universal sparsity theorem.

Stable exterior branches still curve in two dimensions. At noise pass 128
there are no crossings, but projection remainder RMS is 3.67e-6 for u
and 1.07e-7 for w. All that remainder is on stable exterior sites.
Counting branch changes alone therefore cannot certify an affine 2-D model.

## 7. The certificates themselves have a phase

At a stored state, b=t-Dx is the incoming feasible field and Pi(t) is the
newly projected feasible field. Both can give legitimate certificates, but
they represent different stages. The earlier certificate.py uses Pi(t).
That is valid; it is not identical to measuring the incoming Bregman pair.

The trace measures four stages: incoming fields/current u; newly projected
fields/current u; newly projected fields/new u; and next projections/new u.
For the ramp at (.05,40), pass 64 gives gaps per pixel

    .04779 -> .06259 -> .05307 -> .06177.

Some intermediate changes improve the witness and others worsen it.
An endpoint scalar alone cannot say which internal relationship was learned
or which was temporarily disturbed. Selecting a different witness is also
not the same as advancing the state.

## 8. Does the intermediate pattern already determine the answer?

To test this without assuming that 2-D disks are affine, the second probe
uses genuine one-dimensional periodic signals embedded as a 1x64 image.
The y fields remain exactly zero. Projection is scalar clipping, and each
fixed clipping branch/sign pattern makes the complete map exactly affine:

    T(z+h) = T(z) + A h

as long as the segment stays in that pattern. A candidate fixed point must
satisfy

    (I-A)h = T(z)-z = r.                               (11)

The probe explicitly builds the 256-dimensional matrix, solves by dense
least squares, checks linear consistency, checks the candidate's clipping
pattern, and evaluates the actual nonlinear residual and objective gap.
This is standard affine fixed-point/active-set algebra used as an oracle;
it is not a new linear solver or an attribution claim about the original idea.

All examples below use lambda=.05, mu=40:

| State | Incoming gap/pixel | Linear residual RMS | Changed clip sites | Candidate gap/pixel | Interpretation |
|---|---:|---:|---:|---:|---|
| ramp, pass 16 | 1.20960 | 2.00e-14 | 10 | 25.1612 | affine solution leaves its own pattern |
| ramp, pass 64 | .0625907 | 1.12e-14 | 0 | 5.77e-14 | current pattern contains a certified optimum |
| edge, pass 64 | .253603 | .0371776 | 0 | .204660 | fixed-point equations are inconsistent |
| carrier, pass 4 | .126092 | 1.67e-15 | 0 | 8.53e-14 | current pattern contains a certified optimum |

The ramp pass-64 and carrier pass-4 candidates have actual full-state
fixed-point residual RMS 1.48e-14 and 5.29e-15. The carrier construction
here is inferred from the complete intermediate clipping pattern, not from
the prior constant-cartoon Poisson shortcut. Dense cost is substantial;
these numbers demonstrate information already present, not practical speed.

The failed ramp candidate shows why solving a local model is insufficient.
The edge case is more revealing: even though its candidate changes no clip
sites, the equations cannot be satisfied on that pattern. A least-squares
answer must not be reported as a solved fixed point.

## 9. A stalled pattern can encode a necessary future event

Let M=I-A and let h_ls minimize ||M h-r||. The residual

    s = r-M h_ls

lies in ker(M^T), to numerical precision. If s is nonzero, no fixed point
exists in the current affine pattern. Further, s^T A=s^T, so while the
pattern persists,

    s^T(z_{k+1}-z_k) = s^T r = ||s||^2.                (12)

This is a precise linear drift obstruction, rather than a vague claim that
the method has slowed. It identifies a scalar direction that cannot settle
while the current equations remain applicable. It does not, alone, prove
when a crossing occurs or that all trajectories remain bounded.

In the measured edge trajectory, pass 64 waits another **150 ordinary
steps** for its first clipping-pattern change. Starting at pass 128 gives
the same event 86 steps later, at absolute pass 214. The pattern therefore
persists for a long interval even though its linear fixed-point equations
already reveal that it cannot finish the solve.

This is the clearest distinction found in the intermediate state:

* A consistent, admissible pattern can already encode the remaining answer.
* A consistent but inadmissible local answer points outside that pattern.
* An inconsistent pattern carries unresolved drift and cannot be closed.

The previous whole-step gap/factor studies did not distinguish these cases.

## 10. Implications for further work

The evidence supports investigating how to use the intermediate projection
geometry to separate already-resolvable global motion from necessary changes
of local constraints. A plausible next design would solve the consistent
bulk while retaining and resolving the projection events that obstruct it.
It must preserve the full retained vector state, including divergence-free
information, and handle the curvature of exterior disks in two dimensions.

The hard questions are now explicit: can linear consistency and the relevant
drift be detected cheaply; can the next necessary event be located without
walking every intervening pass; and can local event changes be incorporated
without rebuilding an expensive global system? A count of stable pixels, a
gap decrease, or a dense least-squares success is not enough to answer them.

No fast general solution to those questions is claimed here. The result of
this exploration is a tested intermediate-state model and a concrete
separation of two mechanisms of slow progress, together with an example
where the intermediate state already contains a certifiable completion.

## Reproduction and scope

intermediate_state.py: five sources, three parameter pairs, checkpoints
1,2,4,8,16,32,64,128 at 64-square resolution. Results:
meyer_intermediate64.json.

intermediate_chart.py: three 1x64 sources, one parameter pair, checkpoints
4,16,64,128, dense chart probes and up to 2048 forward steps to locate the
first actual branch event. Results: meyer_intermediate_chart64.json.

test_intermediate_state.py checks the within-pass identities and bounds,
the whole-map remainder identity, the Hodge nonintegrability relation,
finite-difference coupling from a divergence-free field, and the successful
and obstructed affine-chart examples. All three test groups passed on the
Mini. Run on the Mini via m4build. All
diagnostic transforms and dense solves are additional research work, not
included in any production timing claim. Existing native and paper artifacts
are unchanged by this investigation.
