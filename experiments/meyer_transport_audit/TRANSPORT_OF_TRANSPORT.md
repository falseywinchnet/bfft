# Carrying transport of transport as a coupled state

September 4, 2026. The user's correction is that transport and its changing
geometry are inseparable, and additional state might let us model their
joint evolution. This investigation establishes an exact finite-increment
representation and tests two compressed implementations. The exact
representation works; the compressed shortcuts do not yet deliver useful
acceleration.

## 1. An exact enlarged state

Let z be the complete six-field state and T the existing ordinary pass.
Define its finite transport increment

    h = T(z)-z.

Carry both z and h. The next state is z+=z+h. The next transport is

    h+ = T(z+h)-T(z).

This is an exact finite change, including every projection event and both
changing driving images. It does not require identifying a stationary
subproblem, adding a scalar step multiplier, or discarding any field.

The new state is computationally redundant: h is a function of z. The
reason to carry it is to expose and reuse the evolution of the update,
not to claim that it adds information absent from the original Markov state.

## 2. Exact finite transfer through a disk projection

Consider two local vectors x and y=x+h, with radii r=|x| and s=|y|.
The disk radius is a. Define

    sigma(r) = min(1,a/r), with sigma(0)=1,
    sigma_bar = (sigma(r)+sigma(s))/2,
    rho = (min(a,s)-min(a,r))/(s-r),
    v = (x+y)/(r+s).

At equal radii choose rho=1 in the disk and rho=0 outside; at the origin
take v=0. The implementation uses stable piecewise divided differences.
Then the symmetric matrix

    B(x,y) = sigma_bar I + (rho-sigma_bar) v v^T          (1)

satisfies

    Pi_a(y)-Pi_a(x) = B(x,y)(y-x).                     (2)

To verify (2), expand sigma(s)y-sigma(r)x into the sum of an average-scale
term and a scale-difference term, and use
|y|^2-|x|^2=(x+y) dot (y-x).

Moreover 0<=sigma_bar,rho<=1 and |v|<=1. One eigenvalue is sigma_bar;
the other is sigma_bar(1-|v|^2)+rho |v|^2. Thus

    0 <= B <= I.                                      (3)

This finite transfer handles radial and angular change and net effects
of crossing the disk boundary. No small-displacement approximation is
needed. It is one exact secant representation, not a unique Jacobian and
not necessarily the path-average Jacobian. Its exactness is for the actual
increment y-x; applying it to unrelated directions is a modeling choice.
The contraction statement concerns the local projection, not the entire
coupled Meyer iteration.

## 3. Transport of the complete increment

Construct B_u and B_w from the two stored vector fields and their actual
increments. Let delta p_u=B_u h_tu and delta p_w=B_w h_tw. With D*=-div,

    h_u+ = S_u[c_u(h_u+h_w) + eta_u D*(h_tu-2 delta p_u)],
    h_w+ = S_w[-c_w h_u+ + eta_w D*(h_tw-2 delta p_w)],
    h_tu+ = D h_u+ + delta p_u,
    h_tw+ = D h_w+ + delta p_w.                         (4)

The source cancels when the two complete passes are subtracted. The second
increment solve still uses the newly computed first increment. Let A_k
denote this complete linear transfer assembled from the current finite
matrices. The exact lifted recurrence is

    z_(k+1) = z_k+h_k,
    h_(k+1) = A_k h_k,
    A_k = A[B(t_u,k,t_u,k+h_tu,k), B(t_w,k,t_w,k+h_tw,k)]. (5)

The transfer field changes with the state and the transported increment.
Keeping A fixed would discard part of this coupled evolution.

There are still two screened solves (four FFTs) per lifted step. The
implementation carries 12 main real fields instead of six; retaining both
symmetric 2x2 transfer fields would add six more planes. The prototype
currently constructs these transfer fields as temporary arrays. This is
an exact representation with extra state, not an established speedup.

## 4. How the transport change itself evolves

Let a_k=h_(k+1)-h_k. Subtract the successive transfer equations:

    a_(k+1) = A_(k+1) a_k + (A_(k+1)-A_k) h_k.        (6)

This is an exact law for the change in transport in the chosen finite
representation. Its two terms must evolve together. The second term is
the effect of the changing transfer geometry on the current transport;
carrying only a_k while freezing that geometry omits it.

It is not a decomposition into independently solvable physical processes.
The transfer matrices are themselves determined by the coupled trajectory.
The term sizes and commutators depend on the chosen secant representation,
although equations (2), (4), (5), and (6) are exact for that representation.

Five 64-square sources at (lambda,mu)=(.05,40) were followed for 128 steps.
The maximum state RMS discrepancy between the lifted and ordinary
trajectories was 1.37e-12, occurring in the ramp. Other maxima ranged from
2.26e-14 to 3.32e-13.

At pass 4, the norm of the changing-geometry term relative to the norm of
a_(k+1) was .643 for the ramp, .447 for crossing, and .501 for noise. These
are ratios of norms, not additive percentages of explained change; terms
can cancel. The edge and carrier had zero such contribution at that sampled
state. Geometry evolution is consequential in some states and absent in others.

In the same chosen transfer representation, reversing the order of two
successive operators changed the predicted second increment by .434,
.236, and .225 of its norm for ramp, crossing, and noise. This swapped
calculation is a diagnostic, not a second physical trajectory. It warns
against treating the successive transfer operators as interchangeable.

## 5. A practical compressed model was also tested

To ask whether extra carried state can save global solves, the first
prototype stores a few complete transport directions Q plus their adjoint
responses. In the retained affine subspace z=z0+Qy, it evaluates

    y+ = Q^T[T(z0+Qy)-z0]                              (7)

exactly, while re-evaluating every pointwise disk at every modeled state.
The only approximation in (7) is restriction to that subspace.

The adjoint identity is inexpensive to state. For a complete test field
q=(q_u,q_w,q_tu,q_tw), compute

    R_w = S_w(q_w+D*q_tw),
    R_u = S_u(q_u+D*q_tu-c_w R_w).

Then <q,T(z)> is a fixed linear functional of z plus a constant and

    <q_tu-2 eta_u D R_u, Pi(t_u)>
      + <q_tw-2 eta_w D R_w, Pi(t_w)>.

Those response fields are prepared once per retained direction. Each
inner model step then requires pointwise projections and reductions, but
no FFT. Tests compare it with the full projected map on arbitrary retained
coordinates and explicitly block FFT calls during the inner rollout.

The directions come from 2, 4, or 8 consecutive complete-state increments.
All training steps, SVD, adjoint preparation, nonlinear evaluations,
reconstruction, and one ordinary refresh are charged in the timings.
A matched frozen-Jacobian model uses the same retained directions and
the geometry at the end of training.

Nonlinear Galerkin reduction and the cost of evaluating its nonlinear
terms are established topics; this is not a claim to invent model reduction.
For context see Chaturantabut–Sorensen, 2010:
https://epubs.siam.org/doi/10.1137/090766498 . The current prototype does
not use DEIM or sample only a subset of the disks.

## 6. What the experiments rejected

The screen covers five 64-square sources, starting passes 4 and 32, retained
depths 2/4/8, and horizons 16/32. Each candidate and the ordinary target
receive one final ordinary pass. Three timing repetitions use the identical
NumPy backend with BLAS thread counts limited. These are exploratory timings,
not shuffled native benchmarks or a same-accuracy speed comparison.

| Retained directions | Nonlinear carried model: median time / ordinary | Median gap / ordinary |
|---|---:|---:|
| 2 | .514 | 3.700 |
| 4 | .693 | 3.434 |
| 8 | 1.130 | 1.902 |

The nonlinear model had a lower gap than its matched frozen model in 40
of the 48 non-carrier comparisons. It beat ordinary transport in none of
those 48. The simple carrier was reproduced to numerical precision. Faster
computation of a poor approximation is not the requested result.

A second version retained all four Bregman fields at full spatial
resolution and reduced only the jointly updated primal pair. It preserved
the exact local memory update and capacity invariant every modeled step.
It still beat ordinary in none of the 48 non-carrier cases, and generally
performed worse than the fully coupled subspace version. Its median
gap ratios were 11.50, 8.25, and 4.96 at depths 2,4,8.

The results reject these particular static low-dimensional restrictions.
They do not reject carrying transport-of-transport. In particular, keeping
more dual memory did not compensate for restricting the primal motion
needed to remain compatible with it. That is evidence for taking the
user's inseparability constraint seriously at the representation level.

## 7. Current conclusion

The exact augmented system (5)–(6) is the sound starting point. It carries
the full spatial transport and its changing transfer geometry together.
The remaining computational question is how to reuse that evolving transfer
over multiple passes while controlling the error of the **whole coupled
state**. A fixed small subspace is not sufficient in the tested cases.

No general accelerated implementation has been obtained. The added-state
recurrence is exact and tested; both attempted shortcuts and their full
costs are retained as negative evidence. Production code and the paper
remain unchanged.

## Files and reproduction

Exact lift: finite_transport_lift.py and test_finite_transport_lift.py.
Its four tests cover finite projection transfer and eigenvalue bounds,
arbitrary complete-state differences, a 128-step trajectory, and (6).
Probe: probe_finite_transport_lift.py; data: meyer_finite_lift64.json.

Compressed experiments: transport_of_transport.py and
test_transport_of_transport.py. Its five tests cover the adjoint identity,
the projected nonlinear map, training-trajectory reproduction and absence
of FFTs in its inner rollout, constant states, and full-memory invariants.
Data: meyer_carried64.json and meyer_carried64_full_memory.json. The later
record includes all four comparisons and supplies the table above.

All nine tests passed on the M4 Mini. Repository AGENTS.md contains the
remote commands. Diagnostic operator comparisons use additional transforms;
they are not included in any claim of free production statistics.
