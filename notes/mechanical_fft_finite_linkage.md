# Exact finite-motion completion of the mechanical FFT

2026-09-04. Companion to [the mechanical cone-FFT derivation](mechanical_cone_fft.md)
and the [standalone simulator](../experiments/mechanical_cone_fft/simulator/README.md).

## What is completed

The prior work proves an N=8 signed Bruun factorization, its positive two-rail
lift after projection, and a forward kinematic gauge yielding convex rows.
It also studies a *small-deflection* beam realization. The missing geometric
step is an exact finite-displacement interpolation member: a rigid bar with
both endpoints restricted to fixed horizontal coordinates cannot rotate
without changing length.

The simulator closes that step with horizontal pin slots and a lateral centre
guide. It then computes the spectrum from the tap coordinates of all 44
members. It does not use an FFT result to prescribe a decorative animation.
This completes the ideal finite-motion kinematic map. It does not complete
fabrication, collision-free routing, elasticity, friction or dynamic analysis.

## 1. Finite interpolation-bar theorem

Let input displacements be a,b in [0,1], a rigid member have length L>0,
and choose a travel T satisfying 0<T<L. Prescribe endpoint heights

    y_A = T a,   y_B = T b.

Constrain the bar's midpoint to a vertical guide at horizontal coordinate c.
Let its endpoint pins move horizontally in slots attached to the two input
shuttles. On the branch with the right endpoint to the right, the constraints
have the unique solution

    h = sqrt(L² - T²(b-a)²),
    A = (c-h/2, T a),
    B = (c+h/2, T b).

The distance |B-A| is identically L. A material tap at fixed fraction beta
along the member is

    P = A + beta (B-A).

Attach that tap, through another horizontal slot, to a vertical output
shuttle. Its displacement is therefore

    q = P_y/T = (1-beta)a + beta b.

No small-angle approximation appears. Slot motion absorbs the horizontal
change; the member never stretches and beta never moves along it.

The branch is nonsingular over the full input box because

    h >= sqrt(L²-T²) > 0.

The page chooses T=0.55L, so the maximum rotation is arcsin(0.55), about
33.37 degrees. Each endpoint can retreat inward by at most
`(L-sqrt(L²-T²))/2`, about 0.08242L; the enlarged view's endpoint slots cover
that range. The tap's lateral excursion is at most
`|beta-1/2|(L-sqrt(L²-T²))`.

The differential relation is also exact:

    dq = (1-beta) da + beta db.

Consequently the member retains its constant interpolation law at finite
rotation. Under ideal workless constraints, virtual work gives the reciprocal
force ratios. This is the usual constrained-mechanics framework described in
[MIT's virtual-work lecture](https://ocw.mit.edu/courses/16-61-aerospace-dynamics-spring-2003/e16936ad266f184f38159855f69fc227_lecture9.pdf);
the formula above supplies the explicit geometry for this construction.

## 2. Completion for a convex feed-forward linkage

A row-stochastic row with finite nonnegative weights is a finite binary tree
of the interpolation members above. Inductively, all intermediate q remain
in [0,1]; the same travel bound protects every member. For an acyclic network,
all member poses and output shuttles are uniquely determined from the inputs.

Thus a finite row-stochastic network has an exact finite-displacement
kinematic realization *given ideal unit displacement transfers between
modules*. Those transfers can be modeled as shared or rigidly connected
shuttles/yokes. The revised page instantiates those transfers as 60 connected rigid yokes in
separate depth planes. Source slots, stems, crosspieces, branches and receiving
slots translate as a single body. The 12 pass-throughs share existing yokes.
Axle pins connect each front-plane lever to its source and receiver slots.
Finite-thickness collision clearance remains a separate engineering problem.

Downstream load alters the forces needed to prescribe the input motion.
It does not alter these rigid constraints. This distinction is why an ideal
linkage is not equivalent to the failed naive cascade of finite springs in
the earlier study. Real compliance and friction require the earlier beam
model or a new physical model, not an arbitrary settling animation.

## 3. Exact N=8 Fourier recovery

Let signed stages be M_s, with the original packed orthonormal Fourier map
`F = P M_2 M_1 M_0`. Write C_s for the canonical positive two-rail lift. Use

    Pi = [I, -I],   Pi C_s = M_s Pi,
    d_0 = 1,
    d_(s+1) = 1 / (C_s (1/d_s)),
    B_s = diag(d_(s+1)) C_s diag(d_s)^-1.

Every B_s is row-stochastic. This is the **displacement** gauge, not the
backward column-stochastic mass gauge. Input encoding is smooth and bounded:

    p_i=(1+x_i)/2,   n_i=(1-x_i)/2,   x_i in [-1,1].

Thus Pi q_0=x. All physical rails stay between zero and one, including the
common-mode offset. The composed linkage satisfies

    q_3 = diag(d_3) C_2 C_1 C_0 q_0,
    P Pi diag(d_3)^-1 q_3 = F x.

The simulator reads this fixed output ruler calibration from the moving
shuttles. Packed order is

    [DC, Re1, Im1, Re2, Im2, Re3, Im3, Nyquist].

The signed stage's natural wire permutation is `[0,4,5,1,2,6,7,3]`. To show
ordinary unnormalized FFT bins, multiply DC and Nyquist by sqrt(8), and each
interior real/imaginary coordinate by 2. The page's bottom moving-ruler readout and
complex-bin cards both use this calibration.

Only these calibrated final positions produce the displayed FFT. A separate
direct DFT checks the answer; it never drives a member or output coordinate.

The complete vocabulary is unchanged:

- 44 bars, by stage [16,24,4];
- 36 midpoint taps and 8 taps at sqrt(2)-1;
- 12 unit pass-through rows;
- 16 nonnegative input rails driven by 8 fixed-pivot input rockers.

## 4. Mathematical corrections and scope

The early physical FFT whitepaper called the canonical lift a monoid
homomorphism. That is false before projection. The counterexample already
present in the later mechanical note is now reflected in the whitepaper:
`C(A)C(B)` may contain common mode absent from `C(AB)`, while their projections
agree exactly. The mass-gauge potential is also correctly described as a
backward DAG potential, rather than generally as a Perron eigenvector.

Positive sheet-mass balance is not automatically an energy or passivity law
for arbitrary optical, electrical or mechanical coordinates. In this model,
rails encode **displacement**. Passivity and force reciprocity follow from
ideal mechanical constraints and virtual work, not from calling displacement
“mass.” Likewise a complex conjugation operation is not a scalar phase delay
on an arbitrary complex envelope, and propagation is not zero latency.
The mechanical simulator makes none of those stronger substrate claims.

## 5. Certificates and user interface checks

`test_mechanics.cjs` evaluates 1,266 input cases: all 256 sign-box corners,
8 basis impulses, 1,000 seeded random inputs, zero and a small constant signal.
It independently checks:

- all calibrated outputs against a direct DFT;
- all intermediate rail differences against the signed Bruun stages;
- nonnegative row-stochastic stage coefficients;
- rigid length and material tap fraction in every bar;
- bounded rail positions;
- orthonormal Parseval energy;
- the bar/tap/pass-through census and rejection of invalid inputs/geometry.

Recorded maximum absolute errors:

| Certificate | Error |
|---|---:|
| FFT coefficient | 3.862e-15 |
| Unit bar length | 2.220e-16 |
| Tap distance fraction | 1.110e-16 |
| Parseval energy | 5.329e-15 |
| Stochastic row sum | 0 |

The existing NumPy `certificate.py` also passes on the M4 Mini, independently
confirming the original factorization and the [16,24,4] inventory.

Browser checks verify the impulse response (all bins 1), alternating response
(Nyquist 8), keyboard sample-slider input, bar selection, desktop layout and
phone layout. The reference DFT check remains below 1e-14 for these cases.

## 6. How to interpret the motion

Input motion is prescribed and eased for readability. Each animation frame
solves all member constraints in topological order. Intermediate members
are not separately eased; doing so would violate the mechanism while it
moves. Slow motion lengthens only the prescribed input transition.

This is a quasistatic kinematic simulator, with no claimed masses, damping,
settling time or natural frequencies. An actual dynamic extension should use
the constrained equations of motion and specified physical masses, joints,
loads and dissipation. It should not add arbitrary per-node spring easing and
call the result physical.

## 7. Connected vertical assembly correction

The earlier visualization used symbolic routing curves and separate sliders.
That did not demonstrate a connected physical system. The revised assembly
uses input levers and rigid transfer bodies throughout, with Fourier output
rulers at the bottom.

For input rocker length L_in and common travel T, use fixed pivot (c, y0) and
opposite endpoint displacements ±Tx/2. Their horizontal separation is
sqrt(L_in² − T²x²). The midpoint stays fixed, the member length stays L_in,
and its slotted endpoint receivers encode (1+x)/2 and (1−x)/2. The rendered
assembly uses L=48 for interpolation bars, T=26.4, and L_in=66.

For a source rail q, every vertex of its yoke is its reference vertex plus
(0, −T(q−1/2)). All distances within that rigid body are invariant. Its source
slot receives the upstream tap pin; each terminal slot receives a downstream
endpoint pin. Slot width 12 covers the entire horizontal pin excursion, with
clearance, over the full input box. Fixed guide brackets connect to the frame.

The bottom ruler rides the negative yoke and its pointer rides the positive
yoke. In screen coordinates, (y_negative − y_positive)/T = p−n. Multiplying
this relative distance by the fixed output calibration recovers the FFT.

`test_assembly.cjs` checks these exact rendered coordinates in 1,257 cases.
Maximum source/receiver vertical mismatch and rigid segment length error are
1.14e-13 SVG units. Maximum calibrated ruler error versus direct DFT is
5.69e-14. These are connectivity and finite-motion certificates, not a claim
that finite-thickness axles can pass through every intervening layer without
additional clearance design.
