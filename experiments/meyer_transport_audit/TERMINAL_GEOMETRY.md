# Finding the bottom through transport feasibility

September 4, 2026. Factor investigation is discontinued at the user's request.
This experiment asks what terminal transport conditions determine the answer.
It does not rescale an iteration or extrapolate its history.

## The target

For D the periodic forward gradient and D* = -div, write

    min TV(u) + lambda/2 ||f-u-D*g||^2,  with |g_i| <= mu.

Its dual maximizes

    <f,q> - ||q||^2/(2 lambda) - mu TV(q),
    q = D*p, |p_i| <= 1.

Equivalently, the dual minimizes ||q-lambda f||^2/(2 lambda) + mu TV(q)
over that convex feasible set. The quadratic makes the optimal q unique.
This does not establish uniqueness of u, g, or the decomposition.
These are consequences of convex duality for the existing model, not claims
of a new Meyer model or a new duality principle. The original model and
projection construction are in Gilles–Osher, UCLA CAM 11-73:
https://ww3.math.ucla.edu/camreport/cam11-73.pdf

At the bottom the following conditions hold together:

    q = lambda (f-u-D*g),
    p_i dot (Du)_i = |(Du)_i|,       |p_i| <= 1,
    g_i dot (Dq)_i = mu |(Dq)_i|,    |g_i| <= mu.

The feasible gap is exactly the sum of the two complementarity deficits
and the squared residual-balance deficit. Each is nonnegative in exact
arithmetic. An upper/lower gap Gamma also bounds the unique field:

    ||q-q*||_2 <= sqrt(2 lambda Gamma).

This follows from strong convexity of the negative dual on its feasible
set and P >= optimal objective. It is not a bound on the texture image.

## A terminal shortcut that failed

Given a current feasible q, setting g_i = mu (Dq)_i / |(Dq)_i| wherever the
gradient is nonzero makes texture complementarity exact. Setting
u = f-D*g-q/lambda then makes residual balance exact. Existing flux is
retained at exactly zero gradient. All bounds remain feasible.

This is an algebraic attempt to impose terminal equations, not an accelerated
step. It fails badly because it does not impose cartoon complementarity.
Tiny nonzero gradients can be numerically reliable as values yet give a
terrible prediction of the terminal flux direction when the limiting
gradient vanishes. At Dq*=0, optimality permits an entire capacity disk;
it does not demand saturation.

At 32-square resolution, 12 source/parameter cases and six checkpoints
(16,32,64,128,512,2048) were measured. At pass 64, the low carrier with
(lambda,mu)=(.05,40) has ordinary gap/pixel 2.7408e-5 and q RMS drift to
pass 2048 of only 6.89e-7. The direct saturated candidate instead has gap
28.6237, entirely in cartoon complementarity to rounding accuracy.
The finite 2048 checkpoint is not treated as an exact oracle.

Different cases have different unresolved conditions. At pass 64, edge
(.05,40) has gap terms approximately (.00242, .67352, .02490), ordered as
cartoon, texture, balance. Noise (.02,20) has (.04516, .00220, .55047).
A single scalar gap obscures this difference in terminal work.

## A terminal construction that succeeds in a certified regime

The objective is nonnegative. If u is constant and the entire centered image
can be represented by a capacity-feasible flux, the objective is zero and
the global bottom is attained. This supplies a direct sufficient test:

    u = mean(f),
    D*D phi = f-mean(f), with zero mean phi,
    g0 = D phi.

If max_i |g0_i| <= mu, then D*g0 = f-u, q=0 is dual-feasible, and both
primal and dual objectives are zero in exact arithmetic. The discrete
periodic Poisson equation uses one forward and one inverse FFT. There is
no trajectory prediction, factor, continuation parameter, or fitted threshold.

The implementation checks capacity and evaluates a feasible gap after disk
projection to account for the computed reconstruction. It reports the actual
floating-point upper bound rather than claiming exact floating-point zero.

For the tested source f=100+2 sin(2 pi (x+2y)/32), the candidate flux peak
is 4.575805. It fits each tested capacity (20,40,80). The resulting gap/pixel
is between 2.34e-31 and 1.17e-30. Ordinary pass 64 has gaps between 1.93e-5
and 5.90e-5. The terminal construction uses a single Poisson solve, rather
than 64 four-FFT passes. This is an operation-count comparison; no native
end-to-end wall-time speedup has been measured.

All nine ramp, edge, and noise cases fail this sufficient Poisson-flux test.
The returned projected candidate is not accepted as an optimum in those
cases. This is a successful terminal regime, not a solution of the full
decomposition problem.

## What the general problem now asks

The Poisson flux minimizes squared flux norm, not peak capacity. Its failure
does not prove that the centered image cannot fit. Every flux g0+h with
D*h=0 transports the same centered image. Such divergence-free changes
redistribute the load without changing its image-space divergence.

The next transport question is therefore whether this freedom can satisfy
capacity, and, when it cannot, which cartoon structure and residual are
required at the optimum. A general terminal solver must determine the free
and saturated flux regions jointly with cartoon complementarity and residual
balance. The numerical experiment rules out simply normalizing a tentative
dual gradient. There is no proved general solver or new novelty claim yet.

This suggests working on capacity feasibility and simultaneous terminal
conditions, keeping the existing global spectral machinery where useful.
The constant-cartoon result is the first executable piece of that direction;
it is not a replacement objective for the general task.

## Reproduction

Source: terminal_geometry.py. Four tests in test_terminal_geometry.py verify
agreement with the existing gap certificate, the failed candidate's two
exactly closed conditions, the sinusoidal capacity certificate, and the
constant-image case. All four passed on the M4 Mini.

Run through m4build from the repository:

    python3 -m unittest experiments.meyer_transport_audit.test_terminal_geometry -v
    python3 experiments/meyer_transport_audit/terminal_geometry.py --size 32 --out /tmp/meyer_terminal_geometry32.json

Copy /tmp/meyer_terminal_geometry32.json back immediately. The complete
record is saved here as meyer_terminal_geometry32.json. No production
preset or manuscript has been changed by this follow-up.
