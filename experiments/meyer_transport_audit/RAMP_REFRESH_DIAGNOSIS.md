# The ramp regression: coupled ringing and certificate-stage sensitivity

The September 5 follow-up separates two effects. Refreshing both memory
branches excites a slowly decaying sign-alternating mode. Measuring only
the next projected witness also substantially exaggerates the regression,
because the refresh has already retained a better feasible witness.
All statements below concern f_i=255 i/128 with periodic differences: a
sawtooth with a large wrap jump, not a nonperiodic affine ramp.

## Controlled reduction and objective measurement

The repeated-row 128-square source has an invariant one-row subspace. The
diagnosis uses its 1x128 representative with the same per-sample objective.
A repeated-row numerical equivalence test checks all four branch ablations.

For any feasible projected or retained memory pair p_u,p_w, define

    q = eta_u D* p_u,  g = (eta_w/c_w)p_w,  v = D*g.

The feasible primal-dual gap decomposes exactly into

    TV(u)-<Du,eta_u p_u>
    + mu TV(q)-<Dq,g>
    + (lambda/2)||f-u-v-q/lambda||^2.                 (1)

These are cartoon complementarity, texture complementarity, and quadratic
balance respectively. Each is nonnegative. Tests check the decomposition
against certificate.py for both incoming and projected feasible fields.

The source can admit different optimal u/v decompositions. Distances from
one long-run reference u must not be called optimality errors. The JSON
labels them reference distances. Gap bounds and their nonnegative components
are used for conclusions about progress.

## Why the original gap convention exaggerates the regression

Let an ordinary endpoint have x+, t_O=Dx++p. Its projected witness is

    q = Pi(t_O) = Pi(Dx++p).

The refreshed endpoint stores t_R=Dx++q. It has the same primal fields,
and its **incoming retained witness is exactly q**. But certificate.py
projects t_R once more and measures r=Pi(Dx++q). Thus the earlier comparison
used q for ordinary and r for refresh, even when refresh already held q.
Both are valid certificates, but r need not be the tighter one.

The exact same-input witness-transport identity is tested. For each method
the diagnosis also computes the best bound from both available stages:

    min(P_incoming,P_projected) - max(Q_incoming,Q_projected).       (2)

Both primal upper bounds are feasible; both dual lower bounds are feasible.
Combining the better bounds is legitimate and is applied to every method.
This changes measurement, not the evolution. Earlier recorded certificates
remain valid; they do not measure the best available bound at each state.

First passage at gap/sample <= 1e-3, checked every pass:

| (lambda,mu) | Ordinary, old convention | Refresh, old convention | Ordinary, best of two stages | Refresh, best of two stages |
|---|---:|---:|---:|---:|
| (.02,20) | 84 | 127 | 84 | 86 |
| (.05,40) | 146 | 181 | 146 | 137 |
| (.1,80) | 316 | 319 | 316 | 285 |

At the tighter 1e-4 target the real low-parameter regression persists:
ordinary 108 passes versus refresh 164 using the better bounds. The old
convention reported 108 versus 216. For (.05,40) the better-bound counts are
197 versus 188, and for (.1,80), 373 versus 343.

These are iteration counts, not new wall-time claims. Earlier validation
sampled only every eight passes; its first-hit counts therefore differ.
Update overhead remains a possible wall-time regression even where this
more complete certificate reveals an iteration improvement.

## The branch interaction is causal

At (.02,20), under the original projected-witness convention, reaching
1e-3 takes 84 ordinary passes, 74 when only u memory is refreshed, 82
when only w memory is refreshed, and 127 when both are refreshed.
Refreshing either branch alone is not sufficient to explain the harm.

The spectral diagnosis constructs only an experimental dense derivative;
it is not a proposed runtime solver. At pass 256, the local derivative
propagates the next 32 actual steps with maximum state RMS errors below
6.6e-13 in all four (.02,20) ablations. The regime is therefore affine
over the measured horizon to numerical precision.

Its strongest negative real eigenvalues are:

| Schedule | Most negative eigenvalue |
|---|---:|
| Ordinary | approximately 0 (roundoff-scale negative values) |
| Refresh u only | -0.401155 |
| Refresh w only | -0.956805 |
| Refresh both | -0.974882 |

The both-refresh mode loses only about 2.5% of its amplitude per pass and
changes sign every pass. The actual full-state increments at pass 384 have
cosine -0.999509 with the following increment, versus +0.999999 for ordinary.
At (.05,40), the both-refresh negative mode is -0.976019 and the measured
increment cosine is -0.999846. Numerical eigenvalues are a local diagnosis,
not a global convergence theorem. Ordinary also has slow positive modes;
the distinction is the strongly excited alternating response, not an absence
of slow dynamics in ordinary transport.

The local derivative explains where this feedback enters. Write A for the
ordinary pass derivative and J+ for projection at the ordinary endpoint.
The extra memory stage changes its field perturbation to

    delta t_R = D delta x+ + J+ delta t_O.             (3)

Where J+=I, delta t_O=D delta x+ + delta p, giving
delta t_R=2 D delta x+ + delta p. The extra gradient contribution is
inserted in both memories and then acts through the next coupled primal
responses. This is an algebraic consequence of the extra stage, not an
adjustable factor. Local projection feasibility does not guarantee good
damping of that coupled feedback loop.

## Where the lost certificate progress appears

At pass 84, (.02,20), the original projected gap decomposes per sample as:

| Method | Cartoon complementarity | Texture complementarity | Balance |
|---|---:|---:|---:|
| Ordinary | 8.144e-4 | 4.97e-5 | 2.6e-6 |
| Refresh both | 9.002e-4 | 2.3447e-3 | 8.84e-5 |

The largest deterioration is in texture complementarity. Relative to the
long-run optimum bounds, ordinary's primal excess and dual deficit are
8.161e-4 and 5.071e-5; refresh's are 1.010e-3 and 2.323e-3. Much of the
reported loss is therefore the quality of the extra projected dual witness,
consistent with the stage comparison above.

Along the full-refresh trajectory, almost all extra w-correction energy
already opposes the first memory correction by pass 64. By pass 128 both
branches have essentially all extra-correction energy at sites where
<beta,e><0. The existing direction gate detects this specific reversal.
It reaches 1e-3 at pass 73 and 1e-4 at pass 103 in the low-parameter ramp,
under either stage convention at those first hits.

## Stop interventions from exactly the same ringing state

Starting from the full-refresh state at pass 128, continue 8 passes with
the same global update but different extra local refresh stages:

| Extra stages continued | Best available gap after 8 passes | Increment cosine |
|---|---:|---:|
| Both | 2.01918e-4 | -0.998400 |
| u only | 8.470e-6 | +0.986864 |
| w only | 1.40373e-4 | -0.997202 |
| Neither | 8.762e-6 | +0.998041 |

Suppressing the extra w-memory feedback quiets the already-excited mode.
There is no new global solve, changed source, new initialization, or new
objective. This supports a repeated feedback mechanism, not merely a bad
first step. Separately, injecting a single refresh into an ordinary state
has relatively small long-term effects after ordinary evolution resumes.

## Implication and limits

The ramp calls for distinguishing retained feasible progress from the next
projection's witness, and detecting when additional memory relaxation begins
to undo the preceding compatibility correction. The existing directional
gate addresses the measured feedback in this source; it is not established
as a universal controller. No production update or default certificate was
changed by this diagnosis. The earlier speed table should be read as times
to its specified projected-witness convention, not times to the best bound
already available in every state.

The two test groups cover branch-ablation equivalence, decomposition and
nonnegativity, witness-stage identity, and repeated-row invariance.
`diagnose_ramp_refresh.py` records trajectories and same-state interventions;
`probe_ramp_ringing.py` records local spectra and increment directions.
`plot_ramp_diagnosis.py` renders the saved JSON locally. Raw records are
`meyer_ramp_diagnosis.json` and `meyer_ramp_ringing.json`.
