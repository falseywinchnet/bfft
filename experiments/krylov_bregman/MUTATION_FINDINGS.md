# What Sinkhorn does not teach us about Meyer, and tested mutations

The Sinkhorn result establishes portability of finite Krylov transport after
supplying the new iteration, its state, tangent, and stopping rule. It does not
establish that Sinkhorn learns Meyer, or that our code has discovered those
problem-specific ingredients. The useful comparison is between the dynamics
exposed by those ingredients.

This follow-up derives a structural difference, implements two families of
mutation with a defect-enrichment variant and a contraction ablation, and
checks full cost. No new mutation is promoted: all are slower on the five
original Meyer cases. One mutation improves local prediction at all 30 anchors,
which identifies a useful direction but does not establish useful acceleration.

## The difference is coupled phase and nonlinear response, not merely curvature

| Property | Sinkhorn used in the transfer study | Original Meyer splitting |
|---|---|---|
| Supplied problem data | Cost kernel, two positive marginals, entropy strength | Image, TV and texture parameters, gradient/divergence, screened operators |
| Complete reduced state | One gauge-fixed log scaling vector | `s=u+w` plus two vector memory fields `t_u,t_w` |
| Constraint primitives | Positive scaling / marginal normalization | Euclidean disk projection and reflection |
| Tangent mechanism | Two positive conditional averaging maps | Screened spatial response coupled to reflected projection derivatives |
| Characteristic local behavior | Nonnegative real spectrum after a suitable similarity | Complex phase pairs, unit directions, changing projection orientation |
| Stopping evidence | Actual transport-plan marginal error | Actual primal/dual gap certificate for the original objective |

The underlying states are globally coupled in both cases. Sinkhorn is not a
collection of independent scalar recurrences. Nor should the full Meyer
splitting map be identified with smooth memoryless mirror descent; the
smooth-objective compatibility theorem in the paper has a narrower scope.

### Sinkhorn has a positive tangent structure

At a finite state let v=exp(y), u=p/(Kv), and s=K^T u. Before gauge removal,

    J = diag(1/s) K^T diag(u/(Kv)) K diag(v).

For the positive diagonal metric M=diag(v*s),

    M J = diag(v) K^T diag(u/(Kv)) K diag(v) = B^T B.

Consequently J is similar to a symmetric positive semidefinite matrix. It
also obeys J 1=1. Its spectrum lies in [0,1]; the gauge quotient removes the
trivial constant direction. This statement is local at each state. It does
not imply one globally fixed Euclidean metric or one globally fixed tangent.

### Meyer retains a turning mode even in a linear regime

For one graph/projection branch, suppose its disk projection is locally the
identity and isolate one longitudinal Fourier mode. In whitened coordinates,
with sigma=sqrt(a)*|D|, the primal/memory update has the two-by-two block

    1/(1+sigma^2) * [[1, -sigma], [sigma, 1]].

Its eigenvalues are (1 ± i*sigma)/(1+sigma^2): contraction accompanied by
rotation. This is a branch calculation, not a claimed diagonalization of the
complete coupled nonlinear Meyer map. The full map additionally couples the
two branches and changes disk projection derivatives with the state. Outside
a disk, DP(t)=r/|t| * (I-n n^T), so a fixed inside/outside label does not freeze
the geometric response.

A small explicit-Jacobian diagnostic confirms this distinction. At prefixes
0,4,32, the 64-dimensional Sinkhorn quotient has no eigenvalues with imaginary
part above 1e-8. The 320-dimensional Meyer quotient on an 8x8 camera image has
132,196,194 such eigenvalues, with maximum imaginary magnitudes
0.471,0.459,0.468. The dimensions and norms differ; this is a structural probe,
not a claim that Meyer has a larger coordinate-independent nonnormality index.
No explicit Jacobian enters any accelerated solve.

## Frozen-tangent accuracy is not actual-trajectory accuracy

`phase_probe.py` separately replays the exact frozen affine tangent and the
actual nonlinear map, then compares depth 4,8,12 finite approximations.
Compression error and nonlinear geometry error can cancel. Improving only
the former can therefore worsen the actual prediction.

For the 32x32 crossing scene after four ordinary passes, predicting 64 passes:

| Depth | Compression error / actual displacement | Geometry error / actual displacement | Total prediction error / actual displacement |
|---:|---:|---:|---:|
| 4 | 3.289 | 3.288 | 0.866 |
| 8 | 2.410 | 3.288 | 1.449 |
| 12 | 1.260 | 3.288 | 2.400 |

The two error vectors, not their scalar norms, add to the total error.
This rejects the simple response "just enlarge the frozen Krylov basis."
It does not say a larger basis is always worse; later anchors can benefit.
Near-stationary cases must be read through their retained absolute errors
and displacement, not a ratio with a vanishing denominator.

## Mutation 1: keep the nonlinear primitive inside the reduced recurrence

For the whitened Meyer quotient, the exact identity is

    F(z+d) = F(z) + J_z d + L q_z(d),

where q consists of the two disk projection remainders
P(t+delta)-P(t)-DP(t)delta, whitened by the existing metric. L is the fixed
screened spatial response of the original operator. It is not learned by
Sinkhorn. `factor_transport.py` derives its adjoint and tests it against the
actual original map, including boundary crossings.

For an acquired orthonormal basis Q, precompute R=L^T Q. Then

    c_next = Q^T(F(z)-z) + (Q^T J_z Q)c + R^T q_z(Qc)

evaluates exactly the projection Q^T(F(z+Qc)-z), without replaying screened
spatial solves at each reduced step. The disk primitives are reevaluated at
every predicted state. This is exact Galerkin evaluation on the acquired
subspace; it is not exact closure of that subspace under the full dynamics.
The nonlinear evaluation still scales with the full number of pixels.

The first variant keeps the original four-dimensional basis and tests horizons
16 and 64. The enrichment variant evaluates one actual off-subspace defect at
reduced horizon eight, appends that direction and its tangent response, then
rebuilds the reduced recurrence. It retains existing adjoint responses and
pays for at most two new tangent and adjoint actions. This is a coupled block
of response directions, not a fit to separate scalar coordinates.

At 30 anchors across five 64x64 scenes, prefixes 4/32/128, horizons 16/64:

- Keeping nonlinear geometry within the original basis improves prediction
  over frozen linear transport at 12/30 anchors.
- Defect enrichment improves prediction over that nonlinear model at 30/30.
- The enriched model improves over frozen linear transport at 26/30.

The orthogonal response matters. Keeping the primitive exact inside an
inadequate state subspace does not by itself keep the future in that subspace.

### Complete Meyer solve costs reject these implementations

Three repeats, shuffled method order after warmup, original gap-ratio target
1e-4, budget 4096 map/tangent/adjoint units, same gap-check cadence. All
acquisition, adjoints, primitive evaluations, probes, settling, and gap checks
are timed. Shared initialization is excluded. All 75 runs reached the target.
Median times in milliseconds:

| Scene | Ordinary | Frozen linear 4/16 | Nonlinear 4/16 | Nonlinear 4/64 | Enriched /64 |
|---|---:|---:|---:|---:|---:|
| Camera | 231.39 | 418.30 | 881.64 | 1869.67 | 2219.87 |
| Permuted camera | 33.36 | 77.83 | 147.38 | 344.75 | 352.79 |
| Barbara | 100.29 | 175.79 | 372.82 | 825.02 | 906.39 |
| Carrier | 25.03 | 47.31 | 99.84 | 217.24 | 231.81 |
| Crossing | 176.70 | 271.86 | 381.90 | 999.44 | 1568.67 |

Enrichment improves model fidelity but does not amortize its cost or establish
better optimization progress. The appropriate next representation target is
the coupled nonlinear response itself and reuse of acquired response directions,
not repeated full-field nonlinear evaluation inside a tiny reduced state.
This suggested next mutation has not yet been implemented or validated.

## Mutation 2: learn finite transport from ordinary increments

`increment_transport.py` uses eight actual ordinary transitions, forms paired
full-state increment matrices, and fits a reduced recurrence of rank at most
four. Its finite geometric sum retains a unit-drift mode without a fixed-point
inverse. Tests verify exact constant drift and a damped coupled rotation.
No Jacobian or scalar rational coordinate is provided to this mutation.

It screens horizons 64/32/16 with live defects at the current, middle, and
penultimate predicted states, using the earlier relative tolerance 0.05. An
accepted jump is followed by an ordinary settling pass. Rejected probes do
not replace the retained ordinary state. All fitting and failed work is charged.
These sparse probes are an empirical screen, not a global certificate.

For Meyer, an optional singular-value cap enforces a contractive reduced
operator in the proven nonexpansive quotient metric. Global nonexpansivity
of a nonlinear map does not imply that one constant contractive matrix fits
all observed increments. We therefore also test the raw uncapped recurrence.
Removing the cap does not resolve the observed failure.

Three-repeat median results, with ordinary and frozen-linear baselines rerun
in the same study:

| Problem | Ordinary time | Increment transport time | Speedup | Accepted jumps per solve |
|---|---:|---:|---:|---:|
| Meyer camera | 231.32 ms | 573.34 ms | 0.40x | 0 |
| Meyer permuted camera | 33.49 ms | 79.71 ms | 0.42x | 0 |
| Meyer Barbara | 100.40 ms | 228.78 ms | 0.44x | 0 |
| Meyer carrier | 24.99 ms | 44.09 ms | 0.57x | 1 |
| Meyer crossing | 175.18 ms | 417.73 ms | 0.42x | 0 |
| Sinkhorn image | 101.82 ms | 48.63 ms | 2.09x | 31 |
| Sinkhorn mixture | 44.06 ms | 19.76 ms | 2.23x | 10 |

The Sinkhorn controls use 512 points, seed 0, epsilon=0.003, target 1e-8.
This is a two-case control, not a repeat of the 24-case Sinkhorn study. Gap or
marginal checks occur every 32 work units here. All 78 runs, including the
uncapped Meyer ablation, reach their targets. The uncapped version still
accepts zero jumps on the four non-carrier Meyer scenes and one on the carrier.
The failed proposals leave ordinary progress intact, but their overhead costs
roughly a factor of two or more in the non-carrier cases.

## Research conclusion and scope

A transferable method needs a complete state contract and a mechanism for
learning coupled nonlinear response directions; success on a positive scaling
map does not establish either for reflected primal/vector-memory dynamics.
The tests distinguish three requirements:

1. Retain the coupled memory and real two-dimensional blocks that can carry
   phase, rather than seeking an independent scalar law for each coordinate.
2. Carry the directional nonlinear forcing and the directions in which it
   leaves the current representation. A scalar curvature bound discards this
   information; exact primitive evaluation confined to the old basis is also
   insufficient.
3. Amortize those directions and compress their evaluation. Improving trajectory
   prediction without reducing full cost is not an accepted acceleration.

The implemented defect-enrichment mutation is useful evidence for the second
requirement. It does not meet the third. No new Meyer accelerator is claimed.
These conclusions apply to the tested representations and tolerances, not to
all possible learned observables or all Meyer regimes.

The methods have established connections to projection-based nonlinear model
reduction and snapshot/operator inference; no novelty is claimed for those
frameworks. See [Peherstorfer and Willcox](https://dspace.mit.edu/entities/publication/6120d5d1-6aa7-4437-a451-4ab5fbdac5d9)
and [Chaturantabut and Sorensen](https://epubs.siam.org/doi/10.1137/090766498).
Here the contribution is the explicit original-Meyer factorization, its tested
adjoint/Galerkin identities, and the matched falsification of candidate mutations.

## Retained evidence

`results/krylov_structure_probe.json`, `krylov_phase_probe.json`,
`krylov_factor_pilot.json`, `krylov_factor_full.json`,
`krylov_increment_pilot.json`, `krylov_increment_full.json`, and
`mutation_summary.json` retain the evidence. Pilot files are not repeated
performance results. `mutation_report.py` regenerates the summary and plot.
The open LaTeX source and original Meyer implementation remain unchanged.
