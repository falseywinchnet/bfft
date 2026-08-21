# First full-scale cross-predictive 1-D result

## What changed

The rejected affine simmer mixed every chord up to a fixed horizon into one
proposal cloud. Long relations could therefore erase a short oscillation, and
the horizon became a hidden model choice.

The new experiment keeps relation scale in the state. At every sample `i` and
every topological lag `s=1,...,floor(N/2)`, three first-jet characteristics
predict without reading `y_i`:

\[
q^0_{i,s}=\frac{y_{i-s}+y_{i+s}}2,\qquad
q^-_{i,s}=2y_{i-s}-y_{i-2s},\qquad
q^+_{i,s}=2y_{i+s}-y_{i+2s}.
\]

The opposite path and two one-sided paths are not noise-model branches. They
are the complete minimal first-jet path family on a line. Each path pays its
endpoint variation plus its cross-predictive error over the interval it claims
to explain. Reciprocal action is conductance, and scale carries the
dilation-invariant measure `ds/s`:

\[
d\Pi_i(s,k)\;\propto\;
\frac{1}{s\,[A_i(s,k)+D_i(s,k)]}\,ds,
\qquad
z_i^{(0)}=\int q_{i,s}^k\,d\Pi_i(s,k).
\]

There is no maximum physical lag, selected scale, winning horizon, named noise
law, smoothing time, or requested denoising duration.

## Intrinsic residual continuation

One transport pass is deliberately conservative and can leave coherent signal
in `r=y-z`. Repeating it a chosen number of times would merely hide physical
time in a loop count. Instead, let `q` be the relation transport of the current
residual and measure

\[
c=\mathbb E[rq],\qquad
v_c=\frac{\operatorname{Var}(rq)}{N},\qquad
e_+=(c^2-v_c)_+.
\]

The admitted transport mass is

\[
\alpha=
\min\!\left(1,
\frac{c}{\mathbb E[q^2]}\frac{e_+}{c^2}
\right)
\]

when `c>0` and `e_+>0`, and zero otherwise. The first factor is the quadratic
line-search transport; the second removes covariance energy attributable to
finite-sample chance. The cap says one continuation cannot carry more than the
residual mass that exists. Every accepted continuation strictly lowers
`E[(y-z)^2]`. Equilibrium is `e_+=0`, not a small chosen improvement or a fixed
number of rounds.

## Broad M4 result

The recorded battery contains:

- six mixed structures: smooth geometry, jump-plus-carrier, chirp, pulses,
  mixed broad/fine transport stress, and a fully oscillatory composite;
- thirteen corruption conditions spanning Gaussian, uniform, Laplace,
  multiplicative, salt-and-pepper, random replacement, and mixed replacement
  plus uniform perturbation;
- three seeds per condition, for 234 corrupted cases; and
- clean-input measurements for every structure.

Noise names generate controls only. The candidate receives only the resulting
observation. Truth is used only by the external battery.

| method | mean noisy MSE | first-difference MSE | case wins |
|---|---:|---:|---:|
| full-scale cross-predictive transport | **0.002398** | 0.000969 | **96** |
| fixed-horizon affine relation | 0.002684 | 0.001120 | 67 |
| legacy Gaussian + support flow | 0.004898 | **0.000574** | 34 |
| median width 5 | 0.005151 | 0.003373 | 32 |
| Gaussian sigma 2 | 0.005912 | 0.000781 | 5 |
| unchanged observation | 0.027491 | 0.051470 | 0 |

On the six clean structures, the candidate has:

- mean MSE `2.24e-5`;
- first-difference MSE `4.86e-5`;
- second-difference MSE `1.44e-4`;
- total-variation retention `0.933`;
- variance retention `0.967`; and
- central-range retention `0.985`.

Every corrupted run reached intrinsic covariance equilibrium. Mean accepted
continuations were `0.483`; the maximum was two; the numerical ceiling was
never reached. The full evidence is `1d_cross_predictive_battery.json`.
The matched mixed-corruption traces are rendered in
`1d_cross_predictive_examples.png`; their generator is
`render_1d_cross_predictive_examples.py`.

This is the first 1-D candidate in this experiment to pass all of the declared
broad gates simultaneously. It is a meaningful step, not an optimality claim.

## W1 readout and the failed continuation lifts

The path action is an L1 action, so its intrinsic scalar barycenter is the
weighted median.  Reading the unchanged first-pass law that way is a real
advance under replacement: on the three-seed gate, MSE at replacement
fractions `0.25` and `0.40` falls from `0.001984` and `0.005314` for the mean
to `0.000622` and `0.001473` for the W1 median.  A maximum branch is unstable
in one dimension.  These matched readouts are
`1d_relation_readout_forms_3seed.json`.

The median cannot simply be reused after every continuation.  Fully W1
residual continuation preserves noisy MSE near the one-pass value and gives
clean aggregate MSE `7.24e-5`, but the clean oscillatory composite remains at
`2.52e-4`.  A mean residual continuation restores that composite to
`1.03e-4`, but mixes incompatible L1 and L2 reductions and restores too much
noisy variation.  `1d_w1_continuation_gate.json` records this incompatibility.

Keeping covariance on each lag/path particle instead of reducing the residual
early resolves the clean side: MSE is `2.12e-5`, total-variation retention is
`0.969`, and variance retention is `0.968`.  It fails the corrupted side,
however: mean noisy MSE is `0.00282`, heavy replacement and mixed errors rise,
and 78 runs reach the numerical ceiling.  Global branchwise covariance rewards
untransported corruption.  The record is `1d_particle_continuation_gate.json`.

Three target-exclusion constructions were then falsified:

1. Independent affine collisions without path variation admit isolated
   zero-action extrapolations and explode.
2. Adding the complete source-path variation makes the action coercive but a
   sparse accidental affine germ can still monopolize reciprocal conductance.
3. Adding the exact Haar-weighted W1 arrival-population potential removes that
   singularity, reducing aggregate MSE from `0.0250` to `0.00161`, but still
   oversmooths clean structure and trails the existing median law.

Finally, deleting every validation site whose predictor reads the target gives
an exactly J-invariant interior action.  It is also rejected: aggregate MSE is
`0.00171` versus `0.00146`, and mixed-corruption derivatives worsen.  Index
deletion reduces effective population but does not transport ancestry.  The
records are `1d_causal_collision_gate.json`,
`1d_causal_path_action_gate.json`, `1d_joint_collision_gate.json`, and
`1d_causal_crossfit_gate.json`.

The accepted public candidate therefore remains the original broad-gate
mean-section/mean-residual control.  W1, particle, collision-population, and
causal-crossfit forms remain named research probes rather than silent changes
to that baseline.

## Positive lineage in the joint information metric

The next experiment retains every lag/path atom and equips it with predicted
value `z`, first jet `j`, exact residual `r=y-z`, and positive mass.  Between
adjacent targets, atoms are parallel transported to their common midpoint.
Their pooled `(z,j,r)` covariance supplies a precision tensor; its determinant
is normalized to one, so it controls anisotropy without introducing a metric
strength.  Positive forward/backward messages use reciprocal Sasaki distance
in that metric.  Branch identity is therefore geometric state rather than a
lag-table index.

Three matched metric forms establish why the full state is necessary:

- value/jet Euclidean transport strongly reduces replacement MSE but deflates
  clean structure;
- an equal Euclidean value/jet/residual direct sum resists deflation but gives
  back most corruption recovery; and
- the determinant-one inverse covariance of the joint arrival law improves
  both error and derivative retention without a mixing coefficient.

The surviving readout is a single W1 median after this positive lineage pass.
It has no continuation count.  On the full 256-sample, six-structure,
thirteen-corruption, three-seed gate (234 corrupted cases):

| method | noisy MSE | first-difference MSE | second-difference MSE | noisy TV ratio |
|---|---:|---:|---:|---:|
| accepted mean/covariance equilibrium | 0.002467 | 0.000940 | 0.002300 | 2.678 |
| local one-pass W1 law | 0.001683 | 0.001092 | 0.002628 | 2.750 |
| **joint information-lineage W1** | **0.001595** | 0.000948 | **0.002050** | **2.471** |

Against the local W1 law, lineage improves MSE by `5.2%`, first-difference
error by `13.2%`, second-difference error by `22.0%`, and excess noisy total
variation by `15.6%`.  It is especially decisive on the former weaknesses:

| condition | accepted equilibrium MSE | local W1 MSE | information-lineage MSE |
|---|---:|---:|---:|
| replacement 0.25 | 0.002337 | 0.000596 | **0.000494** |
| replacement 0.40 | 0.004568 | 0.001517 | **0.001199** |
| mixed 0.25 | 0.003544 | 0.002822 | **0.002736** |
| mixed 0.40 | 0.008565 | 0.005462 | **0.004958** |

Clean MSE is `1.61e-4` versus `1.53e-4` for local W1, while clean TV retention
rises from `0.812` to `0.833` and variance retention stays `0.877`.  Thus the
gain is not a collapse toward the mean.  The older accepted continuation still
has much better clean near-identity (`2.24e-5`, TV `0.933`), so the new path is
not promoted over it yet.

Applying the lifted operator again to its residual restores clean TV to
`0.930` on the focused 256 gate but also restores mixed corruption and is
rejected.  A refinement-invariant two-lineage collision law with mass
`m^2/h` improves clean TV to `0.885`, but over-concentrates heavy mixed cases;
it remains a support diagnostic until distinct root collision identity is
carried explicitly.  The evidence is
`1d_information_lineage_full_256_3seed.json`,
`1d_information_lineage_equilibrium_gate_128.json`, and
`1d_lineage_collision_readout_focused_256.json`.

## What still fails

The result remains provisional for precise reasons:

1. Scale centers exclude their local observation, but the accepted
   conductance action still contains the target's validation residual. Exact
   index exclusion was tested and rejected; positive transported lineage is
   required instead.
2. Heavy replacement remains the principal weakness. The three-path lift
   improved it materially, but the old robust affine cloud still wins many
   individual replacement cases.
3. Mean bias remains visible when replacement values are not centered on the
   latent signal.
4. Noisy total variation is still too high. Value MSE wins more decisively than
   derivative MSE, so the transported jet marginal is not yet explicit enough.
5. Reflection supplies a numerical boundary closure but not a
   continuum-normalized boundary relation measure.
6. The positive covariance obstacle is principled, but the current variance
   estimate assumes enough effective independence among products. The future
   causal forest must provide the correct transported effective population.

## Next lift

The line experiment now identifies the state needed in two dimensions. Each
of the three path families becomes an actual `(z,r,j,mass)` particle. V3 causal
parents carry those particles to common midpoints. Scale conductance is scored
only by observations outside the particle's dependency ancestry. The
horizontal Wasserstein pullback then supplies population and determinant-one
eikonal anisotropy.

The current Python line solver should remain a matched control while that joint
particle transport is built. It is now available in the Dear PyGui 1-D tab,
but it does not replace FMMT or authorize C++ specialization.

The later section-order control also rejects an attractive false analogy with
2-D population phase. Selecting maximum collision branches separately from the
left- and right-causal messages and then averaging them preserves clean total
variation (`0.803` at size 128), but corruption seizes the independent
sections: noisy MSE becomes `0.00838` and total variation `4.40` times truth.
The two line orientations are physical evidence, not alternative numerical
realizations. They must meet in the joint positive action before any branch is
selected. The record is `1d_oriented_collision_section_gate128.json`.

## Joint-section continuation negatives

The left/right messages have now been made to meet before projection in four
parameter-free ways. None replaces the positive-lineage W1 median.

1. A forward/backward max-product law transports density relative to `ds/s`.
   Its mean strongly lowers first- and second-difference error, but it deflates
   noisy variance to `0.564` on the focused gate and loses replacement MSE.
2. A joint W1 value/jet field minimizes exact coordinate medians until
   equilibrium. It lowers noisy MSE and derivative errors, but clean TV falls
   to `0.371`; several clean cases do not resolve before the numerical guard.
3. A convex determinant-one `(z,j,r)` field action removes the artificial
   value/jet coefficient. The exact endpoint residual cancels the observation
   algebraically, duplicating attraction to the latent proposal; clean TV
   falls further to `0.312` and the numerical solve is frequently unresolved.
4. The equal-parent logarithmic pool `sqrt(m_left m_right)` is the literal 1-D
   Hopf--Lax barycenter of the two oriented measures. Its W1 readout reduces
   derivative error but deflates noisy variance to `0.459` and worsens clean
   MSE/TV.

These failures rule out another scalar smoother. The missing 1-D coordinate is
transported residual ancestry between distinct observations. A residual
compared only at its own endpoint satisfies `r=y-z` and therefore cancels `y`
when a candidate latent value is inserted. The next line law must carry the
complete empirical residual distribution across causal identity before any
field projection. The focused records are `1d_hj_joint_focused96.json`,
`1d_joint_w1_value_jet_focused96.json`,
`1d_joint_information_field_focused96.json`, and
`1d_symmetric_parent_focused96.json`.

## Disjoint-shell root/context experiment

The next construction separates causal identity before forming a scalar
estimate.  At every target `x` and every scale `s`, the contextual particle is

\[
z_s(x)=\frac{y(x-s)+y(x+s)}2,
\qquad
j_s(x)=\frac{y(x+s)-y(x-s)}{2s}.
\]

Its conjugate shell at `2s` defines the target-free, affine-exact action

\[
A_s^{ctx}=|z_s-z_{2s}|+s|j_s-j_{2s}|
          +\int |z_s-z_a|\,h_x(da).
\]

Here `h_x(ds)` is the full reflected Haar scale measure restricted to four
distinct, non-target source identities. There is no scale catalogue,
bandwidth, duration, threshold, or corruption case. One-sided affine arrivals
were tested first and rejected: their extrapolation magnifies noise, producing
focused mixed-case TV ratios near four. The nested midpoint/secant law removes
that jaggedness and is exactly affine in the interior, but by itself erases
curvature and deflates clean variance.

The observation was therefore retained as a separate root particle, lifted
through the contextual jet fibre. Equal causal depth is essential. The root
path pays

\[
A_s^{root}=A_s^{ctx}+|y(x)-z_s(x)|,
\]

not merely its terminal discrepancy. The latter unequal-depth comparison gave
the noisy root about `71%` of transported mass and failed immediately. A
unnormalized reciprocal action-density transition was also tested. It is not
a sub-Markov probability kernel: projective normalization occurs only after
the complete arrival update. Under the corrected equal-depth action it gives
slightly better focused MSE and variance than row-stochastic transport, while
the Markov form gives slightly lower derivative error on some mixed cases.
Neither resolves clean deflation, so this normalization choice is not promoted
as physics.

With equal depth and equal root/context source mass, the focused collision
readout nearly matches the previous lineage MSE while restoring variance and
total variation, but the broad gate becomes too jagged. Counting actual
endpoint ancestry gives two contextual parents and one observed root, hence
the non-fitted causal simplex measure

\[
h^{fused}=\frac23 h^{ctx}+\frac13 h^{root}.
\]

On the three-source, four-hard-condition focused gate its collision readout
improves noisy MSE/first-difference/second-difference error from
`0.003503/0.004021/0.01251` to `0.003320/0.003453/0.01101`. On the broad
six-source, thirteen-condition screen it wins condition-mean MSE in six cases:
Gaussian, replacement 25% and 40%, and mixed 10%, 25%, and 40%. The heavy
mixed 40% errors improve from `0.004396/0.003824/0.01085` to
`0.003758/0.002974/0.009256`.

This is not yet the unified law. Aggregate broad MSE worsens from `0.002308`
to `0.002818`, clean TV falls from `0.700` to `0.479`, and salt/pepper 25%
reveals a severe false-support mode. Fixed source multiplicity improves
corruption rejection but cannot by itself express transported independence at
the terminal collision. The next object is therefore a continuous collision
order derived from lineage overlap/effective ancestry, analogous to the 2-D
Hopf--Lax simplex order. It must interpolate by geometry, not by a quality
setting or noise label. The records are
`1d_paired_side_lineage_focused96.json`,
`1d_nested_midpoint_lineage_focused96.json`,
`1d_root_context_equal_depth_broad96.json`, and
`1d_root_context_simplex_broad96.json`. The matched transition control is
`1d_root_context_transition_focused96.json`.

## Effective ancestry and energy-distance root participation

Several stricter root/ancestry constructions now delimit the next equation.
Their failures are structural rather than parameter failures.

1. Hellinger overlap between root and context scale laws is invalid when the
   root has first been duplicated through the contextual jet fibre. Dense
   impulses then manufacture their own ancestry. The focused record is
   `1d_effective_ancestry_geodesic_focused96.json`.
2. Preserving genuinely independent left/right roles fixes that duplication,
   but one-sided affine extrapolation has irreducible noise amplification.
   Neither matched-scale collision nor the full determinant-one
   left-scale-by-right-scale product collision repairs it. The records are
   `1d_independent_side_collision_focused96.json` and
   `1d_independent_side_joint_collision_focused96.json`.
3. A symmetric Richardson particle is target-free and quadratic-exact, but
   its negative outer-shell weights amplify replacement and impulses. Making
   polynomial order a continuous coordinate
   `z_(s,t)=(1-t)z_s+t(4z_s-z_(2s))/3` does move toward lower order under
   corruption, but not far enough. Exact reconstruction order is not the
   missing support variable. The records are
   `1d_symmetric_second_jet_focused96.json` and
   `1d_continuous_curvature_focused96.json`.

The useful invariant comes from scalar energy distance. For the target-free
nested contextual law `mu_x`, define

\[
c_x=\frac{\mathbb E_{Z,Z'\sim\mu_x}|Z-Z'|}
           {2\,\mathbb E_{Z\sim\mu_x}|y(x)-Z|}.
\]

The triangle inequality gives `0 <= c_x <= 1`. The factor two is the exact
energy-distance identity, not a fidelity setting. On the three-source focused
gate, using `c_x` to collide the contextual estimate with the root essentially
ties value MSE (`0.003364` versus `0.003371`) while lowering first- and
second-difference error by about 13% and 16%. It also repairs the dense impulse
failure and moves total variation toward one.

The broad 128-sample, six-source, thirteen-condition, two-seed gate rejects
that direct endpoint. A diffuse corrupted context can have large internal
energy and therefore falsely support the root. Adding the transported
collision concentration

\[
K_x=\int (dm_x/dh_x)^2\,dh_x,\qquad k_x=1-K_x^{-1}
\]

and combining the two independent witnesses by log-odds gives

\[
a_x=\frac{c_xk_x}{c_xk_x+(1-c_x)(1-k_x)}.
\]

This form is diagnostically much better behaved: its mean readout lowers
broad derivative errors and moves noisy TV from `1.256` to `1.031`. Its
collision readout nearly ties MSE (`0.002639` versus `0.002593`). But variance
falls to `0.669` and clean TV to `0.450`, so it still deflates structure.

A continuous Monge section was also tested. The root selects a rank inside the
contextual law rather than contributing amplitude. It cuts dense salt/pepper
25% MSE from `0.004517` to `0.002071`, proving that the target-free measure can
reject impulses, but broad derivative and clean-structure metrics regress.
The law cannot select structure absent from its particle fibre.

The conclusion is narrower and stronger: root membership can be derived from
energy distance and transported concentration without a noise label, but the
context must carry curvature as a *parallel-transported bundle coordinate*.
Local negative-weight reconstruction, raw-root blending, and duplicated-root
lineage are all ruled out. The broad records are
`1d_energy_root_broad128_2seed.json`,
`1d_transport_energy_root_broad128_2seed.json`, and
`1d_energy_root_monge_broad128_2seed.json`.

## Dynamical phase is transported state, not a frequency band

The curvature follow-up first tested the tempting literal lifts. A raw
determinant-one `(value, jet, curvature, residual)` bundle lets a corrupted
four-dimensional cloud define the covariance that judges itself; its focused
collision MSE rises to `0.011287`. Adding the exact W1 curvature population
potential is worse (`0.01828`). Both tests reject local curvature particles,
not curvature itself: second differences formed from negative outer-shell
weights are already corrupted before lineage transport begins.

An exact Hilbert section of the stable positive midpoint law was then solved:

\[
(I+D^*D)u=\mathbb E Z+D^*\mathbb E J.
\]

It has no duration or relative smoothing coefficient and always lowers its
own quadratic energy. Nevertheless, taking `E Z` and `E J` separately loses
their phase relation. Broad noisy MSE changes only from `0.002593` to
`0.002567`, while clean TV falls from `0.712` to `0.364` and noisy variance to
`0.581`.

The joint phase experiment keeps those coordinates coupled. Endpoint
particles are parallel-transported to edge midpoints as

\[
q=(z\mathbin{\pm}\tfrac12j,j),
\]

their covariance supplies a determinant-one precision `P`, and the scalar
section is the unique solution of

\[
\sum_e B_e^TP_eB_eu=\sum_eB_e^TP_e\mathbb E q_e,
\qquad
B_eu=\left(\frac{u_i+u_{i+1}}2,u_{i+1}-u_i\right).
\]

This is local dynamical phase, not Fourier-band phase and not the population
raster phase used for 2-D quadrature. It is materially informative: broad
noisy first-/second-difference errors improve to `0.001417/0.003828`, and
salt/pepper 25% MSE falls from `0.004517` to `0.002047`. Clean phase clouds are
also far more one-dimensional than corrupted ones: mean information
anisotropy is `141.62` clean versus `32.48` noisy, while exact phase-space root
participation falls from `0.533` to `0.276`.

But phase barycentring still shrinks the orbit: clean TV is `0.398`, noisy TV
is `0.574`, and noisy variance is `0.589`. Blending the observed amplitude
back by the phase energy-distance identity overcorrects to TV `1.440` and
worsens noisy MSE to `0.002671`. Thus phase is neither absent nor yet the
readout. The next equation must use phase coherence to alter causal arrival
and collision order *before scalar marginalization*. Amplitude is then read
once from the selected continuous characteristic section. This matches the
V3 lesson: ownership/phase is transported first; smoothing it into an
amplitude confidence destroys the structure it was meant to protect. The
records are `1d_hilbert_value_jet_broad128_2seed.json`,
`1d_phase_sasaki_broad128_2seed.json`,
`1d_phase_authority_broad128_2seed.json`, and
`1d_phase_behavior_broad128_2seed.json`.

The first attempt to put phase into collision order rather than amplitude is
also a useful rejection. For relative forward/backward arrival density `t`,
the effective-history identity `1/(t^2+(1-t)^2)` continuously replaces the
fixed order two. It worsens focused noisy MSE from `0.002308` to `0.003111`.
The defect is exact: balanced marginal arrival is not coupled phase agreement.
Two diffuse or contradictory histories can have equal density and therefore
receive maximum order. The next state must retain the left/right ancestry
coupling itself and measure agreement on paired transported phase paths. The
negative record is `1d_phase_collision_focused96_1seed.json`.
