# Suitability of support-range clipping in CONV

14 September 2026. Strict support-sample range clipping is an additional
application policy in the joint warp, not a consequence of CONV's original
variation-order requirement. It is useful when the required output must remain
inside those sample bounds, but it is unsuitable as an unconditional statement
that all intersample amplitude excursion is erroneous. The policy excludes
known analytic fields and its coefficientwise realization introduces additional
conservatism. Removing the step alone is not a validated replacement.

## Original construction and deployed warp

The paper's “Variation order” section explicitly permits an existing extremum
to move between samples without adding an alternating derivative pair. Its
factorwise theorem bounds derivative-sign variation by the coarse difference
ledger. A one-sign cell is monotone and remains in its endpoint interval;
a cell carrying an admitted sign transition can have bounded amplitude
excursion. The paper separately states amplitude and BIBO bounds.

The warp appendix adds a stronger condition: every cell's potential must lie
inside the min–max interval of its declared two-jet sample support. Its proof is
correct for the prescribed rule. It is a range guarantee, not a fidelity theorem.

The production perspective source implements both stages:

- `web/instruments/conv-perspective/source-kernel.c`, `refine_line`, builds the
  sign ledger and searches admissible transition locations around the source
  knot, so extrema need not remain pinned to sample sites.
- `clipped_cell`, beginning at line 177 in the audited source, takes minima and
  maxima over the clipped 6×6 two-jet source support and clamps every incident
  Bernstein control to them. Shared controls accumulate the intersection of
  incident intervals.
- Build phase 6 invokes that clipping before phases 7–11 perform joint-current
  admission. The later interpolation between a bilinear base and the clipped
  proposal cannot restore an extremum forbidden by the interval.

The public asset fetched from
`https://paymenottowork.com/instruments/conv-perspective/warp-kernel.wasm?v=phase-bank-5`
matched the local build byte for byte. Both SHA-256 hashes were
`b1c90245331dace5a2e3032143f032520668482bd9569cd074698fc56c5d3052`.
The deployed perspective engine therefore contains this restriction. This audit's
numerical ablations execute the Float64 reference; they do not assert numerical
identity with the live Float32 control-storage path.

The separate browser resizer's `conv-kernel.c` retains its signed-current
construction and has output-domain clamping when packing pixels. It does not
contain the perspective source atlas's `clipped_cell` policy. The problem here
is the additional joint warp construction.

## Two mathematical sources of restriction

### The prescribed interval can exclude the correct field

Consider `f(x)=0.9-0.1(x-2.5)^2` sampled at the six integer sites 0,…,5.
The largest sample is 0.875, while the actual maximum is 0.9 at x=2.5.
The coarse differences have one sign transition. Admitting this intersample
maximum requires no extra rise–fall pair, but strict sample-range preservation
forbids it.

More generally, near an off-grid nondegenerate maximum,

\[
 f(x)=M-\frac{\kappa}{2}(x-x_*)^2,
 \qquad x_*=\frac{x_i+x_{i+1}}2,
\]

the nearest samples are `M−κh²/8`. Every reconstruction bounded above by the
sample maximum has pointwise error at least

\[
 \frac{\kappa h^2}{8}
\]

at the true maximum. This is a second-order obstruction on grids retaining
that subcell phase. It does not depend on the quality of the high-order
proposal or on subsequent target integration.

### The coefficient certificate can be stricter than the prescribed interval

The quintic Bernstein representation of `f(t)=4t(1−t)` has controls

\[
 (0,\;0.8,\;1.2,\;1.2,\;0.8,\;0).
\]

Its exact function range is [0,1]. Clamping those controls to [0,1] changes the
value at t=1/2 from 1 to 0.875: a 12.5% peak loss despite the original polynomial
already satisfying the requested range. Exact half-interval subdivision
certifies all subinterval controls within [0,1] and leaves the polynomial intact.
Thus the meaning of the selected interval and the tightness of its certificate
are independent design choices.

Shared control clipping retains source cardinality and C0 continuity. It does
not in general preserve derivative traces, factor-current sign histories, or
cell moments. For a tensor-quintic cell, its integral changes by
`|C|/36` times the sum of the control changes. These properties must be preserved
by the construction if they are part of the required operator.

## Declared ablations

No experimental current concentration or post-sharpening is used. The same raw
collocation proposal is compared under six treatments:

1. **Raw:** proposal before range or joint-current admission.
2. **Clipped:** coefficientwise intersected support ranges, before current admission.
3. **Nominal:** current reference rule, clipping followed by joint-current admission.
4. **Current-only:** unchanged current-cone admission with no sample-range clipping.
5. **Physical box:** coefficient bounds [0,1], followed by the same current admission.
6. **Refined support:** the same cell support intervals certified on exact
   half-cells, admitted simultaneously with the current constraints from the
   feasible degree-elevated bilinear base. An already feasible proposal passes
   unchanged. Otherwise the existing finite grouped-margin rule is used.

The refined-support treatment changes the finite admission allocation as well
as the certificate representation. Its empirical performance is not a theorem
about the best field satisfying refined inequalities. All variants are fixed
finite constructions; no optimization or iterative solver is used.

There are 56 smooth cases: the full 28-case paper family at source sides 17 and
33, including six edge/carrier orientations, their declared offsets/phases,
two radial offsets, and two crossing phases. There are 12 binary sharp cases:
steps and 1.4-source-pixel-wide bars at 0°, 22.5°, and 45°, at both source sizes.
Another 24 sharp cases repeat them over [0.3,0.7] and [0.48,0.52]. Sources are
nodal samples; targets are a fixed 129×129 point grid. Sharp sampled fields may
be underresolved; their analytic discontinuity errors and excursions are both
reported. No assumed source-area averages are substituted.

## Measured results

Geometric means of paired full-domain MSE ratios relative to the nominal rule:

| Treatment | Smooth, side 17 | Smooth, side 33 |
|---|---:|---:|
| Raw proposal | 0.4896 | 0.1475 |
| Clipped, before current admission | 0.6837 | 0.3247 |
| Nominal | 1.0000 | 1.0000 |
| Current-only | 0.7163 | 0.5591 |
| Physical [0,1] box | 0.7162 | 0.5591 |
| Refined support certificate and finite admission | 1.4463 | 1.8071 |

Smaller ratios mean less error. Current-only improves 18/28 cases and worsens
10/28 at side 17; at side 33 it improves 14/28 and ties the remaining 14 to the
stated 1e-6 relative comparison tolerance. The raw proposal improves all 56
smooth cases relative to nominal, but omits the joint certificates and fails
the sharp excursion screen. The current constraints themselves also affect
fidelity; not all baseline error is attributable to range clipping.

The analytic truth leaves at least one cell's selected support interval in
30/56 smooth cases (16 at side 17, 14 at side 33). At the target grid, the sum
of squared distances of truth to its allowed intervals supplies a lower bound
on the MSE of every field satisfying those intervals. These floors are saved
per case. They are independent of the tested admission algorithm.

On binary sharp fields, current-only excursions outside [0,1] reach 0.258723.
The physical [0,1] box suppresses them, with the same MSE as nominal in these
12 cases. On low-contrast sharp fields, that box is inactive: current-only and
physical-box excursions reach 25.8723% of the edge contrast while remaining
inside [0,1]. A valid encoding range alone therefore does not prevent local
halos. The low-contrast paired geometric-mean MSE ratio is 1.00877 for both.

The refined-support variant preserves its range and current certificates but
worsens aggregate smooth fidelity. Its grouped allocation loses useful
cancellation in these cases. The exact polynomial counterexample still proves
that coarse coefficient clipping can reject a field which satisfies the same
range; a less conservative feasible set alone does not make this particular
finite admission procedure choose a better reconstruction.

## Implications for the construction

Strict support-range clipping should be treated as an explicit range policy,
not as CONV's general criterion for faithful, non-ringing reconstruction.
A replacement must retain the permitted intersample extrema and the relevant
current topology, together with bounds justified by the source model.

The original admitted currents provide a natural starting point. On a factor,
`U(t)=y_i+Σ τ_k(t)c_k`, with `0≤τ_k≤1`, gives the finite enclosure

\[
 y_i+\sum_k\min(c_k,0)\le U(t)\le
 y_i+\sum_k\max(c_k,0).
\]

The integrated Bernstein controls give a tighter finite enclosure. Both retain
amplitude supported by the admitted sign-transition structure instead of
requiring its peak to coincide with an observed value. Transporting the actual
factor construction into a joint atlas must preserve the applicable ledger and
amplitude information; a joint gradient cone alone did not supply sufficient
protection in the sharp tests.

If a physical curvature bound `|f''|≤K` is independently justified, a 1-D cell
can enlarge its endpoint interval by `Kh²/8`; tensor bilinear error similarly
admits `(Kxx hx²+Kyy hy²)/8` with bounded pure second derivatives. Finite
sample differences do not by themselves certify those continuous derivative
bounds. Such envelopes require stated source assumptions or an admitted
continuous model, rather than a tuning constant selected from analytic truth.

These findings support replacing the unconditional sample-range policy in a
future verified construction. They do not support turning clipping off in the
live engine as a complete fix. No production source, deployed asset, or paper
text was changed by this audit.

## Reproduction and evidence

```sh
python -m experiments.conv_warp.support_range_audit.study \
  output/support_geometry/conv_support_range_audit
python -m experiments.conv_warp.support_range_audit.study \
  output/support_geometry/conv_support_range_audit/low-contrast --low-contrast
python -m experiments.conv_warp.support_range_audit.report
python -m unittest experiments.conv_warp.support_range_audit.test_range -v
```

Four focused tests cover the feasible-polynomial clipping counterexample, the
legitimate off-grid peak, the h² obstruction, and exact range subdivision.
All 552 stage/variant records retain source samples and validate the stated
current/range constraints to the recorded floating-point tolerances. The
experiment uses the existing local numerical environment; the Mini's available
runtimes had already been checked and lack required packages. Results and
figures are in `output/support_geometry/conv_support_range_audit/`, and a copy
is accessible from the comparison server at `/range-audit/`.
