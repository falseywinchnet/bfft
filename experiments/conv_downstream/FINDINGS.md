# Fixed downstream CONV formulas

Measured 2026-09-07 on the M4 Mini CPU. Version 1's six-source, five-current
construction is fixed throughout. No variant was promoted to production.

The downstream family does not produce a useful general accuracy improvement
on this battery. Full derivative-energy ledger costs about 18–19% more complete
pipeline time for a mean per-image MSE improvement of 0.0109%. Diagonal projection
metrics cost 12–18% more for 0.0054–0.0081%. None appreciably changes the measured
single-carrier transition. Equal blending has a modest thin-strip excursion
benefit at approximately baseline cost, with mixed natural-image changes. It is
the only measured tradeoff here that merits keeping as a fixed optional research
comparison; this is not evidence for replacing V1.

## Fixed formulas and their mathematical status

Let the local field be `F(u)=x0+sum_i a_i tau_i(u)`, with
`tau_i=sum_{j=i+1}^5 B_j^5`, so `tau_i'=5 B_i^4`. Its derivative-energy Gram is

`G_ij = integral tau_i' tau_j' = 25 C(4,i) C(4,j) / [9 C(8,i+j)]`.

Every entry is nonnegative, and each row sums to five. Consequently

`e^T (5I-G) e = (1/2) sum_ij G_ij (e_i-e_j)^2 >= 0`.

This gives V1's Euclidean coefficient cost a principled interpretation: up to an
irrelevant positive factor, it is the row-sum lumped upper bound on continuous
derivative error. Replacing it by a more detailed metric is not automatically
an improvement in reconstruction accuracy.

The fixed alternatives are:

* **Ledger diagonal:** use `diag(G)`, normalized by its center entry, on the
  sign-incompatible raw coefficients. The weights are
  `[35/18,10/9,1,10/9,35/18]`. This diagonal retains unequal basis energies but
  omits correlations; it is not itself the row-sum upper bound.
* **Ledger full energy:** evaluate the full quadratic `r^T G r` for the clipped
  sign-incompatible residual in each of the two neighboring cells, summing the
  two energies. The original eligible transition boundaries and causal ordering
  are unchanged. This measures the residual used by the ledger, not the complete
  displacement after the subsequent mass projection. A rank-one cost update and
  a direct near-tie reevaluation implement that same fixed objective.
* **Projection diagonal:** replace the fibre objective by
  `sum_i w_i (c_i-a_i)^2`, with the slope weights above and the same signed
  orthant and exact mass constraint. For a fixed active face,
  `lambda=(sum_active a_i-delta)/(sum_active 1/w_i)` and
  `c_i=a_i-lambda/w_i`; breakpoints are `w_i a_i`. Five sorted breakpoints give
  six finite regions. No iterative solver is added.
* **Both diagonal:** use the two diagonal changes together. This is their
  fixed combination, not an image-dependent choice.
* **Projection value:** the same finite diagonal projection, using the diagonal
  of the centered value Gram `H_ij=integral (tau_i-u)(tau_j-u)`. Admissible
  projection errors have zero sum, making this gauge represent the same value
  error as the original tail basis. Centering makes reversal symmetry explicit.
  Its diagonal is still a surrogate for the full value Gram.
* **Equal blend:** replace V1's source-grid `beta=Gyy/(Gxx+Gyy)` by `1/2`.
  Both original axis orders and Q1 interpolation remain. The fixed arithmetic
  mean is symmetric under transposition, and avoids source geometry computation.
* **Amplitude blend:** use `sqrt(Gyy)/(sqrt(Gxx)+sqrt(Gyy))`, with `1/2` on
  flat data, retaining Q1 interpolation. This weights directional amplitudes
  rather than squared amplitudes.
* **Exact basin integral:** use the degree-six antiderivative
  `P_i(u)=(1/6)sum_{l=i+2}^6 (l-i-1) B_l^6(u)` and endpoint differences to form
  the existing uniform basin mean. This is an algebraic reformulation of V1's
  three-point Gauss integration of a quintic. Floating-point accumulation differs;
  it is not a bitwise certificate. Measured full time is indistinguishable from V1.
* **Hat basin:** integrate against the target-grid piecewise-linear hat and
  divide by its area. This is a positive, mass-lumped L2 transfer to nodal values;
  four-point Gauss integration handles the degree-six integrand exactly in real
  arithmetic. The hats form a partition of unity, and their area-weighted output
  sum preserves the integral. This changes the averaging operator and broadens
  the effective filter. Its image MSE worsens by 3.46–10.36%, mean 6.84%.

A full non-diagonal fibre metric is not represented by the diagonal tests. A
basis change that diagonalizes its objective also transforms the signed orthant
into coupled inequalities. That does not establish an equally cheap finite
projection. Earlier full-metric solver studies remain diagnostics; this work
adds no solve-based construction.

Blending currents is likewise not a free exact rewrite of blending fields:

`gradient[(1-beta)A+beta B] = (1-beta)gradient A + beta gradient B + (B-A)gradient beta`.

Dropping the last term changes the field and need not give a conservative current.
The tests here change only the stated scalar order weights; they do not claim an
unimplemented joint-potential or current-level fusion.

## Measurement

The source snapshot is `../conv_fixed_orders/v1_native.c`; its SHA256 is saved
with the results. Calls use the legacy two complete axis orders and Q1 blending,
without the experimental terminal-fusion entry point. Only generated temporary
C copies contain the downstream changes. Compilation and exact coefficient
construction occur outside the timing loop.

* Six original-resolution images: camera, text, brick, coins, grass, moon.
  Reduction uses floor(original shape / 8). Both common-baseline reduction plus
  variant synthesis, and each variant's complete round trip, are measured.
* 146 one-dimensional carrier cases: 73 frequencies from 0.1 to 1.9 times axial
  Nyquist, at two fixed phases, with 8x synthesis and a boundary crop.
* 60 two-dimensional single-carrier cases: five orientations, six frequencies,
  two phases; 33 to 257 synthesis and a fixed interior crop.
* 30 step/strip cases: five orientations and three offsets, each for a step and
  a strip of width 1.6 source units. Excursion uses the actual reconstructed
  extrema outside the true [0,1] range; no display clipping is used.
* 15 two-carrier mixtures and 40 reduce/synthesize carrier cases independently
  exercise admission and the alternative basin transfer.
* 31 warmed alternating-order repetitions per method for complete camera
  512→64→512, synthesis-only, and reduction-only time, at four and one workers.
  Baseline four-worker medians: 5.730 ms complete, 1.702 ms synthesis, 4.006 ms
  reduction. Separate-stage medians need not sum to the complete median.

Carrier amplitude and generated residual are measured by explicit two-coordinate
moment formulas after removing the mean. This is measurement, not a reconstruction
solver. Frequencies above the source Nyquist include aliasing, and the exact
Nyquist samples depend on phase; these nonlinear carrier measurements are not a
universal linear transfer function.

See `../../output/support_geometry/conv_downstream/SUMMARY.md` for all complete
cost and image numbers, `comparison.png` for the plot, `conv_downstream4/results.json`
for the full census, and `conv_downstream_mixtures.json` for the additional probes.
The mean quality score is the arithmetic mean of six relative MSE improvements.
It is not a confidence interval or a statistical significance claim. Timing ratios
within about a percent are treated as near-baseline, not proven speedups.

## What the data says

The full energy ledger improves mean image MSE by 0.0109%; its individual changes
range from a 0.0045% regression to a 0.0544% improvement. The diagonal projection
variants improve each natural image by tiny amounts, but their runtime increase
is several orders of magnitude larger than their relative image improvement.
Combining diagonal ledger and projection does not improve that tradeoff.

All 1-D gain changes are below 2.3e-8, and the 2-D single-carrier measurements
agree to numerical precision. The raw field is identical by construction. On
this sampled family, these downstream metrics do not sharpen the transition.
Two-carrier mixtures do activate meaningful differences: the full ledger has a
mean MSE of 0.011586962 versus V1's 0.011588444, but its worst individual relative
regression is 0.302%. This does not justify a general improvement claim.

Equal blending reduces worst thin-strip excursion from 0.270093 to 0.253745
(6.05%). Amplitude blending reaches 0.267609 (0.92%). The metric variants leave
that worst excursion unchanged. Equal blending's average image MSE change is a
0.0004% regression, with both signs across images, and its observed complete-time
ratios are 0.982 at four workers and 0.993 at one. Its advantage is narrowly the
strip result and simpler fixed symmetry, not a demonstrated overall quality gain.

The antiderivative basin formula agrees with the original basin mean within
rounding and offers no repeatable material time improvement. Hat averaging has
near-baseline cost but adds unwanted passband attenuation: the complete-wave
mean MSE rises from 0.028790 to 0.035080, and all six natural images regress.
A more structured averaging basis alone is not evidence of better reconstruction.

## Verification and reproduction

Seven tests pass on the M4 Mini: rational Gram identities; native profile mass,
cardinality and constant preservation; 2,000 weighted-fibre KKT cases; primitive
versus original basin equivalence; full-ledger recurrence versus direct quadratic
cost (bit-equal profiles on the tested random lines); hat area-weighted integral
conservation; and transposition symmetry of all blend rules. These are focused
invariants, not universal topology or bit-equivalence proofs for changed operators.

From the authoritative repository, run through `m4build`:

```sh
python3 -m unittest experiments.conv_downstream.test_downstream -v
CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_downstream.run_study --out /tmp/conv_downstream4 --repeats 31
CONV_NATIVE_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_downstream.timing --out /tmp/conv_downstream1.json --repeats 31
CONV_NATIVE_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m experiments.conv_downstream.probe_mixtures --out /tmp/conv_downstream_mixtures.json
```

Copy outputs back immediately using the host returned by `m4host`. This run used
an isolated `/tmp/conv_downstream_workspace` copy of authoritative source files
on that host because the shared mirror was concurrently being changed by other
work. No remote source changes or package installations were made. Render locally:

```sh
.venv-jpeg/bin/python -m experiments.conv_downstream.report
```
