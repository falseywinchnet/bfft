# Continuous-scale local edge-family transport

## Result

The continuous-scale zonotopic state now preserves local transport freedom,
scale ancestry, posterior-shed ancestry, and both the untransported parent and
positive-transport child branches.  The representation succeeds.  The current
residual evidence does not yet contract it enough to select an image.

This is a useful boundary rather than a negative result: we now have an exact,
refinement-stable representation of what remains unknown, and a direct
measurement that the available empirical residual box cannot resolve that
ignorance lawfully.

## Falsified global scale coefficient

The first implementation retained every heat increment but gave each complete
image-wide increment one coefficient.  Its mechanics were exact:

\[
 S(p+G\alpha)=Sp+(SG)\alpha,
\]

and every pushed generator was reconstructed on the evolved Selling graph to
roughly `1e-14`.  Nevertheless a single pixel constraint acted on the same
coefficient everywhere.  Clean coefficient widths collapsed to `0--.03` and
pixelwise truth coverage was zero.  A global scale coefficient is therefore
not a continuous-scale posterior; it is a hidden whole-image decision.

That control remains in `continuous_scale_zonotope_transport_2d.py` because it
is the cleanest executable falsification of the global representation.

## Local scale-edge family

Let the exact inherited residual and posterior-shed heat measures be

\[
 r_{\rm inherited}=c_0+\sum_g c_g,
 \qquad
 r_{\rm shed}=s_0+\sum_g s_g.
\]

Every lineage field is represented on the current Selling graph
`L=D^TWD`:

\[
 c_g=D^Tf_g+m_g,
 \qquad
 s_g=D^Th_g+n_g.
\]

Each active `(lineage, edge)` flux has its own coefficient in `[0,1]`.
Constant graph modes retain separate coefficients.  The resulting generator
matrix has only two nonzeros for each edge variable, even though it contains
tens of thousands of possible local exchanges.

The safe bounded-residual contractor preserves zero transfer.  It narrows
coefficient intervals only when a local flux would leave the outer enclosure
containing both the target-excluded witness alternatives and the current exact
residual.

## Factorized positive push-forward

Materializing every pushed generator would require an image-by-roughly-90,000
dense matrix at refinement one.  The operator instead uses the factorization

\[
 R=SD^T.
\]

`R[:,e]` is computed once per Selling edge.  A lineage-edge generator pushes
forward as

\[
 f_{g,e}R_{:,e}.
\]

The centre uses the signed sum of contracted lineage fluxes per edge; the
radius uses their absolute sum.  On the evolved graph, each response column is
again represented as one antisymmetric flux pattern.  Scale and source labels
remain as multiplicative coordinates and are never marginalized.

## Parent and child are mixture branches

Positive transport is not truth preserving.  On clean woven chirps, the
pushed branch's pixelwise outer truth coverage falls from `1.000` to `.523`.
This is not numerical loss: the positive smoother genuinely transports the
set away from fine reciprocal texture.

Therefore

\[
 \mathfrak M^+
 =\{\text{identity lineage},\ \text{positive-push lineage}\}
\]

is a mixture of feasible branches.  The pushed child never overwrites its
parent.  Later independent evidence may falsify or contract either branch;
until then both remain.

## 20-pixel screen

Values are pixelwise outer truth coverage before push, after push, and in
either branch.  They are coverage audits, not estimator scores.

| scene | condition | parent | pushed child | either branch |
|---|---|---:|---:|---:|
| Cameraman | clean | 1.000 | .920 | 1.000 |
| Cameraman | mixed .25 | .750 | .658 | .765 |
| tapered hair | clean | 1.000 | .975 | 1.000 |
| tapered hair | mixed .25 | .755 | .728 | .785 |
| woven chirps | clean | 1.000 | .523 | 1.000 |
| woven chirps | mixed .25 | .728 | .678 | .750 |

The pushed branch contributes unique mixed-case coverage of `1.5%`, `3.0%`,
and `2.25%` respectively, while contributing no unique clean coverage.  It is
therefore useful hypothesis expansion, not a descent direction.

At one nested trace refinement, 32 lineages become 62 and local edge variables
roughly double:

| scene/condition | refinement 0 variables | refinement 1 variables | parent coverage change |
|---|---:|---:|---:|
| Cameraman clean | 42,163 | 83,243 | 0 |
| Cameraman mixed | 33,355 | 65,718 | +.005 |
| hair clean | 38,724 | 76,744 | 0 |
| hair mixed | 31,994 | 63,048 | 0 |
| woven clean | 45,042 | 89,612 | 0 |
| woven mixed | 31,714 | 62,684 | +.0025 |

This is the first satisfactory refinement behavior in this branch.  The
enclosure grows slightly under refinement, as expected when child increments
gain independent local coefficients, but it does not collapse or change
qualitatively.

## The current evidence is insufficient

Mean coefficient widths remain `.998--1.000` in nearly every case.  The safe
residual box preserves coverage but supplies almost no contraction once local
degrees of freedom are admitted.  That is the honest result.  Reusing the same
observation through another smoother cannot manufacture the evidence needed
to choose among 30,000--90,000 feasible local transports.

The next contractor must act on independently testable joint coordinates:

1. retain value and oriented first-jet intervals together rather than a
   pixelwise residual box;
2. use disjoint ancestry folds so a target edge never certifies its own flux;
3. apply reciprocal phase as a necessary covector relation on each labelled
   scale lineage, not as a probability or a field multiplier;
4. retain curvature as a separate feasible branch;
5. intersect parent and child branches only with evidence that was not used to
   generate their metric or proposal;
6. outer-reduce branches by containing enclosure when their count grows.

Only such evidence can produce a set-theoretic terminal event: every surviving
branch is stable under further independent push-forward/intersection.  There
is still no justification for a midpoint, posterior mean, or GUI output.

## Artifacts

- `continuous_scale_zonotope_transport_2d.py`: falsified global-scale control.
- `continuous_scale_edge_family_transport_2d.py`: local sparse family and
  factorized push-forward.
- `test_continuous_scale_edge_family_transport_2d.py`: local recomposition,
  high-dimensional contraction, enclosure, evolved-flux, and constant-state
  invariants.
- `continuous_scale_edge_family_transport_20.json`: refinement and coverage
  measurements.
