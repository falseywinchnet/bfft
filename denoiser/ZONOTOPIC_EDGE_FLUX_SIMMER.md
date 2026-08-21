# Zonotopic Selling-edge flux simmer

## Result

The conservative posterior--residual exchange now has a genuinely set-valued
next state.  It is not a denoiser readout and is not in the interface.
Posterior-shed ancestry, reciprocal-phase return, and target-excluded
curvature return remain labelled components.  Each component is a zonotope of
antisymmetric Selling-edge fluxes whose coefficient intervals are contracted
by bounded residual evidence without being converted into probabilities.

The representation and contractor invariants pass.  The first probe also
falsified an important shortcut: a four-direction empirical witness interval
is not a guaranteed residual enclosure.  Treating it as one conclusively
rejected every component, including clean structure.  The safe enclosure must
contain both the target-excluded witness alternatives and the exact current
residual.  With no independent new observation, zero transfer must remain
feasible.

## Edge-flux state

Let `L=B^T W B` be the symmetric Selling Laplacian on an arbitrarily oriented
undirected edge set.  Every zero-mean proposed transfer `d_0` has the
minimum-energy potential solution

\[
 L\phi=d_0,
 \qquad f=WB\phi,
 \qquad d_0=B^Tf.
\]

One constant generator per connected component carries the zero mode.  A
complete proposal is therefore represented exactly as

\[
 d=A\mathbf 1,
\]

where the edge columns of `A` each sum to zero and are divergences of
antisymmetric pair fluxes.  Replacing `1` by unknown coefficients gives

\[
 \mathcal Z=\{A\alpha:0\leq\alpha\leq1\}.
\]

For any `alpha`, the exchange

\[
 p^+=p+A\alpha,
 \qquad r^-=r-A\alpha
\]

preserves `p+r=y` pointwise.  The implementation carries complete coefficient
intervals, not one fitted transport strength.

## Bounded residual contractor

The target-excluded directional witnesses give an empirical centre `mu` and
outer radius `h`.  Their raw interval frequently excludes the exact residual;
it is therefore not a lawful support declaration.  The safe present-state
enclosure is

\[
 \underline r=\min(\mu-h,r),
 \qquad
 \overline r=\max(\mu+h,r).
\]

Generator coefficients are contracted under

\[
 \underline r\leq r-A\alpha\leq\overline r.
\]

Interval Gauss--Seidel applies every pixel inequality to every incident edge
coefficient.  Empty intersection may falsify this outer component.  Nonempty
intersection only preserves coverage; it creates no likelihood and no
component weight.  Because the current residual is included, `alpha=0`
survives every safe first-generation test.

Measured raw exclusion is not a roundoff curiosity: it is 27--48% of pixels
on the three clean scenes and 33--40% under mixed corruption.  The original
four-member box therefore fails the coverage obligation decisively.

## Components

The first generation retains five diagnostic forms:

1. exact posterior-shed lineage;
2. reciprocal-phase residual return;
3. target-excluded posterior-curvature return;
4. the direct sum of shed lineage and reciprocal phase;
5. the direct sum of shed lineage and curvature.

The pure forms show what each source can explain.  The direct sums preserve
source labels internally: their generator matrices are concatenated rather
than their final images being averaged or their authorities being unioned.

## First 32-pixel coverage screen

The table reports the fraction of the oracle truth correction lying inside
the component's *pixelwise outer transfer enclosure*.  Truth is used only for
this retrospective audit.  Coverage of the correlated zonotope itself is no
larger than this value, so these are deliberately optimistic upper bounds.

| scene | condition | shed + phase | shed + curvature | mixture outer union |
|---|---|---:|---:|---:|
| Cameraman | clean | .752 | .604 | .787 |
| Cameraman | mixed .25 | .093 | .193 | .202 |
| tapered hair | clean | .909 | .904 | .926 |
| tapered hair | mixed .25 | .093 | .118 | .145 |
| woven chirps | clean | .297 | .220 | .332 |
| woven chirps | mixed .25 | .121 | .299 | .306 |

The clean/mixed separation is strong for Cameraman and hair.  Woven texture
remains the hard counterexample: its reciprocal oscillation is contracted by
the empirical residual enclosure more strongly than mixed woven corruption.
Curvature explains a different part of mixed woven residual, confirming that
phase and curvature must not be collapsed.

Coefficient widths show the same structure.  For shed-plus-phase the mean
surviving widths are `.962/.809` for clean/mixed Cameraman, `.972/.772` for
hair, and `.528/.713` for woven.  Every component remains feasible and every
full proposal is rejected as unjustified.  That is appropriate: the state has
narrowed possible transport without pretending it knows a unique correction.

Midpoint images are emitted only to catch representation mistakes.  They are
not estimators: choosing the midpoint of ignorance would recreate the false
averaging ontology this work is meant to remove.

## What this changes

The residual cannot be an anonymous image.  It must carry where each portion
came from:

\[
 r=r_{\rm inherited}+r_{\rm shed}+r_{\rm phase}+r_{\rm curvature}+\cdots.
\]

The labels are causal lineages, not named noise classes.  Once the shed field
is marginalized into one scalar residual, later transport cannot know whether
it is returning structure or inventing amplitude.

The present component proposals are still too compressed.  One aggregated
phase-return field does not span enough of clean woven truth, and the bounded
box forgets correlations between pixels.  The next generation should:

1. retain each continuous heat-scale lineage increment as its own generator
   family rather than collapsing the scale measure into one phase field;
2. push each zonotope through the frozen positive Selling resolvent, which is
   a linear map and therefore preserves the generator representation exactly;
3. re-express the pushed generators on the evolved Selling graph while
   retaining their ancestry labels;
4. replace the pixelwise residual box by transported joint value/jet
   enclosures, so coherent texture is not rejected merely for oscillating;
5. branch only when phase and curvature feasible sets are disjoint, and use
   containing outer reduction when the component count grows.

The eventual terminal event is set-theoretic: no surviving component can
contract or exchange additional flux under new transported evidence.  It is
not a chosen cycle count or a scalar fixed point.

The continuous-scale follow-up is recorded in
`CONTINUOUS_SCALE_EDGE_FAMILY_RESULT.md`. It falsifies a global coefficient per
scale, replaces it with local lineage-edge families, performs an exact
factorized positive push-forward, and proves that the identity and pushed
states must remain separate mixture branches.

## Executable artifacts

- `zonotopic_edge_flux_2d.py`: edge decomposition, interval contractor, and
  one-generation mixture construction.
- `test_zonotopic_edge_flux_2d.py`: exact proposal reconstruction,
  antisymmetry, interval retention/falsification, constant state, and
  posterior/residual conservation.
- `probe_zonotopic_edge_flux_2d.py`: clean/mixed truth-coverage audit.
