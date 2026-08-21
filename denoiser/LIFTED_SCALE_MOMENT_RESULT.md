# Fixed-dimensional scale-moment transport

## Compression law

The lineage-edge zonotope is an exact oracle, but it is not a viable runtime
state.  With `N` pixels, `E` Selling edges, and `K` scale generations, it has
roughly `E*K` coefficients; its explicit pushed edge-response has `N*E`
entries.

The compressed state treats scale as a continuous fibre.  Every lineage has a
canonical coordinate `s in [0,1]`, obtained by normalizing the midpoint of its
heat interval in the intrinsic `log(1+t)` chart.  This creates no bands and no
chosen scale.

For a signed ancestry measure `x_g` at scale `s_g`, transport five raw fields:

\[
 m_0=\sum_g x_g,\qquad m_1=\sum_gs_gx_g,
\]

\[
 a_0=\sum_g|x_g|,\qquad
 a_1=\sum_gs_g|x_g|,\qquad
 a_2=\sum_gs_g^2|x_g|.
\]

There are two ancestries—posterior shed and inherited residual—so the scale
state has ten channels regardless of `K`.  Posterior plus the three symmetric
eikonal-metric entries gives fourteen persistent fields; residual remains
`y-p` by exact pointwise conservation.

The raw fields, rather than their ratios, pass through the positive screened
transport. Consequently accumulation commutes with transport. Readouts are

\[
 \bar s=a_1/a_0,\qquad
 \operatorname{Var}(s)=a_2/a_0-\bar s^2,
 \qquad c=|m_0|/a_0.
\]

The transported uncertainty supplied to the eikonal metric is

\[
 U=(a_0^2-m_0^2)+a_0^2\operatorname{Var}(s).
\]

Its first term measures cancellation between competing signed scale actions;
its second measures uncertainty about scale itself.  Both have squared-image
units, require no relative tuning coefficient, and therefore join the residual
second action directly.

## Two-dimensional action fibre

The extra signed moment makes a continuous scale-selective action exact for
every affine response

\[
 a(s)=\theta_0+\theta_1s,
 \qquad
 T_a=\theta_0m_0+\theta_1m_1.
\]

Positivity and contraction over the complete scale fibre reduce to two
endpoint inequalities:

\[
 0\leq\theta_0\leq1,qquad
 0\leq\theta_0+\theta_1\leq1.
\]

Thus the next joint contractor needs only two action coordinates per spatial
location, not one coefficient per scale and edge.  The absolute Hankel matrix

\[
 \begin{bmatrix}a_0&a_1\\a_1&a_2\end{bmatrix}\succeq0
\]

carries the accompanying scale uncertainty and eigenstructure.

## Measured checkpoint

The size-20 matched audit is
`lifted_scale_moment_transport_20.json`. Across clean and mixed cameraman,
tapered-hair, and woven-chirp scenes:

- 32 lineages at refinement zero and 62 at refinement one always become 14
  persistent fields;
- signed zeroth ancestry changes by at most `2.8e-17` under refinement;
- the expanded and lifted full-action normal-contractor decisions agree
  exactly in all six cases;
- the lift is `2.98x` to `3.58x` faster in the current Python prototype;
- its measured core representation is `182x` to `264x` smaller;
- non-signed scale-moment refinement change is `2.6e-5` to `1.6e-4` mean
  absolute image units.

The memory gain is principally the removal of the dense pixel-by-edge response
map.  The lifted push has ten right-hand sides and remains `O(N)` in stored
fields on a bounded-degree Selling graph.

## What is and is not established

Exact:

- pointwise posterior/residual conservation;
- signed zeroth ancestry recomposition;
- affine continuous-scale action from `(m0,m1)`;
- positive transport of the raw scale measure;
- the measured complete-action value/jet normal audit.

Closed rather than exact:

- arbitrary nonlinear responses over scale;
- multimodal scale laws sharing their first two absolute moments;
- correlations between different spatial contractor rows.

This is therefore the appropriate compact state for the next experiment, not
yet a finished denoiser.  The next action should contract the two endpoint
coordinates of the affine scale response using the joint posterior/residual
normal slabs, then transport their Hankel uncertainty with the same eikonal
operator.  Only after that produces a stable point readout should it enter the
GUI or native implementation.
