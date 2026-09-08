# Lifted endpoint action estimator

## Point law

The fixed-dimensional scale lift permits the affine action

\[
 a(s)=u_f(1-s)+u_cs,
 \qquad
 t=u_f(m_0-m_1)+u_cm_1,
\]

where `u_f,u_c in [0,1]` are fine- and coarse-endpoint fields.  The estimate
and residual remain exactly conservative:

\[
 p'=p+t,\qquad r'=r-t,\qquad p'+r'=y.
\]

The rejected first attempt projected these endpoint fields into every joint
normal slab.  It converged slowly toward zero action: the slabs describe the
noise-stability component, not the complete truth set.  That iterative control
is not part of the runtime path.

## Contractor as competing action evidence

For endpoint basis `b`, evaluate its residual and posterior normal gaps:

\[
 d_r=\operatorname{dist}(H_rb,[l_r,u_r]),\qquad
 d_p=\operatorname{dist}(H_pb,[l_p,u_p]).
\]

Removing a basis that the residual normal body expects gives support action
`d_r^2`; adding a basis that the posterior body rejects gives noise action
`d_p^2`.  Squared coordinate covectors pull both actions back to vertices.
The first resolvent carries the four-field positive measure

\[
 A^{(1)}=(E_{r,f},E_{p,f},E_{r,c},E_{p,c})
\]

without collapsing it. Its temporary fine/coarse barycentre makes a
conservative exchange `p_1=p+t_1`, `r_1=r-t_1`. Endpoint uncertainty is not
discarded: the second transport metric receives

\[
 \sigma_a^2=n_f(1-n_f)b_f^2+n_c(1-n_c)b_c^2.
\]

The rebuilt positive resolvent `S_2` carries the first action, while the
provisional posterior/residual produce a second target-excluded action
observation `A^(2)`. They are not counted as independent evidence. Their
equal-counting mean and retained disagreement are useful moments,

\[
 M_A=\tfrac12(S_2A^{(1)}+A^{(2)}),\qquad
 V_A=\tfrac14(S_2A^{(1)}-A^{(2)})^2.
\]

but `M_A` is not the point estimator: its factor one-half cancels at endpoint
division, so it merely pools two correlated raw gap magnitudes. Those
magnitudes are measured against different residual states and are not
exchangeable.

The replacement is a causal action fusion. Write `C=S_2 A^(1)` and `N=A^(2)`.
The phase/scale explanation authority is

\[
 h=\frac{\phi\Phi\bar s}
 {\phi\Phi\bar s+(1-\phi)(1-\Phi)(1-\bar s)}.
\]

The complete four-action directional overlap and component overlap are

\[
 \rho=\frac{\langle C,N\rangle}{\lVert C\rVert\lVert N\rVert},
 \qquad
 \omega_j=\frac{2C_jN_j}{C_j^2+N_j^2}.
\]

The carried measure is causal state. A new component receives authority only
where the current structural explanation is rejected, the complete action
directions agree, and that component is witnessed at both times:

\[
 A_j^*=C_j+(1-h)\rho\omega_jN_j.
\]

All factors are bounded, positive, and parameter-free. In particular, an
orthogonal or one-time action cannot enter merely because its raw gap is
large. Only after this fusion is endpoint support divided:

\[
 n_b=\frac{A^*_{r,b}}{A^*_{r,b}+A^*_{p,b}}.
\]

## Continuous stopping certificate

Fusion authority says whether the new action is admissible; it does not say
whether another endpoint exchange is warranted. That requires quantities with
matching units. The first endpoint mixture has unresolved action variance
`sigma_a^2` in radiance-squared units. The four retained temporal variances
have radiance-fourth units. Define their normalized counting-measure readout

\[
 D_A=\frac14\sum_{j=1}^4 V_{A,j},
 \qquad R_A=(\sigma_a^2)^2.
\]

The continuation coordinate is

\[
 \kappa=\frac{R_A}{R_A+D_A}.
\]

This is neither an iteration ceiling nor a fitted tolerance. When temporal
disagreement dominates the unresolved endpoint uncertainty, `kappa` tends to
zero. When the endpoint remains uncertain but the transported actions agree,
it tends to one. The final fine and coarse endpoints move on their affine
fibres:

\[
 u_b=u_b^{(1)}+\kappa\left(u_b^{(2)}-u_b^{(1)}\right).
\]

Thus every intermediate state remains bounded and the posterior/residual
exchange remains exactly conservative. No pixel makes a discontinuous stop
decision.

If both actions vanish, `n_b=1`: lack of contrary evidence retains the
observation rather than authorizing smoothing.

Normal evidence alone does not distinguish clean structured residual from
accidental noisy agreement.  The final endpoint is the normalized Hadamard
intersection of four bounded transport coordinates:

- `n_b`: joint normal support;
- `phi(x)`: local reciprocal-phase support;
- `Phi`: action-weighted scene phase context;
- `s_bar(x)`: positively transported continuous-scale mean.

Writing `q=n_b*phi*Phi*s_bar`, the endpoint is

\[
 u_b=\frac{q}{q+(1-n_b)(1-\phi)(1-\Phi)(1-\bar s)}.
\]

This is an agreement of support and rejection measures, not an independence
claim or a fitted probability.  It has no threshold, exponent, named noise
law, scale band, or iteration count.

The compact state remains fixed-dimensional: the original 14 lift coordinates
plus four raw endpoint actions and four action-uncertainty coordinates, for 22
per pixel. No number of scale generators or graph edges enters that dimension.
The positive second resolvent preserves each carried action mass to numerical
precision.

## Current result

The size-20 unknown-corruption battery is stored in
`lifted_endpoint_action_transport_20.json`.  It covers clean, Gaussian,
uniform, salt-and-pepper, and mixed replacement-plus-uniform observations for
cameraman, tapered hair, and woven chirps.  Runtime is approximately
`0.38–0.44 s` per image in the current Python/SciPy research form.

Against the noisy observation, the endpoint estimate improves MSE in all 12
corrupted cases.  Against the already-smoothed provisional posterior:

- all salt-and-pepper and mixed cases improve;
- tapered-hair and woven-chirp Gaussian cases improve;
- cameraman Gaussian and all uniform-additive cases regress slightly;
- clean cameraman is essentially unchanged from the provisional posterior,
  while clean hair and woven texture regress modestly.

Before stopping, the causal second action improves all three clean and all six
additive cases at size 20, but no replacement case. The stopping certificate
preserves every clean/additive gain and improves one of six replacement cases,
while substantially reducing the other replacement regressions.

At size 32, the unstopped fusion improves three of three clean, four of six
additive, and zero of six replacement cases. The stopped result improves three
of three clean, five of six additive, and three of six replacement cases. The
equal raw pool had managed only one of six additive and one of six replacement
cases at this size. All twelve corrupted stopped outputs still improve over the
observation, and eight of twelve improve over the initial smoothed posterior.

Coarse endpoint action generally exceeds fine endpoint action, while severe
replacement corruption drives both close to zero.  The desired large-to-small
ordering has therefore emerged from the continuous scale coordinate rather
than a scheduled denoising band.

This point estimator is promising but not ready for the GUI.  Its strongest
result remains replacement/outlier damage. The two-cycle representation
now achieves the intended identity improvement and a substantially more robust
additive-noise gain. The dimensional continuation law is the first viable
stopping certificate: it recovers half of the size-32 replacement cases and
shrinks the remaining losses, but does not yet preserve the first cycle's
outlier advantage universally.
