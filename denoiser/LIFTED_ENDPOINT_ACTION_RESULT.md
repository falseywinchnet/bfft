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
Their raw nonnegative fields are transported through the positive eikonal
resolvent before division:

\[
 n_b=\frac{S E_r}{S E_r+S E_p}.
\]

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

## Current result

The size-20 unknown-corruption battery is stored in
`lifted_endpoint_action_transport_20.json`.  It covers clean, Gaussian,
uniform, salt-and-pepper, and mixed replacement-plus-uniform observations for
cameraman, tapered hair, and woven chirps.  Runtime is approximately
`0.38–0.49 s` per image in the current Python/SciPy research form.

Against the noisy observation, the endpoint estimate improves MSE in all 12
corrupted cases.  Against the already-smoothed provisional posterior:

- all salt-and-pepper and mixed cases improve;
- tapered-hair and woven-chirp Gaussian cases improve;
- cameraman Gaussian and all uniform-additive cases regress slightly;
- clean cameraman is essentially unchanged from the provisional posterior,
  while clean hair and woven texture regress modestly.

Coarse endpoint action generally exceeds fine endpoint action, while severe
replacement corruption drives both close to zero.  The desired large-to-small
ordering has therefore emerged from the continuous scale coordinate rather
than a scheduled denoising band.

This point estimator is promising but not ready for the GUI.  Its strongest
result is replacement/outlier damage; additive noise and clean identity
retention remain the next optimization target.  The next law should retain the
four raw support/rejection actions as a mixture through another conservative
cycle instead of committing immediately to their scalar Hadamard readout.
