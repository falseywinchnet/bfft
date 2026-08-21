# 2-D terminal caustic: localized truth, rejected universal endpoint

## Decision

The active research path is 2-D again. The 1-D work is frozen as negative
evidence and no longer determines the interface or theory agenda.

The causal-simplex Hamilton--Jacobi law remains the strongest unfinished 2-D
theoretical object. Its transported branch barycenter is smooth and
representation-aware, but loses terminal variance and sparse interface
concentration. The discrete branch mode preserves more edge energy but is a
jagged section and cannot be the continuum endpoint.

The first resumed experiment asked whether this loss is partly a change-of-
measure error at scalar projection. The answer is **yes for tapered hair, but
no as a universal denoiser**. The form is retained as a theorem probe and is
not promoted to FMMT or the GUI.

## Scalar pushforward equation

Let the transported order-one terminal path law on characteristic branch
space be

\[
d\mu_x(b)\propto e^{S_x(b)}\,dh_x(b),
\]

where `h` is branch Haar measure and `S` is the gauge-fixed causal HJ action.
The scalar image value is the pushforward `z_x(b)`. If `Q_x(u)` is the quantile
map of `(z_x)_* mu_x`, then its scalar density satisfies

\[
p_x(Q_x(u))=\frac{1}{Q'_x(u)}.
\]

The Hopf--Lax parent simplex already supplies the continuous collision order

\[
\alpha_x=1+\frac{1}{(1-t_x)^2+t_x^2}
\]

for a two-parent arrival, with the established root and one-parent limits.
Transporting that collision through scalar amplitude gives the continuous
caustic barycenter

\[
\hat z_x=
\frac{\int_0^1 Q_x(u)\,Q'_x(u)^{1-\alpha_x}\,du}
     {\int_0^1 Q'_x(u)^{1-\alpha_x}\,du}.
\]

This has no intensity bandwidth, edge threshold, corruption label,
temperature, or duration. Branch labels may be permuted without changing the
answer. Population phase is integrated after the per-realization scalar
section, as required by the earlier ordering result.

## Full external gate

The matched gate uses six structures, clean plus nine corruption generators,
four population phases, 16-square images, four projective directions, and 16
branch quantiles. Generator names never enter the solver.

| form | MSE | SSIM | variance | edge retention |
|---|---:|---:|---:|---:|
| HJ simplex branch barycenter | **0.005764** | **0.7432** | 0.7289 | 0.5669 |
| scalar caustic | 0.006055 | 0.7355 | **0.7607** | **0.5707** |
| integrated FMMT | **0.005250** | **0.7864** | **0.8112** | **0.6475** |

Against the branch barycenter, the scalar caustic wins only 18 of 60 MSE
cases, 15 SSIM cases, and 30 edge cases. It therefore fails the universal
promotion gate.

## The tapered-hair localization

Across all ten tapered-hair conditions, however, the same equation changes
MSE from `0.004799` to `0.004394`, variance fidelity from `0.5463` to `0.6338`,
and edge retention from `0.5139` to `0.5612`, with essentially unchanged SSIM.

At 25% random replacement:

| form | MSE | SSIM | variance | edge retention |
|---|---:|---:|---:|---:|
| HJ simplex branch barycenter | 0.005036 | 0.7639 | 0.5078 | 0.4713 |
| scalar caustic | **0.004366** | 0.7663 | 0.5880 | 0.5136 |
| integrated FMMT | 0.005546 | **0.8075** | **0.8704** | **0.6214** |

So the scalar Jacobian identifies real missing hair structure and even beats
FMMT on MSE in this case, but it does not restore enough edge amplitude or
structural similarity. This is a localization result, not a finished repair.

## What was rejected next

Population-phase variance was tested as uncertainty about transport itself:
the branch barycenter and scalar caustic were combined by their inverse phase
variances, using only machine precision as a floor. That uncertainty did not
predict reconstruction error. In particular it reduced authority for the
caustic on the replacement-hair case where the caustic was actually superior.
Numerical representation stability is therefore not epistemic transport
uncertainty and must not be substituted for it.

## Next 2-D state

The failure split is now sharper. Scalar caustic concentration helps a thin,
low-area transported interface and harms several distributed or mixed
structures. A global choice between branch Haar and scalar amplitude measure
would merely create another hidden image class.

The next estimator must keep the projection map itself uncertain inside the
transport state. Concretely, terminal state should remain a joint measure on

\[
(b,z,j,r,a,\mathcal T),
\]

where `b` is characteristic branch, `z` value, `j` transported jet, `r`
residual, `a` causal ancestry, and `T` the local transport map from branch to
amplitude. Collision should contract uncertainty in `T` using independent
causal path agreement before either branch-Haar or scalar-caustic
marginalization. Population-phase variance is only quadrature error and cannot
play that role.

The executable probe is `terminal_caustic_transport_2d.py`; its invariants are
in `test_terminal_caustic_transport_2d.py`; and the complete record is
`terminal_caustic_full_gate16_phase4.json`.
