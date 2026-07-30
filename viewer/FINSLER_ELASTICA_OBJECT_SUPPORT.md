# Sparse Finsler-elastica object support

This experiment keeps both shipped representations intact:

- the canonical transport cells remain the reconstruction and segmentation
  basis;
- the intrinsic Voronoi partition remains a support measurement.

Only the object-support graph receives new evidence.

## Lifted geometry without an angle volume

The continuous Euler-elastica objective used by Chen, Mirebeau, and Cohen is

\[
  E(\gamma)=\int_\gamma
  \frac{1+\alpha\kappa(s)^2}{\Phi(\gamma(s),\theta(s))}\,ds .
\]

Their orientation lift represents a point by \((x,y,\theta)\). The intrinsic
Voronoi partition already supplies embedded interface arcs and endpoint
tangents, so a dense \(H\times W\times N_\theta\) volume is unnecessary.
Every two-ended intrinsic arc has two directed states. Traversing state \(a\)
costs

\[
  C_{\rm travel}(a)=\frac{\ell_a}{\Phi_a},
\]

and continuing from \(a\) to \(b\) at their shared endpoint costs

\[
  C(a,b)=C_{\rm travel}(b)
  +\alpha\frac{\operatorname{wrap}(\theta_b-\theta_a)^2}
  {\max((\ell_a+\ell_b)/2,1)}.
\]

Immediate reversal along the same arc is forbidden. The resulting directed
line graph is the sparse analogue of the continuous Finsler orientation lift.

The measured speed is

\[
  \Phi_a=\epsilon+(1-\epsilon)
  \left(1-e^{-(w_b b_a+w_c c_a+w_m m_a)}\right),
\]

where \(b_a\) is canonical BFFT boundary confidence, \(c_a\) is the robust
OKLab jump between intrinsic cells, and \(m_a\) is the robust log support-size
jump. Support and colour can corroborate a contour, but the default weighting
keeps the canonical boundary field dominant so that wood grain does not
become an object boundary merely because its support scale changes.

## Threshold-free two-sided closing

Every directed state is a possible source with potential

\[
  U_0(a)=-\tau\log\Phi_a.
\]

One multi-source label-setting march computes the min-plus inf-convolution

\[
  D(a)=\min_r\{U_0(r)+d_F(r,a)\}.
\]

The continuation span is expressed in units of the measured median intrinsic
travel action:

\[
  \tau=\rho\,\operatorname{median}_a C_{\rm travel}(a).
\]

It therefore follows the populated support geometry instead of imposing a
fixed pixel or cell spacing. A weak arc receives completed saliency only when
both directions support it:

\[
  \widehat\Phi_a=
  \sqrt{e^{-D(a)/\tau}e^{-D(\bar a)/\tau}},
\]

where \(\bar a\) is the reversed state. The new evidence is only the positive
lift

\[
  \Delta\Phi_a=\max(\widehat\Phi_a-\Phi_a,0).
\]

The lift is rasterized on the intrinsic interface complex, averaged onto the
literal canonical object interfaces, and used to complete the canonical
boundary witness:

\[
  b'_e=1-(1-b_e)(1-\omega\Delta\Phi_e).
\]

It cannot lower a measured barrier, move a canonical site, or modify the
reconstruction.

## Common-surround association

An intrinsic arc crossing a visible part interface is assigned its dominant
unordered boundary family \(\{A,S\}\). All families propagate simultaneously
in a second lifted geodesic-Voronoi march. If fronts from \(\{A,S\}\) and
\(\{B,S\}\) collide with compatible orientation, set intersection recovers
the common surround \(S\) and proposes the signed relation

\[
  A\sim B,\qquad A\not\sim S,\qquad B\not\sim S.
\]

This is currently exposed as a diagnostic collision field. It is deliberately
not yet allowed to union parent IDs: the black Pikachu ear apex can still lie
inside the same canonical support fragment as the black surround, so accepting
a parent relation before that topology is represented would hide rather than
solve the remaining failure.

## Cost and observed behavior

For the 600 by 400 coffee image, the intrinsic complex has 20,974 arcs,
41,948 directed states, and 92,214 transitions. After JIT warm-up:

- lifted graph construction: about 7 ms;
- two-sided closing: about 70–90 ms;
- one targeted saucer geodesic: about 24 ms.

The coffee result separates the table, saucer, bottom rim, and spoon where the
previous object forest placed most of them in one 184,312-pixel ID. On the
512 by 512 Cameraman control, the object count remains 8 and the old/new hard
partition agreement is 0.995.

The implementation follows the metric construction in
[Chen, Mirebeau, and Cohen, *Global Minimum for a Finsler Elastica Minimal
Path Approach*](https://arxiv.org/abs/1612.00343), but replaces its dense
orientation grid and prescribed-keypoint grouping with the intrinsic embedded
interface complex and simultaneous label-setting marches.

