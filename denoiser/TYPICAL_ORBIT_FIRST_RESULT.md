# First post-FMMT result: local orbit survival before cleanup

## Status

This is the first executable checkpoint under `TRUTH_UNDER_DISTORTION.md`.
It is not promoted to the GUI. The result is positive only in a narrow sense:
we now have an observation-derived structural veto that preserves more of the
Cameraman tripod when FMMT is used as a cleanup after-pass. It does not yet
recover the complete tripod, and it trades MSE/SSIM for that preservation.

## The first idea failed immediately

`typical_orbit_set.py` first implemented the literal global patch proposal:
target-excluded D4-invariant ring moments retrieve similar patches; their
centres become clean-value hypotheses; the scalar readout is the medoid of the
narrowest strict-majority interval.

That construction is rejected. On clean 128-square Cameraman it changes valid
structure and obtains MSE/SSIM/edge/tripod retention
`0.004530/0.7214/0.4362/0.2627`. Similar invariant moments do not authorize
centre-value transplantation. This is the patch-space version of the FMMT
mistake: an epistemic resemblance was converted into an ontic identity.

The code is retained as `denoise_typical_orbit_set` so this failure remains
reproducible.

## Local orbit survival

The replacement begins with no global retrieval and no smooth provisional
chart. For a target `x`, direction `d` in the four projective D4 axes, and
radius `r`, two disjoint one-sided affine charts predict the target:

\[
L_{d,r}(x)=2y(x-rd)-y(x-2rd),\qquad
R_{d,r}(x)=2y(x+rd)-y(x+2rd).
\]

Their bounded feasible component is

\[
I_{d,r}(x)=
[\min(L_{d,r},R_{d,r}),\max(L_{d,r},R_{d,r})].
\]

A crossing edge makes this interval wide; it does not force an average. The
observation is retained exactly whenever

\[
y(x)\in\bigcup_{d,r}I_{d,r}(x).
\]

Only when every target-excluded chart falsifies it is `y(x)` replaced. The
replacement is one actual affine endpoint with maximum interval coverage,
with a medoid used only to resolve equal coverage. No hypothesis mean occurs.

On clean Cameraman, the local law reaches
MSE/SSIM/edge/tripod `0.000104/0.9958/0.9872/0.9755`. Under the mixed
replacement-plus-uniform case it retains `0.9136` aggregate edge response and
`0.9194` tripod response, but its MSE `0.02599` shows that it has deliberately
left most diffuse corruption in place. It is a structural survivor, not a
complete denoiser.

## FMMT as a subordinate after-pass

Each radius also selects the narrowest feasible directional chart. Let its
projective tangent be

\[
u_r=(\cos 2\theta_r,\sin 2\theta_r)
\]

and weight it by the normalized gap between its smallest and second-smallest
feasible interval widths. The cross-scale resultant is

\[
C(x)=\frac{\lVert\sum_r w_r u_r\rVert}{\sum_r w_r}.
\]

The all-scale checkpoint protects the original value only when every radius
has a nonzero directional margin and

\[
C(x)\geq\cos(\pi/4),
\]

the half-cell angle of the four-direction projective quadrature. Locally
falsified samples receive the affine endpoint. FMMT supplies every remaining
pixel. Thus FMMT cleans after the structural state has spoken; it cannot
define that state.

## 128-square M4 result

The table reports MSE / SSIM / aggregate strong-edge retention / tripod
strong-edge retention.

| corruption | FMMT | FMMT + all-scale orbit veto |
|---|---|---|
| clean | `0.000598 / 0.9337 / 0.8575 / 0.7612` | `0.000629 / 0.9354 / 0.8607 / 0.7651` |
| salt/pepper 25% | `0.004775 / 0.8222 / 0.7959 / 0.7468` | `0.005889 / 0.7396 / 0.8031 / 0.7557` |
| uniform 0.24 | `0.004186 / 0.5713 / 0.5270 / 0.3652` | `0.005505 / 0.4410 / 0.5705 / 0.4188` |
| mixed, replacement 8% | `0.004824 / 0.5380 / 0.4915 / 0.3320` | `0.006374 / 0.4037 / 0.5369 / 0.3837` |
| mixed, replacement 25% | `0.006918 / 0.5285 / 0.3931 / 0.2693` | `0.008833 / 0.3673 / 0.4338 / 0.3090` |

Under mixed 8% replacement, the hybrid gains about 9.2% relative aggregate
edge retention and 15.6% relative tripod retention over FMMT. That gain is
visible but incomplete: the focused plate still shows broken tripod legs.
MSE and SSIM regress. This is a useful tradeoff surface, not a win.

At 128 square, local orbit construction takes about `0.02--0.03 s`; the full
hybrid takes about `0.37 s`, essentially the retained FMMT cost. Native work is
not justified yet.

## The phase transition

A preliminary 96-square control admitted directional evidence when only two
of three radii carried compatible projective phase. On mixed 8% replacement it
raised tripod retention to `0.562`, versus `0.362` for FMMT, but MSE rose to
`0.0109`, versus `0.00509`. Requiring all three radii reduced tripod retention
to `0.396` and MSE to `0.00650`.

This is not a request for a user strength slider. It identifies the missing
state. Pointwise phase agreement is trading noise survival against line
survival because a line is not a set of independent protected pixels.

## Next experiment

The next state must transport a **connected oriented bond measure** through
the local orbit charts:

1. retain the signed jump and projective tangent of every coherent chart;
2. join only charts whose tangent phase and one-sided affine continuation
   agree;
3. carry support along the resulting sparse predecessor graph;
4. let connected bond survival veto FMMT, rather than pointwise phase;
5. reconstruct the bond as a signed BV measure, not by reinserting noisy pixel
   values.

That is the route to restoring the tripod as a tripod. The present method can
recognize fragments of its phase, but it does not yet possess the connected
object that those fragments form.

## Reproduction

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_typical_orbit_set

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_typical_orbit_benchmark \
  --size 128 --seed 719 --out /tmp/local_orbit_allscale_128
```

The complete all-scale record is `typical_orbit_allscale_128.json`. The
two-scale phase control is `typical_orbit_twoscale_96.json`.
