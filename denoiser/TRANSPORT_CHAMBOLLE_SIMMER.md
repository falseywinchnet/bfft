# Transport-admissible Chambolle: first simmer

## What classical Chambolle actually contains

For an observation \(y\), scikit's recurrence resolves the ROF problem

\[
  \min_u \frac12\|u-y\|_2^2 + \lambda\sum_x |D u(x)|_2
\]

through a bounded dual flux. Its four separable choices are:

| piece | classical choice | rebuilt choice |
|---|---|---|
| observation | every image gradient drives TV | signed noise resistance drives contraction |
| differential geometry | fixed Cartesian \(D,D^*\) | same exact adjoint pair for this first experiment |
| flux body | one isotropic ball of radius \(\lambda\) | local oriented zonotope measured from transport |
| descent/stop | projected fixed point and energy tolerance | FISTA on flux, objective restart, observer phase balance |

The literal scikit recurrence remains in `dcnt.tv_chambolle_reference`; it
matches `skimage.restoration.denoise_tv_chambolle` to floating-point equality.

## Transport observer and resistance posterior

Four parity-shifted coarse lattices are resized by CONV*. The chart containing
target \(x\) as an anchor is removed, leaving three target-excluded witnesses
\(q_a(x)\). Their median and zonotope are

\[
 m=\operatorname{median}_a q_a,\qquad
 Z_x=m+\sum_a[-|q_a-m|,|q_a-m|].
\]

The interval distance \(r_Z=y-\Pi_Zy\) is resistance to resize transport, not
yet noise. The optimized Meyer ROF proximal decomposes it into coherent
resistance \(r_C\) and removable oscillation

\[
 r_N=r_Z-r_C.
\]

This correction was required experimentally: using \(|r_Z|\) directly as a
TV capacity damaged clean texture almost as badly as classical TV.

## The rebuilt convex action

Let \(K_x\) be an oriented rectangular flux body. Its tangent capacity is the
incident-edge envelope of \(|r_N|\). Its normal capacity is narrowed only by
gradient energy persistent across resize charts and only while predictive
action dominates removable action. There is no edge threshold or noise class.

For one frozen observation generation, the new operator solves

\[
  \min_{p(x)\in K_x}\frac12\|r_N+D^*p\|_2^2,
  \qquad u=y+D^*p.
\]

This asks a conservative flux to explain the signed noise posterior. It does
not give unsigned TV permission to erase whatever lowers image variation.
Projection onto each oriented rectangle is exact. The step is the Chambolle
bound \(1/4\) in two dimensions; acceleration and restart live only on the
static flux problem.

After a solve, CONV* observes \(u\) again. Evolution stops at the measured
phase crossing

\[
  \mathbb E[r_{N,next}^2]
  \leq
  \mathbb E[c_{next}\,s_{next}^2],
\]

where \(s\) is target-excluded predictive scale and \(c\) is chart-persistent
structure coherence. The outer cycle ceiling is only a numerical guard.

## First 32-pixel matched gate

The gate contains Cameraman, geometric interfaces, and woven chirps under
clean, Gaussian, uniform, replacement, and mixed corruption (15 cases).

| method | MSE | SSIM | edge retention | variance ratio |
|---|---:|---:|---:|---:|
| observation | 0.016468 | 0.5375 | 0.9106 | 1.3812 |
| classical Chambolle, weight 0.10 | 0.007045 | 0.5591 | 0.4203 | 0.6460 |
| transport Chambolle | 0.007111 | 0.6255 | 0.7299 | 0.9443 |
| DCNT uncertainty | 0.006696 | 0.6311 | 0.7261 | 0.9707 |
| integrated FMMT control | 0.004088 | 0.6941 | 0.5390 | 0.7722 |

The new form is not an MSE winner yet. It is nevertheless a meaningful
Chambolle correction: compared with classical TV it retains far more edge
action and SSIM, moves clean images much less (mean clean MSE 0.000575 versus
0.00318), and removes the external denoising weight. Its remaining weakness is
under-contraction of dense replacement noise. That is now isolated as the
representation of signed residuals by a local divergence field, rather than
an invitation to strengthen generic smoothing.

The 64-pixel repetition preserves the conclusion: transport Chambolle records
MSE 0.006478, SSIM 0.5904, and edge retention 0.7677; classical Chambolle
records 0.006068, 0.5495, and 0.4196. Clean MSE falls to 0.000267 with clean
edge retention 0.915. The added resolution benefits the transport observer,
but the remaining divergence under-contraction is still real.
