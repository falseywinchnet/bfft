# Complete-extension theorem for compact CONV support

## Setting

Let \(\Omega\) be a compact smooth raster domain with boundary and let
\(g\in C^2(\overline\Omega;\operatorname{Sym}^+_2)\) be CONV's Riemannian
metric.  Uniform positive definiteness follows automatically on the compact
domain:

\[
0<mI\preceq g(x)\preceq MI<\infty.
\]

The boundaryless-extension proposition in *On the Spectrum of Differential
Operators Under Riemannian Coverings* constructs a smooth manifold \(N\)
without boundary and an isometric embedding

\[
i:(\Omega,g)\hookrightarrow(N,\widetilde g).
\]

For a rectangular raster, the extension can be chosen on \(\mathbb R^2\):
extend the metric coefficients through a collar, blend two SPD metrics by a
smooth partition of unity, and set \(\widetilde g=I\) outside a compact set.
Convex combinations of SPD forms remain SPD.  Since the metric is Euclidean
outside a compact set and uniformly SPD inside it, \((\mathbb R^2,\widetilde
g)\) is complete and noncompact.

A literal rectangle has corners and is not a smooth-boundary manifold.  Apply
the extension statement by first extending its coefficient field to an open
neighborhood, then placing the rectangle inside a slightly larger smooth
rounded domain.  The subsequent collar is attached to that smooth boundary;
the metric on the original raster is unchanged.

## Theorem: complete-extension support radius

> **Theorem.** Let \((\Omega,g)\) be a compact \(C^2\) Riemannian domain and
> let \((N,\widetilde g)\) be a complete boundaryless Riemannian extension in
> which \(\Omega\) is embedded isometrically.  Define
> \[
> r_\Omega=\min_{x\in\Omega}\operatorname{inj}_{\widetilde g}(x).
> \]
> Then \(r_\Omega>0\).  For every \(x\in\Omega\), every
> \(v\in T_xN\), and every
> \[
> 0\le t<\frac{r_\Omega}{\|v\|_{\widetilde g}},
> \]
> the arc \(s\mapsto\exp_x(sv)\), \(0\le s\le t\), is the unique shortest
> path between its endpoints.  Consequently any CONV transport construction
> whose complete-extension length is strictly smaller than \(r_\Omega\) uses
> one unambiguous minimizing branch.

### Proof

On a complete boundaryless \(C^2\) Riemannian manifold, the injectivity-radius
function is positive at every point and continuous.  Its restriction to the
compact set \(\Omega\) therefore attains a positive minimum \(r_\Omega\).
By the definition of injectivity radius, \(\exp_x\) is a diffeomorphism on the
open tangent ball of radius \(r_\Omega\) for every \(x\in\Omega\).  Its radial
geodesics are the unique minimizing curves there.  The stated transport result
follows immediately. \(\square\)

## Curvature-explicit corollaries

### Positive curvature

If the extension additionally satisfies

\[
0<K_{\widetilde g}\le k_1,
\]

Toponogov's 1970 theorem yields

\[
\boxed{r_\Omega\ge\frac{\pi}{\sqrt{k_1}}.}
\]

Thus every CONV geodesic support arc of length strictly below
\(\pi/\sqrt{k_1}\) is uniquely minimizing.  At the endpoint, Toponogov gives
minimality; strict inequality avoids cut-locus ambiguity.

### Nonpositive curvature

If \(N\) is simply connected and

\[
K_{\widetilde g}\le0,
\]

Cartan-Hadamard gives

\[
\boxed{r_\Omega=\infty.}
\]

All pairs of points are joined by a unique minimizing geodesic.

### General mixed curvature

No curvature-only formula follows.  Nevertheless the theorem still gives

\[
r_\Omega>0.
\]

This is enough for convergence-scale statements: any compact-support family
whose physical support radius tends to zero eventually lies below
\(r_\Omega\).  It is not enough to certify an arbitrary coarse raster without
an additional quantitative injectivity estimate.

## Why this direction fits CONV

The extension changes no sample value, current, interpolation coefficient, or
runtime operation.  It supplies a mathematical ambient space in which the
already-declared compact transport is interpreted.

The proof exits with the fixed inequality

\[
L_{\mathrm{support}}<r_\Omega,
\]

not with an eigenproblem, sparse factorization, or iterative solve.

## Exact limitation

The generic collar construction preserves the metric on \(\Omega\), but does
not automatically preserve a curvature sign in the collar.  Therefore the
Toponogov radius is available only when a positive-curvature extension is
separately established.  The general compact-extension theorem remains valid
without that stronger condition.

For CONV's source-derived metric, curvature may change sign.  The paper should
therefore state the general \(r_\Omega\) theorem first and Toponogov's explicit
\(\pi/\sqrt{k_1}\) bound as a positive-curvature corollary, not as a universal
property of every raster metric.

## Sources

- V. A. Toponogov, *Theorems on Shortest Paths in Noncompact Riemannian
  Spaces of Positive Curvature*, 1970:
  https://www.sovietrxiv.org/items/ru-197001.37368
- *On the Spectrum of Differential Operators Under Riemannian Coverings*,
  Proposition 3.1 and Lemma 3.2:
  https://link.springer.com/article/10.1007/s12220-019-00196-1
- *Stitching Data: Recovering a Manifold's Geometry from Geodesic
  Intersections*, for continuity and compact-path use of injectivity radius:
  https://link.springer.com/article/10.1007/s12220-021-00815-w
