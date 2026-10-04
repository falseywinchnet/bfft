# Matched CONV shrink and grow banks

This study derives mutually constrained analysis and synthesis banks. The
main case is the 2/3 reduction investigated in **Identify Scaling Algorithm**,
ChatGPT conversation `6aa872e9-8eb0-83ea-bb2c-2fc67b84f341`. The earlier
seven-tap example in that conversation was the centered phase of 3/4
reduction. Ratios and sample conventions are kept explicit here.

The 2/3 compact-pair feasibility calculation has no admissible replacement
to promote. Two exact algebraic FIR pairs satisfy the linear reproduction
and reciprocity equations at a larger support, but both fail conservation
and the step test. The dyadic prototype is a separate cell-average transform.
No public implementation or main-paper source was changed by this study.

## The 2/3 analysis operator

Write `X_q=(x_(3q),x_(3q+1),x_(3q+2))` and
`Y_q=(y_(2q),y_(2q+1))`. The Laurent variable is an advance:
`z X_q = X_(q+1)`. The destination centers are `3q` and `3q+3/2`,
and each physical destination basin has width `3/2`.

Direct integration of the paper's raw quintic two-jet gives the centered
seven-tap row, indexed `-3,...,3`,

\[
 h_0=\frac1{163840}[603,-4466,27797,115972,27797,-4466,603],
\]

and the second row, indexed `-2,...,5`,

\[
 h_1=\frac1{4423680}
 [577,11587,-107343,2307019,2307019,-107343,11587,577].
\]

These are the **uncontracted interior** banks. The implemented CONV operator
also performs source-dependent current admission. The exact test reconstructs
the quintic Bernstein polynomials and integrates their clipped pieces; it
does not take the conversation's coefficients on trust.

Grouping source indices modulo three produces

\[
 A_{pq}(z)=\sum_{k\equiv q\pmod3}h_p[k]z^{\lfloor k/3\rfloor}.
\]

For grow rows `g_r[j]`, define

\[
 G_{rp}(z)=\sum_{j\equiv p\pmod2}g_r[j]z^{\lfloor j/2\rfloor}.
\]

The mutual-pair identity is `AG=I_2`. Its opposite cycle `P=GA` is idempotent
and has rank two. It cannot be the identity on three arbitrary fine phases.

## Joint compact design, without fitted coefficients

Allow **both** analysis rows to change. Retain the centered seven-site support
of `h_0`, the reflected eight-site support of `h_1`, and reflection symmetry.
Impose the physical basin moments through degree four:

\[
 \sum_k h_p[k]k^r
 =\frac23\int_{3p/2-3/4}^{3p/2+3/4}t^r\,dt,
 \qquad 0\le r\le4.
\]

There is one free parameter in each shrink row. With `a=h_0[3]`,

\[
 h_0[2]=-6a-53/10240,\quad
 h_0[1]=15a+293/2560,\quad
 h_0[0]=-20a+4001/5120.
\]

With `b=h_1[1]=h_1[2]`, its other reflected pairs are

\[
 h_1[-2]=-b/5+5347/51200,\quad
 h_1[-1]=b-10627/20480,\quad
 h_1[0]=-9b/5+93641/102400.
\]

Grow must recover each degree-four polynomial from its measured basin
averages. At fine phase `r` the exact constraints are

\[
 \sum_j g_r[j]\left(\frac23
 \int_{3j/2-3/4}^{3j/2+3/4}t^p\,dt\right)=r^p,
 \qquad0\le p\le4.
\]

Reflection gives `g_0[-j]=g_0[j]` and `g_2[2-j]=g_1[j]`.
Substitution into `AG=I_2` leaves polynomial equations over the rationals.

For the centered seven-tap grow supports

* `g_0`: `-3,...,3`;
* `g_1`, `g_2`: `-2,...,4`,

the exact Gröbner basis is `{1}`. **There is no solution in this family**,
even before current admissibility or total conservation is imposed. This
result assumes the stated reflection symmetry, physical phase convention,
degree-four reproduction, and contiguous support. It is not a theorem about
all possible nonlinear transforms or all alternate polynomial orders.

Measuring a common grow radius in fine-grid physical coordinates, all included
low-grid sites give tap counts `(7,6,6)` at radius `9/2`, `(7,7,7)` at radius
`5`, and `(7,8,8)` at radius `11/2`. Each has Gröbner basis `{1}`.
At radius `6`, with tap counts `(9,8,8)`, the finite equations have exactly
two solutions. This is an algebraically established first feasible support
for these linear constraints, not a support sweep chosen by image scores.

Writing `t` for the last coefficient of `g_1`, the two roots are

\[
 t_\pm=\frac{158143523\pm91\sqrt{2563337202145}}{50130938880}.
\]

Every other coefficient is an affine rational function of `t`, recorded in
`two-thirds-feasibility.json`. The integrated white-phase projection loss

\[
 \mathcal E=\frac1{2\pi}\int_{-\pi}^{\pi}
 \|I_3-G(e^{i\omega})A(e^{i\omega})\|_F^2\,d\omega
\]

is obtained exactly as the constant Laurent coefficient, reduced modulo the
quadratic equation for `t`. Its values are `1.047569769161314` and
`2.187890532684944`. The first is the minimum over this two-element feasible
linear family. **Neither is a conservative ringing-free CONV pair.**

Physical total conservation imposes the additional identity

\[
 \frac32[1,1]A(1)=[1,1,1],
 \qquad b=298553/552960-5a.
\]

Adding it makes the radius-six system inconsistent as well. The lower-loss
candidate's physical column masses are
`(1.1217192654616666, 0.9391403672691668, 0.9391403672691668)` instead of
`(1,1,1)`. Its unit-step output reaches `-0.1788031143` and `1.1019534137`.
Its raw grow/shrink cycle is accurate to `8.9e-16`; that numerical identity
does not make the candidate admissible.

## What the canonical dual optimizes

For a full-row-rank analysis bank,

\[
 G_*=A^\sharp(AA^\sharp)^{-1},\qquad
 G=G_*+NQ,\qquad AN=0.
\]

Pointwise in frequency, orthogonality gives

\[
 \|I_3-GA\|_F^2=1+\|(G-G_*)A\|_F^2.
\]

Thus loss one is attained exactly by the canonical dual. This objective does
not impose polynomial reproduction, conservation at every phase of an altered
analysis bank, or current admissibility.

For the **original** CONV analysis above, the canonical dual preserves a
constant but sends the samples of the affine field `f(x)=x`, measured by the
coarse basins, to

\[
 3q+r+\epsilon_r,\qquad
 (\epsilon_0,\epsilon_1,\epsilon_2)
 =\left(0,\frac{29973}{655360},-\frac{29973}{655360}\right).
\]

This is derived by exact formal Taylor arithmetic, not numerical differencing.
The canonical dual therefore fails even affine reproduction. Over 1,025
frequencies its raw coarse-cycle defect is at most `6.7e-16` and its loss differs
from one by at most `2.3e-16`, confirming that those two tests alone miss the
reproduction failure.

There is a stronger constraint for changing both seven/eight-tap analysis
rows. Let `n=A_0 cross A_1`. An orthogonal projection reproduces polynomial
data only if `n^sharp X_polynomial=0`. Expanding this identity through degree
four, together with total conservation, again gives Gröbner basis `{1}` in
`a,b`. Consequently no bank in this analysis family can simultaneously have
the absolute white-phase loss one, degree-four reproduction, and physical
total conservation, even with unlimited grow support. A constrained optimum
must therefore have loss greater than one, or use a different analysis family.

## Admission must be part of the pair

Let `C` denote an admission operation on a proposed fine state. If `AG=I`,
applying admission after an independent dual preserves reciprocity only when

\[
 A\bigl(C(Gy)-Gy\bigr)=0.
\]

The ordinary signed-current projection does not impose this nullspace
condition. Retaining its inequalities does not by itself preserve `AG=I`.
A paired construction must compile both the current inequalities and the
analysis identity into the same feasible state. In a fixed linear chamber,
the form `x=Gy+Nd`, `AN=0`, enforces the analysis identity before admission;
the remaining current constraints act on `d`. Their feasibility and local
realization still require proof. No runtime solver or unproved admission
substitution has been installed in CONV.

## Separate dyadic cell-average pair

The exploratory `pair.py` addresses a different geometry: one parent cell is
exactly the union of two children. Its distinct operators are

\[
 (Rx)_i=(x_{2i}+x_{2i+1})/2,\qquad
 (Sy)_{2i}=y_i-\widehat m_i,\quad
 (Sy)_{2i+1}=y_i+\widehat m_i.
\]

Here `mhat` is obtained using the paper's existing conservative moment
admission. The new seven-parent-site proposal is

\[
 m_i=\frac{201(y_{i+1}-y_{i-1})-44(y_{i+2}-y_{i-2})
                  +5(y_{i+3}-y_{i-3})}{1024}.
\]

It is the unique antisymmetric radius-three proposal reproducing polynomial
cell averages through degree six. Constant and even polynomial conditions
follow by symmetry; the three odd moment equations determine its three
coefficients exactly. Seven sites means seven **parent** taps per grow phase,
not a seven-site fine-grid analysis stencil. The current-admission scan has
the same nonlocal lineage dependence as the existing method.

Admission preserves `RS=I` algebraically because the moment cancels in each
parent sum. With detail `d_i=(x_(2i+1)-x_(2i))/2-mhat_i(Rx)`, synthesis is
exact in the other direction as well. Zero-detail grow retains the existing
one-dimensional sign-variation theorem. Polynomial exactness beyond affine
is a proposal property and holds after admission only where it passes through.
The two-dimensional prototype checks the reverse-order coarse-cycle identity;
it is not a proof of the full joint two-dimensional current cone.

A second exact design minimizes uniform-band **unadmitted moment** error,
not the full nonlinear round-trip loss. For coarse-band frequency `omega`,
the ideal half-cell moment response is `i tan(omega/4)`. Among radius-three
antisymmetric proposals reproducing degree four, all coefficients have the
form `(5t+11/64,-4t-3/128,t)`. Orthogonality of the three sine terms gives

\[
 t=\frac{4096-945\pi}{13440\pi}.
\]

The exact objective is strictly convex (the reduced squared coefficient
distance has second derivative `84`). `BAND_OPTIMAL` records this proposal;
its optimum does not extend automatically through nonlinear admission.

## Reproduction

Run the exact symbolic derivation and independent audit:

```sh
.venv-jpeg/bin/python -m experiments.conv_dual_pair.derive_two_thirds
.venv-jpeg/bin/python -m experiments.conv_dual_pair.audit_two_thirds
.venv-jpeg/bin/python -m unittest experiments.conv_dual_pair.test_two_thirds experiments.conv_dual_pair.test_pair experiments.test_conv_conservative_multiresolution -v
```

The receipts include exact coefficient families, feasibility equations,
canonical polynomial defects, pole data, sampled reciprocity, and source
hashes. These focused symbolic computations take seconds locally and use the
existing SymPy environment. No remote software was installed.

Foundational reference: Wim Sweldens, *The Lifting Scheme: A Construction
of Second Generation Wavelets*, SIAM J. Math. Anal. 29(2), 511–546 (1998),
[author manuscript](https://cm-bell-labs.github.io/who/wim/papers/lift2.pdf).
The numerical coefficients and infeasibility results above are derived in
this repository, independently of the conversational source.
