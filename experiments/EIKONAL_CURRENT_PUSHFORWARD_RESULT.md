# Signed-Eikonal Owner-Current Result

This note records a focused geometry result.  It is not paper text and does
not yet define the complete general-image operator.

## 1. What the FIR observation means

An ideal fractional translation is

\[
({\cal S}_\tau f)(x)=f(x-\tau).
\]

Its Fourier multiplier is \(e^{-i\omega\tau}\), which has unit magnitude at
every frequency.  A polyphase FIR does not have to discover a source
frequency.  It approximates this same spatial translation for every
fractional lattice phase.  A finite signed FIR has

\[
H_{\tau,R}(\omega)=e^{-i\omega\tau}+E_{\tau,R}(\omega).
\]

The first term accounts for the smooth diagonal.  The alternating spatial
lobes of the truncation error account for the edge ringing.  Long support by
itself is insufficient; the essential property is coherent translation over
all lattice phases.

## 2. The corresponding geometric coordinate

For an owner \(C_a\), let the nonnegative variation measure supplied by the
compact CONV jet be

\[
d\mu_a(x)=\|df(x)\|^2\,dx.
\]

Its barycenter and spatial covariance are

\[
m_a={1\over\mu_a(C_a)}\int_{C_a}x\,d\mu_a(x),
\qquad
\Sigma_a={1\over\mu_a(C_a)}\int_{C_a}
(x-m_a)(x-m_a)^T\,d\mu_a(x).
\]

The principal eigenvector \(t_a\) of \(\Sigma_a\) is the owner's long axis,
and \(n_a=Jt_a\) is its normal.  This is an owner-scale spatial moment, not a
local gradient vote and not a frequency estimate.  It contains no fitted
gain, threshold, or selected band.

The transverse zero set is

\[
\Gamma_a=m_a+\mathbb R t_a.
\]

For a Riemannian metric \(M\), the signed owner coordinate is

\[
\phi_a(x)=\operatorname{sgn}(n_a^T(x-m_a))
\inf_{\substack{\gamma(0)\in\Gamma_a,\;\gamma(1)=x\\
                 \gamma\subset C_a}}
\int_0^1\sqrt{\dot\gamma^T M(\gamma)\dot\gamma}\,dt.
\]

Thus \(\phi_a=0\) on an entire transverse level set, rather than at a point
germ.  In a constant Euclidean metric and a straight owner,
\(\phi_a(x)=n_a^T(x-m_a)\).

## 3. Straight-owner current transport

Let \(F_j^x\) be the admitted horizontal CONV profile on source row \(j\).
At signed level \(s\), its intersection with the owner phase is

\[
x_j(s)=m_x+{s-n_y(j-m_y)\over n_x}.
\]

Likewise, a vertical profile \(F_i^y\) intersects at

\[
y_i(s)=m_y+{s-n_x(i-m_x)\over n_y}.
\]

A row crossing represents level-set arclength \(1/|n_x|\), and a column
crossing represents arclength \(1/|n_y|\).  The probe therefore constructs

\[
G_a(s)=
{\displaystyle
 \sum_j {|n_x|^{-1}}F_j^x(x_j(s))+
 \sum_i {|n_y|^{-1}}F_i^y(y_i(s))
 \over\displaystyle
 \sum_j {|n_x|^{-1}}+
 \sum_i {|n_y|^{-1}}},
\qquad
\widehat f(x)=G_a(\phi_a(x)),
\]

where only profiles whose level intersection lies in the source domain enter
the normalized sum.  This is the two-axis quadrature of a level-set fiber,
not a wider Cartesian support window.

Three statements are exact for this construction.

1. **Tangent invariance.**  If \(t_a^Tn_a=0\), then
   \(\phi_a(x+\alpha t_a)=\phi_a(x)\), hence
   \(\widehat f(x+\alpha t_a)=\widehat f(x)\).
2. **Axis symmetry.**  Interchanging the Cartesian axes interchanges the two
   sums and their arclength masses; it does not change \(G_a\).
3. **Range preservation.**  Every admitted line-profile value lies in the
   global source range.  All fiber coefficients are nonnegative and sum to
   one.  Consequently
   \(\min f\leq\widehat f\leq\max f\).

If all pulled-back line currents have the same witnessed sign order, their
positive arclength mixture has that order as well.  Integration therefore
cannot introduce an additional alternating variation pair.  This is the
current-domain analogue of coherent translation without a signed value
kernel.

## 4. Focused numerical result

The audit contains 54 analytic straight ridge and edge cycles, with nine
angles and three offsets at \(129\to33\to129\).  Lanczos-3 is a destination
reference only.

| Method | Geometric-mean MSE | Mean tangential ripple | Maximum range escape |
|---|---:|---:|---:|
| Cartesian CONV | 0.0013972802 | 0.0358740916 | 0.0057882667 |
| Signed FIR-3 | 0.0013348803 | 0.0318840457 | 0.0600808077 |
| Exact signed-Eikonal fiber | 0.0009976724 | 0.0067981224 | 0 |
| Coarse owner moment | 0.0010001730 | 0.0068002313 | 0 |
| Analysis-carried owner moment | 0.0009984275 | 0.0067990512 | 0 |

The coarse owner moment has mean angular error \(0.0431^\circ\) and maximum
error \(0.1039^\circ\).  The analysis-carried moment reduces these to
\(0.0162^\circ\) and \(0.0392^\circ\).  The coarse local structure tensor,
by contrast, has mean angular error \(3.72^\circ\) and substantially worse
MSE.  Owner extent, rather than a local normal estimate, is the decisive
geometric observation.

The measured numbers are finite-precision evidence.  Tangent invariance,
axis symmetry, and the range statement above follow algebraically from the
defined operator and do not depend on those measurements.

## 5. Remaining obligation

The scalar fiber \(G_a(\phi_a)\) is complete for a rank-one owner.  A general
image can also vary along the fiber.  Replacing that variation by its mean
would destroy information, so the full operator must transport the exact
current one-form in an owner chart rather than pretending every owner is
one-dimensional:

\[
df=j_\phi\,d\phi+j_\psi\,d\psi.
\]

V3 supplies the owner, metric, cut locus, and curvature subdivision.  The
next construction must define the companion coordinate \(\psi\), transport
both current components with an exact integrability law, and specify traces
at owner boundaries.  The straight-owner result fixes the normal component
and explains the FIR observation; it does not by itself discharge that
two-dimensional obligation.
