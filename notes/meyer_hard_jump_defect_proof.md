# Why the fused hard jump leaves both halos and texture residue

This note diagnoses the current fixed-cost hard jump against the optimized
64-pass Meyer--Bregman result. It does not introduce a new ownership rule.
In particular, it uses no image-content threshold, support classifier, local
window, or authored cartoon/texture label.

The executable certificate is
`experiments/meyer_hard_jump_defect_proof.py`.

## 1. The reference and the compressed construction

Let (v_star) be the texture returned by the fused 64-pass solver. Its two
ROF maps also leave a survivor (w_star), so for a direct two-product
comparison define

\[
u_star := f-v_star = u_{mathrm{model}}+w_star .
\]

This folds the unreported survivor into cartoon and changes neither the fused
texture nor the allocation seen in a two-product display.

For the hard jump, let

\[
R_K := H^K,qquad P_K:=I-R_K,qquad
H(\xi)=\frac{\lambda}{\lambda+2\lambda L(\xi)}
       =\frac{1}{1+2L(\xi)} .
\]

Here (L\ge0) is the symbol of (-\Delta). Let \(\mathcal J\) denote the
existing oriented jump observation, including Hodge integration, and write

\[
s_0=\mathcal J(f),qquad
m_0=P_K(f-s_0),qquad
r_1=f-m_0,qquad
s_1=\mathcal J(r_1),qquad
m=P_K(f-s_1).
\]

The capacity route and disk readout are a final map \(\mathcal C_\mu\), so
the public texture is \(v_H=\mathcal C_\mu(m)\) and the public cartoon is
\(u_H=f-v_H\).

## 2. Exact identity at the second observation

Since (I=P_K+R_K) and (f=u_\star+v_\star),

\[
\begin{aligned}
r_1
 &=f-P_K(f-s_0)\\
 &=R_Kf+P_Ks_0\\
 &=u_\star+R_Kv_\star+P_K(s_0-u_\star).
\end{aligned}
\tag{1}
\]

Thus the second observation does **not** receive the fused cartoon alone. It
receives three terms:

1. the desired two-product cartoon (u_\star);
2. the finite-depth texture remainder (R_Kv_\star);
3. the high-passed first-observation structural error
   (P_K(s_0-u_\star)).

The carrier residue therefore enters before the second jump is measured. It
is not merely a small error left after an otherwise correct observation.

## 3. Exact identity at the final raw texture

The pre-capacity hard texture satisfies

\[
\begin{aligned}
m
 &=P_K(f-s_1)\\
 &=P_K(u_\star+v_\star-s_1)\\
 &=v_\star-R_Kv_\star+P_K(u_\star-s_1).
\end{aligned}
\]

Therefore

\[
\boxed{
m-v_\star
=
\underbrace{P_K(u_\star-s_1)}_{\text{structural halo/mismatch}}
-
\underbrace{R_Kv_\star}_{\text{texture retained in cartoon}}
}
\tag{2}
\]

and, with

\[
q_\mu:=\mathcal C_\mu(m)-m,
\]

the final public error is exactly

\[
\boxed{
v_H-v_\star
=P_K(u_\star-s_1)-R_Kv_\star+q_\mu .
}
\tag{3}
\]

Equation (3) separates the three mechanisms without fitting anything:

- (R_Kv_\star) is the fused texture that the finite resolvent leaves in
  hard-jump cartoon;
- (P_K(u_\star-s_1)) is the structural error emitted as a signed halo;
- (q_\mu) is the additional change made solely to restore the bounded-flux
  capacity contract.

The first two defects are consequently not unrelated. The first is present
in the second observation by (1), where it can perturb (s_1); the resulting
error in (s_1) is then differentiated into the second defect by (2).

## 4. Why the structural term is necessarily a halo

The screened resolvent is a positive, mass-one convolution. Its (K)-fold
kernel (h_K) is also positive and has unit mass. Hence

\[
P_Ke=e-h_K*e
\]

has zero mean. For the one-dimensional step
(e(x)=A\mathbf 1_{x\ge0}),

\[
(P_Ke)(x)=
\begin{cases}
-A\sum_{y\le x}h_K(y), & x<0,\\
 A\sum_{y>x}h_K(y), & x\ge0.
\end{cases}
\tag{4}
\]

It is negative on one side, positive on the other, and localized at the
front: exactly the observed paired halo. In two dimensions the same formula
holds along the local interface normal. Thus a structural mismatch passed
through (P_K) cannot appear as a neutral scalar correction; its zero-mean
form is a bright/dark contour pair.

## 5. Why finite virtual depth necessarily retains carrier

For a Fourier carrier

\[
v_\star(x)=A\cos(\xi\cdot x+\phi),
\]

the retained term is

\[
R_Kv_\star=\rho_K(\xi)v_\star,qquad
\rho_K(\xi)=\left(\frac{1}{1+2L(\xi)}\right)^K.
\tag{5}
\]

For every finite (K) and every finite lattice frequency,
(0<\rho_K(\xi)\le1). A finite scalar resolvent therefore retains a nonzero
carrier component in cartoon even when the jump observation is perfect.

Increasing (K) only trades the two terms in (2). It reduces
(R_Kv_\star), but drives (P_K) toward the identity on every non-DC mode,
so more of any structural-observation error is emitted directly. No choice
of one scalar depth removes both terms for overlapping cartoon and texture
spectra.

## 6. Numerical certificate

At (256^2), (lambda=0.05), (mu=40), and (K=12), the script evaluates
pure edge, pure carrier, symmetric support, checker support, multiscale
crossing, and junction crossing. Across all six scenes, the maximum absolute
residual of identities (1)--(3) is between (8.5\times10^{-14}) and
(1.4\times10^{-13}).

The measured RMS sizes of (R_Kv_\star) are (0.90)--(1.12) intensity
units. The structural term ranges from (0.44) on the pure carrier to (5.28)
on the thin-junction scene. The capacity correction is zero on the pure-edge,
symmetric-support, and checker-support controls; it is independently visible
on carrier crossings. These values are serialized in `proof.json`, and the
corresponding fields are rendered in `defect_atlas.png`.

## 7. Consequence for the next fast method

The defect is not a bad threshold. It is the replacement of a nonlinear
64-pass dual state by a scalar virtual resolvent plus a jump surrogate.
Content-dependent admission rules can hide one term only by introducing a
new classifier, while changing (K) moves error between the two exact terms.

A valid next acceleration must therefore approximate or solve the fused
Meyer KKT/fixed-point state itself—especially its reflected dual fields and
texture-side survivor—at reduced cost. It must not decide ownership by a new
image-content law.
