# Seventeen-body BFFT transport formulation

## The numerical target

The reported Mira value is a certified strict lower bound,

\[
s(17) > 4.468292,
\]

not a construction of that side.  The verified reference chart in this
folder gives the independent upper bound

\[
s(17) \le 4.675530093604551\ldots.
\]

The computational problem is therefore to lower the upper bound.  Pairwise
SAT clearance is useful for a final feasibility audit, but it is not the
energy evolved below.

## Equidistant configuration space

Let \(\mathcal P_L\) be the continuous pose space of one unit square in an
\(L\)-by-\(L\) floor plan.  A complete labelled packing state is

\[
q=(q_1,\ldots,q_{17})\in\mathcal X_L=\mathcal P_L^{17}.
\]

After discretization, associate every \(q\) with the coordinate ket
\(\lvert q\rangle\) in \(\ell^2(\mathcal X_L)\).  For distinct complete
configurations,

\[
\|\lvert q\rangle-\lvert r\rangle\|_2=\sqrt 2.
\]

This is the equidistant lift.  We do **not** place a simplex of trial layouts
inside a 51-coordinate Euclidean pose chart.  The basis of the full
seventeen-particle product space is already orthogonal and equidistant.

## Physical energy, not clearance

For the footprint \(S(q_i)\) of square \(i\), use the physical area energy

\[
V_L(q)=
\sum_{i<j}|S(q_i)\cap S(q_j)|
+\lambda\sum_i|S(q_i)\setminus[0,L]^2|.
\]

It vanishes exactly on feasible packings and varies with the amount of
interpenetrating material.  No maximum separating axis, runner-up owner,
candidate score, or clearance margin appears in the evolution.

## Projective Banach contraction

Let \(G_\varepsilon=\exp(\varepsilon\Delta_{\mathcal P})\) be a strictly
positive one-particle heat kernel.  Its seventeen-body lift factors:

\[
G_\varepsilon^{(17)}=
G_\varepsilon\otimes\cdots\otimes G_\varepsilon.
\]

One symmetric imaginary-time transfer is

\[
K_{\beta,\varepsilon}=
e^{-\beta V_L/2}
G_\varepsilon^{(17)}
e^{-\beta V_L/2}.
\]

Both factors act on the entire configuration amplitude at once.  On a finite
pose lattice with \(\varepsilon>0\), every entry of \(K\) is positive.  The
normalized map

\[
B(\psi)=\frac{K\psi}{\|K\psi\|_1}
\]

is a strict Birkhoff contraction in Hilbert's projective metric

\[
d_H(f,g)=\log\max_q\frac{f(q)}{g(q)}
-\log\min_q\frac{f(q)}{g(q)}.
\]

Thus each fixed \((\beta,\varepsilon)\) stage has one positive Perron ray and
Banach iteration converges to it from every strictly positive start.  Cooling
means converging each stage before increasing \(\beta\) or decreasing
\(\varepsilon\).  In the zero-kinetic limit, amplitude concentrates on the
global minimizers of \(V_L\); it does not descend through one Euclidean basin.

This statement applies to the exact operator.  Any low-rank compression must
report its projective residual and truncation error; otherwise it can silently
destroy the contraction.

## Where BFFT enters

The one-particle heat operator is diagonal in a floor-plan spectral basis, so
all seventeen tensor axes can be advanced with the normalized Bruun/DIP walk.
The pair energy also has a natural low-rank factorization.  If
\(\chi_q(x)\) is a square footprint and \(c_a(q)=\langle b_a,\chi_q\rangle\)
in a BFFT floor-plan basis, then

\[
|S(q)\cap S(r)|
=\langle\chi_q,\chi_r\rangle
=\sum_a c_a(q)\overline{c_a(r)}.
\]

Consequently the seventeen-body potential has a compact pairwise operator
representation.  A generic tensor train was measured and rejected: its
one-dimensional ordering discarded too much Schmidt weight.  The physical
packet interaction graph instead has induced treewidth three near the verified
chart.  Min-sum generalized-distributive-law elimination therefore computes
the exact zero-temperature result over (16^{17}) packet configurations with
tables no larger than (16^4).  This is the appropriate compression of the
interaction operator: eliminate on its sparse topology rather than force it
through a one-dimensional tensor ordering.

## One-way Euclidean measurement

The contraction is completed in \(\mathcal X_L\).  Only then is one joint
configuration measured and mapped to centers and square phases.  Exact SAT is
run afterward as a feasibility audit.

The measured frontier experiment also applies deterministic multiscale
restriction: a measured packet state becomes packet zero of the next finer
chart, and that complete chart is again solved exactly.  This preserves one
achieving preimage and never uses clearance, but it is no longer one global
contraction over the union of all rungs.  Its result must therefore be called
a coarse-to-fine preimage diagnostic, not a proof that the continuous global
minimum was found.

`configuration_transport.py` is the exact binary-alphabet control.  It holds
all \(2^{17}=131072\) joint states, proves that distinct basis configurations
are equidistant, measures projective contraction, and checks its terminal
state against the enumerated global minimum of physical overlap area.  Its
two-pose alphabet is intentionally too small to improve the packing bound.
`exact_packet_transport.py` is the current larger control: it represents
Bruun/DIP pose packets for all seventeen identities and exactly eliminates
their physical pair-factor graph.
