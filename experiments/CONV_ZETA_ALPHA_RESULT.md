# Universal zeta and alpha result

## Subsequent decision recovered from the conversation

September 5, 2026: the numerical sweep below is retained as historical
evidence, but its recommendation to replace the projection metric was
withdrawn in the originating task, **Find Eikonal Laplace evidence (2)**
(`01a0482d-88fa-7512-9ab0-a70d873bf464`). The final follow-up recognized
that the dense profile metric restores the expensive face projection that
CONV* deliberately eliminated. The measured small, nonuniform gains did
not justify that cost. The accepted engineering decision was to retain
the existing proposal and Euclidean fibre projection. No universal
optimality theorem follows from this finite parameter sweep.

The original analysis and its superseded recommendation follow.

## Definition

The moment-preserving six-tap proposal family is

\[
h_k(\zeta)=h_k^{(0)}+\zeta_k(-1,5,-10,10,-5,1),
\]

with reversal and conservation reducing the coefficient vector to

\[
\zeta=(\zeta_0,\zeta_1,-2(\zeta_0+\zeta_1),\zeta_1,\zeta_0).
\]

Every member has exactly the same response on sampled polynomials through
degree four before admission.  The projection family is

\[
W_\alpha=W_0+\alpha W_1,
\qquad \alpha\geq0,
\]

where \(W_0\) is exact integrated squared profile error and \(W_1\) is exact
integrated squared derivative error.

## Search

Every projection was solved by explicit enumeration of all 31 signed faces of
the five-current fibre.  No approximate numerical optimizer was used.

The 1-D population contained 128 sine/chirp probes over two source lattices,
eight phases, and frequencies from \(1/16\) through \(3/4\) Nyquist.  The 2-D
population contained the fixed sixteen diagonal-interface, carrier, curved,
and crossing cases at \(17\to65\).

## Conflicting nonzero optima

The 1-D lexicographic minimax optimum was approximately

\[
(\zeta_0,\zeta_1,\alpha)=(0.028,-0.058,\infty).
\]

Relative to current Euclidean CONV, it reduced worst normalized MSE by
\(6.86\%\), mean normalized MSE by \(5.11\%\), and geometric-mean normalized
MSE by \(5.37\%\).  It primarily improved sinusoids between one-half and
five-eighths Nyquist.  It worsened chirps by \(2.15\%\) to \(4.24\%\).

On the 2-D population, the same coefficients lost all sixteen cases and
increased geometric-mean MSE by \(4.22\%\).

Conversely, a representative 2-D optimum near

\[
(\zeta_0,\zeta_1,\alpha)=(-0.005,-0.030,0.0625)
\]

won all sixteen 2-D cases and reduced geometric-mean MSE by \(2.72\%\), but
worsened all 128 1-D probes; its 1-D geometric-mean error increased by
\(20.26\%\).

Thus nonzero proposal nullspace motion is not universal.  The two signal
geometries demand opposing fifth-difference corrections.

## Universal result

The supported universal proposal is

\[
\boxed{\zeta^\star=(0,0,0,0,0).}
\]

This is not a failure to optimize.  It is the unique tested point that does
not insert a preference for one-dimensional spectral continuation or
two-dimensional factor transport.

For fixed \(\zeta=0\), replacing Euclidean current distance by the exact
profile metric \(W_0\), hence

\[
\boxed{\alpha^\star=0,}
\]

gave:

| Population | Change from Euclidean projection |
|---|---:|
| 1-D mean normalized MSE | \(-0.524\%\) |
| 1-D geometric-mean normalized MSE | \(-0.311\%\) |
| 1-D worst normalized MSE | unchanged to numerical precision |
| 2-D geometric-mean MSE | \(-0.157\%\) |
| 2-D case wins | 11 of 16 |
| 2-D worst individual regression | \(+0.046\%\) |

The improvement is small, but \(W_0\) is independently determined by the
representation: it minimizes the integrated squared difference between the
raw and admitted quintic profiles.  It therefore has a formal justification
that Euclidean distance between Bernstein currents lacks.

## Invariants

Both \(\zeta=0\) and projection through \(W_0\) retain

- exact endpoint-current conservation;
- the identical ordered signed fibre;
- the existing variation theorem;
- constant and affine reproduction;
- reversal symmetry on the zero-sum current-error subspace;
- zero range excursion on exact step probes.

The result supports changing only the projection metric.  It does not support
changing the FIR proposal coefficients.
