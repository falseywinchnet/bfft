# Mathematical restart: relational transport, refinement, and evidence

Status: a new foundation and a specification of the missing operator. This is
not the theory of the existing particle tracker. That tracker and its scores
are a comparison baseline. No performance result is claimed for this document.

## 1. Observation contract and two independent notions of progression

Data are (t_i,y_i), y_i in R^3. Physical time t parameterizes the target's
motion. Refinement index k parameterizes our changing explanation of a fixed
observation record. A k-step is not another measurement and not elapsed motion.
Noise may include bias and dependence; any probabilistic calculation below
states a narrower sensor assumption explicitly.

The unknown object is a compatible family of oriented displacements over time,
including counterfactual future displacements. For an absolutely continuous
path, write

    gamma(t) = a + integral_[t_min,t] j(s) ds.

Here j is an oriented current, considered over an interval rather than reduced
to its value at the latest time. Equivalently, keep the finite displacements
J(s,t)=integral_[s,t] j, satisfying J(s,t)+J(t,u)=J(s,u). This identity retains
relationships between intervals. It alone supplies no denoising or prediction:
every noisy polygonal path satisfies it as well.

Using j in L2 is an explicit finite-energy regularity class. It includes curved
motion and stops, but it is not all conceivable motion. No bound on its norm,
turn rate or future variation is inferred merely by writing this representation.

## 2. A finite representation without selecting a motion catalog

Choose orthonormal functions phi_l on a finite physical interval containing the
history and forecast interval, and write j_N(s)=sum_l c_l phi_l(s), c_l in R^3.
An increasing dense sequence of such spaces can approximate any L2 current.
This is a representational resolution, not a fixed collection of straight,
circular or helical laws. For coefficient error delta c the path error obeys

    ||delta gamma(t)|| <= sqrt(t-t_min) ||delta j||_L2,

by Cauchy-Schwarz. Thus L2 convergence of currents gives uniform path
convergence on a fixed finite interval when the anchor is fixed.

Let A_il=integral_[t_min,t_i] phi_l(s) ds. The observation operator is exactly

    y_i = a + sum_l A_il c_l + e_i.

Irregular observation times remain in A. Missing measurements omit rows; they
are not filled with synthetic observations. A finite-dimensional trial space
may constrain continuations through its basis: any conclusion dependent on
that restriction must survive increasing resolution or carry truncation error.
In particular, future-supported basis functions expose unobserved future
freedom instead of silently closing it.

## 3. Zak as an exact coordinate map for relationships

For N=e*q coefficients per spatial coordinate, define the normalized finite
Zak map

    Z[delta,r] = (1/sqrt(e)) sum_(m=0)^(e-1)
                  c[r+m*q] exp(-2*pi*i*delta*m/e),
    delta=0,...,e-1; r=0,...,q-1.

Write Z=U C, with C an N-by-3 real coefficient matrix. U is unitary, so
C=U*Z and ||C||_F=||Z||_F. Valid complex Z obey the conjugacy constraints
implied by real C. No information is added. The Zak lattice's cyclic indexing
is a coordinate convention; it does not identify the physical future with the
start of the history. Applying Zak to arbitrary basis coefficients does not
automatically give the physical time-frequency localization of a Gabor basis.
That localization requires a compatible basis/lattice choice and its own proof.

For two resolutions/lattice factorizations on the same coefficient space,

    Z_b = U_b U_a* Z_a.

The representations are coupled descriptions of one current, not independent
observations. Cross-lattice consistency must be retained, but cannot itself
identify noise: transformed noisy coefficients also satisfy it exactly.

A relational diagnostic is the Hermitian matrix

    G = Z Z*,   G_pq = sum_(d=1)^3 Z_pd conjugate(Z_qd).

Off-diagonal entries retain phase and cross-component relationships that
coefficient magnitudes discard. Under a physical rigid rotation C -> C R^T,
G is invariant. G alone identifies shape only up to an orthogonal spatial
transformation; retain oriented C/Z and anchor a to answer boundary queries.
The rank-at-most-three fact is automatic for every N-by-3 matrix, including
noise. It is not a denoising prior.

The Zak/Gabor paper motivates efficient analysis/synthesis and compression.
It does not establish a motion law, a universal phase-coherence prior, or that
compression preserves a particular future boundary event.

## 4. Observation-defined distinguishability

Consider two candidate full paths gamma and eta fixed independently of the
noise used to compare them. Under independent e_i~N(0,sigma_i^2 I_3), define

    D_y(gamma,eta)^2 = sum_i ||gamma(t_i)-eta(t_i)||^2/sigma_i^2.

The subscript y denotes the observation design, not dependence on realized
noise. With general known positive-definite joint covariance R, the expression
is delta^T R^-1 delta for the stacked mean difference delta.

The Gaussian observation laws have KL divergence D_y^2/2. Under equal prior
probabilities the optimal binary decision error is Phi(-D_y/2). To verify:
whiten the observations and project onto the difference of their means; the
log-likelihood ratio has variance D_y^2 and mean +/-D_y^2/2. Thresholding at
zero gives the stated error. This is a two-fixed-hypothesis result, not a
bound for selecting an arbitrarily data-fitted trajectory.

Consequences:

* Higher sigma reduces distinguishability; additional informative independent
  observations add squared distinguishability. Duplicate correlated estimates
  require their joint covariance and need not add independent evidence.
* Repeating a refinement computation does not change D_y.
* If two paths agree at all observed times, D_y=0, even if their futures differ.
  Their observations cannot favor one over the other under this sensor model.
* Zak preserves this quantity when the observation operator and covariance
  are transformed correctly. Treating transformed coefficients as independent
  evidence would change the statistical problem.

In coefficient coordinates q=(a,C), let B be the observation matrix. The
quadratic evidence geometry is M=B*R^-1 B. Its kernel contains all coefficient
changes invisible to the observations. In Zak coordinates M transforms by
unitary conjugation on the current block. The singularity is an identifiability
fact, not a defect to be eliminated by arbitrary numerical regularization.

This is resistance to changes in an explanation of the observations. It does
not determine resistance to future changes in the target's physical motion.
The latter requires additional, declared structural information.

## 5. What BTB would mean here

BTB supplies the relaxed refinement pattern

    z_(k+1) = (1-mu_k) z_k + mu_k F(z_k).

Here z must denote an explanation of the current history and its relationships.
No new observation likelihood is multiplied in simply because k increases.
Pereg's error-contraction premise, if applicable to a clean z*, gives

    ||F(z)-z*|| <= q ||z-z*||, q<1,
    ||z_(k+1)-z*|| <= [1-mu_k(1-q)] ||z_k-z*||.

The contraction premise is the difficult content, not a property obtained by
naming F a denoiser. For variable steps the product of contraction factors must
tend to zero; mu_k bounded below by a positive constant is sufficient. Merely
0<mu_k<1 is insufficient when sum_k mu_k is finite.

A global strict contraction has at most one fixed point: if F(a)=a and F(b)=b,
then ||a-b||<=q||a-b|| forces a=b. It therefore cannot simultaneously preserve
several indistinguishable admissible motions as distinct fixed points.

The appropriate target is branchwise or setwise refinement. For a locally
smooth family M of admissible explanations, an ideal differentiable F would
fix M, act as identity on tangent directions, and contract only designated
transverse corruption directions. Differentiating F(z)=z along M proves the
tangent identity. In suitable local coordinates a desired linearization is

    DF(z) = identity on T_z M,
            Q_z on the chosen normal directions, ||Q_z||<=q<1.

These normal directions are not known merely because a signal is expressed in
Zak coordinates. Moreover, a direction invisible to observations cannot be
labeled corruption solely from the same observations. Genuine alternatives
must remain in the represented family. Setwise contraction alone does not
calibrate probabilities along that family.

## 6. The Ghost of Newton requirement

The MRA paper studies a specific alignment/EM model. Its initialization-phase
persistence and finite-sample deterioration do not transfer automatically as
theorems about this construction. They expose the question every refinement
must answer: did the data identify the recovered relationship, or did the
algorithm reinforce its own alignment with this finite noise realization?

Three different assertions must remain separate:

1. The update becomes small.
2. The iteration approaches its mathematical fixed-point set.
3. The fixed-point set retains the true, observationally supported structure.

Neither 1 nor 2 implies 3. Even F(z)=z0 is maximally contractive and can be
wrong. A coherent phase pattern is not sufficient evidence either: the MRA
noise-only analysis gives a concrete setting in which initialized phase
structure survives.

Evidence for learned relationships must account for data-dependent selection.
Independent held-out measurements, a justified conditional predictive test, or
a uniform complexity-controlled bound can do this. Arbitrary splits of a
correlated sensor stream are not automatically independent. Repeated use of
one observation block can optimize an objective but cannot multiply its
likelihood as though each use were another measurement.

## 7. Counterfactual transport and the unresolved construction

Retain an admissible set K_y of complete currents/explanations, with each
assumption that restricts it declared. Its possible positions at T are

    R_y(T) = {a + integral_[t_min,T] j(s) ds : (a,j) in K_y}.

A probability on this set requires an additional justified probability law.
It does not follow from set membership, refinement iteration counts, or a
minimum-resistance path alone. Future-supported variations show that the set
can be unbounded without restrictions on future motion.

The remaining research object is a concrete F (or a branch-preserving family
of Fs) and its structural set M. It must exploit relationships that are both
useful for continuation and supported by a declared source model plus data.
None of the three papers selects that physical structure for us. Copying image
RFN onto coordinates, thresholding Zak coefficients, or imposing Brownian turn
increments would each be a new modeling choice requiring its own justification.

The next construction should therefore answer, in order:

* Which cross-interval/cross-phase relationship is preserved by admissible
  motion and altered by the specified corruption?
* How does a finite update act on that relationship while respecting realness,
  spatial equivariance and the observation operator?
* Which directions contract, which remain neutral, and which must stay as
  distinct alternatives?
* How is its truth-related improvement distinguished from fitting reused noise?
* What justified continuation law links the recovered past relationship to a
  future interval, with an explicit allowance for new motion?

This document establishes the representation and identifiability geometry,
and narrows the operator specification. It does not claim that the final
refinement operator, its contraction premise, or its future law has been solved.

## References

* Alimi and Tannor, Audio Compression using Periodic Gabor with Biorthogonal
  Exchange: Implementation Using the Zak Transform (2025):
  https://arxiv.org/abs/2503.22703
* Pereg, Back to Basics: Fast Denoising Iterative Algorithm (2024 version):
  https://arxiv.org/abs/2311.06634v2 — Sections 3.1–3.2 and Appendix A.
* Balanov, Huleihel and Bendory, Expectation-Maximization for Low-SNR
  Multi-Reference Alignment (2026 version):
  https://arxiv.org/abs/2505.21435v2

The Zak-current construction, observation metric, and branch-preserving
specification above are our proposed synthesis. They are not attributed to
these papers as an already published combined method.
