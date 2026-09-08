# Relational currents inside a Kalman system

This is a mathematical continuation of FOUNDATION_RESTART.md, not an implemented
tracker or a performance claim. The derivations below are our synthesis; the
cited papers do not present this combined construction.

## 1. State and observation

Keep q=(a,vec C), with gamma(t)=L_t q=a+sum_l c_l integral phi_l.
The finite basis covers the observed interval and possible continuation.
A single observation is y_i=B_i q+e_i, B_i=L_(t_i). A joint observation block
uses stacked B and its actual joint noise covariance R. Bias can be a retained
latent variable or part of correlated R, with the prior cross-covariances
retained when necessary. The simple formulas here assume measurement noise
independent of q conditional on the information already assimilated.

The state is the current representation, not necessarily position/velocity/
acceleration. For a fixed basis the static coefficient state has identity
propagation until the representation changes or genuinely new physical
uncertainty is introduced. Conditioning it updates past and future together:
in conventional terminology this includes smoothing and prediction.

Given a proper Gaussian prior q~N(m0,P0), P0 positive definite, independent
Gaussian sensor error gives the exact posterior

    S = B P0 B^T + R
    K = P0 B^T S^-1
    m1 = m0 + K(y-Bm0)
    P1 = P0 - P0 B^T S^-1 B P0.

Equivalently J1=P0^-1+B^T R^-1 B, h1=P0^-1 m0+B^T R^-1 y,
P1=J1^-1 and m1=P1 h1. Singular priors require a support-aware formulation,
not arbitrary inversion or artificial certainty. Without Gaussianity these
mean/covariance formulas give the best affine update under the corresponding
second-moment assumptions, not an exact general posterior.

## 2. The current covariance is the continuation mechanism

For an unobserved query T,

    E[gamma(T)|y] = L_T m0 + L_T P0 B^T S^-1(y-Bm0)
    Cov[gamma(T)|y] = L_T P0 L_T^T
                        - L_T P0 B^T S^-1 B P0 L_T^T.

The decisive object is L_T P0 B^T: covariance between the future query and the
observed path. An innovation changes the future only through this object.
Zero cross-covariance gives no update to that future marginal in this Gaussian
model. A nonzero value is a substantive prior relationship, not information
created by the update. A likelihood-null direction can nevertheless change
marginally through prior correlations with observed directions.

In continuous notation let K_j(s,u)=Cov(j(s),j(u)). Covariance of two finite
increments is the double integral of K_j over their respective intervals.
Thus a kernel on currents defines relations between entire displacements,
including curved motion, with no required derivative-chain state.
The anchor and its correlations also contribute to position covariance.
Any proposed kernel must be positive semidefinite, have a justified source
model, and allow the intended motion changes. Writing an arbitrary kernel
would merely relocate an unjustified continuation assumption.

## 3. Zak coordinates preserve the same Kalman inference

Let T be the identity on the anchor and the normalized finite Zak transform on
current coefficients. Work in independent real coordinates for the valid Zak
subspace (or retain both covariance and pseudo-covariance in a full complex
formulation). T is then an orthogonal coordinate isomorphism:

    z=Tq; mz=Tm; Pz=T P T^T; Bz=B T^T; Kz=T K.

Transforming the posterior back gives exactly the original posterior. Merely
performing Kalman conditioning in Zak coordinates cannot improve its accuracy.
Keeping only diagonal Pz generally changes the model: off-diagonal entries
carry uncertain relations between phase, interval and spatial components.
Signal Gram entries ZZ* and the belief covariance Pz are different objects;
nonlinear functions of the state cannot be appended as independent Gaussian
measurements. Any lifted state must retain its induced dependencies.

The potential benefit is an efficiently representable, justified relational
prior/transition in these coordinates. Physical time-frequency meaning still
requires a compatible basis and lattice, as specified in FOUNDATION_RESTART.

## 4. A rigorous BTB-like refinement with exactly one observation likelihood

Introduce artificial assimilation time beta in [0,1], with physical time fixed:

    p_beta(q) proportional to p0(q) p(y|q)^beta.

In the linear Gaussian case this remains Gaussian and

    P_beta^-1 = P0^-1 + beta B^T R^-1 B
    m_beta = P_beta(P0^-1 m0 + beta B^T R^-1 y)
    dm_beta/dbeta = P_beta B^T R^-1(y-Bm_beta)
    dP_beta/dbeta = -P_beta B^T R^-1 B P_beta.

The last two equations follow by differentiating the first two. At beta=1
this is exactly one ordinary Kalman observation update. Positive finite steps
delta beta summing to one can be done exactly by using effective R/delta beta
in each conditional update. This factors one likelihood into powers, not
independent copies. Adaptive step sizes retain this identity when the total is
one and B,R and the prior model remain fixed.

This gives a mathematically controlled relaxed assimilation path, not a new
claim of denoising improvement. Going beyond beta=1 sharpens a different
posterior through overcounting unless a separately justified target is intended.
Repeating the original R update k times instead gives J=P0^-1+k B^T R^-1 B.
It manufactures certainty from a fixed record.

## 5. Connecting a genuine denoiser to a probabilistic structure

The missing BTB F cannot be an arbitrary mean transformation followed by an
unchanged covariance. One possible rigorous interface is a proximal operator:

    D_tau(v) = argmin_u { V(u) + ||u-v||^2/(2 tau) }.

Here V is an explicitly declared structural negative log prior; this is a
mathematical characterization, not a choice of motion law or an implementation
prescription. With quadratic V(q)=(q-m0)^T P0^-1(q-m0)/2,

    D_tau(v) = m0 + (I+tau P0^-1)^-1(v-m0).

The same operator specifies the prior geometry and its uncertainty. A
fixed point of

    q = D_tau(q - tau B^T R^-1(Bq-y))

satisfies P0^-1(q-m0)+B^T R^-1(Bq-y)=0 and hence equals the Kalman posterior
mean. A BTB-style relaxation of this fixed-point map has the same fixed points;
convergence requires appropriate step/relaxation conditions. Iterating computes
one posterior objective and does not accumulate new measurement information.
Posterior covariance is (P0^-1+B^T R^-1 B)^-1, not the spread of iterates.

For an affine candidate D(v)=m0+L(v-m0), symmetric 0<L<I is sufficient to
recover a Gaussian prior precision tau^-1(L^-1-I). L must be symmetric in the
chosen Euclidean metric, or satisfy the corresponding condition in a declared
metric. Generic contractions, phase selectors or image denoisers need not
have this form. Eigenvalue 1 means zero prior precision along that direction,
which is not a proper full-space Gaussian prior by itself. Eigenvalue zero
corresponds to a hard restriction/limiting precision, requiring support care.

A nonlinear structural D can potentially define a non-Gaussian prior, but its
proximal/probabilistic interpretation and uncertainty require proof. A local
Kalman approximation only represents one local region; it cannot exactly
retain separated admissible branches. A collection of branches is an option,
not an automatic instruction to return to the earlier mode catalog.

## 6. New physical time and unexplained future motion

Changing coordinates/basis must transport the full belief. Exact invertible
re-expression transports covariance by congruence. Projection into a smaller
space can lose information and must not erase omitted-motion uncertainty.

An actual transition q_next=A q+w with independent Gaussian w and known Q
would give m_next=A m and P_next=A P A^T+Q, but A and Q are physical modeling
content. They have not been derived by the Zak transform or BTB relaxation.
Window shifting is not itself such a physical law. Fresh future-supported
coefficients must receive explicit uncertainty and any past/future correlations
must be justified. If structural parameters theta are inferred, uncertainty
in theta must also survive:

    Cov(q|y) = E_theta[Cov(q|y,theta)]
                 + Cov_theta(E[q|y,theta]).

A plug-in covariance omits the second term. Learning a relationship from the
current innovation and then treating it as independently established prior
information risks precisely the noise-reinforcement issue motivating the
Ghost of Newton audit. Use a coherent joint posterior, prequential evidence,
or an independently justified construction; do not count an empirical-Bayes
fit as a separate observation.

## 7. Construction status

The state, exact observation update, Zak covariance transformation, and
single-likelihood refinement path are now specified. They are compatible.
The substantive unsolved part is a physically and statistically justified
current relationship operator/kernel (including new future motion), with a
computational representation that preserves the needed cross-covariance.
This is where a benefit would have to arise. Kalman conditioning supplies the
accounting and a testable interface; it does not supply that source law.

References: FOUNDATION_RESTART.md for the three motivating papers. Classical
Kalman reference: R. E. Kalman, A New Approach to Linear Filtering and Prediction
Problems (1960), https://people.math.harvard.edu/archive/116_fall_03/handouts/Kalman1960.pdf.
