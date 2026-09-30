# An evolving conditional-response state that descends

The user's suggestion is to build a higher-level state system capable of
carrying the changing rule, drawing intuition from functional gradient descent
with adaptive representations and this repository's self-context project.
Neither reference is treated as the proposed algorithm itself.

This experiment supplies an exact lifted system, a descent identity, and an
inexpensive response-memory realization tested on the same six transport
problems. The response-memory realization is limited-memory BFGS, a known
quasi-Newton method; its defect-based admission is a small experimental change.
This is an explicit baseline for the intuition, not a claim of a new optimizer
or of having solved the original long-horizon closure problem.

## 1. What the higher state carries

Let K=exp(-C/epsilon), with positive unit-mass marginals a,b. Define

    F(y) = sum_i a_i log sum_j K_ij exp(y_j) - b.y,
    P_ij(y) = K_ij exp(y_j) / sum_l K_il exp(y_l),
    c = P^T a,
    gradient_y F = c-b.

The collection of conditional distributions P determines the current AND its
response to perturbations. For any displacement h, its exact update is

    P'_ij = P_ij exp(h_j) / sum_l P_il exp(h_l).

There is no affine scalar-law fit in this identity. Each source row carries a
whole conditional distribution. The curvature and ordinary Sinkhorn rule are

    H = diag(c) - P^T diag(a) P,
    J = diag(c)^(-1) P^T diag(a) P = I-diag(c)^(-1) H.

Consequently changing source responsibilities changes the rule automatically.
This keeps interactions that projecting onto a single eigenvalue discards.
It does not show that a mixture component count equals a needed state rank.

For a time-dependent y, the exact lifted differential identity is

    dot P_ij = P_ij [dot y_j - sum_l P_il dot y_l].

It preserves row sums and compatibility with the exponential family when
initialized compatibly. Thus (y,P) is a closed representation of the physical
state. P is redundant information, not a new degree of physical freedom. The
lift makes the response law explicit; it does not magically reduce its cost.

## 2. A descent law on that state

Use fixed mass coordinates z=sqrt(b)*y and

    g = (c-b)/sqrt(b).

For any symmetric positive-definite response operator M on the nonconstant
subspace, choose dot z=-M g. Then

    dot F = -g^T M g < 0                         if g != 0.

The constant gauge vector in these coordinates is sqrt(b); the implementation
projects it out. This identity remains true when M changes with the current
state and its history. Differentiating F introduces no dot M term because F
is a function of the physical state, and M determines its instantaneous
velocity. This is a Lyapunov function for physical descent, not a strictly
decreasing objective for all redundant representation coordinates.

Discrete steps use an Armijo test on the actual F difference, not on a
quadratic model's prediction. The exact difference is

    F(y+h)-F(y) = sum_i a_i log sum_j P_ij exp(h_j) - b.h.

For small h the implementation uses log1p(P expm1(h)), avoiding subtraction
of two nearly equal full objective values. With positive-definite M,
backtracking terminates in exact arithmetic at a nonstationary finite state.
The recorded numerical tests are floating-point checks, not interval proofs.

## 3. The finite response-memory state

The inexpensive additional state is a list of at most eight pairs

    s_k = z_(k+1)-z_k,
    t_k = g_(k+1)-g_k.

They measure how the rule responded to moves the system actually made. A
standard inverse BFGS update satisfies M_new t_k=s_k and preserves positive
definiteness for s_k.t_k>0. The two-loop implementation applies the resulting
operator in O(n m) work and O(n m) storage for memory m. No dense inverse,
eigendecomposition, fixed point, reference amplitude, or future trajectory is
used. Eight pairs are not eight physical modes or an eight-dimensional closed
trajectory: the operator includes an isotropic action on the full space.

The experimental adaptive variant admits a valid pair when

    ||M_k t_k-s_k|| / ||s_k|| > 0.1,

or when no pair has yet been stored. Once full, it drops the oldest pair.
This ratio is an observed secant-consistency defect. It is NOT a certified
relative gradient error and provides no bound on unseen future directions.
The threshold and memory length were fixed before running the six cases.

Controls are ordinary Sinkhorn, natural-coordinate gradient descent, a changing
scalar curvature step size, standard always-update L-BFGS, and a frozen
response context that keeps its first eight pairs forever. The scalar control
uses s.t/(t.t) as its next scale. Success of that control does not contradict
the previous affine theta(a) impossibility: these are different scalar models.

## 4. Relationship to the two references

[Csillag et al., Functional Gradient Descent with Adaptive Representations](https://arxiv.org/html/2606.16926v1)
refines the representation of a functional gradient until a computable error
bound is small relative to the approximation. Its convergence results require
the stated smoothness, compatibility, error-control, and, for global rates,
PL assumptions. The transferable idea is that representation adequacy belongs
inside the dynamics. We do not inherit its theorem for a secant-defect test.

The repository's `ML_experiment/models.py` self-context path constructs a
response-weighted context, anchors the augmented chart to the actual input,
and reallocates the metric and responses. Here the feedback is temporal:
accepted state changes produce response measurements that change the next
motion operator. This is an analogy, not an implementation or validation of
the neural self-context architecture. No neural training was performed.

## 5. Cost and scope

The first version explicitly rebuilt P using log-domain softmax. That version
reduced step counts but usually lost to lean BLAS Sinkhorn in wall time.
`context_descent128_explicit.json` retains that measurement; the initial
three-repeat screen is also retained and is not the final timing record.

The final implementation uses the exact factorization

    P = diag(1/(K v)) K diag(v).

It stores no extra dense dynamic conditional table. A state/gradient evaluation
uses two kernel-vector products, and each objective trial one more. The supplied
kernel remains dense. Setup includes exp(logK), response-memory updates,
rejections, line search, history recording, and an independent final log-domain
marginal certificate. Severe scaling-range problems trigger a log-domain
fallback. Lean ordinary Sinkhorn has no Armijo or history overhead and includes
kernel setup and the same final certificate. This is the performance comparator;
the common-evaluator ordinary control diagnoses iteration behavior.

All methods start at y=0, with n=128, seeds 0,1 and epsilon=.01,.003,.001.
The stopping condition is max_j |c_j/b_j-1| <= 1e-9. The earlier anchor study
used a tighter, different stopping rule, so its counts 274/919/3053 and
178/566/1656 are not the ordinary counts in this benchmark. The input problems
are unchanged. The benchmarks run on the M4 Mini with one BLAS thread and five
complete timing repeats. Millisecond timings describe these implementations
and inputs, not hardware-independent complexity guarantees.

This implementation descends the same transport objective but changes the
trajectory. It does not emulate H ordinary iterations, certify a skip, predict
three held-out eigenvalue anchors, or prove closure of a small autonomous rule
model. Those remain distinct requirements of the earlier accelerator project.

## 6. Retained result

Every retained method/case result (7 methods x 6 problems) converged; timings use five complete repeats. All 18 focused and neighboring regression tests passed on the Mini. The new six-test suite checks the exact retilting identity, gradient, lifted differential rule, positive inverse response, gauge invariance, descent, factorization, and independent marginal agreement.

Median complete wall time is in milliseconds. Counts are accepted steps.

| Seed | epsilon | Ordinary steps | Adaptive steps | Frozen steps | Lean ordinary ms | Scalar ms | Adaptive ms | Standard L-BFGS ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01 | 182 | 35 | 117 | 3.140 | 2.180 | 3.089 | 3.140 |
| 0 | 0.003 | 610 | 75 | 1086 | 9.930 | 5.358 | 6.713 | 6.826 |
| 0 | 0.001 | 2002 | 163 | 5273 | 33.514 | 11.680 | 14.204 | 14.108 |
| 1 | 0.01 | 123 | 35 | 108 | 2.115 | 1.829 | 2.966 | 2.984 |
| 1 | 0.003 | 393 | 70 | 405 | 6.472 | 4.425 | 6.000 | 6.010 |
| 1 | 0.001 | 1155 | 132 | 6793 | 19.261 | 10.047 | 11.556 | 12.364 |

The richer adaptive memory takes fewer accepted steps than the scalar control in all six cases, but **the scalar control is faster in all six**. Adaptive memory beats lean ordinary Sinkhorn in five cases (one by only a few percent), loses in one, and is essentially ordinary L-BFGS on four cases. The two altered paths are seed 0/epsilon .01 and seed 1/epsilon .001; the first still takes 35 steps, while the second takes 132 instead of 140. There is no convincing distinct algorithmic gain from the defect gate.

Frozen early response memory is poor, sometimes much worse than ordinary iteration. This compares one particular early frozen context; it does not rule out a better fixed preconditioner.

The experiment establishes a concrete response-state system with an exact physical descent law and successful non-oracle discrete realization. It does **not** establish that richer directional memory is necessary for fast solution: the cheaper evolving scalar controller wins this small timing battery. The useful remaining research target is a context that carries nonlinear rule evolution cheaply enough to outperform these established controls, with an independently checked closure error if long-horizon skipping is required.

Raw data: `results/theory/context_descent128_factored.json`. The final test transcript is `results/theory/context_descent_verification.txt`.
