# Krylov polynomial transport for Richardson-Lucy / ML-EM

Transfer of the `codex/krylov-bregman-research` polynomial transport (and the
entropic closure analysis of `experiments/entropic_transport_closure`) to
Poisson deconvolution, a classically slow EM iteration. `core.py` is an
unmodified copy of that branch's `experiments/krylov_bregman/core.py`.

## Structure (tested in `test_rl.py`)

Log coordinates z=log x, periodic normalized Gaussian blur A, background b>0:
F(z) = z + log(A^T(y/(A e^z + b))). With P~ = diag(1/m)[A diag(x) | b]
(row stochastic, background column carries zero displacement) and
Q = diag(1/r) A^T diag(y/m) (row stochastic), r = A^T(y/m):

* J = I - Q P, self-adjoint in the metric diag(x r) = diag(x_next).
* Exact remainder: F(z+d)-F(z)-Jd = -Q L_P(d~) + L_Q(-e), both log-moment
  terms in [0, osc(d~)^2/8], so ||remainder||_inf <= osc(d~)^2/8 with
  osc taken over d and 0. This is the Sinkhorn bound (13) transferred.
  Measured sharpness: actual/bound ~ 0.08-0.12 for 1x-256x displacements.
* NOT transferred: I-QP is not an inf-norm contraction, so the accumulated
  per-step sum is not a proved trajectory certificate here. It is used as a
  horizon gate; every jump is also objective-guarded (monotone NLL).

## Horizon gate

`certified_horizon`: largest H in {16..1024} with
sum_j [ (sum_i |c_ij| osc(q_i))^2/8 + sum_i |c_ij| ||E_i||_inf ] <= tau ||s_H||_inf,
using triangle-inequality bounds so each step is O(depth), not O(n depth).
The first exact-vector version was correct but slow (results/krylov_em_cert.json).

## Results (M4 Mini, 128x128, 2 seeds x sigma{1.5,3} x counts{100,1e4})

Target = NLL reached by ordinary RL after N sweeps; speedup = wall time vs
those N ordinary sweeps; all overheads (guards, Arnoldi, gate) timed.
Single timing per run; no repeats yet.

| N=20000 | guarded Anderson(4) | Krylov d8/h256 fixed | certified d8, tau=10 |
|---|---|---|---|
| sigma 3   | 4.1-12.5x | 8.6-9.4x | 7.0-7.5x |
| sigma 1.5 | 6.1-8.6x  | 2.4-5.0x | 7.3x (all four) |

Fixed horizon 1024 hits the budget on sigma 1.5 / 100 counts; the gate never
fails. Unguarded Anderson NaNs or stalls in 4/8 deep cases. Guarded
Anderson remains fastest on 4/8 (all 1e4-count cases). tau was chosen from
{0.1,1,10,100} on these same cases: a held-out validation is still owed.

Files: `results/krylov_em_v1.json` (N=3000), `krylov_em_deep.json`,
`krylov_em_cert.json` (exact-vector gate), `krylov_em_cert2.json` (final gate).
