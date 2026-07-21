# Bregman Meyer decomposition, fused: sub-second Gilles-Osher + the soft-rung law

Date: 2026-07-21.  Status: complete measurement record.
Code: `experiments/meyer_bregman.py` (engine + all experiments; CLI sections
`engine|transition|iters|transport|budget|ladder|barbara|barbara_ladder`).
Figures: `experiments/out/meyer_{iterations,transition,transport,ladder_rig,barbara,barbara_ladder,barbara_zooms}.png`.
Paper: Gilles & Osher, "Bregman implementation of Meyer's G-norm for cartoon +
textures decomposition", arXiv 2410.22777.  Image: Barbara 512^2 (the paper's
own Fig. 1 test image), Set12/09.

## 1. The reduction that makes the paper ours

Gilles' Algorithm 3 (with `P_ROF(g, c) = argmin J + (c/2)||g - w||^2`; the
second argument is the fidelity coefficient, fixed by their Prop-2 proof)
collapses exactly, by the substitution `w = lambda z`, to a two-projector
alternation:

    u <- ROF(f - v, lambda)          # cartoon step: ball radius 1/lambda leaves u
    v <- P_{G_mu}(f - u)             # texture step: G-ball radius mu
    with P_{G_mu}(g) = g - ROF(g, 1/mu)   (Chambolle residual identity)

The u-step moves at most a G-sliver of size 1/lambda from u to v per pass, so
the outer loop is a slow geometric texture transfer — structurally the same
regime as the two-lattice alternating projections, which is why the program's
levers were testable here.

Conventions: images [0,255] float64, periodic BC, forward-difference TV,
FFT-diagonal u-solve (Alg. 2 verbatim).  lam=0.05, mu=40-60 give a Fig-2-like
split; the paper's "lambda = mu = 1000" is not reproducible under any scaling
we tried and its convention is left as unknown.

## 2. Minimal iterations: measurements

Rig: 256^2 synthetic cartoon (shaded disk/rect, step edges) + two windowed
oscillations (T=12 and T=3, amplitude 20; G-norms ~38 and ~9.5), known
ground truth.  Cost currency: inner Bregman sweeps (1 sweep = FFT u-solve +
shrink + b-update); truth stick = FGP reference fixed point (120 exact
outers, disk-cached; its own residual outer drift 2.0e-3 = err floor).

| variant | sweeps | wall | fp-resid | PSNR u / v |
|---|---:|---:|---:|---|
| Gilles Alg.3, cold nested (inner tol 1e-4) | 21,071 | 18.7 s | 2.4e-3 | 35.01 / 36.21 |
| fused: warm interleaved 1-sweep passes | 810 | 0.84 s | 2.0e-3 | 34.61 / 35.73 |
| fused + heavy-ball beta=0.88 | 810 | 0.9 s | 5.9e-2 | 27.73 / 24.47 |

- **The interleave IS the algorithm: 26x fewer sweeps to a better fixed-point
  residual, same fixed point to 8.6e-3.**  Nested inner convergence on a
  wrong outer state is pure waste (inexact-Uzawa folklore, measured).
- **Primal heavy-ball momentum is harmful here** (beta<=0.6 neutral, >=0.8
  limit-cycles).  The two-lattice beta=0.88 law belongs to nonconvex radial
  projections; these prox operators are firmly nonexpansive and already
  contractive under interleaving.  Momentum polarity flipped — see §3.
- A separate seed is redundant: 1-sweep interleaving from zero is itself a
  graduated seed (seed_sweeps=5 changed nothing).
- Barbara 512^2, same protocol: nested tight = 63,633 sweeps / 361 s;
  fused 0.9 s budget = 60 passes, ||u - u_Gilles||/||f|| = 1.0e-2,
  **~400x wall speedup**, visually identical at 1x (differences are
  edge-hugging ripples visible only at 20x amplification).
  Context: ONE Chambolle-projector v-step at tol 1e-4 took 4000 p-iterations
  — the paper's Bregman-vs-Chambolle speed claim reproduces in passing.

### Budget shootout (sub-second gate, 0.8 s cap, 256^2)

All memoryless levers tested at fixed wall budget: eta_v=10c (best inner
accuracy per sweep at weak fidelity), KM over-relaxation omega in 1.3-1.9,
lambda-continuation /4-/16, 40-sweep seeds, and combinations.  Result: every
variant lands in err 9.0e-3-1.6e-2, PSNR u 34.1-34.5 — **the plain
interleave at eta=10c is on the plateau; the split is PSNR-converged by
~130 passes and the remaining error is outer-transfer distance that no
per-pass decoration removes.**  Half-resolution boot on Barbara: worse
(1.4e-2 vs 1.0e-2) — 2x downsampling destroys the near-Nyquist tablecloth
check; honest negative.

## 3. The transport polarity (from the "fiberwise Jacobian" prompt)

External prompt asked whether branch-aware/transport-style Jacobian handling
speeds Bregman descent.  Honest scoping: Split Bregman has no pushforward,
no change of variables, and single-valued prox maps — the fiberwise /
multi-branch formalism has no referent in this problem.  The one rigorous
transport object is Meyer's own: **the G-norm is a flux norm** (v = div p,
||p||_inf <= mu), and ROF's dual lives on that flux:
`p* = argmin_{|p|<=1} ||g + div p / c||^2`, u = g + div p/c.

Measured (T1, isolated texture step at weak fidelity c=1/60, error vs
converged step after 800 inner iterations):

| solver | err @ 800 it | note |
|---|---:|---|
| Chambolle p-iteration (no momentum) | 1.9e-1 | the paper's baseline |
| FGP = same iteration + FISTA on the flux | 5.1e-3 | **40x: momentum belongs on the flux** |
| Split Bregman eta=2c | 3.1e-2 | |
| Split Bregman eta=10c | 4.0e-3 | best; creeping tail |

**Momentum polarity law (measured):** heavy ball on the primal image
diverges; FISTA on the dual flux is the correct accelerated form.  Same
levers, opposite sides of the duality gap.

But in the full pipeline the flux does NOT win: carrying FISTA momentum
across outer passes (changing objective) is harmful (err 7.6e-2), restarting
momentum each pass while keeping the flux recovers it (1.9e-2), and the SB
interleave still leads at equal wall time (1.1e-2) because its FFT u-solve
is exact per sweep and its (d,b) state encodes the edge set, which transfers
across passes.  Stale momentum on a moving flow is the same failure measured
twice (primal beta, dual FISTA-carry) — the user's constraint "transport
flow changes at every step" is the correct summary; history-based outer
extrapolation (Anderson) was rejected on those grounds without test.

## 4. The soft-rung law: Meyer scale ladders are ramps, not cells

`exp_transition`: capture fraction of P_{G_mu} on pure full-field cosines vs
mu/||g||_G.  10-90% transition width = e^2.2-2.6 = **9-14x in mu** (T=4,8,16
px all alike).  Theory agrees: ROF on a cosine soft-thresholds AMPLITUDE, so
capture ~ min(1, mu/G) — a linear ramp whose 10-90 width is exactly 9x.
Compare the STFT aperture ladder's sharp cells (r* in (4,8], task-10):
**G-norm rungs have no critical ratio; band cross-talk decays only linearly
in scale separation.**

Consequences, all measured on the two-texture rig (G ratio 4):

- Two-way band routing telescopes: summed bands above/below a split depend
  only on the STRADDLING rung — interior rungs cancel exactly (r=2 and r=4
  ladders produce bitwise-identical layers).  Ladder design for a pair
  reduces to placing one rung.
- With soft rungs the geometric-midpoint placement rule fails: rung at
  0.79 x G_fine (the r=8 ladder's accidental placement) beat the midpoint
  rung on total cross-leak (0.20/0.37 vs 0.14/0.65) — optimal placement
  skews toward the finer texture because the ramp is what it is.
- Separation at G-ratio 4 is partial no matter the ladder (x-leak >= 0.2).
  Clean separation needs pair ratio >~ the transition width (~14x), BUT the
  usable span is compressed from above too: at T=48 (G=153) the "texture"
  scale collides with cartoon features and the u/v split itself degrades
  (PSNR u 25.0 vs 31.8) — mu_max ~ 2.5 G_c drags cartoon into v.
- **Net: the Meyer axis supports ~2 genuinely distinct texture bands per
  image.  The scale-ladder theorem does NOT transplant from windowed-
  spectral families to ROF scale space; sharp multi-band routing needs
  two-lattice-style windowed-DFT families.**  This is the negative that
  makes the two axes' roles precise: G-ladder = graded transport-soft
  display; STFT ladder = sharp coverage.

## 5. What the ladder is still good for: ghost quarantine (Barbara)

3-rung ladder {40, 10, 2.5} on the converged Barbara texture layer
(bands = differences of independent ROF solves; NOTE Bregman (d,b) states
must not be carried across rungs — eta/c-scaled state pins the next rung at
the previous fixed point and empties the band; measured, fixed):

- mid band (mu 10->2.5): scarf stripes + tablecloth check, nearly pure
  textile — the layer a texture application should read;
- coarse band (mu 40->10): **the paper's Fig-2 contour-ghost artifact
  (face/edge leakage) concentrates HERE**, quarantined away from the
  textile band;
- fine band (<2.5): weave detail + sensor grain.

Single-mu Gilles v mixes all three inseparably.  The ladder does not remove
the edge-ghost; it isolates it in a known band at ~1.8 s extra (512^2).

## 6. Standing algorithm (what to reuse)

`a2bc_budget(f, lam, mu, budget_s)`: zero init; per pass ONE Split-Bregman
sweep on `ROF(f - v, lam)` (warm state) + ONE sweep on `ROF(f - u, 1/mu)`
with eta = 10/mu (warm state); v = (f-u) - w; stop on wall clock.  No
momentum, no seed, no continuation, no history.  256^2 -> 0.8 s, 512^2 ->
0.9 s at ~1e-2 from the converged Gilles fixed point (PSNR-indistinguishable).
Band the texture afterward with independent ROF solves per rung.

Open (not pursued): outer-transfer acceleration that respects the moving
flow (rejected: Anderson/secant — flow changes per step); mu-selection from
image content; relaxed per-region texture weights (edge-gated w in [0,1],
our 4c generalization) for ghost suppression rather than quarantine.
