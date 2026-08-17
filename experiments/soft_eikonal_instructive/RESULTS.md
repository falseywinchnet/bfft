# Soft Eikonal instructive-compartment results

## Verdict

The parameter-free **self-context reallocation** is the standout. It makes a
first Eikonal allocation, backprojects that allocation into the layer input,
and then reallocates using the resulting model-generated contextual guess.
Across the six-task confirmation it improved learning-curve area on **6/6**
tasks, mean held-out score by **+0.023**, and mean tail score by **+0.017** at
exactly the same parameter count as the unchanged soft Eikonal. Its price is
about **1.75x CPU time**.

This is not an extrapolation solution. No tested variant retained the spiral or
checkerboard rule outside the observed region. The result is narrower and more
interesting: iterative, model-generated acquisition context improves how the
Eikonal pool acquires several difficult structures without adding a privileged
task basis.

## Confirmation protocol

- M4 Mini CPU; Torch CPU only.
- Six tasks: inner-region spiral and checkerboard, radial stripes, 2-D ripple,
  the user-supplied multiscale 1-D function, and localized 1-D steps.
- Width 36, 800 AdamW steps, batch 256, three paired seeds.
- 9 variants × 6 tasks × 3 seeds = **162 confirmation runs**.
- Every variant has exactly the unchanged soft Eikonal parameter budget:
  7,814 parameters for two-class 2-D tasks, 7,777 for scalar 2-D regression,
  and 7,741 for scalar 1-D regression.
- Classification score is mean class recall. Regression score is
  `1 / (1 + normalized MSE)`. Learning AUC is validation score integrated over
  the 800-step trajectory. Tail score uses the explicitly unobserved region
  where a task has one.

An earlier width-16, 400-step, two-seed screen contributed another **108 runs**
and selected no variants: all nine proceeded to confirmation.

## Exact-budget comparison

Values are three-seed means. “Test” is outward continuation for spiral,
checkerboard, and the two 1-D functions; it is held-out interpolation for
radial stripes and ripple.

| Problem | Ordinary MLP | Eikonal | Self-context | Hard allocation | Relational secant | Garnish instruction | Paired-zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| Checkerboard test | **.508** | .460 | .453 | .447 | .461 | .462 | .457 |
| Checkerboard observed-region validation | .840 | **.960** | .958 | .966 | .955 | .957 | .817 |
| Spiral test | .340 | .351 | .342 | .362 | .350 | **.388** | .372 |
| Spiral tail class retention | .312 | .320 | .327 | .332 | **.347** | .288 | .305 |
| Radial stripes | .517 | .831 | **.911** | .831 | .657 | .719 | .519 |
| Ripple | .506 | .968 | .969 | **.970** | .960 | .916 | .718 |
| Multiscale 1-D test | .138 | .220 | .319 | .294 | **.369** | .295 | .179 |
| Multiscale 1-D tail | .187 | .278 | .385 | .324 | **.427** | .334 | .225 |
| Localized steps test | .918 | .963 | .937 | .961 | **.981** | .966 | .943 |
| Localized steps tail | .871 | .926 | .883 | .919 | **.958** | .935 | .889 |

## Learning speed and cost

| Variant | Mean learning AUC | AUC wins vs Eikonal | Mean score delta | Mean tail delta | CPU seconds/run |
|---|---:|---:|---:|---:|---:|
| **Self-context** | **.867** | **6 / 6** | **+.023** | **+.017** | 6.05 |
| Hard allocation (`T=.55`) | .857 | **6 / 6** | +.012 | +.006 | 3.49 |
| Eikonal control | .844 | — | — | — | 3.46 |
| Relational secant | .838 | 4 / 6 | -.003 | +.002 | 3.56 |
| Allocation secant | .832 | 2 / 6 | -.015 | -.018 | 9.13 |
| Garnish instruction | .822 | 1 / 6 | -.008 | -.025 | 9.25 |
| Soft allocation (`T=1.8`) | .822 | 0 / 6 | -.002 | +.001 | 3.48 |
| Paired-zero | .719 | 0 / 6 | -.101 | -.113 | 3.67 |
| Ordinary LELU MLP | .682 | 0 / 6 | -.144 | -.140 | **.51** |

No composite score was used. Fit, learning speed, continuation, class
retention, parameter count, and runtime remain separate measurements.

## What the proposals actually did

### 1. Acquisition auxiliary: supported

The self-context version does not receive a derivative, label, neighbor, or
future sample. Its first allocation makes a contextual guess from the current
activation using the model's fixed primitive directions; that guess perturbs
the latent input before a second allocation. This is a continuous within-layer
loop and adds no parameters.

It is strongest where a point benefits from being interpreted relative to an
emerging reusable structure: radial bands, ripple phase, and the multiscale
curve. Its three-seed radial result is also much more stable than the base
(.911 ± .008 versus .831 ± .091).

The ablations say the gain really lives in the conditioned allocation. On
radial stripes, removing the correction, making allocation uniform, or giving
each point another point's allocation reduces self-context by .404, .392, and
.415 respectively; the corresponding base-Eikonal drops are .321, .329, and
.326. Intriguingly, self-context does **not** merely sharpen the final gate. It
raises mean allocation entropy (up/down: .758/.509 versus .712/.384) while
making the learned metric much less singular on multiscale 1-D (median
condition numbers roughly 12,955/904 versus 127,233/17,471). The best current
interpretation is a better-conditioned, more distributed coordinate choice,
not winner-take-all filter selection.

### 2. Harder allocation: useful, but a smaller result

Reducing the allocation temperature from 1.0 to .55 improved learning AUC on
all six tasks at no runtime cost. It is a strong practical default, but its
effect is smaller and more task-dependent than self-context. This says some of
the base pool's weakness is allocation indecision, not missing representational
capacity.

### 3. Output secants: specialized continuation pressure

The relational loss explicitly supervises `f(x_i)-f(x_j)` against
`y_i-y_j`. It is the best continuation variant on both 1-D functions and the
best spiral tail-class variant, but damages radial stripes. It induces a
relationship, so it is less structure-agnostic than self-context and should be
treated as a specialized training objective rather than the new default.

### 4. Garnish instruction: an informative near miss

Two randomly perturbed views create an averaged output and its task-error
derivative; the true-input stream receives that detached derivative rather than
a direct target loss. It improves mean spiral test score (.388 versus .351),
but spiral tail class retention falls (.288 versus .320), ripple worsens, and
runtime rises to 2.7x the base. It can push a decision surface outward without
preserving both classes. That is not the desired vectorized acquisition signal.

### 5. Authentic double-width pairing: rejected in this form

The paired model was trained on `(x_i, x_j)` and `(x_i, 0)` and was always
evaluated on `(x, 0)`. Its input and output are genuinely doubled while its
total parameter count remains exact. It loses learning speed on every problem
and collapses near chance on radial stripes. Pairing alone creates a harder
joint representation problem; it does not force the first slot to extract a
useful relational scene basis.

## The important failure

The observed-region checkerboard fit remains visually excellent for Eikonal
(.960 validation), yet its unobserved-region score is .460 and every method has
zero strict tail-survival bins. Spiral behaves the same way: nearly perfect
inner fit and roughly chance outward continuation. The ordinary MLP is less
beautiful inside the checkerboard but happens to remain nearer chance outside.

Therefore none of these results supports “the Eikonal learned the global rule.”
Self-context improves acquisition and some continuation; it does not supply the
missing evidence needed to identify an arbitrary periodic continuation from an
inner scene.

## Reproducible artifacts

- `results_screen/results.json`: 108-run first screen.
- `results_confirm/results.json`: 162-run confirmation.
- `results_confirm/summary.json`: paired-seed aggregates.
- `results_confirm/probes.json`: prediction fields and 1-D curves.
- `results_confirm/diagnostics.json`: matched/uniform/mismatched/base-only
  allocation ablations.
- `run_screen.py`, `run_visual_probes.py`, and `run_diagnostics.py`: M4 CPU
  entry points.
