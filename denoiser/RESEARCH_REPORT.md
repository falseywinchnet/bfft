# Integrated FMMT support birth

## Why this branch exists

PIT-R10 established a specific failure of FMMT and ordinary denoisers: stochastic
fine-scale error can be reduced in total variance while some of its energy condenses
into broad, low-frequency pits in low-information regions. Once that happens, the
reconstruction itself makes the pit look like structure.

The post-pass PIT rule helped, but it acted after FMMT had already allowed the false
support to influence geometry and transported measures. This branch moves the rule
upstream.

The design statement is:

> coarse energy in a provisional estimate is not evidence for coarse support.

## What changed

Everything after FMMT's provisional bootstrap remains recognizably FMMT. The new
state transition occurs before the eikonal graph is finalized.

### 1. Provisional bootstrap

The ordinary robust FMMT histogram transport produces `x0_provisional`.

### 2. Cross-predictive coarse support

Two complementary checkerboard observation lattices independently fit an affine
local relation and a quadratic relation. For each lane, the bounded predictive
advantage is

    d = s * tanh((|y-a| - |y-q|) / s)

where `s` is a robust local scale. Coarse curvature earns support only when the
quadratic relation improves held-out prediction on both lanes. The rule is evaluated
at physical scales 8 and 12 pixels.

A remote bootstrap-residual tail raises the evidence required for a new coarse birth.
It never becomes a likelihood.

### 3. Fine-support ancestry

Four disjoint residue lattices independently measure the 1.5 -> 3 pixel scale-space
band. Dense agreement among those lanes supplies fine ancestry. This lets distributed
high-entropy structure protect its descendants without naming it "texture".

### 4. Observation reliability

Two representation failures reduce the authority of the support witness instead of
creating a noise-family branch:

- **hard-bound censoring:** when substantial observation mass collapses onto 0 or 1,
  interior geometric evidence is censored. A smooth authority transition occurs over
  censored mass 0.14 -> 0.22;
- **remote support fragmentation:** when the 90th/50th absolute bootstrap-residual
  ratio becomes enormous, the observation is too fragmented to rewrite the robust
  bootstrap. Authority transitions over ratio 16 -> 32.

The two authorities multiply. Thus the original FMMT state is inherited exactly on
the 61% salt/pepper holdout, while the supplied website-noise Cameraman remains fully
eligible (tail ratio 12.31, censored mass 0.104).

### 5. Heritable bootstrap support

Split the provisional chart into

    x0 = fine + coarse,
    coarse = Gaussian_1.5(x0).

Define

    S = max(cross_predictive_coarse_support, fine_ancestry)
    g = smooth((1-S) * witness_authority).

Only the coarse state evolves. One conservative four-neighbor flux step is

    F_ij = dt * (g_i + g_j)/2 * (x_j - x_i),    dt = 0.18,

with equal and opposite updates. The retained checkpoint uses 128 finite steps. This
is a support horizon, not a convergence solve. The fine state is carried unchanged.

### 6. Eikonal barrier admission

The same evidence has a distinct geometric role. For an edge `i-j`, ordinary FMMT
uses a contrast-dependent crossing cost. We multiply only that contrast term by

    b_ij = average(1 - authority * (1-S)).

Supported contrast and unreliable/censored observations therefore retain ordinary
FMMT resistance. Reliable unsupported contrast becomes easier for the ordered front
to cross. The graph gate by itself was negative; it is retained only after bootstrap
support certification, where it gives a small consistent improvement.

### 7. What was deliberately left unchanged

After this point FMMT is unchanged:

- robust residual-scale construction;
- atom/local empirical signal packet rule;
- empirical residual packet;
- same ordered Dijkstra/eikonal fronts for signal and residual measures;
- additive observation coupling;
- posterior mean and entropy inertia.

No R6/AREL mechanism is present. No PIT operation runs on the final image.

## Matched severe/unknown-noise screen

Four 96x96 natural images (camera, moon, coins, page), each under full-field
uniform additive, random-value replacement, mixed replacement+uniform, Gaussian,
Laplace and multiplicative corruption. Clean truth is evaluation-only.

| method | mean MSE | mean SSIM | low-complexity curvature ratio s=4 | s=8 |
|---|---:|---:|---:|---:|
| plain FMMT | 0.00381260 | 0.656863 | 0.83190 | 0.99250 |
| FMMT + PIT-R10 post-pass* | 0.00369546 | 0.672112 | 0.75153 | 0.95226 |
| **integrated FMMT** | **0.00360911** | **0.689573** | **0.70752** | **0.91924** |

`*` matched to the current local PIT-R10 implementation in this checkpoint.

Integrated FMMT beats plain FMMT in 20/24 MSE and 22/24 SSIM cases, and beats the
matched PIT post-pass in 21/24 MSE and 22/24 SSIM cases.

The curvature ratio is the scale-normalized Laplacian RMS of reconstruction error
inside truth-defined low-complexity regions, divided by the same quantity in the
corrupted input. Values above one mean denoising has concentrated *more* error into
that coarse scale than the input contained.

## Original FMMT 28-case holdout

The original four natural images x seven corruption families were rerun with the
same seeds.

| method | mean MSE | mean SSIM |
|---|---:|---:|
| plain FMMT | 0.00391936 | 0.749316 |
| **integrated FMMT** | **0.00388810** | **0.757239** |

The 20% and 61% salt/pepper groups are exactly inherited from plain FMMT because
support-witness authority falls to zero. Continuous/mixed groups improve on average.

## Support-falsification screen

Seven synthetic truths were designed to separate flat fields, affine gradients,
genuine broad blobs, genuine smooth shadows, steps, periodic texture and mixed
structure. Five severe corruption processes yield 35 cases.

| method | mean MSE | mean SSIM |
|---|---:|---:|
| plain FMMT | 0.00240902 | 0.618979 |
| FMMT + PIT-R10 post-pass | 0.00210416 | 0.664349 |
| **integrated FMMT** | **0.00192628** | **0.709008** |

The periodic truth is effectively unchanged by the new support rule. Broad real
blobs/shadows improve rather than being flattened, which is the main falsification
of the trivial "smooth low entropy" explanation.

## Cameraman pitting regression

On 128x128 Cameraman with severe full-field uniform corruption:

| method | MSE | SSIM | curvature ratio s=4 | s=8 |
|---|---:|---:|---:|---:|
| FMMT | 0.0056034 | 0.46200 | 1.217 | 1.193 |
| FMMT + PIT-R10 | 0.0052914 | 0.48593 | **1.018** | **1.128** |
| **integrated FMMT** | **0.0052526** | **0.49462** | 1.029 | 1.132 |

The integrated system wins distortion/perceptual metrics, while the final-image
post-pass is marginally more aggressive on this particular coarse-curvature metric.
That distinction is retained rather than hidden.

## Supplied website-noise Cameraman

No exact clean reference is assumed for this upload. The diagnostic state is:

    residual tail ratio      12.3089
    hard-bound mass          0.1038
    support witness authority 1.0

The integrated result visibly reduces the broad sky mottling relative to plain FMMT
without a second denoising stage. On this 256x256 file in the current environment:

    plain FMMT total          ~1.99 s
    FMMT then PIT-R10 total   ~5.75 s
    integrated FMMT total     ~2.60 s

## Interpretation

PIT was not fundamentally a correction operator. Its useful content was a rule for
**what is allowed to become support**. Moving that rule before eikonal geometry makes
it stronger and cheaper:

    observation witnesses
       -> provisional support admission
       -> bounded support evolution
       -> eikonal barrier admission
       -> ordinary FMMT measure transport
       -> one reconstruction.

The remaining open problem is very coarse pitting. The present support horizon
continues to help low-complexity regions as it grows, so a future version should
replace the fixed 128-step horizon with a transported support budget or stopping law.
That is explicitly left open rather than tuned further in this checkpoint.
