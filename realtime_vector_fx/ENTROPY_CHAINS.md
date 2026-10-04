# Optional entropy Chains — 2026-09-10

Enable the **Chains** checkable group in **Entropy-Guided Decorrelation Stretch**.
This extends the existing filter; no additional filter instance is needed.
Existing settings default to Chains off. Turning it on starts with RGB amount 1,
brightness amount 0, and contrast amount 0, preserving the existing RGB operator.
The existing decorrelation and entropy-allocation settings still define that
RGB operator. The new RGB amount blends its complete result with its input.

| Option | Default | Range / meaning |
|---|---:|---|
| Chains | Off | Enable independent stage amounts and composition |
| RGB color amount | 1 | 0–1, strength of the existing RGB treatment |
| Brightness stretch amount | 0 | 0–1, bounded source-derived midtone adjustment |
| Entropy contrast stretch amount | 0 | 0–1, entropy-driven brightness-range allocation |
| Brightness reference (midtone) | 0.5 | 0.1–0.9, desired reference in encoded SDR luma |
| Application order | RGB → Contrast → Brightness | All six permutations available |

Start with the existing RGB tuning and raise the tone amounts gradually. There
is no new universal preset. Overall **Effect strength** blends the completed
chain with the input, separately from the three stage amounts.

## Shared observation and independent estimation

There is one capture and one sparse lattice at each analysis update. Each stage
uses the same unpremultiplied source samples and source-derived local entropy.
No stage measures an earlier stage's adjusted output, and the displayed output
is never fed back into the estimators. Input means the image entering this
filter: upstream OBS filters are already included.

The RGB stage is unchanged, including its internal decorrelation followed by
RGB-channel range allocation. Its allocation histogram still belongs to that
complete RGB operator. The brightness/contrast estimators use the original RGB
samples, independent of the RGB settings, chain amounts, and application order.

Let Y = 0.2126 R + 0.7152 G + 0.0722 B in the existing encoded SDR display space.
This is a display-luma coordinate, not linear-light luminance or physical exposure.
A 64-bin unweighted valid-source histogram estimates median p by interpolation
within its occupied bin; p is bounded to [0.01, 0.99]. Sources whose valid Y range
is no larger than the configured noise floor get identity tone transforms.
This flat-scene guard does not constitute a statistical noise detector.

## Brightness

For brightness reference r, use the bounded odds gain

    g = clamp(r (1-p) / (p (1-r)), 1/4, 4)
    B(Y) = g Y / (1 + (g-1) Y).

The curve is strictly increasing and preserves black and white. Without the
gain cap, B(p) = r. The cap bounds extreme corrections. With other stages active,
the final frame's measured median need not equal r: all stages were learned
from the original source. This is a smooth exposure-like display adjustment.

## Contrast

For each original-source luma bin with population n and local-entropy sum H,
form e = H/(n+4), and mass e² n^0.25. Smooth twice with [1/4, 1/2, 1/4], then use
the existing positive-density rule 0.2 + 0.8 min(8, 64 mass / sum(mass)).
Its normalized cumulative integral is F. With insufficient entropy evidence,
F is the identity. The actual contrast curve is anchored at the original median:

    K(Y) = p F(Y)/F(p)                         when Y <= p
           p + (1-p) (F(Y)-F(p))/(1-F(p))      otherwise.

Thus K(0)=0, K(p)=p, K(1)=1, with positive slopes. High-entropy supported brightness
intervals expand at the expense of lower-information intervals within each
side of the anchor. The two sides can have different derivatives at p. The
density cap is before normalization and is not a final slope limit of eight.

## Color reconstruction, amount and order

Each full tone stage places RGB at its requested Y along the neutral axis. If
that would leave the SDR cube, contract the original chroma vector RGB-Y(1,1,1)
by the largest common factor in [0,1] that fits. This preserves the requested
luma and chroma direction in this coordinate system. It can reduce chroma near
gamut boundaries; it does not promise perceptual hue invariance. Neutral inputs
stay neutral in the direct transform. The legacy RGB stage retains its original
clipping and its existing influence on brightness.

For every stage T, amount a means T_a(x)=x+a(T(x)-x), applied to that stage's
own input. Zero is an exact stage bypass and one applies the complete transform.
The default target is B_b(K_k(C_c(x))). All six orders are implemented. The
operations generally do not commute; independent estimation is not a claim of
independent visual effects.

Composition happens at the existing 33³ LUT nodes. The finished target LUT
receives the existing time-based interpolation and maximum-change bound, then
is packed through the existing SIMD path and rendered with the unchanged shader.
No additional full-frame passes, shader samples, capture, or readback are added.
Extra cost occurs at CPU analysis / target construction. The optional legacy
branch retains its own original LUT construction loop.

Control edits enter the next analysis update and then follow profile adaptation.
**Freeze current color assignment** continues to hold the entire LUT exactly;
chain-control edits take effect after unfreezing. Overall strength remains an
immediate final blend. HDR/float-source bypass and alpha handling are unchanged.

## Verification

`rvfx_entropy_chain_tests` checks all-six-order RGB-only float/upload equivalence,
independent source estimators, all-zero identity, analytic stage composition and
amount semantics, meaningful order changes, monotone grayscale tone curves,
midtone anchoring, endpoints, finite gamut, chroma direction, flat/transparent
inputs, and off-grid LUT error. The existing frozen-reference test still checks
255,296,448 legacy float values plus the exact packed upload tables.

The OBS/Metal smoke additionally checks new property/default registration,
all six actual GPU compositions against the CPU LUT, alpha, freeze, and
zero-amount identity. It also retains the prior filter checks.

The 33³ LUT is an approximation between vertices. On the retained synthetic
source, using RGB decorrelation 0.09/allocation 0.11 and 60,000 uniformly sampled
colors across six orders, mean/max absolute largest-channel error against direct
composition was 0.000429/0.014347 with both tone amounts at 0.15, and
0.001079/0.075446 at full tone amounts. These are normalized RGB errors, before
output quantization, not perceptual-quality scores or universal error bounds.
Strong curves and gamut boundaries can visibly expose the approximation.

Retained build/test evidence and measured costs are in
`output/entropy_stretch/chains/`. Run the root AGENTS.md entropy CMake/CTest
command on the M4 Mini; both CMake and Make include the new chain test suite.

## Retained results

- Release CTest: all four suites passed, including the frozen-reference check.
- AddressSanitizer and UndefinedBehaviorSanitizer: chain suite passed.
- Actual OBS 32.2.1 / Metal, 1920 × 1080: all six chain orders agreed with the
  CPU LUT within one 8-bit channel code, with exact alpha preservation. The
  prior identity, gamma, adaptation, freeze, bypass and color-space tests passed.
- M4 CPU, 128 × 72 synthetic source, average of 100 analysis/target builds:
  approximately 0.25 ms legacy, 0.95 ms with both tone stages active at 0.15.
  These are per-analysis-update costs, not per-video-frame GPU timings.

The extra tone work is optional and does not add shader passes. The current
33³ LUT resolution and temporal law remain shared by both modes. No new
performance guarantee for arbitrary scenes, other platforms, or HDR is implied.
