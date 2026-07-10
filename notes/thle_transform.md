# THLE: the truncated half leading edge transform, fixed and adaptive

Status 2026-07-09: all claims below validated to machine precision (or
measured and reported where the honest answer is quantitative).
Executable record: `experiments/thle_discovery.py` (numpy only, 0.1 s).
Source conversation: "Truncated Half Leading Edge Transform" (ChatGPT export,
2026-07-09) — its central proposals survive validation with two corrections
owned below.

## 0. The two targets

**Fixed THLE-α** — the DFT square sliced down the anti-diagonal:

    y[k] = Σ_{n=0}^{N-1-k} x[n] · exp(-2πi(k+α)n/N),   k = 0..N-1

Bin 0 has full support; bin N-1 has one sample.  α = 0 is the DFT edge,
α = 1/2 the ODFT (half-bin / BODFT) edge.

**Adaptive leading-edge transform** — the real goal.  Per bin k there is a
marching family C(k,τ) = Σ_{n<τ} x[n] e^{-iω_k n} (the cosine/sine
correlation pair B + iA; observable phase arctan(A/B)).  The desired output
is C(k, τ_k) at the slice where the bin maximally correlates
(score |C|²/τ), for **all** bins, at FFT-like cost — explicitly *not* N
sliding correlators (no DDC bank), and *not* N separate transforms.

## 1. Fixed THLE: the walk is a chirp valid-correlation (T1)

With γ = exp(-iπ/N), the quadratic completion
2(k+α)n = (n+k+α)² − n² − (k+α)² gives

    a[n] = x[n]·γ^{-n²}
    b[m] = γ^{(m+α)²},  m = 0..N-1        ← the FINITE chirp packet
    y[k] = γ^{-(k+α)²} · (reverse(a) ⋆ b)[N-1+k]      (linear convolution)

**The diagonal truncation is never applied as a mask: it is the valid
overlap of the finite chirp packet.**  This is a Bluestein-shaped butterfly
whose destination is not the Fourier identity but the diagonal
chirp-correlation identity.  Validated 1e-13 vs the direct sum for
α ∈ {0, ½}, N ∈ {256, 1024}, real and complex input.  Cost O(N log N),
one complex FFT of length ≥ 2N−1 (the house FFT).

**Non-closure of the radix walk (T2).**  The standard radix-2 combine on
shared THLE children reproduces exactly 1 of 256 bins (k = 0).  The child
support bound floor((N−1−k)/2) depends on the PARENT k, not k mod N/2, so
outputs k and k+N/2 need different children — no sharing exists.  The chirp
correlation IS the butterfly walk for this target; there is no other.

**Invertibility (T3, corrected claim).**  |det(THLE)| = 1 exactly (the
anti-diagonal pivots are unimodular), so THLE is algebraically a bijection —
*and yet* its condition number grows exponentially:

    N = 64:  cond ≈ 5.2e5      N = 128: cond ≈ 1.2e8     N = 256: cond ≈ 2.3e11

(≈ ×450 per doubling).  Back-substitution round-trips at 8e-11 (N=64),
1e-7 (N=128), 3e-4 (N=256), and diverges at N=1024; the O(N log N)
series-reciprocal inverse is exact algebra but its reciprocal-chirp
coefficients grow to ~1e9 and float precision dies.  **The leading-edge
slice genuinely erases late-sample information in floating point.**  This is
a property of the object (the ChatGPT derivation's "full rank, not merely a
projection" is true only in exact arithmetic) — practical inversion needs
small N, extra apertures, or regularization.

## 2. The jet law: how a paused walk moves on the manifold (T4)

Every parameter of the family (α, or the apex τ) enters only through packet
phases; every cell of the chirp walk is linear.  Therefore the 2-jet
(C, ∂C, ∂²C) transports **exactly** through the whole walk — channel-wise
convolution with (b, ∂b, ∂²b) and product rule at the post-chirp.  Validated
at 1e-13 against exact analytic derivatives of the direct sum (not finite
differences).  This is the concrete form of the conversation's "phase
pressure" machinery: P = Im(C̄·C_τ), Q at the stationary point, Newton step
Δτ = −P/Q — available inside the walk without re-entering the input.
(The DIP-jet variant — three coupled fields through the d/e diagonal cells —
follows the same closure argument; not separately implemented.)

## 3. The manifold theorem: paused walks tile the (k,τ) plane (T5)

**T5a (lattice identity).**  For block length L = 2^t and on-grid bins
g = f·N/L:

    C(g, m·L) = Σ_{b<m} DFT_L(block_b)[f]        exactly (4e-15 measured)

because on-grid waves have an integer number of cycles per block — the
inter-block twiddle e^{-2πigbL/N} = 1 identically.  No Dirichlet leak ON the
lattice (leak exists only off-grid; cf. the T3'-adjacent correction in
`dip_vertical_identification.md` — that correction concerned off-grid crops
and stands).

So the dyadic pyramid of contiguous-block spectra — log N levels, N values
per level — is an **exact N·log N-point sampling of the entire leading-edge
manifold**, concentrated fine-in-time where frequency is coarse and
fine-in-frequency where time is coarse.  The "one period for the final
wave" hyperbola of the original prompt is the f = 1 column of this pyramid.

**T5b (one-level BODFT identity; corrected).**  Parent spectra of a
concatenated pair [u|v] use a DFT and ODFT child channel:

    parent[2f]   = DFT_h(u)[f] + DFT_h(v)[f]
    parent[2f+1] = ODFT_h(u)[f] − ODFT_h(v)[f]        (1e-14 measured)

The half-bin transform is exactly the missing sibling channel for constructing
that **one parent DFT**.  It does not, by itself, make the channel pair
recursive: constructing ODFT_{2h}([u|v]) requires quarter-bin and
three-quarter-bin child transforms; recursively carrying those introduces
eighth-bin channels, and so on.  Therefore the checked-in per-level rebuild is
honestly O(N log²N).  An exact O(N logN) all-level tower needs a larger
fractional-channel factorization or a different reuse theorem; it is open.

## 4. The adaptive transform (the discovered object)

The answer to "which transform yields correlation-selected phase slices for all
bins without DDC work" is **not a linear transform**.  Maximizing arctan itself
would be branch-dependent and amplitude-blind; arctan(A/B) is the phase
observable at the selected support, while selection uses |C|²/tau. It is a
three-stage nonlinear
object whose linear interior is the paused walk:

1. **Manifold**: the block pyramid of §3 (walk passes, paused at every
   level).
2. **Row-shared landscapes**: per level, running block sums give the score
   |C|²/τ per (row, boundary); each row keeps its best boundary.  O(N) per
   level, shared by all N/L bins that proxy to a row.
3. **Per-bin level choice with a coherence stop**: a bin trusts a level only
   while its proxy row keeps ≥ rel · (its best score over levels).  When the
   level's frequency grid outruns the component's bandwidth the proxy dies
   — this *detected collapse* is the stopping rule, replacing blind
   descent.  The finest trusted level localizes the boundary to ±L.
4. **Activity gate**: bins under act·mean|x|² have no coherent leading edge
   and default to τ = N (plain FFT readout).  Real input walks only the
   independent half; conjugate bins mirror.
5. **Per-bin windowed refinement, anchor-shared**: one truncated transform
   per distinct window start (a handful) anchors C(:, lo) for every member
   bin; each bin then walks its own window by the endpoint recurrence
   C(k,τ+1) = C(k,τ) + x[τ]e^{-iω_k τ}, extending the window geometrically
   whenever the local argmax sits on an edge.  Exact inside the searched
   window, but not a proof against a disconnected, higher maximum elsewhere.
6. **Egress**: the refinement already holds C(k, τ_k) exactly; one full FFT
   serves the defaults.

Measured at N = 1024 vs the brute-force manifold (score-gated coherent
bins):

| signal | coherent | τ err med / p90 | score found/max med / p10 | phase err | endpoint ops | FFTs |
|---|---|---|---|---|---|---|
| events (tone + off-grid burst + short burst) | 30 | 0.0 / 0.0 | 1.0000 / 1.0000 | 0.00° | 34k (DDC = 1.05M) | 5 |
| linear chirp 50→400 | 85 | 0.0 / 0.0 | 1.0000 / 1.0000 | 0.00° | 87k | 7 |

All three planted events recovered exactly (full tone → τ = 1024, burst
ending 500 → 502 = brute optimum, 56-sample burst ending 656 → 661 = brute
optimum).  The chirp — worst case, every bin has its own boundary — still
runs 11× under the DDC bank; the events signal 30×.  Cost is
signal-adaptive: work concentrates where boundaries exist.

**Failure mode owned**: the |C|²/τ criterion charges an event for the silent
prefix before it (leading-edge support always starts at 0), so late short
events score low and gate sensitivity (act) trades detection against wasted
refinement on noise bins.  This is a property of the leading-edge criterion
itself, not the search.

## 5. Relation to the DIP

This is the DIP program run on the dual decimation axis.  DIP levels are
dyadic Zak transforms — comb supports in time, exact for record bins; the
leading-edge pyramid is contiguous-block supports in time, exact for
level-grid bins.  The pause–manipulate–continue concept
(`dip-zak-alignment-law`, ADIP proposal) is validated here in its
contiguous form: the paused states ARE the manifold, selection is packet
metadata, and the only nonlinearity is the per-bin choice of where to
egress.  The BODFT closure (T5b) says the two house primitives (Bruun real
walk + BODFT half-bin walk) are jointly exactly what a recursive
leading-edge tower needs.

## 6. Open

- Fractional (sub-sample) τ via the jet law of §2 (phase-pressure Newton at
  the chosen slice) — machinery validated, not yet wired into selection.
- α = ½ egress (half-bin adaptive variant): pre-twist x by e^{-iπn/N} and
  everything above carries through; not separately measured.
- Two-sided apertures (onset AND offset per bin) — leaves the "leading
  edge" family, meets the aperture-ladder work.
- Production kernel: discover an all-level reuse factorization (the DFT/ODFT
  one-level identity alone is insufficient), then combine it with the per-row
  landscape scan and endpoint refinement.  A Numba port remains a useful
  intermediate benchmark.
- Regularized/multi-aperture inversion of fixed THLE (its exponential
  conditioning makes the plain inverse a small-N tool only).
