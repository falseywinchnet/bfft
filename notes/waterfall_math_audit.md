# Waterfall math audit: paused-DIP and FCT

Date: 2026-07-09

This note records the corrections made while moving the STFT-era experiments
into the streaming IQ waterfall.  It deliberately separates identities,
heuristics, and display choices.

## 1. What the FCT actually returns

For complex IQ `x[t]`, the useful family is

    C(k,tau) = sum_{t=0}^{tau-1} x[t] exp(-2 pi i k t / N).

The selector searches for a high value of `|C|^2/tau`.  The phase
`arg C = atan2(A,B)` is an observable at that selected support; maximizing the
angle itself would be amplitude-blind and dependent on the phase branch cut.

The checked-in FCT guarantees that its emitted value equals `C(k,tau_k)` at the
emitted integer `tau_k` (machine precision in tests).  Its dyadic proxy and
local refinement are a search heuristic.  They recover the planted reference
events, but do not prove a global argmax for arbitrary multimodal signals.

For complex IQ, I and Q cannot be run through two independent FCT selectors and
combined afterward: selection is nonlinear, so the channels can choose
different supports.  `fct_forward_complex` computes the complex block spectra
linearly with two real BFFTs, then makes one shared `tau` decision.

## 2. Complexity correction

The implemented dyadic pyramid rebuilds block FFTs at every level:

    sum_t (N/2^t) O(2^t t) = O(N log^2 N).

The identity

    DFT_2h([u|v])[2f]   = DFT_h(u)[f] + DFT_h(v)[f]
    DFT_2h([u|v])[2f+1] = ODFT_h(u)[f] - ODFT_h(v)[f]

is correct for constructing one parent DFT.  It does not recursively close the
pair `(DFT, ODFT)`: the parent ODFT requires quarter-bin child transforms, the
next level eighth-bin transforms, and so on.  An exact O(N log N) all-level
factorization remains open.

## 3. Waterfall timing is not STFT timing

An STFT frame owns one center time.  An FCT frame owns one origin and a
frequency-dependent endpoint `origin + tau[k]`.  Plotting every bin at the
origin or center causes false time smear.  The viewer therefore analyzes enough
preceding origins to cover the visible range, scatters each bin to its selected
endpoint, and keeps the strongest claimant for a display cell.

The displayed FCT magnitude is

    M(k) = |C(k,tau)| sqrt(N/tau).

This is the `|C|^2/tau` score on an N-point FFT power scale: white-noise level is
approximately independent of support, while coherent energy earns the expected
sqrt(tau) confidence.  `tau/N` is retained as a second measurement and shown as
a warm tint for short supports.  It is not folded into intensity.

The prefix search creates a multiple-comparisons effect.  Under a noise null,
the largest prefix score grows approximately as `log N`; the research default
activity gate `1.5 mean|x|^2` therefore produces false short supports in a
wideband recording.  The viewer uses `(log N + 4) mean|x|^2` as a conservative
false-alarm floor.  This is a display operating point, not a universal theorem.

## 4. Active-delta correction for complex IQ

The earlier experiment began from magnitudes, so PGHI supplied a phase seed.
The IQ viewer already has complex samples and measured complex STFT endpoints.
Discarding those phases and invoking PGHI invents an unnecessary phase field.
The waterfall's direct seed now uses the measured long-aperture complex
endpoints, followed by `back_to_level` and `tap3_adj`.  Magnitude-only PGHI APIs
remain available for their original reconstruction use.

The previous fine-time readout used

    L[k] * clamp(S[e,q] / max_e S[e,q], floor, 1),

which does not conserve amplitude or power and left-aligns each short bin when
it is repeated onto the long grid.  The corrected readout uses

    w[e,q] = |S[e,q]|^2 / sum_e |S[e,q]|^2
    |F[e,k]| = |L[k]| sqrt(interp_q_to_k w[e,q]).

Thus `sum_e |F[e,k]|^2 = |L[k]|^2`.  A small uniform-power mixture replaces the
old lower clamp, and circular linear interpolation centers the short frequency
grid on the long one.

## 5. What remains research

- Replace leading-only support with a two-sided `(onset, offset)` aperture.  It
  removes the silent-prefix penalty for late bursts but needs a 2-D support
  search or a suffix/interval factorization.
- Quantify the FCT selector's miss probability on adversarial multimodal prefix
  landscapes; report score ratio, not only exactness at emitted `tau`.
- Derive a statistically calibrated activity gate from a desired false-alarm
  rate, accounting for correlated prefix trials and frequency bins.
- Explore a causal taper family whose normalization is carried with each prefix.
  A fixed symmetric Hann is incompatible with early-prefix comparison, while a
  rectangle has high sidelobes.
- Find the true all-level transform reuse law.  The DFT/ODFT one-level identity
  is a clue, not closure.
- Preserve phase in a future complex active-delta readout, not merely magnitude,
  if the product needs coherent downstream detection rather than visualization.

The viewer labels the checked-in implementation **FCT support-search
(experimental)**.  It is a useful baseline and detector, but not the intrinsic
phase-space transform described by the larger research ambition.
