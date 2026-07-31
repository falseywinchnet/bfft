# Meyer texture harmonic G-ball relaxation

This experiment never attempts to recognize a JPEG artifact by frequency.
The source is decomposed once:

```text
source = Meyer cartoon + Meyer texture + model residual
```

Only the texture enters a smooth, overlapping Fourier frame. For harmonic
`k` at support-grid position `p`, its actual horizontal and vertical phase
advance is measured by joint-channel complex correlation across the full
support grid. The coefficient is demodulated by that measured transport.
A genuine carrier—including the sidebands created when a smooth window sees
an off-bin frequency—then has a constant coefficient field whether it is
low-frequency, upper-Nyquist, or anywhere between. Ringing that does not
propagate with that phase remains non-ground support-grid energy.

One graph-diffusion step averages only this demodulated field. Its local gain
is continuous:

```text
gain = diffusion * (1 - phase_consistency)^power
```

There is no frequency threshold and no per-cell candidate scan. A controlled
The graph mean includes the centre coefficient, so `diffusion=1` moves an
interior coefficient only one fifth of the way to its four-neighbor mean.
The operator now exposes the full convex range `[0, 5]` without another
iteration or scan. The default is `1.5`, a 30% neighbor step.

The relaxed field is overlap-added and optionally reprojected with the Meyer identity
`P_Gmu(x) = x - ROF(x, 1/mu)`. Recomposition replaces texture only; cartoon
and the original Meyer model residual are copied exactly.

Explicit reprojection is disabled by default. The input texture already comes
from the G-ball, while a cold finite-sweep ROF projection moves it
substantially. Use `--gball-sweeps` only as a separate constraint diagnostic,
not as part of the winning default path.

The experiment also generates a clean analytic scene with a genuine
off-bin harmonic, JPEG-compresses it, and compares the relaxation to known
ground truth. This distinguishes real artifact removal from a merely smoother
appearance on the supplied card.

The default uses eight Meyer iterations. On the controlled scene the result
improves monotonically through approximately eight iterations and then
saturates: 33.80 dB at one, 33.90 dB at four, 34.02 dB at eight, and
34.03 dB at sixteen. This is texture-allocation convergence rather than an
extra spatial smoothing loop.

With the eight-iteration, `diffusion=1.5` default, the controlled JPEG
improves from 33.48 to 34.05 dB. The fitted genuine-carrier amplitude error
increases by only 0.64 percentage points relative to truth. On the supplied
card, 8.03% of Meyer-texture energy is relaxed while the decoded
block-boundary ratio falls from 1.229 to 1.163.
Because no clean source exists for that card, the latter is structural
evidence rather than a restoration-quality proof.

Run:

```sh
.venv/bin/python experiments/meyer_harmonic_gball_relaxation.py \
  /Users/quentinkuttenkuler/Downloads/1500x500.jpeg
```
