# Vowel-seeded articulatory inversion

The source inventory is no longer a collection of phone recordings or
independently rendered phone snippets. Its source material is a bounded space
of physically possible vocal-production trajectories.

## State

A trajectory has two kinds of coordinates.

- Speaker coordinates change slowly: vocal-tract length, section proportions,
  habitual fundamental-frequency range, and glottal-source character.
- Gesture coordinates change within a vocalization: tongue-body position and
  diameter, lip aperture, localized constriction position and diameter, velum,
  voicing intensity, and tenseness.

The initial forward model is a continuous-phase LF-like glottal source feeding
a 44-section Kelly--Lochbaum waveguide. Area changes alter reflection
coefficients continuously. There are no phone boundaries in the synthesizer
and no waveform concatenation.

## Acquisition from a flat vowel

1. Bound a vocalization around a persistent voiced nucleus, with limited
   context on each side. The first `daveandsimon.wav` bound is now frames
   `22:90`; `85:90` is deliberate terminal context.
2. Estimate F0 and the first three formants. Express the formants as semitone
   offsets from F0. These relative-register coordinates factor out much of the
   speaker's absolute pitch.
3. Find the most stationary voiced window. It is an acoustic fixed point, not
   yet a vowel label.
4. Search bounded vocal-tract configurations whose waveguide resonances match
   that fixed point. Keep a posterior family because the inverse is not unique.
5. Use that posterior as the speaker-conditioned origin from which gestures
   may spread backward and forward.

For Simon's first vocalization the current stationary seed is approximately
`F0=137.5 Hz`, `F1=653.7 Hz`, `F2=1209.0 Hz`, and `F3=2225.2 Hz`. The first
coarse tract solution is tongue index `12.0`, tongue diameter `2.4`, and lip
diameter `1.0`. It is a seed hypothesis, not the declaration “AO.”

## Spreading into a gesture lattice

From the seed, construct short smooth perturbations in every anatomically
allowed direction. Each proposal includes its duration, velocity, and
acceleration. A bidirectional beam retains proposals according to

`acoustic likelihood × movement prior × duration prior × continuity prior`.

The acoustic likelihood compares measured and rendered trajectories, not
static fingerprints:

- F0 and voicing continuity;
- F1--F3 register positions and their derivatives;
- resonance bandwidth and antiresonance evidence;
- energy/turbulence only when the proposed constriction predicts it;
- the Fourier-unrolled ridge field as a residual diagnostic, not the primary
  identity coordinate.

The movement prior penalizes impossible speed, acceleration, discontinuity,
and simultaneous incompatible constrictions. It does not force a unique mouth
shape when multiple shapes explain the same acoustics.

## Dictionary registration

The acoustic engine never directly emits a word. After a trajectory beam has
been constructed, dictionary pronunciations select compatible paths through a
gesture graph. A dictionary entry supplies ordering and broad manner/place
constraints; it does not supply the waveform.

For “all,” the hypothesis is one uninterrupted vocalization:

1. sustained posterior vowel configuration;
2. continued glottal excitation;
3. tongue-body retraction plus an apical/lateral constriction;
4. the resulting downward resonance motion into dark `/l/`;
5. intensity release.

Competing dictionary entries must explain the same observed motion with their
own permitted gesture sequence. Rapid pruning is possible because the stable
vowel seed fixes the speaker/register gauge before the combinatorial search.

## Known missing physics

The first model is oral and one-dimensional. A convincing `/l/` requires a
lateral side branch and its antiresonance; nasals require the existing velar
branch to be activated. These are explicit forward-model deficiencies rather
than reasons to return to waveform templates. The next refinement is a
two-path lateral junction at the tongue-tip constriction, followed by
trajectory search over the `/ɔː/ -> /ɫ/` terminal movement.
