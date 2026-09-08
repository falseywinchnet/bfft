# Ostensibly front-end experiments

This folder develops the non-neural speech front end from an exact numerical
oracle before attempting a fused ODFT implementation.

The first oracle is deliberately literal:

1. Hann STFT of a real waveform.
2. `numpy.fft.irfft` down the frequency axis.
3. A second default-length `numpy.fft.irfft` down the resulting row axis.
4. Retain the low-row crop and take its absolute value.
5. Build several sub-hop phase lattices, transport them into the zero-centred
   lattice with the repository's overlapping Fourier-circle atlas, and take a
   per-pixel maximum in that common gauge.
6. Keep the complete registered maximum for phone comparison.  A two-pass
   Meyer texture/cartoon cascade remains an audit experiment, but is not part
   of the promoted engine path: harmonic existence routes a candidate while
   the whole registered patch, nuisance texture included, supplies geometry.

`fourier_unrolled_densified.py` adds a multiscale aperture family without
pretending the reassigned-power image is a waveform. Each of `N=1024`, `2048`,
and `4096` first receives the complete phase-lattice fusion above. Its rows are
then mapped to the common `N=2048` physical gauge by
`r_2048 = r_N*(2048-1)/(N-1)`. Exact three-FFT Hann reassignment supplies only
cross-aperture registration landmarks; a reassigned FFT bin maps to an
unrolled row by `r_2048 = k_hat*2*(2048-1)/N`. The untouched transported
linear fields are finally combined by a literal per-pixel maximum.

That maximum is retained as a deliberately blind control.  The finite
research arm in `supported_hodge_fusion.py` carries the support-bearing idea
from the earlier finite-Zak fusion through the repository's later Meyer
jump-measure refinement.  The short aperture supplies the time derivative and
the long aperture supplies the row/frequency derivative, but only where a
second aperture independently supports the event.  This mixed vector field is
an observation, not yet an image.  Its longitudinal Hodge projection is the
nearest integrable scalar field; the divergence-free transverse remainder is
reported as contradictory evidence.  One feed-forward residual remeasurement
removes aperture blur that disagrees with the first integrable readout.  There
is no iterative optimizer and no terminal maximum in this candidate.

On the first second of `daveandsimon.wav`, the blind maximum raised the
strongest-ridge q10 by 42% but changed the measured void fraction from 4.76%
to 9.52%.  The supported Hodge arm raised q10 by 30% and the median by 25%
while retaining the 4.76% void fraction.  Its rejected transverse component
contained 4.25% of mixed derivative energy; 14.98% of pixels survived both
independent support and the finite residual remeasurement.  These are
diagnostics on one real scene, not a transcription-accuracy claim.

The visually sharper construction selected as the working phone-comparison
baseline is intentionally simpler: a 3x3 box average of the multiscale
registered maximum is the cartoon term, and the square root of reassigned
support power is placed in the same amplitude gauge by matching the 99.5th
percentile and added as texture. `reassigned_texture_baseline` is the single
implementation used by both the bounded full-recording builder and the
synthetic whole-phone battery. The N=2048 field, blind multiscale maximum,
support power, two additive components, and promoted `trace_field` remain
separately stored for audit.

The frozen 39-phone synthetic battery now includes all five ARPAbet
diphthongs.  With a 24-class metadata route, its 195 held-out synthetic cases
measure 100% route recall, 60.51% top-1, 93.33% top-3, and 98.46% top-5.  These
numbers are a synthetic geometry gate only.  They do not establish transfer to
recorded conversation.

The original harmonic-only activity gate retained 19.89 of 60.53 seconds and
emitted 241 phone regions. This is incompatible with the approximately
592--608 CMUdict phones in the two listener transcripts. The corrected builder
therefore retains harmonic existence as one witness and unions it with an
independent, two-population centered waveform log-RMS witness. The unchanged
saved trace field then retains 44.13 seconds and emits 494 phone regions. Run
that cheap resegmentation without rebuilding the Fourier field with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_resegment_recording \
  /path/to/corrected_full_recording.npz \
  /path/to/corrected_full_recording.json \
  /path/to/daveandsimon.wav --out /tmp/ostensibly_resegmented
```

`word_lattice.py` adapts the old Kolmogrov coupled history-occupancy recurrence
to phone sequences. CMUdict pronunciations are exact external anchors. The
two-coordinate hash executes only the stored `n-1`, `n`, and `n+1` radius-one
plans and can only propose candidates; complete top-k acoustic evidence is
read again by dynamic programming. A 4,096-pronunciation audit measured 100%
candidate recall for exact, substitution, deletion, and insertion probes, but
q95 candidate fanout was 1,078. This is why the address is never used as a
similarity score. Homophones remain grouped as explicit ambiguity classes.

The first independent terminal-geometry calibration uses the timed phone
labels distributed with CMU ARCTIC. Those labels were generated by the CMU
Sphinx/FestVox pipeline and were not hand-corrected, so they are an independent
benchmark rather than perfect truth. Five phonetically complementary
utterances supply 191 crops and all 39 phones per speaker. Analytic 8 dB
templates transfer very poorly to either BDL or SLT (about 4--6% top-1 and
18--23% top-5), proving that the synthetic generator is not a valid model of
recorded-phone geometry. A real BDL bank routed SLT at 94.76% but achieved only
17.28% top-1 and 50.26% top-5 even with minimum-witness matching. Routing is
therefore useful; the terminal geometry and analytic generator are the larger
bottlenecks.

`multiscale_phone_geometry.py` retains the complete normalized phone raster at
nonzero weight and embeds it together with two Gaussian scale-space views and
12 pooled complex-Gabor magnitude views. The frozen component weights are
`0.1`, `4`, `8`, and `2`. On the same-utterance BDL+SLT-to-CLB screen this raised
top-1 from 16.23% to 28.80% and top-5 from 57.59% to 66.49%. That screen was
later identified as lexically confounded because all speakers read the same
sentences. The stricter authoritative holdout uses three different CLB
utterances (116 timed phones, 38 represented phone types): raw top-1/top-5 are
11.21%/43.10%, while the unchanged multiscale geometry reaches 12.93%/46.55%.
Route recall is 89.66%. The gain survives, but is much smaller; the earlier
66.49% must not be reported as general phone accuracy. Coupling each acoustic
witness to the weak timing penalty
`0.08 * abs(log(query_frames / witness_frames))` raises the disjoint top-3 to
37.07% and top-5 to 55.17% (top-1 13.79%). The coefficient was selected on the
BDL↔SLT calibration folds and frozen before this holdout. This confirms that
duration is useful evidence, not merely routing metadata.

An untouched fourth speaker, RMS, was then evaluated on the same three
sentences without changing the bank, descriptor, or duration coefficient. Its
raw, multiscale/Gabor, and duration-coupled top-5 recalls are 38.79%, 47.41%,
and 54.31%; duration-coupled top-3 is 41.38%. RMS route recall is 83.62%
versus 89.66% on CLB. The close agreement of the two disjoint-speaker results
supports a measured generic top-5 level of roughly 54--55%, while also showing
that routing is now a secondary bottleneck.

The activity/boundary audit separates missed support from misplaced cuts. On
CLB, the original activity gate proposes only 45 regions for 116 labeled
phones and covers 43.5% of phone centers. At a ±40 ms tolerance, its boundary
precision/recall/F1 are 86.8%/49.6%/63.1%: detected cuts are usually near a
real boundary, but most boundaries never become eligible. A development
setting that accepts 40 ms evidence, bridges 250 ms gaps, and pads support by
40 ms raises CLB to 84 regions, 84.7% center coverage, and 67.6% boundary F1.
Frozen on RMS, it raises center coverage from 39.9% to 82.3% and boundary F1
from 62.9% to 66.7%. The activity fix generalizes; novelty-based splitting
remains undercomplete.

A separate calibration lane freezes the BDL+SLT real-speech witnesses and
matches Dave/Simon regions without adapting to either speaker. The original
494-region segmentation's
optimistic monotone forced-alignment ceiling with duration coupling reaches
77.57% top-5 against the Gemini listener transcript and 78.74% against the SOL
transcript, versus
61.66% and 60.85% for the earlier synthetic-bank baseline. This is not
independent accuracy because the aligner may select favorable timestamps. The
resulting ambiguity-preserving word lattice contains 2,831 edges. Its consensus
landmark coverage does not improve monotonically with the phone ceiling and
the retained words still occur at implausible times. No sentence path is
selected. The next research target remains recorded-phone invariance and
boundary support, not a wider dictionary search.

Applying the independently developed activity setting to the saved
Dave/Simon field yields 559 regions and 54.10 active seconds, versus 494 and
44.13 seconds previously. It increases matched forced-alignment pairs but does
not materially improve top-5 or landmark timing, so region count alone is not
the remaining answer.

Build the candidate lattice and run the ceiling audit with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_word_lattice \
  /path/to/radio_whole_patch_top_k.json /path/to/cmudict.dict \
  --out /tmp/ostensibly_word_lattice

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_reference_phone_audit \
  /path/to/radio_whole_patch_top_k.json reference_transcripts.md \
  /path/to/cmudict.dict --out /tmp/ostensibly_reference_audit
```

Run the frozen generic-bank matcher after producing ARCTIC fingerprints with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_generic_multiscale_radio_match \
  /path/to/radio_query_fingerprints.npz \
  /path/to/resegmented_recording.json /path/to/arctic_calibration \
  --training-speakers bdl,slt --out /tmp/ostensibly_generic_radio
```

Run and render the one-second comparison with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  experiments/ostensibly_frontend/run_fourier_unrolled_densified.py \
  /path/to/daveandsimon.wav --out /tmp/ostensibly_unrolled_densified
```

For `N=2048`, the second inverse has `4094` rows and the current speech crop is
the first `256` rows.  The displayed arrays are vertically flipped so row zero
is at the top.

Run the first-second probe with:

```sh
PYTHONPATH=. python3 experiments/ostensibly_frontend/run_frontend_probe.py \
  /path/to/daveandsimon.wav --out /tmp/ostensibly_frontend
```

The input to the Meyer operator is linear amplitude normalized to `[0,255]`.
Logarithmic scaling is used only while rendering the diagnostic image.

The original single-split probe is retained as a failed control.  Run the
corrected first-second experiment with the SciPy/BFFT environment, then render
its arrays with the plotting environment:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  experiments/ostensibly_frontend/run_uncertainty_cascade.py \
  /path/to/daveandsimon.wav --out /tmp/ostensibly_uncertainty

PYTHONPATH=. python3 \
  experiments/ostensibly_frontend/render_uncertainty_cascade.py \
  /tmp/ostensibly_uncertainty/uncertainty_cascade_arrays.npz \
  --out /tmp/ostensibly_uncertainty
```

`trace_geometry.py` applies a vertical top-hat to the cartoon, extracts
persistent connected ridges, and reuses the PDF glyph experiment's
signed-distance/Fourier-circle descriptor.  The accompanying registration
probe measures invariance to gain, polarity, STFT-window phase, and noise:

```sh
PYTHONPATH=. python3 experiments/ostensibly_frontend/run_registration_probe.py \
  /path/to/daveandsimon.wav --out /tmp/ostensibly_frontend
```

The full-recording runner computes the exact low rows in bounded column blocks,
applies one global Meyer split, detects persistent-ridge speech intervals, and
proposes phone-sized boundaries from changes in normalized cartoon geometry:

```sh
PYTHONPATH=. python3 experiments/ostensibly_frontend/run_full_recording.py \
  /path/to/daveandsimon.wav --out /tmp/ostensibly_full
```

`run_vowel_inventory.py` is the first reference-matching lane. It synthesizes
English hVd/hVt vowel carriers with several installed macOS voices, extracts a
central vowel nucleus, frequency-registers its lowest persistent ridge, and
emits a top-k vowel set for every proposed phone interval. It is intentionally
reported as vowel-only evidence, not as a transcript.

`run_direct_vowel_inventory.py` removes the carrier-crop assumption by using
the Speech Manager's explicit `[[inpt PHON]]...[[inpt TEXT]]` mode. Its radio
hypotheses are gated off unless leave-one-voice-out classification exceeds 60%
with a positive median nearest-impostor margin.

Modern `say` voices did not pass that gate and produced implausibly long
"single-phone" recordings, so `run_acoustic_inventory.py` supplies the valid
fallback: deterministic glottal excitation through declared formant resonators,
several pitch/vocal-tract probes, and an explicit 250–3000 Hz SSB-like channel.
It is synthetic test geometry, not a trained speech generator.

`run_acoustic_radio_match.py` freezes two nuisance realizations as the
reference witnesses, validates on a third, and only then emits ranked vowel
sets for the radio phone proposals. The gate is defined on held-out top-3
recall because the downstream design consumes top-k sets rather than hard
phone labels.

`synthetic_consonants.py` adds explicit stop closure/burst, affrication,
frication/voicing, nasal antiresonance, and liquid/glide resonance probes. The
separate `run_consonant_inventory.py` gate measures them before they may enter
the radio lattice.

## Accepted occupation-cloud baseline

The current geometry lane begins at the weighted Carmona-style stochastic
occupation field in `crazy_climber_geometry.py`.  It keeps the first 128
unrolled rows, linearly interpolates them at 4x row and 8x frame resolution,
and deterministically lifts square-root occupation mass into 32,768
`(row, frame, height)` points.  The square root is the Hellinger amplitude of
the empirical occupation measure; it retains weak recurrent harmonics without
forming a binary mask or allowing one bright ridge to consume the whole cloud.

`occupation_point_cloud.py` registers a candidate to a reference with only
five declared degrees of freedom: row scale, frame scale, row shift, frame
shift, and linear pitch drift.  It does not resize a raster or permit a free
3-D rotation/nonlinear warp.  Reproduce the first BDL `/R/` versus Dave/Simon
`/R/` probe with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_first_phone_cloud_fit \
  experiments/ostensibly_frontend/out/sparse_gabor_first_pair/sparse_gabor_first_pair.npz \
  --out experiments/ostensibly_frontend/out/occupation_point_cloud_first_pair
```

The first measured pair reduces the robust symmetric Chamfer residual from
`1.5989` to `0.6760` (57.7%).  Its fitted time scale is `0.7977`, close to the
11/14 duration ratio; row scale is `0.8979` and pitch drift is `-0.0298`
row/frame.  This is a successful registration probe, not yet evidence of phone
discrimination: the next gate must compare the positive `/R/` fit against
identically segmented non-`/R/` references and same-phone examples from more
speakers.

### Discrimination and bottleneck audit

That gate falsified hard identity matching from one occupation patch.  The
duration-matched BDL battery contains 70 witnesses spanning 39 phones and does
not use geometry while selecting references.  It also reproduces the saved BDL
positive-control trace byte-for-byte before scoring anything.

The original 3-D Chamfer fit ranks `/R/` 25th of 39 labels; the exact purported
positive is 48th of 70 instances.  Distance has Spearman correlation `0.803`
with reference height mean because occupation was encoded both as point density
and as z height.  Removing z leaves `/R/` 26th, proving that local
nearest-neighbor coverage is also a shortcut.  Sliced Wasserstein preserves
empirical mass and explicitly distinguishes clouds with identical support but
different multiplicity, yet it ranks the radio interval's `/R/` hypothesis
34th.

The radio hypothesis itself is not a valid positive: `0.0747--0.2240 s` is the
first of six consecutive activity regions covering `0.0747--0.8000 s`, the
opening of “All right, okay,” rather than an independently aligned `/R/` crop.
The decisive control therefore queries SLT's labeled second `/R/` in the same
ARCTIC sentence against BDL.  Sliced Wasserstein ranks `/R/` 21st and 58 of 70
affine fits hit a bound.  The remaining mismatch is not global pitch or time
scale; it is speaker-dependent nonuniform row geometry.

`marginal_copula_cloud` constructs the empirical occupation copula by replacing
row and frame coordinates with their marginal CDF values.  This removes
arbitrary monotone frequency/time warps without using a phone label.  It reduces
the same SLT `/R/` rank to 15th and removes spread/height correlations, but the
corresponding same-sentence BDL `/R/` is still 48th of 70.  The accepted
occupation field is therefore useful evidence, but one isolated field is not a
speaker-invariant hard phone identifier.  The next system gate must retain
top-k uncertainty and combine relational trajectory evidence across neighboring
phone regions; no word decoder should consume a hard label from this matcher.

The immediate-context follow-up keeps the left, center, and right occupation
copulas in ordered time lanes and separately measures the row-quantile
transport laws at both phone boundaries.  On the controlled SLT `/R/`, center
evidence ranks `/R/` 13th of 38 labels, ordered context 12th, and the two
boundary laws plus center 8th.  An empirical worst-channel percentile happens
to rank `/R/` 3rd, but the nine-phone holdout falsifies it as a general fusion:
its median label rank is 11 and its median exact-correspondence rank is 14.
Independent evidence channels are proposal mechanisms, not commensurate terms
in one terminal distance.

`unique_support_cloud` supplies the complementary geometric comparison.  It
quantizes the copula at `256 x 256`, removes stochastic-sample multiplicity,
and retains cells supported by at least two samples.  Its symmetric Gaussian
miss cost is the analytic limit of superimposing many random two-cell pitch/time
offsets.  This prevents energy density from becoming the score while tolerating
small registration errors.  Across nine fixed SLT-to-BDL phone queries it has
median label rank 7, four top-5 hits, and six top-10 hits.  It is particularly
useful for `/P/`, `/S/`, and `/K/`, but ranks `/R/` 26th; it is complementary
rather than universal.

The accepted interface is consequently a provenance-preserving proposal union.
With center, ordered-context, boundary-transport, and support-geometry channels,
top-5 per channel retains 6 of 9 targets in a median 13-label set.  Top-8 retains
9 of 9 in a median 19-label set.  This is enough to hand uncertainty to a
phoneme-sequence or word-level geometric comparator, but it is not evidence for
a hard per-phone recognizer.  The fixed audit is saved as
`out/occupation_context_battery_slt_r/multi_query_context_audit.json`.

### Provenance lattice and shared-gauge word geometry

`word_lattice.py` now lifts each phone interval into a provenance-preserving
rank map instead of collapsing channel distances.  Its lexical semiring orders
uncovered phones, edit count, worst proposal rank, total proposal rank, and
supporting-channel count lexicographically.  A positional pronunciation index
routes the complete pinned CMUdict inventory without exhaustively scoring it.
On the initial matched-sentence audit, route-and-score takes `0.8--10 ms` after
a one-time `0.5--0.8 s` index build.

Whole-word occupation geometry must use one shared frequency gauge.  Applying
an independent copula to every phone destroys between-phone pitch/formant
motion.  With time canonicalized inside each phone but frequency ranked jointly
over the complete word, the exact cross-speaker BDL spans for “followed,”
“line,” and “railroad” rank first among all same-length spans.  The two-phone
“the” is a counterexample at rank 94; short spans do not provide enough
relational structure for this gauge to be universal.

The first synthetic word comparator superimposes equal-mass raw examples for
each proposed phone and scores the routed pronunciation by shared-gauge sliced
Wasserstein distance.  `8 x 128` points per phone and 32 projections preserve
“railroad” rank 4 while reducing its terminal scoring time from `7.87 s` to
`0.99 s`; smaller forms make the rank unstable.  Full resolution does not fix
the remaining word errors, so point count is not the active bottleneck.

The original five-sentence boundary atlas contained BDL's recording of the
query sentence.  That is useful as a cross-speaker geometry control but is not
an honest generic-inventory evaluation.  The leave-sentence-out audit shows
the real coverage constraint: in the remaining four sentences, “followed” and
“railroad” have no observed target diphones and “line” is missing one.  Their
exact indexed routes therefore disappear.  Across the 119 locally available
non-query BDL sentences, all target diphones are present (762 distinct observed
diphone types), so this is inventory sparsity rather than a dictionary failure.

Expanding the measured bank to 14 non-query sentences makes “followed,” “the,”
and “line” routable, but unconditional synthetic references rank only 27/231,
12/37, and 23/70.  Constructing references from the best observed boundary
witnesses improves the first two while hurting “line.”  Selecting witnesses
with full shared-gauge diphone SWD gives target ranks 9, 5, and 34.  “Line” is
beaten by plausible endpoint alternatives such as `W AY N` and `L AY L`.
Consequently the current bottleneck is context-conditioned generic reference
construction and endpoint phone ambiguity, followed by inventory coverage.
It is not SWD runtime, cloud resolution, or CMUdict lookup.  The relevant
leave-out artifacts are `out/occupation_word_span_*_14_pair_swd.json` and
`out/synthetic_word_geometry_*_14_pair_swd_witness.json`.

Overlap-consistent diphone stitching fixes the reference-construction failure.
Each selected diphone is treated as a local frequency chart.  Adjacent charts
are registered by a positive affine fit between the shared phone's row
quantiles; the two shared-phone measures are then merged at equal mass before
one final whole-word gauge.  Under the 14-sentence boundary leave-out this
moves “followed” from rank 9 to 4 and “line” from 34 to 3.  “The” remains rank
6 because a two-phone word has no overlapping chart to reconcile.

The phone proposal results above still forced the corresponding BDL phone from
the query sentence.  `run_occupation_context_battery.py` now supports
`--exclude-query-reference` and a shared external cache.  With the complete
phone stage rerun against the 14 non-query sentences, several targets fall well
outside top 8: the worst best-channel ranks are 10 for “followed,” 13 for “the,”
13 for “line,” and 33 for one of “railroad”'s `/R/` intervals.  The fully
disjoint proposal baseline is therefore top 24 per channel plus one uncovered
phone.  The uncovered phone is recovered by exact boundary proposals rather
than widening every position to top 33.

A fixed top-128 cap on every full-diphone SWD atlas is the main runtime gate.
It retains every development target while cutting the routed sets to 227, 48,
127, and 82 candidates for “followed,” “the,” “line,” and the railroad
occurrence-depth oracle.  Their whole-word ranks are 6, 9, 5, and 13.  The
untouched internal word “for” is the complete disjoint holdout: its `F AO R`
homophone class ranks 5 of 160 by whole-word SWD and 2 by both mean and
worst-diphone SWD.  These are useful word lattices, not final hard word labels.
Artifacts are `out/synthetic_word_geometry_*_fully_disjoint.json` and
`out/synthetic_word_geometry_for_holdout.json`.

The active correctness bottleneck is now disjoint phone-proposal recall and
rare-context diphone depth.  Utterance-edge handling is no longer an open
failure: the context battery substitutes a deterministic zero-mass silence
chart on the absent side and applies the same left/center/right topology to
every reference.  On the untouched sentence-final word “chances”
(`CH AE N S AH Z`), terminal `Z` is admitted at the fixed top-24 cutoff and
the whole phone sequence survives the one-uncovered-phone rule.  The target is
nevertheless rejected before geometry scoring because `CH-AE` has no witness
in the 14-sentence disjoint boundary bank; its other boundary ranks are
3, 3, 31, and 38 under the unchanged top-128 cap.  This separates the edge
problem from the actual remaining coverage problem.  The artifacts are
`out/occupation_word_span_chances_holdout_pair_swd.json` and
`out/synthetic_word_geometry_chances_holdout.json`.

The synthetic word runner records a per-target route audit: dictionary-anchor
presence, per-phone channel ranks, phone-stage admission, each boundary rank,
and the exact stage at which a target disappears.  A null target therefore no
longer conflates missing dictionary coverage, proposal recall, boundary depth,
or geometric ranking.  SWD now has an exact preprojected representation, and
the word scorer compiles each query and all of its lane windows once.  On the
unchanged 155-candidate “chances” route this reduces scoring from `9.09 s` to
`4.65 s` (48.8%) while preserving the direct distance bit-for-bit; compiling
the query projections takes `0.026 s`.  `run_compile_boundary_atlas.py` now
writes the generic reference representation once, and
`run_query_boundary_atlas.py` emits the same boundary-ranking ABI consumed by
the word scorer.  The 14-sentence BDL atlas contains 439 diphone occurrences
and 260 pair types in a 30.2 MB fixed-quantile file; it builds in `2.76 s`.
All five “chances” boundaries query in `0.054 s`.  Against the full
4,096-point diagnostic, every top-128 set and all 260/260 selected witnesses
agree at every position; Spearman rank correlation rounds to 1.00000.  Four
finite target ranks change only from `[3, 3, 31, 38]` to `[4, 4, 32, 38]`.
The compiled artifact and compatible query are
`out/bdl_14_boundary_atlas_q256_p64.npz` and
`out/compiled_boundary_query_chances.json`.

The remaining runtime component is sentence-path decoding over successive
ambiguity-preserving word edges.  Sentence context must choose among the
retained word lattice; forcing an isolated top-1 word would discard measured
uncertainty.

That sentence-path component is now implemented by `sentence_decoder.py` and
`run_decode_word_lattice.py`.  It is an acoustic-only k-best DAG: global paths
beat greedy edges, homophone sets remain grouped, disconnected regions become
explicit `<unk>` spans, and punctuation comes only from measured time gaps.
The existing 60-second radio lattice (494 proposed phone regions and 2,831
word edges) decodes 16 complete paths in `6.19 s` with no uncovered regions.
Its best path is nevertheless 209 mostly nonsensical words.  Path search is
therefore not the current accuracy bottleneck.

`run_transcript_phone_alignment.py` aligns both supplied listener transcripts
to uncertain phone regions without assuming timestamps.  The old
multiscale/Gabor-duration matcher has only 10.4--11.6% aligned top-1 phone
recall and 51.3% top-5 recall, with normalized edit cost about `0.61`.
Reclassifying the identical regions through the compiled marginal-copula
occupation atlas raises top-5 only to 54.2% and improves edit cost to
`0.566--0.584`; top-1 stays near 10.7%.  Adjacent-region boundary geometry
raises the best top-5 result to 58.3% and lowers the best edit cost to about
`0.545`.  This is useful but not sufficient sharpening.

At the frozen top-24 proposal budget, the proposal-conditioned listener
alignment can place a listener phone in 98.4--98.6% of regions.  This is a
useful routing upper bound, but later controls show that it is not independent
recall: the dynamic program chooses its mapping using those same proposals.
`run_stream_word_lattice.py` now routes those top-24 center/boundary proposals
through top-128 diphone postings, including one insertion/deletion plan for
the measured region-count deficit.  On a 100-region pilot, 899/900 spans are
routable, but median lexical fanout is 1,751 classes (maximum 8,846).  Actual
boundary ranks substantially improve some listener words—`anyway` 44→1,
`nice` 38→6, `you` 7→1, and `hand` 4→1—but others remain badly confused:
`okay` ranks 1,162 and `hold` 522.  Retaining 64 lexical classes and sweeping
the word-fragmentation penalty still puts zero listener anchors in the best
global path.  The active bottleneck is therefore acoustic rank concentration
and whole-word geometric rescoring, not phone support, dictionary routing, or
sentence decoding.  The next controlled experiment is compiled whole-word
geometry on transcript-aligned spans before it is allowed into streaming
selection.

The relevant artifacts are
`out/generic_multiscale_duration_word_lattice_39/sentence_paths.json`,
`out/generic_multiscale_duration_radio_39/transcript_phone_alignment.json`,
`out/bdl_14_phone_atlas_q256_p64.npz`, and
`out/occupation_radio_phone_atlas_39/`.

### Radio whole-word falsification and active bottleneck

The transcript-aligned whole-word control falsifies the proposed terminal
shared-gauge SWD.  On the first 20 Gemini listener words, stitched generic
diphone witnesses recover only 3 top-5 targets; a four-witness superposition
does not improve that count.  Adding an independent SLT atlas increases the
average isolated-phone top-1 recall from 10.7% to 12.7% and raises the best
two-listener top-5 fusion from 55.8% to 59.9%, but fixed-span whole-word
retrieval falls to 1/20 top-5.  Reference population and one-speaker bias are
therefore measurable secondary effects, not the missing terminal invariant.

Global witness coherence does not repair the result.  A dynamic program over
eight diphone occurrences per boundary penalizes the affine-registration
residual between the two versions of each overlapping phone.  It still yields
only 1/20 top-5 and takes 41.6 seconds.  Permitting the existing one-phone
lexical edit in geometry makes all 20 targets scoreable but yields zero top-5
hits.  Exact phone-count conditioning had been hiding the failure rather than
solving it.

The alternate waveform-supported segmentation contains 559 regions and covers
54.1 seconds, versus 494 regions and 44.1 seconds in the active lattice.  Its
count independently agrees with the 563--564 listener phones and improves
top-24 normalized alignment edit cost from about 0.142 to 0.105, but its
shallow phone recall and whole-word rank are worse: 11--13 surplus regions
remain and the first-20 whole-word audit is again only 1 top-5.  Segmentation
must eventually remain a lattice, but it is not the terminal accuracy
bottleneck.

The strongest negative control compares identically formed whole-word clouds
inside the radio recording.  Under the Gemini alignment, 87 repeated word
tokens produce 1/7/23 top-1/top-5/top-20 retrieval hits; exact random
expectations are 1.35/6.54/23.15.  Affine frequency/time gauging temporarily
raises this to 4/13/32, but the independently worded SOL alignment reverses
the ablation.  Conditioning both listeners on exact phone/region cardinality
leaves both copula and affine variants at chance.  The listener word
alignments themselves agree on only ten exact region spans and no repeated
word, so these are representation falsification controls rather than a
timestamp-grade benchmark.  The raw word clouds and all variants are in
`out/occupation_radio_phone_atlas_39/listener_word_*`.

Continuous lexical composition is useful but insufficient.  Ranking every
same-length CMUdict pronunciation by robustly normalized center-phone and
diphone-boundary distances gives 10/108 combined top-5 targets, versus 4/108
for isolated centers, but median target rank is still 259.  The full
39-distance centroid idea also fails: a cross-speaker BDL/SLT profile atlas
scores only 6.6% top-1 and 33.9% top-5 under leave-one-out on its own 906
generic fragments, then degrades radio top-5 recall to 43--45%.  The complete
distance vector is not a phoneme-invariant coordinate.

These controls locate the active bottleneck in the terminal geometry itself.
The occupation representation is an excellent proposal generator: listener
phones remain present about 98% of the time at depth 24.  Neither whole-cloud
SWD, affine marginal density, density-free global support, coherent diphone
stitching, nor distance-profile centroids concentrate that support into word
identity.  The next terminal experiment must compare ordered local ridge
topology across phone/diphone/triphone windows, with explicit edit transport,
rather than another global empirical-measure summary.

The first ordered-local replacement is now measured.  A conditional ridge
signature divides each phone into 32 physical-time slices and records 16
frequency-occupation quantiles per slice, plus its time-mass profile and local
trajectory derivative.  It permits only a two-slice shift.  On all 453
SLT-to-BDL generic phones it improves the compiled copula-SWD control from
47/168 top-1/top-5 and median rank 9 to 52/191 and median rank 7.  The combined
BDL+SLT atlas is 0.81 MB and queries all 494 radio regions in 2.64 seconds.  It
also transfers: listener top-5 rises from 52.7% to 56.1% for Gemini and from
56.1% to 59.9% for SOL.  Top-24 recall is lower, so this is a sharper local
proposal address, not a replacement for the deep copula safety net.

The same construction improves generic diphone ranking from 45 to 59 top-5
hits over 439 SLT-to-BDL boundaries, but that gain does not transfer to radio:
its two-listener boundary top-5 averages 56.0% versus 57.4% for boundary SWD.
Consequently conditional-phone, copula-phone, and SWD-boundary evidence remain
separate.  On one fixed top-24 transcript alignment their top-8 union retains
62.3%/68.6% of listener phones with median fanout 14; per-channel global
realignment had materially overstated shallow recall.  The artifacts are
`out/bdl_slt_28_conditional_ridge_t32_q16.npz`,
`out/bdl_slt_28_conditional_boundary_t32_q16.npz`, and
`out/occupation_radio_phone_atlas_39/transcript_top8_proposal_union_audit.json`.

A resolution ablation does not justify replacing the 32-by-16 signature.  A
16-by-8 form is 3.2 times smaller and classifies the 494 radio regions in 0.80
seconds, but its reverse-speaker result is mixed and radio top-5 falls from
56.1/59.9% to 53.6/53.5%.  Rank-fusing the two resolutions also fails to
recover the loss.  The 32-by-16 conditional phone channel remains the active
geometric baseline.

Duration now has the deliberately narrower role requested by the experiment:
it decides what is plausible to compare, never how two ridge geometries are
scored.  `phone_duration_gate.py` stores all robust log-duration occurrences;
the two-speaker inventory is only 3.3 KB.  On timestamped cross-speaker phones,
admitting the 24 nearest duration envelopes before preserving the unchanged
conditional ordering improves both directions: 60/177 becomes 64/185 and
52/191 becomes 54/195 top-1/top-5.  Retaining both the ungated and admitted
conditional proposals as distinct provenance channels raises the fixed radio
top-8 union from 62.3/68.6% to 66.4/72.3%, at a median fanout cost of 14 to 16.
The compiled artifacts are `out/bdl_slt_28_phone_duration_gate.npz`,
`out/occupation_radio_phone_atlas_39/radio_duration24_conditional_ridge_compiled.json`,
and `out/occupation_radio_phone_atlas_39/transcript_top8_four_channel_duration_union_audit.json`.

The listener audit has also exposed a more important epistemic limitation.
Aligned directly to one another, Gemini and SOL agree on 435 of 512 paired
phones (85.0%), including exact runs as long as 99 phones.  When each sequence
is independently aligned through the acoustic proposal lattice, however, they
assign the same phone to only 62 of 494 regions (12.6%).  Therefore neither the
98% depth figure nor small changes in per-channel globally realigned recall are
timestamp-grade ground truth.  A duration-and-order-only attempt produced 149
co-located consensus anchors at one setting but was unstable under nearby
weights, so it is retained as a diagnostic rather than promoted.  Timestamped
ARCTIC cross-speaker tests remain the clean representation assay; radio
proposal unions are reported only on one frozen mapping, with fanout, until
there are independent human timestamps or a stable non-acoustic alignment.

### Timestamped sentence bottleneck assay

`run_arctic_sentence_assay.py` now measures the complete acoustic-to-word
collapse on fourteen official ARCTIC prompts with authoritative phone
timestamps.  The prompt pronunciations and labeled regions have identical
counts for all 453 phones; eight sentences match CMUdict exactly and the other
six differ by one alternate phone choice.  Each direction uses only the other
speaker's conditional, copula, boundary, duration, and raw-occurrence banks.
This is the primary non-circular development assay; radio transcripts are no
longer used to tune terminal geometry.

The clean result confirms that proposal support is genuinely deep.  At eight
proposals per channel, the provenance-preserving phone union recalls 375/453
(82.8%) BDL phones and 353/453 (77.9%) SLT phones.  At depth 24 it recalls
453/453 and 452/453.  The original hard boundary intersection nevertheless
admits only 89/128 and 83/128 true words.  `query_aligned` now accepts a bounded
one-boundary uncertainty budget and records the missing transition in
`RelationalSequenceCost`; this recovers 105/128 and 98/128 words without hiding
the uncertainty.  A positional-phone posting screen has also been restored
before candidate scoring.  It is an exact acceleration: the final candidate
set is unchanged.

Routing recall is not terminal accuracy.  With one uncertain boundary, the
current route ordering produces only 8/31 and 11/29 top-1/top-5 words in the
two directions.  Continuous sums of conditional phone distances are worse,
as is worst-local lexicographic ordering.  This falsifies the hypothesis that
integer ranks merely discarded a useful margin.

The decisive positive control compares complete timestamped word surfaces.
An opposite-speaker recording of the same word retrieves 25--29 of 128 words
top-1 and 65--72 top-5 among the 90 prompt word types.  Thus the ordered ridge
field does contain word identity when one global frequency gauge is applied
after the phones form a word.  Independent isolated-phone normalization had
destroyed that relation.

`generic_word_template_bank.py` implements the generic equivalent.  It stores
512 raw points for every phone occurrence (1.5--1.6 MB per speaker), chooses
complete deterministic phone realizations, applies one word-level frequency
gauge, and only then forms the conditional surface.  Four realizations over a
bounded top-128 routed pool do not replace the route ordering, but provide a
repeatable independent terminal proposal channel.  Route/generic top-1 union
improves from 8 to 15 and 11 to 14; top-5 union improves from 31 to 40 and 29
to 33.  The combined two-speaker bank is 3.1 MB at
`out/bdl_slt_28_generic_word_template_bank_p512.npz`.

`run_stream_word_lattice.py` can now retain route and generic-word winners
with separate ranks, distances, and `proposal_channels`; it does not invent a
fusion score.  A 20-region radio ABI/timing pilot routes 78 spans and builds
3,471 distinct pronunciation templates successfully.  It currently takes
35.4 seconds, of which 24.6 seconds is generic template construction.

`generic_word_signature_cache.py` now persists those compiled conditional
surfaces under a bank fingerprint and exact realization/resolution ABI.  A
float16 cold build and warm reload produce identical routed edges; float16 also
preserves all 78 former float64 proposal sets.  The 3,471-pronunciation cache is
32.4 MB.  Warm generic verification falls from 23.57 to 4.57 seconds (5.2x),
and total routing falls from 33.85 to 15.70 seconds.  The equivalent mean-
agreement stream takes 13.57 seconds, including 3.88 seconds of generic
verification.

The realization experiment distinguishes two meanings of superposition.
Taking the minimum distance over four independent generic words gives
9/17 and 8/15 top-1/top-5 words in the two timestamped directions.  Requiring
mean distance agreement improves both directions to 10/18 and 10/16.  A
literal equal-weight Wasserstein quantile barycenter is mixed at 8/16 and
9/18; median aggregation is similarly asymmetric.  The streaming verifier
therefore defaults to mean realization agreement, while exposing minimum,
median, and barycenter modes explicitly.  This is an evidence aggregation over
repeated complete word realizations, not a fusion of the route and geometric
channels.

The active runtime target is now span admission and route enumeration.  The
active accuracy target is the generic equivalent itself: real opposite-speaker
whole-word surfaces retrieve roughly 66--72/128 top-5, while generic words
reach only 16--18.  A simple cross-channel selector can improve top-1 to
12/128 in both directions, but that is not sufficient to choose a sentence.

The first direct generic-equivalent correction is coarticulation-aware
realization.  Version 2 banks retain each occurrence's immediate phone
neighbors.  A proposed word may condition interior phones on both known
neighbors and edge phones on the one inward neighbor it legitimately knows;
one-phone words remain unconditional.  This avoids falsely treating unknown
cross-word context as silence.  In the controlled 14-sentence assay, 199
interior phones have exact two-neighbor support, 248 word-edge phones have
one-neighbor support, and six one-phone words have none.

With mean realization agreement, contextual pools improve generic retrieval
from 10/18/32 to 14/24/44 and from 10/16/29 to 12/24/39 at
top-1/top-5/top-20.  Route/contextual-generic union ceilings are 17/42/69 and
17/41/61.  The contextual radio cache remains deterministic, occupies 31.8
MB for the 3,471 pronunciations touched by the 20-region pilot, and routes warm
in 14.55 seconds (4.14 seconds of generic verification).  This promotes
context-conditioned mean agreement to the current generic baseline.  The
remaining caveat is deliberate: these controlled reference inventories use
the same prompt set in the opposite voice, so the next generalization assay
must compile context from disjoint utterances.

A first disjoint-utterance control uses 610 BDL occurrences from 19 recordings
that share none of the 14 SLT query prompts.  The compiler can now build missing
step-3 clouds transiently per utterance and retain only the 512-point bank
occurrence, avoiding a large raw-cloud cache.  On this matched bank, mean
agreement changes from 8/17/38 unconditioned to 11/20/32 contextual; median
changes from 10/17/31 to 12/22/34.  Thus neighbor conditioning improves early
precision outside the memorized prompts, but hard maximum-context selection
can reduce deeper recall.  The support audit explains why: only 61 query
phones have exact two-neighbor inventory support, 295 have one neighbor, and
92 have none; 53 of 127 representable words contain at least one unsupported
phone.  The next generic-equivalent target is therefore a support-weighted
mixture of exact, one-sided, and unconditional realizations rather than hard
replacement by the deepest available pool.

That deterministic 2/1/1 exact/one-sided/unconditional mixture is now an
explicit cache policy.  On the disjoint assay its mean rule scores 11/18/38:
it preserves hard-context top-1 and unconditional top-20, but gives up two
hard-context top-5 hits.  The three generic policies are therefore retained as
provenance, not collapsed into a fabricated universal score.  Their mean
top-1/top-5/top-20 proposal union ceiling is 12/24/47; adding the route channel
raises that ceiling to 18/39/61.  Median-policy union is 14/26/44, or
20/41/61 with route.  Selecting among these proposals remains the terminal
accuracy bottleneck.

### Fragment-tolerant, support-stratified word admission

The radio marker sequence oversegments real phones: the known “okay” interval
at regions 6–12 contains six fragments for the three-phone pronunciation
`OW K EY`.  `query_grouped_aligned` now enumerates contiguous region-to-phone
partitions for candidate admission while leaving the complete, unmerged cloud
for geometric verification.  Grouping alone is insufficient because K and EY
are absent from every retained per-region top-8 list.  Allowing two uncovered
phones and two uncovered transitions admits “okay,” but only at global route
rank 13,933 (rank 434 inside its weak-support stratum).

`GenericPronunciationCatalog` is the efficient second gate.  It stores
low-resolution whole-word signatures and vectorizes exact shifted conditional
ridge distance without consulting the proposed phone labels.  The current
3-phone catalog contains 3,725 pronunciations, compiles in 5.84 seconds, and
queries in about 10 ms.  For the known interval, “okay” is coarse rank 128
globally but rank 8 inside its marker-support stratum.  Full `64 x 24`
verification places it rank 16 in that stratum.  The stream lattice therefore
retains geometry winners per `(phone length, uncovered phones, uncovered
transitions, edit count)` stratum instead of adding incomparable marker and
geometric scores.  With a top-20 stratum cutoff, the correct “okay” edge is now
present with `proposal_channels = ["generic_word:unconditional"]` and no false
route provenance.

This establishes the next measured bottleneck: the correct acoustic word is
admissible, but the interval retains 225 ambiguity classes across support
strata.  Terminal lexical/sequence selection must prune those classes without
collapsing their epistemically different admission regimes.

### Radio-domain terminal-geometry audit

The next audit separates four plausible failures instead of tuning the same
score repeatedly.  First, the point cloud's third coordinate is not redundant:
sampling multiplicity represents `field**0.5`, while stored height retains the
local field amplitude.  `conditional_ridge_signature` therefore has an
optional `saliency_power`; power `0.5` approximately restores the original
field measure.  Symmetric query/reference weighting is negative on the radio
control: “okay” changes from rank 129/225 to 147/225, and forced “all right”
from 236/236 to 230/236.  It remains a diagnostic and is not promoted.

Second, applying the nonlinear morphology/ridge/occupation lift once to the
literal whole radio interval, rather than stitching six independently lifted
fragment clouds, improves “okay” from 129/225 to 67/225 and “all right” from
236/236 to 230/236.  This proves that fragment stitching is lossy, but it is
not the terminal fix.

Third, `run_synthetic_word_positive_control` sends complete macOS speech words
through the same SSB, Fourier-unrolled, ridge, occupation, and 32,768-point
pipeline.  Against the top 24 generic whole-trace competitors, Daniel's whole
word “okay” is only rank 22/25; “cowie” is first.  Complete coarticulation thus
does not close the clean/synthetic-to-radio domain gap.  Affine row gauging,
which preserves internal harmonic spacing while removing global pitch shift
and scale, also trades errors: it improves forced “all right” from 236/236 to
195/236 but worsens whole-trace “okay” from 67/225 to 144/225.

Finally, the coarse pronunciation catalog had been compiled only for length
three.  The length-five catalog contains 21,486 classes, compiles in 43.15
seconds, occupies a `21486 x 4 x 16 x 8` surface tensor, and ranks the complete
catalog in about 55 ms.  This removes a real admission capability gap, but the
geometry remains negative: `AO L R AY T` is rank 2,612/2,877 in its exact
marker-support stratum with stitched fragments and 2,087/2,877 with the whole
radio patch.  The catalog is retained as an audit artifact, not added to the
streaming baseline.

These controls localize the active bottleneck to radio-domain transport of
ridge identity.  Candidate enumeration and coarse search are already fast
enough; neither saliency weighting, fragment merging, whole-word
coarticulation, random pitch/time superposition, nor a less invariant row gauge
currently produces a pronunciation ordering stable across both known phrases.

The File Manager Kolmogrov history-tuple work remains correctly scoped to this
candidate stage: bounded deletion/insertion addressing proposes neighbors and
an exact verifier must decide.  The speech `PronunciationProposalIndex` already
implements that typed edit-plan role.  Kolmogrov's own contract does not rank
candidates, and treating its filename hash as an acoustic distance would move
the unresolved verifier problem rather than solve it.

### Continuous segmental verifier and closure audit

The measured radio-floor transport is not a stable identity map.  A noise
profile estimated from the longest inactive run has a 99.5-percentile
noise/active ratio of 0.374.  Adding phase-shifted copies of that correlated
floor to synthetic word rasters before morphology can improve the small
“okay” control from rank 22/25 to 16/25, but the same settings worsen or leave
unchanged the other attempted phrases.  The transported references become
more radio-like without becoming reliably more word-specific, so
`radio_raster_transport.py` remains an ablation rather than the baseline.

`segmental_pronunciation_geometry.py` replaces one global word score with a
candidate-conditioned contiguous partition.  Each proposed phone must explain
one or more observed fragments, and the best partition minimizes the worst
within-span atlas rank before the mean rank.  Raw distances are deliberately
not compared across different spans: their empirical floors vary from about
0.032 to 0.168 on this six-fragment word.  The within-span rank is the
distribution-free gauge of the unchanged conditional-ridge distance.

The first run exposed a boundary bug with acoustic meaning.  Between radio
regions 7 and 8 are 12 inactive frames, or 128 ms, at the expected `/K/`
closure.  Independently lifting fragment spans omitted that gap at a phone
cut.  `run_segmental_pronunciation_ablation.py` now forms a continuous raster
partition and records whether every inactive gap is assigned right, split at
its midpoint, or assigned left.  Giving the closure to the following phone
changes `/K/` from rank 12--27 to rank 2.  `/EY/` is rank 5.  `/OW/` remains
rank 26, so `OW K EY` is still only 171/197 among admitted three-phone classes
and 3,099/3,725 in the exhaustive length-three catalog.  The failure is now
localized to one phone coordinate rather than the entire word.

The local diagnosis identifies information discarded by the existing
signature.  `conditional_ridge_signature` rank-gauges every cloud's frequency
coordinate, preserving ordered occupation topology while erasing physical
band spacing.  A bounded time-warp probe does not repair `/OW/` (rank 24--25).
By contrast, a physical adjacent-quantile interval probe places `/OW/` rank 2
while remaining invariant to a global pitch translation; it does not preserve
the strong `/K/` and `/EY/` rankings.  `physical_interval_geometry.py` now
implements and tests this complementary coordinate, but it is not fused or
promoted.  The next controlled gate is cross-speaker per-phone calibration of
topological versus physical-interval geometry, followed by a pronunciation
rank only if that calibration predicts the coordinate without consulting the
radio label.

### Cleanup/SHARK speech anchors

`cleanup_shark_vad.py` now supplies a deterministic registration layer before
phone comparison. Cleanup's sorted three-frame logit-correlation statistic is
tracked against a quiet-population center and non-collapsing deviation floor.
A conservative Cleanup-style MAN/ATD gain and stationary noise estimate produce
the extracted magnitude field. SHARK-style 25 ms normalized pitch periodicity
and raw-spectrum peak/background prominence supply the independent voice
anchor. Prominence is deliberately measured before denoising: measuring it on
zeroed bins creates a circular false background and spurious 100+ dB peaks.

The state machine requires two voiced hits in three frames, backfills at most
120 ms of structurally supported unvoiced onset, retains structurally supported
unvoiced frames for 200 ms after the last pitch anchor, and crops at the last
measured content plus 25 ms rather than exporting the whole hangover. Isolated
pitch hits and unanchored structure are rejected.

Word-sized vocalization bounds are now exported separately from that inclusive
speech gate.  They discard isolated voiced hits, bridge only sub-25 ms pitch
dropouts inside a persistent voiced nucleus, then add a bounded 192 ms lead and
75 ms analysis tail.  The asymmetry prevents an energy-only noise component from pulling
an onset arbitrarily backward while retaining weak pre-periodic articulation.
On `daveandsimon.wav` the first stable nucleus is frames 40--83 and its
vocalization analysis bound is therefore 22--90; frames 85--90 deliberately
retain a small post-vocalization context for terminal-gesture fitting. The
obsolete ridge/energy union had
incorrectly admitted the preceding noise at frames 4--21.

### Vowel-seeded articulatory source material

The unmatched recorded-phone and independently rendered-phone banks are no
longer the active source model. `articulatory_tract.py` provides a deterministic
continuous-phase glottal source and 44-section Kelly--Lochbaum vocal-tract
waveguide, with smooth tongue, lip, and constriction trajectories. The model
is derived from Neil Thapen's MIT-licensed Pink Trombone construction; the
license is retained in `THIRD_PARTY_NOTICES`.

`articulatory_acoustics.py` estimates F0 and F1--F3, expresses formants in
semitones relative to F0, locates a stationary vowel seed, and measures static
waveguide resonances in batches. `run_articulatory_all_probe.py` uses Simon's
first `22:90` vocalization to calibrate a speaker-conditioned tract seed and
then renders one continuous held-vowel-to-dark-L trajectory. The complete
inversion and dictionary-registration design is in
`ARTICULATORY_INVERSION.md`.

The Cleanup certifier is now calibrated on 28 noisy BDL/SLT recordings rather
than fixed by inspection of the radio sentence. A label-held-out audit scores
speech-frame coverage, crop edges, and voiced/unvoiced landmarks; timing labels
are consulted only after state inference. A 3-by-3 hysteresis grid separates
certification of SHARK pitch from certification of unvoiced spectral/energy
structure. Under a one-sided 95% frame-precision lower bound of 0.99, every one
of 14 leave-one-utterance-out folds selects voice strength 0 and structure
strength 0.125. Crossfit precision is 99.69% (lower bound 99.55%), recall is
92.97%, and F1 is 96.21%. Pitch-only recall is 84.61%; fully ungated structure
has 96.37% precision and fails the certified floor. Cleanup therefore certifies
non-pitched continuation but does not attenuate the independent pitch anchor.

On `daveandsimon.wav`, the calibrated detector turns 88 disconnected old
ridge-support islands into 29 contiguous speech crops and retains 97.59% of
their support. The first three crops are 0.363--0.949 s, 1.547--2.251 s, and
2.539--6.613 s. The second crop is wider than the earlier conservative control
because it retains an additional measured compartment through frame 211. That
is speech segmentation, not a claim that one word fills the complete crop.

Reproduce the state arrays, denoised magnitude, intervals, and comparison
statistics with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_cleanup_shark_vad \
  /path/to/daveandsimon.wav \
  experiments/ostensibly_frontend/out/reassigned_texture_full/corrected_full_recording.npz \
  --out experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/cleanup_shark_vad
```

`debounced_state_landmarks` carries those estimates into the segmental
verifier. For the 1.557--2.165 s “okay” crop it removes one-frame pitch
flicker but retains the stable sequence
`U 146:152, V 152:156, U 156:159, H 159:165, U 165:177, V 177:203`.
Consequently the closure, release, and sustained vowel restart are candidate
cut coordinates rather than a transcript-conditioned gap assignment. The
unchanged topological verifier improves `OW K EY` from 171 to 153 of 197
admitted classes and from 3,099 to 1,865 of 3,725 exhaustive classes. Exposing
every old support edge inside the same crop is a negative control (178/197),
so only the debounced state landmarks are retained.

With the calibrated hysteresis, the same radio region has landmarks
`145,152,156,160,163,177,199,203,211`. Forcing `okay` to consume the complete
145:211 speech crop expands the exact pronunciation population to 47,898 and
worsens its state-23 rank to 14,848. This is a deliberately rejected word-
boundary assumption. The generic composer already evaluates every measured
subspan: under the unchanged state-23 proposal stratum it retains `OW K EY` at
rank 1,337 on frames 145:199 and rank 2,064 on 145:203. The latter endpoint is
the old crop boundary, now correctly represented as an internal landmark.
Sentence-level path selection must decide which adjacent word owns 199:211;
the VAD neither consumes nor labels it.

The physical-interval coordinate is now calibrated independently of the radio
label. `run_calibrate_geometry_fusion.py` evaluates 906 BDL-to-SLT and
SLT-to-BDL phone queries over 14 utterances. Topology alone retrieves 368
top-five instances with median rank 8; physical spacing alone retrieves 442
with median rank 6. The policy chosen in 12/14 leave-one-utterance-out folds is
the conservative maximum of their rank percentiles: its held-out result is
454 top-five with median rank 5. Applied unchanged to the state-bounded radio
word, it yields a physically plausible partition `OW 146:156, K 156:177,
EY 177:203`, phone ranks `20, 2, 25`, admitted rank 159/197, and exhaustive
rank 1,875/3,725. An arithmetic fusion reaches 144/197 and 1,247/3,725 but is
not promoted because cross-speaker calibration selected the intersection
policy. The conservative fusion is also not promoted over state-only topology,
because it slightly worsens 153/1,865 to 159/1,875 by moving `/EY/` from rank 2
to rank 25. The remaining failure is therefore radio-domain vowel transport,
not candidate enumeration, closure ownership, or an uncalibrated fusion
weight.

`population_interval_alignment.py` tests the simplest possible transport
error without using phone labels: one row-unit multiplier is fitted between
the complete generic and radio adjacent-interval populations by the closed-form
log-Wasserstein quantile translation. Across 416,995 reference and 228,151
radio intervals it estimates `0.913696`, reducing quantile RMS discrepancy from
0.140401 to 0.107546 (23.4%). Applied frozen to every radio query, however, the
conservative word result only changes from 159/1,875 to 158/1,866. It remains
worse than state-only topology at 153/1,865 and is not promoted. The marginal
scale mismatch is real, but a single global dilation is not the recognition
nuisance; the remaining error lies in conditional interval shape, trajectory,
or occupation mass, especially for the radio vowel.

The three physical terms were therefore separated under the same bounded
temporal registration. `run_physical_component_audit.py` evaluates interval
spacing, median-ridge trajectory, occupation mass, and their fixed mixtures on
all 906 cross-speaker queries. The existing top-one-first calibration objective
selects mass alone in every leave-one-utterance-out fold: 138 top-one, 429
top-five, median rank 6. The full physical mixture has broader but weaker
top-one retrieval (122 top-one, 442 top-five), showing that interval and
trajectory evidence help a wider shortlist while sometimes reversing the
nearest class. Recalibrating fusion specifically for the selected mass channel
chooses equal-weight geometric topology-by-mass in every fold. It reaches 163
top-one and 458 top-five with median rank 5, improving the prior full-physical
maximum at 153 and 454.

Applied frozen to the radio word, geometric topology-by-mass preserves the
measured state compartments and improves the target from 153/1,865 to
146/1,823. Its selected cuts are `OW 146:156, K 156:165, EY 165:203`; phone
ranks are `25, 19, 5`. This complementary rule is promoted as the current
segmental verifier baseline. It is still a weak recognizer: the remaining work
is to transport the unresolved `/OW/` and `/K/` geometry, not to claim that
temporal mass has solved phone identity.

### Empirical top-k coverage and state phone DAG

A phone-conditioned probability remapping was tested and rejected before
radio evaluation. `empirical_phone_compatibility.py` measures two finite
statistics for each candidate: conformal survival among genuine examples and
the enrichment of the threshold event `score <= observed` among genuine versus
impostor examples, including raw counts and a 95% Wilson ratio bound. Nested
utterance-held-out selection nevertheless degrades the generic geometric rank
from 163/458 top-one/top-five to 98/335. Normalizing away phone difficulty also
normalizes away real discriminative geometry. The artifact is retained as a
falsification control and is not connected to radio inference.

`phone_rank_coverage.py` supplies the useful probabilistic layer without
reordering anything. It compiles the exchangeable cross-speaker distribution
of the true phone's rank under frozen geometric topology-by-mass. Top-five
coverage is 50.55% (95% interval 47.30--53.80%), top-16 is 83.44% with an
80.88% lower bound, and top-24 is 93.49% with a 91.69% lower bound. The minimum
depths whose 95% lower confidence bounds exceed 50%, 80%, 90%, and 95% are 6,
16, 23, and 28. These are coverage guarantees for a retained set, not invented
per-phone posterior probabilities.

`run_state_phone_lattice.py` now turns every Cleanup/SHARK speech crop into a
label-free directed acyclic graph. Debounced state landmarks are nodes;
contiguous spans of at most three state compartments are candidate phone
edges. Each edge contains the unchanged topology-by-mass ranking, its component
ranks, and the measured coverage annotations. At the 95% conservative target,
the first two radio crops produce 18 edges in 5.8 seconds and retain 28 phones
per edge. A post-hoc audit finds the complete `OW K EY` route in the second
crop at edges `0:2, 2:4, 4:6`, ranks `25, 19, 5`; no target label was consumed
when constructing the DAG. This is the current reusable acoustic frontend for
dictionary composition.

`state_word_lattice.py` composes that DAG with the exact CMUdict pronunciation
trie. It permits no phone edits and applies no language prior. The resulting
full-span crop contains 28,537 pronunciation classes; pure geometric ordering
places the `OW K EY` homophone class at rank 18,716. This is an important
negative result: exact dictionary structure cannot manufacture concentration
that is absent from the phone geometry.

Duration and Cleanup/SHARK statistics are consequently retained as independent
proposal strata rather than folded into the identity distance. The state
profile contains 11 bounded features: pitch periodicity, spectral prominence,
Cleanup probability, fused voice score, unvoiced score, state occupation, and
energy SNR. On 906 cross-speaker ARCTIC queries it has only 81 top-one and 311
top-five identities, but its manner-like shortlist is useful: state depth 23
has a 90.47% lower-95% coverage bound and depth 28 has a 95.06% bound. It must
therefore decide what to compare, never how similar two phone geometries are.

The named-word audit is strictly post-hoc. With the label-free graph held
fixed, duration depth 24 moves `okay` from rank 18,716 to 6,386 and state depth
23 to 3,680. An initially tempting intersection priority moved it to 1,395,
but the matched cross-speaker audit rejects that result: duration-24 and
state-23 jointly admit only 707/906 true phones (78.04%, lower-95% 75.22%).
The defensible proposal union admits 899/906; it marks a phone unsupported only
when both channels reject it. That safe union ranks `okay` 12,846, worse than
either individual proposal channel. Cleanup/SHARK state evidence is therefore
retained as framing and auditable proposal provenance, but no duration/state
fusion is promoted as an identity ordering. Phone geometry remains the active
bottleneck.

A matching denoising-aware extraction control tests whether that state evidence
should suppress point-cloud mass in weak time slices. The support envelope is
the maximum of Cleanup probability, the voiced sigmoid, and unvoiced score;
five suppression floors are applied identically to every reference and query
cloud. The floor-1 control exactly reproduces 163/458 top-one/top-five. Full
suppression falls to 156/453, and leave-one-utterance-out selection chooses the
unchanged floor in 11/14 folds; its crossfit result is 154/454. Time-only
reweighting is rejected. A future denoising control must apply Cleanup's
frequency-dependent gain to the complex spectra before the unrolled transform,
not erase complete time slices after occupation geometry has been extracted.

That frequency-dependent control is now implemented and rejected as an
identity transport. `CleanupSharkAnalysis.spectral_gain` exposes the exact
product of Cleanup's statistical and noise gains. The same real gain is
interpolated onto every FFT aperture and applied to the complex spectra before
double-IRFFT unrolling; reassigned occupation support receives the corresponding
gain-squared energy. An all-ones gain reproduces every fused field, support map,
and cached point cloud exactly.

A five-utterance screen was chosen by exact set cover so that all 39 phones are
represented. At gain floor 0.5, global spectral gain appeared to improve the
354-query screen from 54/182 to 65/195 top-one/top-five and therefore earned a
full evaluation. The complete 906-query result reversed the apparent gain:
156/440 versus the raw baseline's 163/458, with median rank worsening from 5 to
6. In paired queries, 337 improved, 216 tied, and 353 worsened. Leave-one-
utterance-out selection retained raw geometry in 13/14 folds; crossfit selection
also fell to 154/457. The per-phone audit suggests why: periodic classes such as
`OY`, `JH`, `EH`, `W`, and `R` often improve while weak/unvoiced evidence such as
`TH`, `V`, `HH`, `P`, and stops often worsens, although rare-phone estimates are
necessarily noisy.

A stricter state transport then applied the half-gain only on frames certified
`VOICED`, leaving quiet, unvoiced, and hangover spectra at exact unity. It too
failed the label-complete screen: 47/168/244 top-one/top-five/top-ten versus
54/182/252 raw, with 100 paired improvements, 109 ties, and 145 degradations.
It therefore did not earn a full run. Denoising changes speaker-dependent ridge
geometry even when confined to pitched frames. The promoted boundary is now
explicit: Cleanup/SHARK supplies temporal extraction, state landmarks,
confidence statistics, and candidate-proposal evidence; unchanged raw
Fourier-unrolled geometry supplies phone identity distance.

### Oracle word gate and balanced occurrence tails

`phone_rank_matrix.py` freezes the complete 39-label permutation for all 906
cross-speaker phone queries, rather than retaining only each labeled target's
rank. Its zero-quantile topology-by-mass matrix reproduces every target rank in
the preceding calibration artifact exactly. With true word boundaries and true
phone counts supplied, `oracle_word_geometry.py` then enumerates every exact
same-length CMUdict pronunciation. This is a deliberately favorable identity
gate: segmentation and phone-count inference have already been solved, and no
language prior is used.

The gate rejects sentence decoding as the current remedy. Leave-one-utterance-
out selection chooses sum-then-worst phone-rank composition in all 14 folds,
but retrieves only 20/42/50 of 256 words at top-one/top-five/top-ten; median
rank is 125.5. A pooled genuine-versus-impostor rank likelihood reaches
18/41/50 and is rejected. Sentence search cannot manufacture the missing
acoustic concentration even under oracle boundaries.

That audit exposed an estimator bias in the phone atlas. Each label formerly
used its closest occurrence, giving common phones many more chances to realize
an accidental extreme. Even after correct outputs are removed, reference-
occurrence count and false-winner frequency have Spearman correlation 0.9063
and 0.8841 in the two directions. For example, `AH` has 47 witnesses and is the
wrong winner on 88 of 453 BDL-atlas queries, while several singleton labels
never win. `occurrence_statistics.py` therefore applies the
same within-label distance quantile before labels are ranked. Quantile zero is
an asserted exact reproduction of the old minimum.

A 6-by-6 topology/mass grid selects topology quantile 0.05 with mass minimum
on the full generic set, improving 163/458 to 166/468 top-one/top-five. Nested
selection is not stable enough for promotion: it chooses topology 0, 0.05, and
0.1 in 1/9/4 folds and scores 153/466 crossfit. The balanced form also fails
the oracle terminal gate at 15/42 top-one/top-five. It is therefore retained
only as a distinct proposal statistic: “does the label's lower occurrence
distribution fit?” versus the baseline's “does any occurrence fit?”

That distinction is useful. At fixed top-five depth, their generic union
retains 500/906 targets, 42 more than the minimum alone. At the word level the
best post-hoc union ceiling is 21/51 rather than 20/42, but no cross-channel
score is invented. Both channels independently require rank 28 for a 95%
lower-bound coverage target. Applied frozen to the first two radio crops, their
union expands each edge from 28 candidates to a median 29 and maximum 30,
adding 31 balanced-only labels over 27 edges. On the known `OW K EY` route,
minimum ranks are 19/5/6 and balanced-tail ranks are 15/6/2. The state DAG
records both ranks and `geometry_proposal_channels`; its primary `rank` remains
the unchanged minimum geometry.

Two parameter-free corrections clarify why the balanced tail must remain a
proposal. Full occurrence-distribution dominance ranks a label by the exact
probability that one of its occurrences is closer than a non-label occurrence.
It removes the count/false-winner correlation (fused rho -0.242 and -0.350),
but washes away the local match and falls to 75/256 top-one/top-five.
Correcting only the closest witness by the null probability of seeing that
minimum in `n` draws preserves more locality at 151/396, yet still trails the
163/458 baseline and retains rho 0.705/0.711. Neither statistic is admitted to
the radio graph. Their value is diagnostic: the closest context-specific
witness contains real identity, while its unequal extreme-value advantage is
also real.

An independent routing control then uses duration exactly in its permitted
role: for every candidate label, it admits only the 1, 2, 4, or 8 reference
occurrences nearest the query duration and applies the unchanged geometric
minimum afterward. This also fails cleanly. The four settings score 66/305,
84/319, 104/363, and 138/412 top-one/top-five; unrestricted witnesses retain
163/458, and every leave-one-utterance-out fold selects unrestricted. Duration
can select plausible phone labels and temporal spans, but it is not a useful
within-label witness router. The next comparable-occurrence test must be
candidate-conditioned neighbor context at word level, where the proposed
sequence legitimately supplies the routing relation.

Reproduce the label-balanced calibration, coverage law, complete rank matrix,
and oracle word gate with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_occurrence_quantile_geometry \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/bdl_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/slt_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_slt_physical_atlas.npz \
  --baseline-audit experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass.json \
  --out experiments/ostensibly_frontend/out/cross_speaker_occurrence_quantile_geometry.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_phone_rank_coverage \
  experiments/ostensibly_frontend/out/cross_speaker_occurrence_quantile_geometry.json \
  --configuration 'topology:0.05|mass:0' \
  --out experiments/ostensibly_frontend/out/cross_speaker_phone_rank_coverage_q05.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_compile_phone_rank_matrix \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/bdl_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/slt_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_slt_physical_atlas.npz \
  --topology-occurrence-quantile 0.05 \
  --out experiments/ostensibly_frontend/out/cross_speaker_phone_rank_matrix_mass_q05.npz

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_oracle_word_geometry_audit \
  experiments/ostensibly_frontend/out/cross_speaker_phone_rank_matrix_mass_q05.npz \
  /tmp/ostensibly_cmudict.dict \
  experiments/ostensibly_frontend/out/arctic_sentence_assay_bdl_to_slt.json \
  experiments/ostensibly_frontend/out/arctic_sentence_assay_slt_to_bdl.json \
  --out experiments/ostensibly_frontend/out/cross_speaker_oracle_word_geometry_mass_q05.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_duration_routed_geometry \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/bdl_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/slt_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_slt_physical_atlas.npz \
  --baseline-audit experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass.json \
  --out experiments/ostensibly_frontend/out/cross_speaker_duration_routed_geometry.json
```

Reproduce the calibration and conservative radio verifier with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_geometry_fusion \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/bdl_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/slt_14_conditional_ridge_t32_q16.npz \
  --out experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_segmental_pronunciation_ablation \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/stream_word_lattice_grouped3_catalog_stratified32_generic12_warm.json \
  experiments/ostensibly_frontend/out/reassigned_texture_full/corrected_full_recording.npz \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/region_clouds_p8192.npz \
  experiments/ostensibly_frontend/out/bdl_slt_28_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/generic_pronunciation_catalog_bdl_slt_len3_r4_t16q8.npz \
  --phone0 6 --phone1 12 --target-phones OW,K,EY \
  --boundary-source state-landmarks \
  --speech-state experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/cleanup_shark_vad_m4.npz \
  --physical-atlas \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_slt_physical_atlas.npz \
  --physical-weight 1 --fusion-policy maximum --points 32768 \
  --out experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/segmental_pronunciation_okay_state_calibrated.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_estimate_population_interval_alignment \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_slt_physical_atlas.npz \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/region_clouds_p8192.npz \
  --out experiments/ostensibly_frontend/out/radio_population_interval_alignment.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_physical_component_audit \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_slt_physical_atlas.npz \
  --out experiments/ostensibly_frontend/out/cross_speaker_physical_component_audit.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_geometry_fusion \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/bdl_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/slt_14_conditional_ridge_t32_q16.npz \
  --physical-interval-weight 0 --physical-trajectory-weight 0 \
  --physical-mass-weight 1 \
  --out experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_phone_compatibility \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/bdl_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/slt_14_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_slt_physical_atlas.npz \
  --out experiments/ostensibly_frontend/out/cross_speaker_phone_compatibility.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_phone_rank_coverage \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass.json \
  --out experiments/ostensibly_frontend/out/cross_speaker_phone_rank_coverage.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_spectral_gain_geometry \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  --gain-floors 0.5 \
  --out experiments/ostensibly_frontend/out/cross_speaker_spectral_gain_geometry_full.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_select_spectral_gain_geometry \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass.json \
  experiments/ostensibly_frontend/out/cross_speaker_spectral_gain_geometry_full.json \
  --out experiments/ostensibly_frontend/out/cross_speaker_spectral_gain_geometry_selection.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_spectral_gain_geometry \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  --utterances arctic_a0007,arctic_a0041,arctic_a0095,arctic_a0117,arctic_a0120 \
  --gain-floors 0.5 --gain-mode voiced-only \
  --voice-certifier-strength 0.25 --structure-certifier-strength 0.25 \
  --out experiments/ostensibly_frontend/out/cross_speaker_spectral_gain_geometry_voiced_screen.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_audit_speech_anchors \
  /private/tmp/ostensibly_arctic_expanded \
  --out experiments/ostensibly_frontend/out/cross_speaker_speech_anchor_hysteresis_sweep.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_select_speech_anchor_certifier \
  experiments/ostensibly_frontend/out/cross_speaker_speech_anchor_hysteresis_sweep.json \
  --out experiments/ostensibly_frontend/out/cross_speaker_speech_anchor_hysteresis_selection.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_state_phone_lattice \
  experiments/ostensibly_frontend/out/reassigned_texture_full/corrected_full_recording.npz \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/cleanup_shark_vad_m4.npz \
  experiments/ostensibly_frontend/out/bdl_slt_28_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_slt_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_phone_rank_coverage.npz \
  --crop0 0 --crop1 2 \
  --out experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/state_phone_lattice_crops0_2_coverage95.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_phone_state_profiles \
  /private/tmp/ostensibly_arctic_expanded \
  --cache experiments/ostensibly_frontend/out/cleanup_shark_arctic_cache \
  --out experiments/ostensibly_frontend/out/cross_speaker_phone_state_profiles.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_phone_proposal_intersection \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/cross_speaker_phone_state_profiles.json \
  experiments/ostensibly_frontend/out/bdl_14_phone_duration_gate.npz \
  experiments/ostensibly_frontend/out/slt_14_phone_duration_gate.npz \
  --depths 24:23,24:28 \
  --out experiments/ostensibly_frontend/out/cross_speaker_phone_proposal_intersection.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_calibrate_temporal_evidence_reweighting \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  experiments/ostensibly_frontend/out/cleanup_shark_arctic_cache \
  --floors 0,0.25,0.5,0.75,1 \
  --out experiments/ostensibly_frontend/out/cross_speaker_temporal_evidence_reweighting.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_state_phone_lattice \
  experiments/ostensibly_frontend/out/reassigned_texture_full/corrected_full_recording.npz \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/cleanup_shark_vad_m4.npz \
  experiments/ostensibly_frontend/out/bdl_slt_28_conditional_ridge_t32_q16.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_bdl_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_geometry_fusion_mass_slt_physical_atlas.npz \
  experiments/ostensibly_frontend/out/cross_speaker_phone_rank_coverage.npz \
  --duration-gate experiments/ostensibly_frontend/out/bdl_slt_28_phone_duration_gate.npz \
  --state-profile-atlas experiments/ostensibly_frontend/out/cross_speaker_phone_state_profiles_combined_atlas.npz \
  --crop0 0 --crop1 2 \
  --out experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/state_phone_lattice_crops0_2_duration_state_coverage95.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_compose_state_word_lattice \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/state_phone_lattice_crops0_2_duration_state_coverage95.json \
  /tmp/ostensibly_cmudict.dict --state-admission-rank 23 \
  --candidates-per-span 4096 \
  --out experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/state_word_lattice_crops0_2_state23_exact.json

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_audit_state_word_candidate \
  experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/state_phone_lattice_crops0_2_duration_state_coverage95.json \
  /tmp/ostensibly_cmudict.dict okay --crop-index 1 \
  --duration-ranks 0,24 --state-ranks 0,23,28 \
  --out experiments/ostensibly_frontend/out/occupation_radio_phone_atlas_39/state_word_okay_duration_state_audit.json
```

Compile and query the disjoint boundary atlas with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_compile_boundary_atlas \
  /private/tmp/ostensibly_arctic_expanded \
  --utterances arctic_a0007,arctic_a0014,arctic_a0033,arctic_a0041,arctic_a0052,arctic_a0066,arctic_a0078,arctic_a0088,arctic_a0095,arctic_a0102,arctic_a0108,arctic_a0112,arctic_a0117,arctic_a0120 \
  --raw-cache experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  --out experiments/ostensibly_frontend/out/bdl_14_boundary_atlas_q256_p64.npz

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_query_boundary_atlas \
  /private/tmp/ostensibly_arctic_expanded \
  experiments/ostensibly_frontend/out/bdl_14_boundary_atlas_q256_p64.npz \
  --query-start-index 36 --phone-count 6 \
  --raw-cache experiments/ostensibly_frontend/out/occupation_raw_cloud_cache \
  --out experiments/ostensibly_frontend/out/compiled_boundary_query_chances.json
```

Reproduce the radio and controlled cross-speaker batteries with:

```sh
PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_occupation_cloud_battery \
  /tmp/ostensibly_arctic \
  experiments/ostensibly_frontend/out/sparse_gabor_first_pair/sparse_gabor_first_pair.npz \
  --out experiments/ostensibly_frontend/out/occupation_cloud_battery_bdl

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_occupation_cloud_battery \
  /tmp/ostensibly_arctic \
  experiments/ostensibly_frontend/out/sparse_gabor_first_pair/sparse_gabor_first_pair.npz \
  --query-arctic-speaker slt --query-arctic-utterance arctic_a0019 \
  --query-phone R --query-phone-ordinal 2 \
  --distance-mode sliced_wasserstein --height-metric-scale 0.000001 \
  --marginal-copula \
  --out experiments/ostensibly_frontend/out/occupation_cloud_battery_slt_r_copula

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.run_occupation_context_battery \
  /tmp/ostensibly_arctic --query-phone R --query-phone-ordinal 2 \
  --out experiments/ostensibly_frontend/out/occupation_context_battery_slt_r

PYTHONPATH=. .venv-jpeg/bin/python \
  -m experiments.ostensibly_frontend.summarize_occupation_context_battery \
  experiments/ostensibly_frontend/out/occupation_context_battery_slt_r/holdout_*.json \
  experiments/ostensibly_frontend/out/occupation_context_battery_slt_r/occupation_context_battery.json \
  --out experiments/ostensibly_frontend/out/occupation_context_battery_slt_r/multi_query_context_audit.json
```
