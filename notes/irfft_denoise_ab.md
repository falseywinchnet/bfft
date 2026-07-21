# The irfft-frame object as a denoising substrate: measured

Experiment: `experiments/irfft_denoise_ab.py`.  Signal: first 2 s of
daveandsimon.wav (48 kHz, peak-normalized), N = 512, hop 128, Hann, white
noise added at a known SNR.  Scored on the span the overlap-add fully
covers.

## What was being tested

`np.fft.irfft` on a real vector is a DCT-I: it reads the vector as a half
spectrum with zero imaginary part and returns an even-symmetric signal of
length 2(M-1).  Applied to a windowed time frame of length N it yields
2(N-1) real samples carrying the frame's phase as spatial geometry rather
than as a separate array.  The map is exact; `rfft(y).real` inverts it, and
taking the real part is precisely the projection onto the even component,
so a modified object needs no separate symmetrization.  Verified: frame
round trip 5.6e-16, full analysis/synthesis round trip 4.4e-16, OLA
identity 3.3e-16.

The claim: laying phase out as geometry should let a scale-selective 2-D
decomposition beat the same decomposition applied to an ordinary
spectrogram.

## The comparison

Identical decomposition (`bfft.meyer`), identical parameters, identical
number of fine layers dropped, in each of:

- **A irfft** — the object, (T, 2(N-1))
- **B1 logmag** — log|X| with the noisy phase reused; the standard
  spectrogram-denoising setup
- **B2 reim** — Re X and Im X decomposed independently, (T, N/2+1) each
- **W wiener** — no decomposition: a Wiener gain from a noise floor taken
  from the quietest 10% of frames

Layers coarse to fine are [cartoon, band_coarse, band_mid, band_fine,
residual]; `drop = k` discards the k finest.  Every image is affinely
rescaled to [0, 255] before the split, and the rescaled standard deviations
are comparable across domains (irfft 5.35, Re X 5.71, Im X 5.94), so the
normalization does not favour one domain.

## Results

Best setting per method, SNR in dB, LSD lower is better:

| input | method | mu | drop | SNR | gain | segSNR | LSD |
|---|---|---|---|---|---|---|---|
| 10 dB | W wiener | - | - | 14.59 | +4.62 | 11.13 | 52.9 |
| 10 dB | A irfft | 20 | 2 | 11.62 | +1.65 | 9.42 | **26.1** |
| 10 dB | B2 reim | 20 | 2 | 11.69 | +1.72 | 9.24 | 27.9 |
| 10 dB | B1 logmag | 40 | 2 | 7.86 | -2.11 | 5.27 | 63.0 |
| 0 dB | W wiener | - | - | 5.84 | +5.87 | 2.38 | 62.6 |
| 0 dB | A irfft | 40 | 2 | **8.40** | +8.42 | 5.37 | 49.7 |
| 0 dB | B2 reim | 40 | 2 | **8.43** | +8.45 | 5.47 | 47.0 |
| 0 dB | B1 logmag | 160 | 2 | 1.53 | +1.55 | -1.74 | 71.4 |

`drop = 0` reproduces the input to 0.00 dB in every domain, which is the
sanity check that the layers sum exactly.  `drop = 1` (the residual alone)
changes almost nothing.  `drop >= 3` destroys speech.

## Findings

**1. The representation is a null.**  A and B2 agree to within 0.1 dB in
every condition tested (8.40 vs 8.43 at 0 dB, 11.62 vs 11.69 at 10 dB,
9.34 vs 9.08 at 10 dB with mu = 40), across five values of mu and three
input SNRs.  The irfft object confers no measurable denoising advantage
over decomposing Re X and Im X in the ordinary STFT domain.  This is not
surprising in hindsight: both are real linear reparametrizations of the
same frame, and while total variation is not preserved between them, the
scale structure the ladder acts on evidently is.

**2. Carrying phase into the decomposition does matter, and by a lot.**
Both phase-carrying representations beat log-magnitude-with-reused-phase by
6.9 dB at 0 dB input and 3.8 dB at 10 dB.  The standard move -- decompose
the magnitude, keep the noisy phase -- is the weakest of the three by a
wide margin.  This is the part of the original intuition that survives:
phase belongs inside the decomposition.  It simply does not have to be the
irfft view that puts it there.

**3. The decomposition beats Wiener at low SNR and loses at high SNR.**
At 0 dB it is +2.6 dB SNR and 13 LSD points better; at 10 dB it is 3.0 dB
worse on SNR while being *half* the log-spectral distance (26.1 vs 52.9).
The two metrics disagree at 10 dB and neither should be quoted alone.
Wav files for both conditions are written to `experiments/out/`.

**4. The mechanism is visible and is the one claimed.**  In the 0 dB
figure's second row, band_coarse holds the speech events as compact shaped
patches (rms 4.37) while band_fine is spatially uniform noise (rms 4.17)
at nearly the same energy.  Energy does not separate them; shape does.
This is the same behaviour as the cartoon layer holding every galaxy with
a shape while the texture layer holds what approaches noise.

## What was not tested

An oriented feature.  Speech in this object is oriented ridges, and the
scale ladder is isotropic -- the same limit that put the orientation
segregation field at chance (see `experiments/meyer_segregation.py`).
Nothing here says an orientation-aware treatment of the object would also
be a null; it says the isotropic scale ladder is.

Soft attenuation.  `drop` is a hard delete of whole layers; a per-band
Wiener gain was not tried, and the gap to Wiener at 10 dB is the sort of
gap a soft rule usually closes.
