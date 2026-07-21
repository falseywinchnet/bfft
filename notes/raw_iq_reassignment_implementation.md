# Raw Complex-IQ Reassignment Readout

## Standalone implementation specification

This document completely specifies the deployed single-aperture raw-IQ
reassignment readout. It requires no DIP solver, PGHI seed, waveform recovery,
tile cache, ladder, or unified projection. Its input is complex IQ. Its output
is an FFT-shifted time-frequency power raster in decibels.

The transform uses three full-length complex FFTs per input frame. The first
FFT measures coefficient power. The second measures the coefficient's time
location inside the frame. The third measures its fractional frequency offset
from the nominal FFT bin. Accepted coefficient power is bilinearly scattered
to the resulting fractional time-frequency coordinate.

## 1. Required conventions

Use these conventions exactly to reproduce the deployed implementation.

- `N` is an even FFT length. The deployed BFFT wrapper requires a power of two
  with `N >= 4`.
- `H` is a positive output hop in complex samples.
- `R` is the number of output time rows.
- Input is `nsamples` complex samples stored as interleaved float32 values:
  `re[0], im[0], re[1], im[1], ...`.
- Internal window, FFT, ratio, and accumulator calculations use float64.
- FFT means the unnormalized forward complex DFT:

      X[k] = sum from j=0 to N-1 of x[j] * exp(-i*2*pi*j*k/N)

- All `N` complex output bins are required. Do not discard the negative
  frequencies as one would for a real-input spectrum.
- `start` is the input-buffer index of the center of output row zero.
- The center of output row `r` is `start + r*H` in input-buffer coordinates.
- An input index outside `[0, nsamples-1]` supplies the complex value zero.
- Frequency bins are cyclic modulo `N`. Output time rows are clamped to
  `[0, R-1]`.

## 2. Exact windows and coefficients

For `j = 0, 1, ..., N-1`, construct the symmetric Hann window:

    g[j] = 0.5 - 0.5*cos(2*pi*j/(N-1))

Define the centered sample coordinate and time-moment window:

    tau[j] = j - (N-1)/2
    tg[j]  = tau[j] * g[j]

Because `(N-1)/2` is half-integer for even `N`, do not replace it with integer
division or `N/2`.

Construct the derivative window with the exact `numpy.gradient` stencil used
by the deployed implementation:

    dg[0]   = g[1] - g[0]

    dg[j]   = 0.5 * (g[j+1] - g[j-1])
              for j = 1, ..., N-2

    dg[N-1] = g[N-1] - g[N-2]

The `0.5` in the interior stencil and the one-sided endpoint coefficients are
part of the required convention. Do not substitute an analytic Hann
derivative if bitwise-equivalent behavior with the deployed renderer matters.

## 3. Frame extraction

For output row `r`, define its buffer-local center:

    c = r * H

For each window position `j`, obtain:

    input_index = start + c - N/2 + j

    if 0 <= input_index < nsamples:
        x[j] = complex_iq[input_index]
    else:
        x[j] = 0 + 0i

The center is therefore between samples for an even symmetric window in the
usual Hann sense, while frame addressing uses the integer index shown above.
Use the formula literally; do not introduce an additional half-window shift.

## 4. The three FFTs

Construct three complex sequences from the same IQ frame:

    ordinary[j]   = x[j] * g[j]
    time_moment[j]= x[j] * tg[j]
    window_deriv[j]=x[j] * dg[j]

Compute three full complex, unnormalized forward FFTs:

    Y[k]  = FFT(ordinary)[k]
    YT[k] = FFT(time_moment)[k]
    YD[k] = FFT(window_deriv)[k]

Their meanings are:

1. `Y`: the ordinary Hann STFT coefficient. Its squared magnitude supplies
   the power to be deposited. Its complex phase is the reference phase for
   both coordinate estimates.
2. `YT`: the first time moment of that same coefficient. Its in-phase
   projection onto `Y` supplies the energy centroid's displacement, in
   samples, from the nominal frame center.
3. `YD`: the coefficient formed with the derivative of the analysis window.
   Its quadrature projection onto `Y` supplies the component's frequency
   displacement, in FFT bins, from nominal bin `k`.

The three FFTs must use the same input samples, frame center, FFT sign,
normalization, and bin ordering.

## 5. Per-frame acceptance gate

For every bin, calculate ordinary coefficient power:

    E[k] = real(Y[k])^2 + imag(Y[k])^2

Find the largest power in the current frame:

    Emax = max over k of E[k]

The exact deployed acceptance threshold is:

    threshold = 1e-8 * (Emax + 1e-30)

Process bin `k` only when:

    E[k] > threshold

The strict greater-than comparison is intentional. `1e-8` is a deployed
numerical policy, equivalent to rejecting coefficients more than 80 dB below
the frame's peak power. It is not a theorem of reassignment. An implementation
seeking exact compatibility must use it; an application may expose it as a
parameter only if compatibility is not required.

## 6. Exact reassignment coordinates

For every accepted bin, use the exact stabilized inverse:

    inverse_power = 1 / (E[k] + 1e-30)

Calculate the real part of `YT * conjugate(Y)` without constructing a complex
temporary:

    time_cross = real(YT[k])*real(Y[k])
               + imag(YT[k])*imag(Y[k])

Calculate the imaginary part of `YD * conjugate(Y)`:

    frequency_cross = imag(YD[k])*real(Y[k])
                    - real(YD[k])*imag(Y[k])

The reassigned buffer-local sample coordinate is:

    reassigned_sample = c + time_cross * inverse_power

The reassigned fractional output-row coordinate is:

    reassigned_row = reassigned_sample / H

The reassigned fractional frequency-bin coordinate is:

    reassigned_bin = k
                   - frequency_cross * inverse_power * N / (2*pi)

`N/(2*pi)` and the minus sign follow from the required negative-exponent FFT
convention. If the FFT library uses the opposite forward-transform sign, the
frequency correction sign must be re-derived rather than copied.

The time correction is measured in samples before division by `H`. The
frequency correction is measured in bins. No sample-rate factor is needed for
raster placement. To report hertz after placement, interpret an unshifted bin
coordinate `b` as `b*sample_rate/N`, wrapping coordinates above `N/2` into
negative frequency.

## 7. Conservative bilinear power deposition

Initialize a float64 accumulator `P` of shape `[R][N]` to zero for every
render call.

For an accepted coefficient, calculate:

    row0_raw = floor(reassigned_row)
    row1_raw = row0_raw + 1
    row_fraction = reassigned_row - floor(reassigned_row)

    bin0_raw = floor(reassigned_bin)
    bin1_raw = bin0_raw + 1
    bin_fraction = reassigned_bin - floor(reassigned_bin)

Clamp time indices independently:

    row0 = clamp(row0_raw, 0, R-1)
    row1 = clamp(row1_raw, 0, R-1)

Wrap frequency indices independently:

    bin0 = ((bin0_raw mod N) + N) mod N
    bin1 = ((bin1_raw mod N) + N) mod N

Define weights:

    time_weight0 = 1 - row_fraction
    time_weight1 = row_fraction
    bin_weight0  = 1 - bin_fraction
    bin_weight1  = bin_fraction

Deposit the ordinary coefficient power `E[k]`, not `abs(Y[k])` and not the
power of either auxiliary FFT:

    P[row0][bin0] += E[k] * time_weight0 * bin_weight0
    P[row0][bin1] += E[k] * time_weight0 * bin_weight1
    P[row1][bin0] += E[k] * time_weight1 * bin_weight0
    P[row1][bin1] += E[k] * time_weight1 * bin_weight1

The four weights sum to one. If clamping maps both time destinations to the
same edge row, both contributions are still added, so accepted power remains
conserved. Bilinear deposition is used because the output has exactly two
continuous coordinates: time and frequency.

## 8. FFT shift and decibel output

For output column `q = 0, ..., N-1`, map to the internal unshifted bin:

    internal_bin = q + N/2
    if internal_bin >= N:
        internal_bin -= N

The exact deployed output conversion is:

    output_db[r][q] = 10 * log10(P[r][internal_bin] + 1e-12)

`1e-12` is the deployed output power floor, so an empty cell is `-120 dB`.
It is a display policy, not part of the reassignment coordinate identity.

## 9. Complete reference pseudocode

    require N >= 4 and N is a power of two
    require H > 0 and R > 0

    precompute g[N], tg[N], dg[N] exactly as specified
    P[R][N] = 0 using float64

    for r in 0 .. R-1:
        c = r * H

        for j in 0 .. N-1:
            s = start + c - N/2 + j
            x = IQ[s] if 0 <= s < nsamples else 0+0i
            ordinary[j]    = x * g[j]
            time_moment[j] = x * tg[j]
            window_deriv[j]= x * dg[j]

        Y  = unnormalized_forward_complex_fft(ordinary)
        YT = unnormalized_forward_complex_fft(time_moment)
        YD = unnormalized_forward_complex_fft(window_deriv)

        Emax = max(real(Y[k])^2 + imag(Y[k])^2 for all k)
        threshold = 1e-8 * (Emax + 1e-30)

        for k in 0 .. N-1:
            E = real(Y[k])^2 + imag(Y[k])^2
            if E <= threshold:
                continue

            inv = 1 / (E + 1e-30)
            tcross = real(YT[k])*real(Y[k]) + imag(YT[k])*imag(Y[k])
            fcross = imag(YD[k])*real(Y[k]) - real(YD[k])*imag(Y[k])

            tr = (c + tcross*inv) / H
            fb = k - fcross*inv*N/(2*pi)

            r0raw = floor(tr); r1raw = r0raw + 1
            f0raw = floor(fb); f1raw = f0raw + 1
            rt = tr - floor(tr)
            ft = fb - floor(fb)

            r0 = clamp(r0raw, 0, R-1)
            r1 = clamp(r1raw, 0, R-1)
            f0 = positive_modulo(f0raw, N)
            f1 = positive_modulo(f1raw, N)

            P[r0][f0] += E*(1-rt)*(1-ft)
            P[r0][f1] += E*(1-rt)*ft
            P[r1][f0] += E*rt*(1-ft)
            P[r1][f1] += E*rt*ft

    for r in 0 .. R-1:
        for q in 0 .. N-1:
            k = (q + N/2) modulo N
            output_db[r][q] = 10*log10(P[r][k] + 1e-12)

## 10. Native C integration

The existing library exposes:

    iqw_ra* iqw_ra_create(int N)
    void iqw_ra_set_bilinear(iqw_ra* engine, int enabled)
    void iqw_ra_render_mem(
        iqw_ra* engine,
        const float* interleaved_iq,
        int64_t nsamples,
        int64_t start,
        int64_t H,
        int R,
        float* output_db)
    void iqw_ra_destroy(iqw_ra* engine)

Minimal call sequence:

    engine = iqw_ra_create(N)
    assert engine is not null
    iqw_ra_set_bilinear(engine, 1)
    allocate output_db[R*N]
    iqw_ra_render_mem(engine, iq, nsamples, start, H, R, output_db)
    iqw_ra_destroy(engine)

One engine owns reusable FFT plans and work buffers. Reuse it across render
calls of the same `N`. Do not call one engine concurrently from multiple
threads because its work arrays and power accumulator are mutable. Use one
engine per concurrent thread or protect it with a lock.

## 11. Optional multiscale readout

The single-aperture method above is complete. A multiscale display repeats it
with several window lengths and is a separate display policy.

For each rung length `Nr`, render its own `[R][Nr]` dB raster using the same
time centers and hop. To remove the exact coherent-power advantage of a longer
symmetric Hann window relative to base length `Nb`, subtract:

    gain_db = 20 * log10((Nr - 1)/(Nb - 1))

from every rung cell. This follows from the exact symmetric-Hann coherent gain
`sum(g) = (N-1)/2` and the fact that the raster accumulates squared magnitude.

Resample every rung's frequency axis to a common grid in linear power, not by
interpolating dB. The current viewer then takes the per-cell maximum. This
multiscale maximum is not required for raw reassignment and should not be
presented as waveform super-resolution.

## 12. Acceptance tests

An independent implementation should pass all of the following.

1. Empty input: every output cell is exactly `10*log10(1e-12) = -120 dB`,
   within the output float32 rounding error.
2. Global phase invariance: multiplying all IQ samples by any unit complex
   number leaves every output cell unchanged within numerical FFT error.
3. Power conservation: before the final dB conversion, the sum of `P` equals
   the sum of `E[k]` over all accepted bins and frames, within floating-point
   accumulation error, including coefficients clamped to edge rows.
4. Centered on-bin tone: a tone exactly at bin `k`, symmetrically contained in
   a frame, has approximately zero time correction and zero frequency
   correction at its dominant coefficient.
5. Fractional-bin tone: the derivative-window correction moves the dominant
   coefficient toward the tone's known fractional bin, with the sign shown in
   Section 6.
6. Shifted impulse: moving an impulse inside a frame by `d` samples moves the
   time reassignment by approximately `d` samples.
7. FFT shift: unshifted bin zero appears at output column `N/2`; unshifted bin
   `N/2` appears at output column zero.
8. Native parity: for the same float32 IQ buffer, `N`, `start`, `H`, and `R`,
   compare the entire dB raster to the native `iqw_ra_render_mem` output.
   Differences should be attributable only to the chosen FFT implementation's
   floating-point rounding.

## 13. Which constants are exact and which are policies

Required by the chosen reassignment convention:

- Hann coefficients `0.5` and `0.5`.
- Center `(N-1)/2`.
- Interior derivative coefficient `0.5` and one-sided endpoint differences.
- Negative-exponent FFT convention.
- Frequency factor `N/(2*pi)` and its sign.
- Conjugate-product real and imaginary parts.
- Bilinear weights derived from the fractional coordinates.

Exact deployed numerical policies, but tunable in a different product:

- Relative power gate `1e-8`.
- Gate stabilizer and ratio denominator `1e-30`.
- Output power floor `1e-12`.
- Time clamping rather than discarding out-of-range deposits.
- Bilinear rather than nearest-cell deposition.

There are no additional hidden coefficients in the deployed raw-IQ readout.
