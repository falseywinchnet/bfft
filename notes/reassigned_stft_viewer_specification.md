# Current Reassigned STFT Viewer Mode

## Exact algorithm and integration specification

This document specifies the live **Reassigned STFT** mode in the BFFT IQ
waterfall viewer. It operates directly on complex IQ. It does not run DIP,
PGHI, a shared seed, magnitude-family projections, ladder reconstruction, or
waveform recovery.

It is a conventional time-frequency reassigned spectrogram with three related
STFTs per frame. Its distinctive deployed policy is **nearest-cell power
deposition**. The viewer's separate **Raw IQ through SR readout** stage uses the
same three-FFT coordinate rule but enables bilinear deposition and can add
multiscale ladder readouts.

## 1. Live viewer data path

For the selected FFT length `N`, hop `H`, first displayed center `position`,
and display height `R = 640`, the viewer performs:

    need = (R - 1)*H + N
    source_start = position - N/2
    z = read need complex samples beginning at source_start

The buffer therefore contains enough samples for `R` centered frames. Inside
the buffer, the first frame center is `N/2`; subsequent centers are separated
by `H`.

If **Remove DC per frame** is enabled in the current application, the actual
implementation subtracts one complex mean from the complete requested buffer:

    z = z - mean(z)

Despite the UI label, this is currently block-mean removal, not an independent
mean for every frame.

The complex float64 source buffer is converted to interleaved float32 IQ:

    iq = [real(z[0]), imag(z[0]), real(z[1]), imag(z[1]), ...]

The viewer calls its persistent `Reassign(N)` engine:

    engine.render_mem(iq, start=N/2, hop=H, n_rows=R)

The engine returns an `R by N` fft-shifted float32 power raster in decibels.

## 2. Required FFT and indexing conventions

- `N` must be a power of two and at least 4 in the deployed BFFT backend.
- `H` is a positive integer number of complex samples.
- Input is complex IQ. All `N` complex FFT bins are retained.
- The forward FFT is unnormalized and uses the negative exponential:

      X[k] = sum over j of x[j] * exp(-i*2*pi*j*k/N)

- Window and ratio calculations use float64 internally.
- The input buffer is interleaved float32.
- Output is float32 dB.
- Samples outside the supplied buffer are treated as complex zero.

For output row `r`, define the buffer-local nominal center:

    c = r*H

For window index `j = 0, ..., N-1`, the engine reads:

    s = start + c - N/2 + j

If `s` is outside `[0, nsamples-1]`, use zero. In the viewer call, `start=N/2`,
so row zero is centered on the first sample identified by the viewer's
`position` marker.

## 3. The three analysis windows

Construct the symmetric Hann window exactly:

    g[j] = 0.5 - 0.5*cos(2*pi*j/(N-1))

Construct the centered time coordinate and time-moment window:

    tau[j] = j - (N-1)/2
    tg[j]  = tau[j]*g[j]

For even `N`, `(N-1)/2` is a half-integer. Do not replace it with integer
division or `N/2`.

Construct the derivative window with the exact deployed finite difference:

    dg[0]   = g[1] - g[0]

    dg[j]   = 0.5*(g[j+1] - g[j-1])
              for j = 1, ..., N-2

    dg[N-1] = g[N-1] - g[N-2]

## 4. The three FFTs and what they represent

For each frame, form three windowed complex sequences:

    ordinary[j]    = x[j]*g[j]
    time_moment[j] = x[j]*tg[j]
    window_deriv[j]= x[j]*dg[j]

Then compute:

    Y[k]  = FFT(ordinary)[k]
    YT[k] = FFT(time_moment)[k]
    YD[k] = FFT(window_deriv)[k]

Their roles are exact and separate:

1. `Y` is the ordinary complex Hann STFT. Its squared magnitude is the power
   that will be moved. Its phase is the reference for both reassignment
   coordinates.
2. `YT` is the first time moment of the same localized coefficient. Its
   in-phase relationship with `Y` gives the coefficient's displacement in
   samples from the nominal frame center.
3. `YD` is the derivative-window STFT. Its quadrature relationship with `Y`
   gives the coefficient's fractional displacement from nominal FFT bin `k`.

The auxiliary transforms do not contribute displayed power. Only `abs(Y)^2`
is deposited.

## 5. Per-frame coefficient gate

For all bins, calculate:

    E[k] = real(Y[k])^2 + imag(Y[k])^2
    Emax = maximum E[k] in this frame

The exact deployed gate is:

    threshold = 1e-8*(Emax + 1e-30)

Process bin `k` only when:

    E[k] > threshold

Thus coefficients at or below 80 dB beneath the frame's peak power are not
reassigned. `1e-8` is an engineering policy, not a reassignment theorem.

## 6. Exact reassignment rule

For each accepted coefficient:

    inv = 1/(E[k] + 1e-30)

Compute the real part of `YT*conjugate(Y)`:

    time_cross = real(YT[k])*real(Y[k])
               + imag(YT[k])*imag(Y[k])

Compute the imaginary part of `YD*conjugate(Y)`:

    frequency_cross = imag(YD[k])*real(Y[k])
                    - real(YD[k])*imag(Y[k])

The reassigned time coordinate, measured in output rows, is:

    reassigned_row = (c + time_cross*inv)/H

The reassigned frequency coordinate, measured in unshifted FFT bins, is:

    reassigned_bin = k - frequency_cross*inv*N/(2*pi)

The minus sign corresponds to the required negative-exponent forward FFT.
The conjugate products cancel common complex amplitude and global phase, so
the coordinates depend on local phase derivatives without explicit phase
unwrapping.

## 7. Nearest-cell deposition: the defining current-mode policy

The live Reassigned STFT mode does **not** enable the engine's bilinear flag.
Every accepted coefficient is deposited into one output cell:

    output_row = round_to_nearest_integer(reassigned_row)
    output_bin = round_to_nearest_integer(reassigned_bin)

Then:

    output_row = clamp(output_row, 0, R-1)
    output_bin = positive_modulo(output_bin, N)
    P[output_row][output_bin] += E[k]

The implementation uses C++ `std::lround`, whose halfway cases round away from
zero. A compatibility implementation should not silently substitute a
banker's-rounding function.

Nearest-cell deposition has these consequences:

- A coefficient remains maximally concentrated in a single raster cell.
- Deposited power is conserved.
- Fractional coordinate motion appears as discrete cell jumps.
- Sparse holes and row/bin quantization are possible.
- The result can flicker when a trajectory crosses a half-cell boundary.

By contrast, the Raw IQ through SR readout calls
`iqw_ra_set_bilinear(engine, 1)` and shares the same power between four adjacent
time-frequency cells. That is the principal mathematical difference between
the two single-aperture raw reassignment forms.

## 8. FFT shift and decibel conversion

After processing every frame and coefficient, output column `q` reads internal
unshifted bin:

    k = q + N/2
    if k >= N:
        k -= N

The exact dB conversion is:

    output_db[r][q] = 10*log10(P[r][k] + 1e-12)

An empty cell is therefore `-120 dB`. The result is ordered from negative to
positive frequency, with DC at output column `N/2`.

## 9. Viewer display processing after reassignment

The reassignment engine returns physical dB values, but the visible image also
passes through the viewer's display transfer function.

1. Select the current normalized frequency interval `[f_low, f_high]`.
2. Resample that interval to 1024 display columns:
   - when shrinking, use max pooling so narrow peaks are not dropped;
   - when enlarging, interpolate in linear power and convert back to dB.
3. Convert dB to color intensity using the selected display mode.

The default **Amplitude (auto)** display computes the 99.5th percentile of a
4-by-4 subsample of the current dB image, calls it `reference_db`, and maps:

    amplitude = clamp(10^((db - reference_db)/20), 0, 1)

Optional gamma is then applied, followed by the selected color lookup table.

Because the percentile is recomputed for every displayed block, two transforms
can have different absolute power yet look similarly bright. Quantitative
comparisons should use saved linear-power rasters or a shared fixed dB range,
not independent automatic contrast.

## 10. Current hop choices

The viewer exposes:

    H = N/8
    H = N/4
    H = N/2

Hop changes the density of nominal analysis centers and the conversion from
time displacement in samples to output-row displacement. It does not alter the
window length or nominal FFT frequency spacing.

For comparison with the historical Dave-and-Simon readout, note that the
reference used `N=1024` and `H=32`, or `H=N/32`. That exact hop is not currently
one of the viewer's three selectable Reassigned STFT hops.

## 11. Minimal native integration

The native API is:

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

To reproduce current Reassigned STFT mode:

    engine = iqw_ra_create(N)
    do not enable bilinear mode
    iqw_ra_render_mem(engine, iq, nsamples, N/2, H, R, output_db)
    iqw_ra_destroy(engine)

The default engine state is nearest-cell deposition. Reuse one engine for
successive calls at the same `N`; it owns its FFT plan and mutable work arrays.
Do not invoke one engine concurrently from multiple threads.

## 12. Relationship to the recommended raw reassignment form

Both forms use exactly the same three FFTs, gate, coordinate equations, FFT
sign, stabilizers, power source, and dB floor.

Current Reassigned STFT mode:

    raw complex IQ
    -> one aperture N
    -> three-FFT reassignment
    -> nearest-cell deposition
    -> no ladder

Raw IQ through SR readout:

    raw complex IQ
    -> base aperture and optional additional apertures
    -> three-FFT reassignment at each aperture
    -> bilinear deposition
    -> optional coherent-gain normalization
    -> frequency-grid resampling
    -> optional per-cell maximum across apertures

For a simple external SDR integration, a single aperture with bilinear
deposition is the recommended default. It removes nearest-cell quantization
without requiring the ladder or any latent reconstruction.

## 13. Acceptance tests

1. Empty input produces `-120 dB` in every cell.
2. Multiplying all IQ by a unit complex constant leaves the complete raster
   unchanged within FFT rounding error.
3. Before dB conversion, accumulated output power equals the sum of all
   accepted `E[k]`, including edge-clamped coefficients.
4. A centered on-bin tone has approximately zero time and frequency correction
   at its dominant coefficient.
5. A fractional-bin tone moves to the known fractional bin with the sign in
   Section 6, then rounds to the nearest output bin.
6. An impulse displaced inside the frame moves by the corresponding number of
   samples before division by `H` and nearest-row rounding.
7. Unshifted bin zero appears at output column `N/2`.
8. With bilinear disabled, every accepted input coefficient deposits into
   exactly one time-frequency cell.
9. Native parity compares the entire float32 dB raster against
   `iqw_ra_render_mem` using identical IQ, `N`, `start`, `H`, and `R`.

## 14. Exact constants summary

Mathematical/discrete convention:

- Hann coefficients: `0.5`, `0.5`.
- Centered time origin: `(N-1)/2`.
- Derivative interior coefficient: `0.5`.
- Derivative endpoint coefficient: `1` through one-sided differences.
- Frequency factor: `N/(2*pi)`.
- Forward FFT sign: negative exponent.
- Reassigned-frequency correction sign: negative.
- Current deposition: nearest cell using `std::lround`.

Deployed numerical/display policies:

- Relative coefficient gate: `1e-8`.
- Gate and ratio stabilizer: `1e-30`.
- Output power floor: `1e-12`.
- Empty-cell output: `-120 dB`.
- Viewer display width: 1024 columns.
- Viewer display rows: 640.
- Default automatic amplitude reference: 99.5th percentile.

There are no additional hidden coefficients in the current Reassigned STFT
signal-processing kernel.
