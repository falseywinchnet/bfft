# The Viewer’s Two Raw Reassignment Modes

This document records exactly what the current program does in two independent
configurations. Each section is complete by itself.

---

# 1. Super-resolution → Raw IQ through SR readout

## Purpose

This configuration sends the original complex IQ directly into the
super-resolution display readout. It does **not** compute a DIP/PGHI seed, run
the unified magnitude projections, create reconstructed tiles, warm-start a
solver, or perform an asynchronous readiness fill. The readout is immediate,
so this mode has no reconstruction sweep.

## User-controlled geometry

Let:

    Nbase = selected “long win”
    Nshort = Nbase/4
    H = Nbase/4
    R = 640 output time rows

The display hop is fixed to `Nbase/4` in super-resolution mode.

If **slow rung** is off, the readout uses only `Nbase`.

If slow rung is `2x` or `4x`, define:

    tile = max(8192, 2*Nbase)
    Nup = min(rung_multiplier*Nbase, tile)

The readout then uses these aperture lengths:

    Nup
    Nbase
    Nshort

Duplicate lengths are omitted. The additional apertures affect only the
readout in raw mode; no multiscale latent solver runs.

## Raw sample request

Let `Ndisplay` be the largest active readout aperture. With the slow rung off,
`Ndisplay = Nbase`.

The program reads:

    sample_start = position - Ndisplay/2
    sample_count = (R-1)*H + Ndisplay

`position` is therefore the center sample of output row zero. Samples requested
outside the recording are supplied as zero by the source reader.

If **Remove DC per frame** is enabled, the current code subtracts one complex
mean from this entire requested sample block:

    x = x - mean(x)

Despite the UI wording, this is block-mean removal, not a separate mean for
every analysis frame.

The complex source samples are converted to interleaved float32 IQ:

    real(x[0]), imag(x[0]), real(x[1]), imag(x[1]), ...

## Reassignment engine created for each aperture

For every active aperture length `N`, the program creates a persistent BFFT
reassignment engine and explicitly enables bilinear deposition:

    engine = iqw_ra_create(N)
    iqw_ra_set_bilinear(engine, 1)

Every engine is rendered on the same time centers:

    iqw_ra_render_mem(
        engine,
        interleaved_iq,
        sample_count,
        first_center = Ndisplay/2,
        hop = H,
        rows = R,
        output_db)

Using `Ndisplay/2` as every engine’s first buffer-local center ensures that all
apertures are centered on the same original source sample, `position`.

## Exact three-FFT reassignment calculation

The following calculation is repeated independently for every active aperture
and output frame.

For aperture length `N`, construct the symmetric Hann window:

    g[j] = 0.5 - 0.5*cos(2*pi*j/(N-1))

Construct the centered time-moment window:

    tau[j] = j - (N-1)/2
    tg[j] = tau[j]*g[j]

Construct the discrete derivative window:

    dg[0] = g[1] - g[0]
    dg[j] = 0.5*(g[j+1] - g[j-1]), for 1 <= j <= N-2
    dg[N-1] = g[N-1] - g[N-2]

For each centered IQ frame, compute three full, unnormalized complex FFTs using
the negative-exponent forward convention:

    Y  = FFT(x*g)
    YT = FFT(x*tg)
    YD = FFT(x*dg)

The three transforms represent:

1. `Y`: the ordinary complex Hann STFT. Its power is what gets displayed.
2. `YT`: the first time moment of `Y`. It supplies the within-window time
   displacement.
3. `YD`: the derivative-window STFT. It supplies the fractional displacement
   from the nominal frequency bin.

For every bin:

    E = real(Y)^2 + imag(Y)^2
    Emax = largest E in the current frame
    threshold = 1e-8*(Emax + 1e-30)

Only bins satisfying `E > threshold` are processed.

For an accepted coefficient:

    inv = 1/(E + 1e-30)

    time_cross = real(YT)*real(Y) + imag(YT)*imag(Y)

    frequency_cross = imag(YD)*real(Y) - real(YD)*imag(Y)

    reassigned_sample = nominal_local_center + time_cross*inv

    reassigned_row = reassigned_sample/H

    reassigned_bin = nominal_bin
                   - frequency_cross*inv*N/(2*pi)

Only the ordinary coefficient power `E` is deposited. `YT` and `YD` determine
its destination but contribute no displayed power of their own.

## Bilinear power deposition

The reassigned row and frequency bin are fractional. The program finds the two
surrounding time rows and two surrounding frequency bins and deposits `E` into
their four combinations using linear fractional weights.

If:

    reassigned_row = r0 + rt
    reassigned_bin = k0 + ft

where `r0` and `k0` are floors and `rt`, `ft` are fractions, deposits are:

    P[r0,   k0  ] += E*(1-rt)*(1-ft)
    P[r0,   k0+1] += E*(1-rt)*ft
    P[r0+1, k0  ] += E*rt*(1-ft)
    P[r0+1, k0+1] += E*rt*ft

Time rows are clamped to the available `0 ... R-1` interval. Frequency bins
wrap modulo `N`. The four weights sum to one, so accepted coefficient power is
conserved.

The accumulator is FFT-shifted and converted to dB:

    output_db = 10*log10(reassigned_power + 1e-12)

An empty output cell is therefore `-120 dB`.

## Combining multiple apertures

Every aperture returns `R` rows but may have a different number of frequency
columns. Each result is resampled to `Ndisplay` columns.

- When shrinking a frequency axis, the program uses max pooling.
- When enlarging a frequency axis, it interpolates in linear power, not dB.

If **Normalize rung gain** is enabled, each aperture result has this exact dB
offset removed before fusion:

    gain_db = 20*log10((N-1)/(Nbase-1))

This compensates for the exact coherent-power gain of symmetric Hann windows
of different lengths.

The final multiscale readout is then:

    final_db[row, column] = maximum normalized dB value
                            across all active apertures

If slow rung is off, there is only the base aperture and no multiscale maximum.

## Final viewer display

The program selects the requested frequency interval and resamples it to 1024
screen columns. Downsampling uses max pooling; upsampling uses linear-power
interpolation.

The default **Amplitude (auto)** display obtains a reference from the 99.5th
percentile of a 4-by-4 subsample and maps:

    intensity = clamp(10^((db-reference_db)/20), 0, 1)

It then applies optional gamma and the selected color map.

## Summary of this mode

    original complex IQ
    -> one or several three-FFT reassignment engines
    -> bilinear power deposition
    -> optional Hann gain normalization
    -> optional per-cell maximum across apertures
    -> display transfer function

No reconstructed waveform exists in this configuration.

---

# 2. Reassigned STFT

## Purpose

This configuration sends the original complex IQ into one conventional
time-frequency reassigned STFT. It uses one selected FFT length and nearest-cell
power deposition. It does not use the super-resolution aperture controls,
ladder, gain normalization, DIP/PGHI seed, unified projections, reconstructed
tiles, or asynchronous readiness fill.

## User-controlled geometry

Let:

    N = selected FFT size
    R = 640 output time rows

The hop selector chooses one of:

    H = N/8
    H = N/4
    H = N/2

The selected analysis-window combo is disabled in this mode because the
reassignment engine always uses its own symmetric Hann window.

## Raw sample request

The program reads:

    sample_start = position - N/2
    sample_count = (R-1)*H + N

`position` is the center sample of output row zero.

If **Remove DC per frame** is enabled, the code subtracts one complex mean from
the complete requested block:

    x = x - mean(x)

This is block-mean removal in the current implementation.

The complex samples are converted to interleaved float32 IQ.

The program uses one persistent reassignment engine:

    engine = iqw_ra_create(N)

It does **not** call `iqw_ra_set_bilinear(engine, 1)`. The engine’s default
nearest-cell deposition remains active.

It renders:

    iqw_ra_render_mem(
        engine,
        interleaved_iq,
        sample_count,
        first_center = N/2,
        hop = H,
        rows = R,
        output_db)

## Exact three-FFT reassignment calculation

For every output frame, construct:

    g[j] = 0.5 - 0.5*cos(2*pi*j/(N-1))

    tau[j] = j - (N-1)/2
    tg[j] = tau[j]*g[j]

    dg[0] = g[1] - g[0]
    dg[j] = 0.5*(g[j+1] - g[j-1]), for 1 <= j <= N-2
    dg[N-1] = g[N-1] - g[N-2]

Compute three full, unnormalized complex FFTs using the negative-exponent
forward convention:

    Y  = FFT(x*g)
    YT = FFT(x*tg)
    YD = FFT(x*dg)

They represent:

1. `Y`: ordinary Hann STFT and the power source.
2. `YT`: first time moment and the time-location witness.
3. `YD`: derivative-window STFT and the fractional-frequency witness.

For every bin:

    E = real(Y)^2 + imag(Y)^2
    Emax = largest E in the current frame
    threshold = 1e-8*(Emax + 1e-30)

Only bins with `E > threshold` are processed.

For each accepted bin:

    inv = 1/(E + 1e-30)

    time_cross = real(YT)*real(Y) + imag(YT)*imag(Y)

    frequency_cross = imag(YD)*real(Y) - real(YD)*imag(Y)

    reassigned_sample = nominal_local_center + time_cross*inv

    reassigned_row = reassigned_sample/H

    reassigned_bin = nominal_bin
                   - frequency_cross*inv*N/(2*pi)

The deposited quantity is the ordinary coefficient power `E`.

## Nearest-cell power deposition

This mode rounds both reassigned coordinates to a single output cell:

    output_row = std::lround(reassigned_row)
    output_bin = std::lround(reassigned_bin)

`std::lround` rounds halfway cases away from zero.

The row is clamped to `0 ... R-1`. The frequency bin wraps modulo `N`. Then:

    P[output_row, output_bin] += E

No power is distributed to neighboring cells. This produces the most compact
raster placement, but fractional motion occurs as discrete row/bin jumps and
the raster can contain quantization holes.

The result is FFT-shifted and converted to dB:

    output_db = 10*log10(reassigned_power + 1e-12)

An empty output cell is `-120 dB`.

## Final viewer display

The program selects the requested frequency interval and resamples it to 1024
screen columns.

- Downsampling uses max pooling.
- Upsampling interpolates in linear power.

The default **Amplitude (auto)** transfer uses the 99.5th percentile of a
4-by-4 image subsample as `reference_db` and maps:

    intensity = clamp(10^((db-reference_db)/20), 0, 1)

Optional gamma and the selected color map are then applied.

## Summary of this mode

    original complex IQ
    -> one three-FFT reassignment engine
    -> nearest-cell power deposition
    -> display transfer function

There is no latent reconstruction and no multiscale fusion in this
configuration.

---

# Concise distinction

Both modes use the same three FFTs, phase-derivative coordinate equations,
coefficient gate, stabilizers, FFT shift, and dB floor.

The exact operational difference is:

    Super-resolution → Raw IQ through SR readout
        bilinear deposition
        base aperture plus optional short/upward apertures
        optional coherent-gain normalization
        optional per-cell multiscale maximum
        fixed super-resolution hop Nbase/4

    Reassigned STFT
        nearest-cell deposition
        exactly one aperture
        no aperture normalization or fusion
        selectable hop N/8, N/4, or N/2
