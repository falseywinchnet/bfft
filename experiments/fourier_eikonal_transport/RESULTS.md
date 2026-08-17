# Positive-control result ledger

This experiment is a mechanics test, not evidence that the method discovers
unknown structure. The synthetic latent spectra are smooth shell sections, so
their generator already satisfies the kind of connection that shell transport
is designed to exploit. The controls identify when a supplied transportable
connection can be used safely; they do not establish that such a connection
can be learned without task-aligned assumptions.

The stabilized Fourier-shell atlas was evaluated on 40 independent latent
spectra with five observations and 10% relative noise. All methods used the
same observations, estimated complex connection, and atlas seeds.

| method | whole NMSE median | whole NMSE p90 | blind-wedge NMSE median | paired wedge wins vs isotropic |
|---|---:|---:|---:|---:|
| direct fusion | 0.0647 | 0.2034 | 1.0000 | 12.5% |
| isotropic atlas | 0.0216 | 0.0604 | 0.3325 | - |
| shuffled metric | 0.0249 | 0.0621 | 0.4134 | 12.5% |
| Eikonal atlas | **0.0134** | 0.0682 | **0.2841** | **65.0%** |

The initial multiplicative-amplitude version had catastrophic path errors. The
stabilized version transports phase, bounds transported amplitude by observed
shell quantiles, and shrinks mutually inconsistent chart predictions. This
reduced mean whole-spectrum NMSE from 0.0479 to 0.0277 and removed the extreme
whole-spectrum outliers, though Eikonal's p90 remains slightly worse than the
isotropic atlas.

## Regime sweep

Twenty paired trials were run for every combination of 3, 5, or 8 observations
and 5%, 10%, or 20% relative noise.

- Correctly located Eikonal transport won whole-spectrum error in 60-95% of
  trials across every cell.
- Median whole-spectrum error was 16-55% lower than isotropic transport.
- Shuffling the metric consistently removed most or all of this gain.
- Blind-wedge recovery was strongest at 5% noise: paired win rates were 70%,
  75%, and 85% for 3, 5, and 8 observations.
- At 20% noise, blind-wedge win rates fell to 55%, 45%, and 40%. The inferred
  connection is then too uncertain to justify long phase transport, even though
  metric-weighted denoising still improves the observed spectrum.

## Limited answer to the transport question

Transport is useful, but two different effects must be separated. The metric
reliably improves denoising by routing atlas support through coherent shell
regions. Extrapolation into a genuinely blind wedge works only when the
cross-observation connection is sufficiently well estimated. More filters or
more transport cannot repair an uncertain connection; the method needs an
explicit transport-confidence gate or a shorter-path fallback.

Within this positive control, Fourier circles are not merely a radial task prior. They are
the domain on which repeated observations define both a connection and its
uncertainty. Eikonal geometry adds value by deciding where that connection may
be trusted.
