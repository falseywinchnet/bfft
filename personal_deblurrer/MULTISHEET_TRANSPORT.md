# Positive multi-sheet appearance transport

## Why the one-field model stops

A single dense field assigns one source coordinate to each sensor coordinate.
That is sufficient for smooth deformation and, with joint adjoint coverage,
can continue through an individual fold. It cannot express a foreground and a
background that carry different appearances and different motion at the same
sensor location. Adding nearby flow particles to one exposure cloud was
measured and rejected: it mixes one appearance along several paths rather than
transporting several appearances.

## Exchangeable positive formation law

For observation `i` and motion sheet `s`, let `A_i,s` be the existing positive
spatial exposure operator, `x_s` a distinct latent appearance, and `pi_i,s` a
continuous sensor ownership measure. The forward model is

```text
y_i = sum_s pi_i,s A_i,s x_s
pi_i,s >= 0
sum_s pi_i,s = 1.
```

There is no winning class. Simultaneously permuting `x_s`, `pi_i,s`, and
`A_i,s` leaves the model and reconstruction invariant. For fixed ownership,
the matched positive update is

```text
x_s <- x_s
       [sum_i A_i,s^*(w_i pi_i,s y_i / yhat_i)]
       / [sum_i A_i,s^*(w_i pi_i,s)]

yhat_i = sum_s pi_i,s A_i,s x_s.
```

This is the same normalized forward/adjoint transport used by the single
latent solver, lifted from one image state to an exchangeable collection of
image states. Reference ownership is either supplied as a continuous measure
or induced by matched adjoint coverage. `multisheet_transport.py` contains the
executable law and audits the positive simplex, coverage, residual, entropy,
uncertainty, and permutation role.

## Representation oracle

`multisheet_results/results.json` is deliberately a known-measure control. It
uses a stationary background appearance and one independently moving
foreground appearance, continuous antialiased opacity, exact positive fields,
two captures at +/-3 pixels, and read noise sigma 0.002. Neither the geometry
nor ownership is estimated in this control.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Best capture | 22.350 dB | 0.8587 |
| Unregistered average | 24.987 dB | 0.8747 |
| Known-measure multi-sheet oracle | 55.020 dB | 0.9990 |

All six sources improve over averaging; the minimum and mean gains are 27.591
and 30.033 dB. The M4 Mini run takes 0.18 seconds. The selected artifact has
SHA-256
`08c5733c725927641b3283b95b373afa9676aaa61f0c5c3c8408945e8dfcc360`.

This result proves that distinct latent appearance is the missing
representation, not that unrestricted blind ownership is identifiable from a
pair. `FLOW_FIBER_ESTIMATION.md` records the next checkpoint: global and
overlapping local Fourier-circle phase charts now supply blind positive support
for straight, folded, and curved layered motion. Forward/reverse
cross-prediction distributes a soft measure over five tensor-quadrature plans,
and continuous authority suppresses the deliberately poor raw atlas on smooth
single-connection deformation. Native ABI v5 batches only the verified
positive operators; estimation and authority remain auditable Python.
