# Blur anomalies in one transport law

## No anomaly catalogue enters reconstruction

The synthetic generator now includes displaced radial centers, faint ghost
copies, rotation, shear, and anisotropic scale exposure. Those names describe
how a controlled observation was made. Reconstruction still receives one
positive conditional transport

```text
y(p) = integral x(q) K(p,dq).
```

Every affine atom has a destination-to-source map `q=A p+a`. Mixtures are
positive measures over those maps. Composition multiplies their matrices and
transports their offsets, so a radial/rotation/ghost observation is one `K`,
not three solver stages. The matrix-free forward and exact scatter adjoint
retain only the affine atoms; their state does not grow with image area.

`observation_anomalies.py` supplies these controlled measures:

- `translation_mixture_measure`: finite positive camera/ghost offsets;
- `ghost_measure`: identity mass plus one faint displaced copy;
- `rotation_exposure_measure`: binomial angular exposure about any center;
- `shear_exposure_measure`: continuous affine shear coordinates;
- `astigmatic_scale_measure`: anisotropic scale exposure in a rotated chart;
- `radial_scale_measure` from `composed_transport.py`: isotropic log-scale
  exposure about any center.

The workbench additionally exposes decentered double radial, radial plus
rotation plus ghost, and a compound lens anomaly. These options exist only
under **Blur image**. An unknown **Use as-is** image receives none of their
names or parameters.

## Sampling anomalies are constraints, not displacement

Clipping, quantization, and missing pixels are not positive spatial blur. A
maximum-code sample says the unencoded transport prediction is at least a
threshold. A quantized sample supplies an interval. A missing sample supplies
no evidence. `ObservationBounds` therefore carries

```text
lower <= Kx <= upper,
precision >= 0.
```

At every descent pass the predicted observation is projected into its
admissible interval:

```text
target = clamp(Kx, lower, upper).
```

Only the distance from `Kx` to `target` drives the matched adjoint. Predictions
already inside an interval receive no invented residual, saturated pixels are
inequalities rather than false white equality data, and dead pixels have zero
precision. The optimal positive line step uses the same active interval
residual. This is an analytic censored likelihood, not empirical detection of
a codec or sensor type from appearance.

The Dear PyGui controls use `Linear quantization levels` and `Dead-pixel
pattern period`. They are enabled only for known consolidated synthetic
operators with read and shot noise disabled, because an exact quantizer
preimage must not silently claim to bound an unmodelled noise realization.

## Measured 96x96 checkpoint

The M4 Mini battery uses the six denoiser scaffold sources and 64 positive-line
passes:

| Unified observation | Raw observation | Consolidated inverse |
| --- | ---: | ---: |
| displaced unequal double radial | 28.049 dB / 0.8468 | 36.417 dB / 0.9450 |
| radial + rotation + ghost | 29.248 dB / 0.8829 | 44.735 dB / 0.9837 |
| astigmatic scale + shear + ghost | 31.211 dB / 0.9336 | 46.724 dB / 0.9948 |

The sensor-anomaly control applies gain 1.8, 32 quantization levels, roughly
2% missing pixels, and source-dependent saturation:

| Sensor treatment | PSNR / SSIM |
| --- | ---: |
| encoded damaged observation | 10.890 dB / 0.2921 |
| incorrect equality inverse | 10.548 dB / 0.1984 |
| bounded interval inverse | 20.549 dB / 0.7531 |

The equality control is important: ordinary deconvolution amplifies the wrong
claim and performs worse than leaving the damaged raster alone. Interval
transport restores ten decibels without inventing measurements at clipped or
missing samples.

These remain known-operator controls. They prove that the shared basis and
inverse can carry the anomalies once their evidence is available; they do not
claim that one unknown image uniquely identifies its transport.

## Next anomalies

The next lifted-state gates are channel/wavelength-dependent transport,
depth/visibility-dependent defocus, and turbulence whose row measure evolves
over exposure time. Channel aberration should be one operator on
`position x wavelength`, not three unrelated RGB deconvolutions. Occlusion
must carry ownership and cannot be represented by a normalized row cloud
alone. Turbulence must compose time-varying geometry and mixing rather than
being replaced by a wider static kernel.

Run the controls with:

```sh
python3 -m unittest personal_deblurrer.test_observation_anomalies
python3 -m personal_deblurrer.run_observation_anomaly_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_observation_anomalies
```
