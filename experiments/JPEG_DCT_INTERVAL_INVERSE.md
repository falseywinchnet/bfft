# JPEG quantization-interval inverse

The source JPEG does not reveal a clean coefficient. It reveals an interval:

```text
Q (q - 1/2) <= c < Q (q + 1/2)
```

`jpeg_dct_interval_inverse.py` builds the phase-correct v3 restriction/lift
proposal and clips it into each observed interval in one closed-form
projection. This separates two limitations that an ordinary deartifacting
comparison conflates:

- **Prior limitation:** Does choosing the geometrically simplest legal point
  inside every interval produce a visibly better latent image?
- **Representation limitation:** Can baseline JPEG express those within-bin
  positions under the original byte budget?

The experiment emits:

- `precision_center_control.jpg`: original bin centres represented with
  quantizers of one.
- `latent_interval_projection.jpg`: projected within-bin positions represented
  with quantizers of one.
- `rate_matched_center_control.jpg`: bin centres represented with quantizers
  approximately half as large, then utility-pruned to the source byte budget.
- `rate_matched_interval_projection.jpg`: the latent projection under the same
  representation and byte constraint.

The precision pair isolates the inverse prior. The rate-matched pair isolates
whether spending bits on within-bin placement beats spending them on the
ordinary bin centres.

Run:

```sh
.venv/bin/python experiments/jpeg_dct_interval_inverse.py \
  /Users/quentinkuttenkuler/Downloads/1500x500.jpeg
```

## Supplied-card result

The experiment rejects the current prior as a restoration criterion:

- The quantizer-one centre control decodes exactly to the supplied JPEG.
- The interval projection moves 1,087,990 of 1,143,040 latent coefficients.
- Its precision representation grows from 96,475 to 210,441 bytes and worsens
  decoded block-boundary and flat-region measures.
- At the source byte budget, the centre control is 48.01 dB from the source
  decode while the interval projection is 43.98 dB and visibly loses detail.

The reason is structural. A zero quantized coefficient still represents a
wide interval, and the v3 restriction/lift generally proposes a nonzero point
inside it. Quantization consistency declares that invented energy legal; it
does not identify it as truth. Consequently the JPEG forward model is a
constraint but not an artifact detector. A useful inverse needs evidence
independent of the corrupted block coefficients, such as a block-phase
invariance or paired cross-boundary continuation measurement.
