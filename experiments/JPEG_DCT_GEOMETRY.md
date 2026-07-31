# JPEG coefficient-geometry experiment

`jpeg_dct_geometry_reassembly.py` operates on the native quantized coefficient
arrays of a JPEG. It does not decode to RGB and then re-encode the experimental
result.

The default phase-correct operator works per native JPEG component:

1. Dequantize and inverse-transform the real 8×8 blocks onto the component's
   native sample lattice.
2. Restrict that lattice by exactly two with a 2×2 box.
3. Lift it to the original lattice with owner-masked eikonal Lanczos using the
   v3 structural quotient.
4. Transform the lifted samples back into the source-aligned 8×8 DCTs.
5. Measure signed local coherence of every original `(u, v)` mode over the
   block lattice.
6. Relax only each mode's incoherent fraction toward the lifted coefficient,
   with increasing freedom toward high spatial frequencies. DC is exact by
   default.
7. Requantize with the source component's original quantization table and
   write the integer coefficients back into the original JPEG structure.

Restriction must happen on the sample lattice. Directly averaging equal
`(u, v)` modes across four child blocks is phase-wrong: an 8×8 DCT basis is
anchored to its block, so four child modes mix into the parent basis under a
2× scale change. The optional `--operator mode_plane` is retained as a
diagnostic demonstration of that failure.

The source's progressive mode, 4:2:0 sampling factors, dimensions, and
quantization tables are retained. Its ICC APP2 payload is copied without
alteration. The control file performs an identity
coefficient round-trip, which distinguishes writer differences from changes
caused by the operator.

By default the operator is rate-neutral in coefficient support. Any net-new
nonzero modes produced by the geometry lift must displace the weakest
low-coherence modes, ranked in dequantized energy with a high-frequency cost.
This makes the experiment an exchange of representational support instead of
allowing restoration quality to come from an unconstrained entropy increase.
It then searches the same ranking until the actual progressive JPEG, including
the preserved ICC payload, does not exceed the source byte count. This second
constraint is necessary because equal nonzero counts do not imply equal
Huffman-coded length.

Install the experiment-only coefficient reader in the repository virtual
environment:

```sh
.venv/bin/python -m pip install jpegio
```

Run the supplied image:

```sh
.venv/bin/python experiments/jpeg_dct_geometry_reassembly.py \
  /Users/quentinkuttenkuler/Downloads/1500x500.jpeg
```

Outputs are written to `experiments/out/jpeg_dct_geometry/`. The JSON report
includes byte size, exact coefficient-change counts, decoded fidelity, an
8×8 block discontinuity measure, and flat-region Laplacian energy. These
metrics describe the intervention but cannot establish restoration quality
without the pre-JPEG source.

The default strengths deliberately stay mild. On the supplied card, `0.12`
changes roughly five hundredths of one percent of all coefficients before the
rate exchange, retains about 59 dB decoded fidelity, and slightly lowers
flat-region high-frequency energy. Larger values are useful controls but
quickly become ordinary detail removal.
