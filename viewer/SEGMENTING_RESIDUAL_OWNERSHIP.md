# Segmenting residual ownership trace

The residual display is not showing pixels outside the Voronoi partition.
The canonical hard partition assigns every output pixel exactly one nonnegative
cell ID, and the affine/ridge reductions consume every labeled pixel.

## Pikachu bottom-frame trace

At the black-to-white transition on row 440 of the default 475-square build:

- 431 pixels carry the source jump;
- 236 pixels (54.8%) have the same owner immediately above and below it;
- same-owner RGB MSE is 0.1526, versus 0.0319 where the owner changes;
- the four dominant crossing cells are IDs 1020, 1021, 1022, and 1024.

The frozen population density is concentrated at the contour. A germ can
therefore be emitted on the discontinuity. Continuous first-arrival
initialization injects each germ into its four enclosing lattice pixels. Germ
1021 is at `(232.19, 439.28)`, so its initial label is present on both black
row 439 and white row 440. Increasing the later boundary action cannot split
that already-shared label.

This is now measured directly by:

- `Same-owner source jump`;
- `Interface-aligned source jump`;
- `Germ injection source jump`;
- `Residual energy + cell boundaries`;
- `Cell mean residual energy`.

The status line also reports unowned pixels and the fraction of source-jump
mass lying inside cells. Default Pikachu has zero unowned pixels.

## Readout defect and correction

Contour-born cells are intended to recover an internal discontinuity through
the finite ridge readout. The former ridge used 41 offsets over normalized span
5. At the default population spacing this puts adjacent offsets about 1.8
pixels apart. The offending cells select the correct 90-degree normal but place
the ridge at the wrong row; increasing sharpness alone therefore regresses.

The canonical readout now uses 161 finite offsets and `kappa=16`. It remains one
bounded scan with no continuous optimizer. On the full default Pikachu path:

| Measurement | Previous | Revised |
|---|---:|---:|
| final objective | 0.002302 | 0.001541 |
| critical-row RGB MSE | 0.09797 | 0.03219 |
| final PSNR | about 27.8 dB | 29.68 dB |

At longest side 384 the same change improves Coffee and Astronaut; Cameraman
rejects the ridge and retains its affine result through the existing objective
gate.

The remaining structural question is whether contour density should continue
to create internal-ridge cells, or be dualized into paired one-sided germs so
the contour becomes a literal ownership interface.
