# Operation Pink Floyd: overhead parent-volume baseline

The accepted implementation was compiled and run on the M4 Mini at 256×256.
The retained Fourier backend and analytic Gaussian oracle produced
byte-identical PPM images.

![Accepted overhead volume render](pink_floyd_volume_256.png)

| quantity | result |
|---|---:|
| spectral boundary packets | 65 |
| maximum retained modes, y / z | 12 / 17 |
| volume-density evaluations | 2,578,181 |
| Fourier terms evaluated | 76,311,236 |
| Fourier render, M4 | 123.01 ms |
| analytic reference render, M4 | 78.24 ms |
| full-resolution image difference | byte-identical |
| source energy | 1.00000000 |
| sheet half-packet | 0.50000000 |
| prism-entry reflection | 0.04789909 |
| glass absorption | 0.00391921 |
| prism-exit reflection | 0.04234818 |
| prism output energy | 0.40583352 |
| blue / green / red exit angles | -27.847 / -25.382 / -24.330 degrees |

The ledger closes to one at printed precision. The 3.517-degree blue-to-red
separation is produced by the N-F2 boundary solve; no screen-space color
offset, visible-beam primitive, or camera-directed haze is present.

The dogfood render verified four distinct pieces of ownership:

- the camera is nearly overhead and the true 3-D prism is directly beneath it;
- the source packet divides at the shared prism/sheet boundary;
- the prism boundary owns a retained internal packet and exposes it only at
  camera characteristics that actually cross it;
- the invisible membrane only modifies the parent sheet face response;
- wavelength lanes become camera-visible only after that face scatters them.

The retained representation is still intentionally unoptimized. Its dominant
cost is reconstructing the same mode families while scanning 65 lanes at many
face samples, not geometry intersection. Stage 2 therefore begins with
projected support ownership and shared spectral/mode evaluation.
