# Autofocus metric forensics

## The identifiability split

There are three different problems that should not share one scalar:

1. **Active contrast autofocus** compares the same support at several lens
   positions. A raw high-frequency score is valid because scene content
   cancels between observations.
2. **Phase autofocus** obtains a signed displacement from two sub-aperture
   images. A rendered RGB photograph does not contain that pair.
3. **Single-frame defocus** estimates optical blur while the unknown scene
   spectrum is a nuisance variable. A raw sharpness score is not an absolute
   blur measurement.

The previous object-support experiment used a calibrated coherent-edge blur
measurement as an interface term. It was correctly small and consequently
changed little. Raising its weight without changing its role would turn
intrinsic texture frequency into false depth.

## Active sweep controls

`experiments/autofocus_metric_forensics.py` evaluates raw Tenengrad, modified
Laplacian, reblur loss, and normalized variants. On a fixed fractal scene
blurred by sigma `0, 0.8, 1.6, 3.2`, these scores are monotone:

- raw Tenengrad;
- modified Laplacian;
- reblur loss;
- variance-normalized reblur loss.

Variance-normalized Tenengrad is rejected. It rises from `0.396` to `0.485`
between sigma 0 and 0.8 because the variance denominator contracts faster
than the gradient numerator. The failure is retained as a regression test.

For an actual controllable lens, raw reblur loss is the preferred cheap
objective in this codebase. It uses only one Gaussian blur and one reduction,
has the correct sweep ordering, and avoids a second-derivative noise penalty.
It is not used to compare unrelated image regions.

## One-frame texture defocus

Locally approximate the latent two-dimensional spectrum by

```text
P(k) = C |k|^-alpha.
```

Gaussian optical blur `sigma` and a Gaussian derivative aperture `s` produce
gradient energy

```text
E(s) = K (s^2 + sigma^2)^-q,
q = (4 - alpha) / 2.
```

Hence

```text
log E(s) = c - q log(s^2 + sigma^2).
```

The intercept absorbs contrast, `q` absorbs the unknown natural spectral
slope, and a short fixed ladder selects `sigma`. The regression residual is
an identifiability confidence.

A single sinusoid is a counterexample: its narrow angular spectrum can mimic
blur curvature. Structure-tensor coherence therefore gates the texture
model. Coherent support is routed to the exact calibrated step-edge reblur
estimator; locally isotropic texture uses the power-law estimator.

Measured weighted-median texture estimates after added Gaussian blur:

| image | sigma 0 | sigma 0.8 | sigma 1.6 | sigma 3.2 |
|---|---:|---:|---:|---:|
| astronaut | 0.0 | 1.4 | 2.0 | 4.0 |
| coffee | 0.0 | 1.0 | 2.0 | 4.0 |
| Pikachu | 0.0 | 0.0 | 1.4 | 2.8 |

Pikachu at sigma 0.8 remains below the scale ladder's reliable resolution.
That is reported as unresolved rather than sharpened into a false estimate.

The edge and texture estimates are now fused on frozen transport support.
At 256 pixels, evidence coverage above 0.02 changes from sparse edge ridges
to:

| image | edge + texture coverage |
|---|---:|
| astronaut | 62% |
| coffee | 65% |
| Pikachu | 24% |

## Signed chromatic cue

Longitudinal chromatic aberration can encode defocus sign through the relative
red/green/blue edge widths. The implementation requires a common physical
edge in all channels, rejecting material-colour boundaries.

The synthetic channel-blur ordering test is positive. Real rendered controls
are effectively null:

- astronaut weighted red-minus-blue log-scale: about `+0.0001`;
- Pikachu: about `-0.0003`;
- coffee: about `-0.018`, but with only `0.020` mean common-edge confidence.

The likely causes are weak lens aberration, demosaicing, and camera chromatic
correction. Chromatic focus is exposed as a forensic view but does not affect
objects.

## Why the object IDs still look similar

At 192 pixels, a focus-interface weight of 0.5 changes the partition only
after permutation-invariant matching:

| image | partition agreement with focus weight 0 |
|---|---:|
| astronaut | 98.7% |
| coffee | 99.3% |
| Pikachu | 99.9% |

This is expected. Focus difference was anchored to an already-decisive
interface and therefore supplied corroboration, not new walls.

Using focal evidence before highpoint formation was rejected. It amplified
small focus fluctuations into hundreds of new objects:

- astronaut: 86 to 531;
- coffee: 85 to 994;
- Pikachu: 14 to 53.

The corrected role is a **veto after geometric persistence**. Focus can reduce
the persistence of an already-existing, confidently defocused highpoint. It
cannot create a highpoint, alter cell geometry, or create an interface. At
weight 0.35 this removes one astronaut peak and changes no coffee or Pikachu
peak on the 160-pixel controls.

## BFFT scale survival is not autofocus

The existing `persistent_activity / local_null_activity` field measures
cross-scale survival. On astronaut, mean `null_confidence` increases as added
blur increases:

```text
sigma 0.0: 0.627
sigma 0.8: 0.758
sigma 1.6: 0.877
sigma 3.2: 0.961
```

Blur makes adjacent scales agree. Therefore this field is useful resolution
reliability but has the opposite ordering from a focus score. The viewer
exposes persistence, fine-scale null activity, and their ratio separately so
the distinction remains visible.

## Current decision

- Use **raw reblur loss** for a real active lens sweep over fixed support.
- Use the **coherent-edge + isotropic power-law texture estimate** for
  one-frame relative defocus.
- Keep chromatic ordering as a signed diagnostic only.
- Use focus only as a veto on already-persistent object highpoints.
- Do not expect focus to decide semantic association. Coffee cup, plate, and
  table can all be within the same focal slab; face and shuttle can both be
  sharp. That missing relation is not recoverable by increasing a focus
  weight.

Primary references:

- Bahat, Efrat, and Irani, *Non-Uniform Blind Deblurring by Reblurring*:
  https://openaccess.thecvf.com/content_iccv_2017/html/Bahat_Non-Uniform_Blind_Deblurring_ICCV_2017_paper.html
- Xu, Quan, and Ji, *Estimating Defocus Blur via Rank of Local Patches*:
  https://openaccess.thecvf.com/content_iccv_2017/html/Xu_Estimating_Defocus_Blur_ICCV_2017_paper.html
- Ho and Chen, *On the Distinction between Phase Images and Two-View Light
  Field for PDAF of Mobile Imaging*:
  https://doi.org/10.2352/ISSN.2470-1173.2020.14.COIMG-390
- Rucker and colleagues, *Accommodation to Wavefront Vergence and Chromatic
  Aberration*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3081412/

