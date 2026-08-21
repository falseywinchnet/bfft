# Eikonal Super Resolver

This is the first interactive scaffold for single-observation spatial
super-resolution. It is separate from the repository's multi-observation STFT
super-resolution work.

The lab uses the same image catalogue as the V3 segmenter application. It
retains a working high-resolution image as scoring truth, generates a controlled
2× or 4× low-resolution observation, and gives only that observation to the
reconstructor. Four reduction controls are exposed:

- literal point decimation;
- block-average prefiltering followed by decimation;
- Lanczos prefiltering followed by decimation; and
- source-measured Eikonal prefiltering into the smaller lattice.

`Eikonal prefilter` is intentionally not information-equivalent to point
decimation. It measures a tensor on the retained source and uses the local
Eikonal chart to integrate directionally related source samples into each LR
pixel, with same-support clamping. It models a richer forward observation in
which fewer pixels can carry more organized source information. The UI calls
this out in its status and tooltip so its results are not confused with an
inverse method recovering more information from the same observation.

The initial reconstruction is the repository's local Eikonal-chart Lanczos
operator. The default mode estimates a structure tensor directly on the
observation. The slower `Current V3 structural owners` mode builds the live V3
segmentation and forbids reconstruction taps from crossing its structural
owners. Neither mode generates a missing-frequency posterior yet.

## Provenance boundary

The earlier `experiments/v3_super_resolution.py` was last committed on
2026-07-31 (`93ec7f8`). Segmenting V3 continued changing through 2026-08-02
(`cd6eddd`), so the old experiment's continuous lift and posterior math are
relic-grade relative to the current segmenter. This folder does not import or
reuse that math. Its only V3 path calls the live `build_segmenting_v3` and reads
the owners and tensor returned by that call. The Eikonal interpolation kernel
comes directly from the maintained `port_needed/eikonal_lanczos.py` primitive.

The UI reports RGB MSE, mean local RGB SSIM, and high-pass (`sigma=1`) MSE for
both ordinary Lanczos and Eikonal reconstruction. `Fine error difference` is a
signed view: teal means the Eikonal reconstruction reduced fine-band absolute
error, and magenta means it increased it. The truth participates only in these
post-reconstruction measurements.

Run the application with the MacBook GUI environment:

```sh
.venv-jpeg/bin/python -m experiments.super_resolver.app
```

Run the core tests on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.super_resolver.test_super_resolver
```

This folder is intentionally a scaffold. Denoising, blur estimation/inversion,
and any learned or probabilistic high-frequency prior are later stages and are
not silently approximated here.
