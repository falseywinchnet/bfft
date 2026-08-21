# Personal deblurrer: exposure-transport blur estimation and removal

This experiment treats blur as transport during a finite exposure. A blur
kernel is a positive unit-mass measure on image displacement, not an arbitrary
signed stencil. For a global translation path `gamma(t)`,

```text
y(x) = integral x0(x - gamma(t)) dt
Y(k) = H(k) X(k)
H(k) = integral exp(-i k dot gamma(t)) dt.
```

The optical transfer function is therefore the characteristic function of an
exposure-path measure. Fourier circles are the natural audit surface: each
circle shows which directions and radii the capture transported, attenuated,
or did not measure.

## Executable method

1. `kernels.py` constructs positive, exactly normalized analytic and sampled
   exposure paths. Forward and adjoint circular transport close to floating
   point precision. Wronski's `[1/4, 1/2, 1/4]` filter, its repeated binomial
   powers, and its separable product are represented as the same positive
   displacement measure. `analytic_support.py` derives the measure's centroid,
   covariance eigenframe, third/fourth cumulants, and exact Fourier-eikonal
   attenuation flow without choosing a blur family; characteristic zeros are
   returned as unsupported directions. See `WRONSKI_OPERATOR.md`.
2. `estimation.py` estimates a pair of registered blur operators from the
   phase-preserving cross-observation closure

   ```text
   Y0 H1 - Y1 H0 = 0.
   ```

   This cancels the unknown latent scene without dividing by it. Residuals are
   pooled per Fourier circle. A blur common to both captures is an exact gauge;
   the estimator fixes a maximally covered representative for computation and
   separately reports that the common blur was not identified.
3. `circles.py` computes literal joint spectral coverage. Relative weights
   answer *which capture* to trust; absolute coverage answers *whether any
   capture* measured a coefficient. The joint dead fraction is returned with
   every reconstruction.
4. `solver.py` minimizes one shared-latent objective for all observations. Each
   pass performs one exact Fourier linear solve, one isotropic flux projection,
   and one persistent Bregman update. There are no nested inner solves,
   continuation schedules, or primal momentum. When the joint dead band is too
   large, the flux descent abstains and uses a conservative Tikhonov inverse.
5. `run_benchmark.py` uses the six-source image scaffold from `denoiser` and
   keeps clean truth evaluation-only. It includes single-blur, complementary
   motion, mixed motion/defocus, identical-blur, and three-angle tomography
   controls.
6. `uncertainty.py` retains a robust finite law over positive path pairs,
   transports its credible branches through the inverse, and returns the
   mixture image, spatial standard deviation, credible limits, entropy,
   retained probability, and a limited noise/model-discrepancy audit.
7. `workbench.py` is a Dear PyGui laboratory for loading images, synthesizing
   Gaussian, defocus, line, curved, random-path, radial-scale, double-radial,
   rotation, and rolling-shutter exposure blur with sensor gain plus read or
   shot noise, running
   known-operator deblurring, and estimating/deblurring an
   explicitly chosen registered pair through one continuous positive
   flow-atlas consensus. Its dedicated relative-aberration panel selects an
   explicit same-scene capture set, runs the blind affine-aberration recovery,
   reports quadratic-jet crossfit and stationary-candidate authority, warns
   about the common-lens gauge, and renders raw and fitted relative tensor
   fields as diagnostic tabs. Equal image dimensions are never treated as
   registration. The
   finite global-kernel posterior remains a headless estimation control, not a
   visible reconstruction mode.
8. `decomposition.py` is the active single-observation path. It factors every
   positive exposure law into deterministic centroid transport followed by a
   centered mixing measure, removes the shift first, and then applies positive
   forward/adjoint center transport. Reflection padding removes periodic edge
   wrap; an absolute Fourier-coverage projection suppresses unsupported
   exposure nulls instead of converting them into ringing. A transported local
   fixed-point gate additionally gives zero correction authority to constant
   regions, suppressing positive null-space echoes without selecting an edge
   or blur family. `curvilinear.py`
   orders every positive PSF atom in one quadratic Eikonal exposure tube and
   applies the same reflected gather with its exact scatter adjoint to center
   clouds, lines, and curves. Two endpoint seed gauges are transported only
   for uncertainty. A differentiated line recurrence is merely a continuously
   weighted auxiliary constraint: covariance, tangent coherence, residual
   demand, descent action, and lattice alignment drive its authority without a
   blur-family switch. Every descent stage shares one robust noise-discrepancy
   stop and may spend up to 32 exact passes.
9. `native/path_operator.cpp` optionally executes the already-verified flat
   and spatial gather/scatter plans through versioned C ABI v6. Its batched
   entry points cross the Python boundary once for all exchangeable flow
   sheets while preserving their distinct appearances and contribution
   counts. ABI v6 additionally runs generated covariance operators concurrently
   across captures. `native_backend.py` loads it when present; the NumPy
   implementation remains the exact fallback and test oracle. Native code
   owns no estimation, authority, or stopping policy.
10. The padded positive basin reuses one circular OTF, carries its previous
    forward state, and uses the nonredundant real half-spectrum. Its diagnostic
    ledger records the exact forward/adjoint counts and the removed redundant
    evaluations.
11. `spatial_transport.py` generalizes the same positive measure to a
    displacement cloud at every sensor pixel. Its single-atom limit is
    deterministic warp and its constant multi-atom limit is the existing
    global blur. It analytically inverts the barycentric map first, transports
    the centered exposure into latent coordinates, and applies one matched
   bilinear gather/scatter descent. Native ABI v6 accelerates those verified
   scalar and batched plans and generates a nine-point positive covariance
   measure from compact eigenaxis fields without materializing spatial plans.
12. `spatial_estimation.py` estimates a continuous rotational trajectory from
    forward/reverse evidence for every observation pair, closes all edges in
    one cycle-consistent solve, and transports every capture into one shared
    latent descent. Relative rotation is measured; common rotation and common
    exposure remain an explicit gauge. Identical captures therefore abstain.
13. `dense_estimation.py` removes the rotational-manifold restriction. One
    multiscale two-component field simultaneously carries translation, affine
    motion, shear, rotation, and smooth local deformation. Forward/reverse
    cycle closure supplies uncertainty; a metric-harmonic solve transports
    connection confidence through supported Eikonal paths. It produces the
    same `SpatialExposureField` consumed by `spatial_consensus.py`, so no motion
    estimate selects a reconstruction branch.
14. `spatial_consensus.py` accepts a positive spatial precision measure for
    every observation and derives latent ownership from adjoint joint coverage.
    Individual folds omit the invalid coordinate preconditioner but retain the
    same positive normal equation. A multi-view solve abstains only where joint
    coverage fails, rather than treating one folded map as a global veto.
15. `multisheet_transport.py` lifts that positive normal equation to several
    exchangeable latent appearances. A continuous simplex measure composites
    their independently transported predictions; simultaneous permutation of
    appearances, measures, and fields changes nothing. Its present benchmark
    is a known-measure representation oracle, not blind sheet estimation.
16. `flow_fiber_estimation.py` places one exchangeable positive tensor measure
    over displacement scale and global/local Fourier-circle atlas coordinate,
    while retaining the dense spatial connection as the common gauge.
    Forward/reverse cross-prediction supplies soft ownership. Coherence,
    disagreement with both atlas coordinates, local support sparsity, and
    Jacobian fold non-closure contribute continuous authority. Every active
    atlas receives one warm multi-appearance sweep.
17. `radiometric_transport.py` estimates one symmetric relative-exposure gauge
    from full-distribution quantile transport and attaches a continuous
    positive precision to clipped sensor samples. That precision is carried
    through dense geometry, atlas cross-prediction, ownership backprojection,
    and reconstruction. `Rolling shutter exposure` is another spatial positive
    path field, not a solver family.
18. `relative_mixing_transport.py` estimates the identifiable centered-mixing
    covariance difference from low-frequency Fourier-magnitude transport after
    deterministic center estimation. Signed covariance eigenspaces become an
    exchange-symmetric pair of positive measures. Common blur is reported as a
    gauge rather than assigned to a preferred capture. `real_capture_evaluation.py`
    preserves input hashes and audits forward closure, uncertainty, local
    envelope excursion, and Fourier-circle amplification.
19. `multicapture_transport.py` closes relative exposure, Fourier-circle
    center, and centered-mixing covariance on the complete observation graph.
    A minimum-trace positive covariance program fixes only the computational
    gauge; all captures remain positive measures in one shared solve. Cached
    observation spectra remove pair-local FFT duplication, exact positive-line
    descent supplies the optimal bounded step along each transport direction,
    and compact global native plans avoid replicated spatial coefficients. Its
    next local mode closes the same graph independently on overlapping Fourier
    charts and blends positive covariance gauges into one spatial measure,
    without choosing a chart or family. See `MULTICAPTURE_TRANSPORT.md`.
20. `quartic_shape_transport.py` transports the next relative log-magnitude
    cumulant without changing covariance. Fourier-circle crossfit and an exact
    zero-mean common-shape gauge control two positive axis side masses per
    capture or local chart. ABI v5 introduced the resulting spatial side-mass
    fields; ABI v6 preserves that representation while batching captures.
    Unstable real fourth-order evidence abstains exactly instead of
    choosing a blur shape; see `QUARTIC_SHAPE_TRANSPORT.md`.
21. `full_quartic_transport.py` fits all five coordinates of the symmetric
    fourth-cumulant tensor and realizes them as one convex mixture of positive
    rotated measures. Held-out predictivity, fold agreement, and a continuous
    null-floor taper transport uncertainty into shape authority. A joint
    positive program carries their common fourth-order gauge without choosing
    a capture. The remaining all-blurred gauge failure is explicit; see
    `FULL_QUARTIC_TRANSPORT.md`.
22. `quartic_gauge_posterior.py` reconstructs covariance and relative-K4
    positive measures together. Forward closure and absolute outer
    Fourier-circle redistribution determine continuous posterior mass; neither
    gauge is selected away. `CompactGlobalExposureField` factors constant
    deterministic translation from centered mixing without spatial broadcast,
    and its bounded parallel FFT batch keeps memory below a twelve-image
    transform slab.
23. `multicapture_posterior.py` transports center, positive inverse, and FMMT
    noise measures in one continuous posterior. Closure evidence concentrates
    as `1/sqrt(capture_count)`; overlapping chart authority becomes a spatial
    inverse-mass field; coherent closure residual suppresses noise transport;
    and fine structure replicated across captures is protected. There is no
    blur/noise/source branch. The 23-source generalization battery includes 17
    V3-era scikit-image files strictly as a chronological data holdout with no
    method inheritance.
24. `run_ringing_benchmark.py` measures broad halo RMS, oscillatory error, and
    local phase displacement on six natural sources plus canonical step and
    sparse-point controls. The selected fixed-point authority improves line,
    curve, and random-path PSNR and SSIM while removing repeated line echoes.
    First/second exposure moments share one transport call, and numerically
    inert straight-line recurrences are not executed.
25. `composed_transport.py` makes the spatial positive operator closed under
    exact discrete composition. Sequential reflected/bilinear transports
    become one row measure with one exact adjoint, so reconstruction receives
    no outer/inner factorization. Affine scale atoms supply an additive
    log-scale chart for radial exposure; a twice-applied radial operator is one
    convolved measure. The current benchmark is a known-measure representation
    and inverse gate, not a blind estimate. See
    `COMPOSED_OBSERVATION_TRANSPORT.md`.
26. `observation_anomalies.py` constructs ghost, rotation, shear, decentered
    radial, and rotated astigmatic-scale measures in that same affine
    transport. Saturation, quantization, and missing pixels instead become
    per-sample admissible intervals and precision: they constrain the one
    transport without masquerading as displacement. The workbench exposes the
    geometric compounds plus optional bounded sensor damage. See
    `OBSERVATION_ANOMALIES.md`.
27. `aberration_recovery.py` estimates a relative lens-aberration field from
    several same-scene observations without clean truth or blur-family labels.
    Pairwise Fourier-circle cancellation yields a local covariance atlas; a
    checkerboard-crossfit complete quadratic jet audits whether that atlas has
    the coordinate structure of affine lens transport. Reconstruction retains
    the raw atlas, so the diagnostic jet cannot erase unsupported anomalies.
    A lens component common to every capture remains an explicit gauge. The
    workbench exposes this exact path over explicitly checked observations and
    visualizes every recovered raw atlas and fitted jet without using synthetic
    truth. See `ABERRATION_RECOVERY.md`.

The workbench gives a loaded file exactly two input roles:

- **Blur image** preserves the loaded pixels as synthetic truth and creates a
  separate working observation with a known forward operator.
- **Use as-is for deblurring** copies the loaded pixels into an unknown real
  observation and makes no truth or kernel claim.

Deblurring never mutates either buffer. The status ledger reports the input
fingerprint audit and, only in synthetic mode, measured before/after PSNR.

The workbench also exposes the scikit-image fixtures from the Segmenter V3
portfolio as convenient source images.  They are data inputs only; the
segmenter, its labels, and its reasoning are not imported into deblurring.

The first pair benchmark remains a global, registered, periodic checkpoint.
The active desktop single-observation path uses reflect boundaries. The
implemented spatial controls cover known positive fields, estimated global
camera rotation, smooth single-layer dense flow, positive multi-view
visibility ownership under individual folds, and a known-measure positive
multi-sheet representation with distinct latent appearances. The active pair
path now infers a blind positive Fourier-circle flow atlas for straight,
folded, curved, and accelerated rolling-shutter layered motion while preserving
smooth spatial deformation to within the measured authority tolerance.
Complementary exposure and clipping are transported in a symmetric radiometric
gauge. Shared rolling-shutter gauge, lens aberration, turbulence, and broad
real-camera generalization are not claimed solved.

The intended lens-aberration continuation is a joint descent, not a blur
classifier: the sharp scene, deterministic ray map, sparse ghost mass, diffuse
center mass, and their uncertainty are updated as one forward model. The
current known-operator ringing checkpoint supplies the trust law required
before that operator itself is allowed to move.
The first immutable Köhler pair checkpoint reduces ringing when relative
mixing is enabled but remains below the untouched observation average on the
limited scene-1 web-JPEG check. It is retained as a failed pair acceptance
gate. The subsequent exchange-symmetric 12-capture checkpoint improves over
the aligned average and every individual capture on that same limited
reference without Fourier ringing amplification. Its adaptive local covariance
atlas further improves the global graph from 27.893 to 28.255 dB by transporting
only cumulant-valid, uncertainty-weighted chart deviations. It is a successful
single-scene field checkpoint, not broad real-camera evidence; see
`REAL_CAPTURE_CHECKPOINT.md` and `MULTICAPTURE_TRANSPORT.md`.

The uncertainty-aware continuation retains the same center-first atlas and
adds no classifier. Across 23 sources and five fixed blur/noise formations it
improves PSNR over center transport by 0.429 dB on average; complementary and
spatial mixing gain 0.683 and 1.109 dB. On the limited real burst it assigns
98.52% mean inverse mass and reaches 28.293 dB / 0.84189 SSIM, 0.020 dB below
the unconstrained atlas while retaining an explicit center uncertainty reserve.

The measured battery is reported in `RESULTS.md`; `UNCERTAINTY_RESEARCH.md`
separates the uncertainty types and records the next method; the collected
primary-paper set and blur-generator taxonomy are under `papers/`.

Run the composition-closure and double-radial gate with:

```sh
python3 -m unittest personal_deblurrer.test_composed_transport
python3 -m personal_deblurrer.run_composed_transport_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_composed_transport
```

## Chronology boundary

Repository history places `two_lattice_blur_study.py` on 2026-07-11 and
`transport_focus_forensics.py` on 2026-07-28, before `segmenting_v3.py` first
appeared on 2026-07-30. They are historical controls only. V3 itself supplies
no representation, segmentation, labels, or algorithm to this experiment; it
has no methodological relationship to deblurring and is solely the chronology
cutoff requested for the audit.

The active construction comes from the later program summarized in
`RESEARCH_SYNTHESIS.md`: coverage before machinery, complementary constraint
families on one latent, positive mass transport, exact adjoints and ledgers,
continuous characteristics rather than a direction catalogue, and warm
one-sweep descent in the variable that actually carries transport.
`SHIFT_MIX_FOUNDATIONS.md` records the corrected deterministic-transport then
centered-mixing ontology and its single-image identifiability boundary.
`CURVILINEAR_EIKONAL.md` gives the curve coordinate, Jacobian, exact adjoint,
normalized boundary update, and endpoint-seed uncertainty law.
`CONTINUOUS_EXPOSURE_DESCENT.md` gives the unified operator, continuous
constraint authority, discrepancy stop, execution-plan reuse, and trust law.
`SPATIAL_EXPOSURE_TRANSPORT.md` gives the spatial formation law, barycentric
coordinate factorization, Jacobian/fold ledger, and current exactness boundary.
`DENSE_FLOW_TRANSPORT.md` gives the continuous flow objective, reverse-cycle
law, metric confidence transport, symmetric gauge, and current ownership gap.
`VISIBILITY_OWNERSHIP.md` gives the positive latent ownership measure, direct
joint fold law, falsified cycle-rejection control, and moving-layer result.
`MULTISHEET_TRANSPORT.md` gives the exchangeable multi-appearance formation
law, matched positive descent, representation oracle, and blind-estimation
boundary.
`FLOW_FIBER_ESTIMATION.md` gives global/local Fourier-circle support,
forward/reverse cross-prediction, continuous atlas authority, rejected
controls, and the straight/curved/preservation batteries.
`RADIOMETRIC_TRANSPORT.md` gives the exposure gauge, continuous sensor
censoring law, accelerated rolling-shutter boundary, and selected batteries.

## Reproduce

Run on the M4 Mini from anywhere in the repository:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest \
  personal_deblurrer.test_exposure_transport_deblur \
  personal_deblurrer.test_uncertainty \
  personal_deblurrer.test_shift_mix_decomposition

/Users/ultimussecundai/.local/bin/m4build -- \
  make -C personal_deblurrer/native clean all

ssh m4mini-awdl \
  'cd /Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539 && \
   python3 -m unittest personal_deblurrer.test_native_operator'

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_benchmark \
  --size 96 --seeds 2 --out /tmp/personal_deblurrer_full

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_shift_mix_benchmark \
  --size 96 --out /tmp/personal_deblurrer_shift_mix

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_path_chart_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_path_chart

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_ringing_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_ringing

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_spatial_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_spatial

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_spatial_estimation_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_spatial_estimation

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_dense_estimation_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_dense_estimation

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_visibility_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_visibility

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_aberration_recovery_benchmark \
  --size 96 --passes 64 \
  --out /tmp/personal_deblurrer_aberration_recovery

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_multisheet_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_multisheet

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_flow_fiber_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_flow_fiber

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_circle_fiber_generalization \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_circle_generalization

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_curved_flow_fiber_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_curved_flow_atlas

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_radiometric_flow_atlas_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_radiometric_atlas

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_rolling_shutter_flow_atlas_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_rolling_atlas

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_full_quartic_battery \
  --size 96 --passes 32 \
  --out /tmp/full_quartic_positive_directional_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.profile_compact_global_transport \
  --size 800 --atoms 153

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.profile_spatial_batch \
  --size 96 --plans 5 --repeats 1000 --out /tmp/spatial_batch_profile.json
```

Copy the remote `/tmp` result directories back immediately after their runs.
The checked-in `results/`, `shift_mix_results/`, and `path_chart_results/`
directories contain the selected batteries.

Launch the desktop workbench on the MacBook with its Dear PyGui environment:

```sh
.venv-jpeg/bin/python -m personal_deblurrer.workbench
```

## Claim boundary

The present positive control demonstrates:

- exact positive exposure transport and adjoint accounting;
- phase-preserving relative kernel estimation from registered pairs;
- large recovery gains when blur families cover complementary Fourier bands;
- a measurable advantage of the warm flux solve over a closed-form inverse in
  those covered cases; and
- explicit abstention for common-blur ambiguity and large joint dead bands;
- recovery of a relative affine-aberration atlas across same-scene captures,
  with the common lens component retained as an explicit gauge.

It does not yet establish broad superiority, single-image blind field
identification, arbitrary non-translational multi-sheet support, or real-capture
performance. It also does not establish single-image blind aberration recovery
or recovery of a lens component shared by every capture. Those are promotion
gates, not implied conclusions.
