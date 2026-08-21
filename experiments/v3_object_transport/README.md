# V3 object transport

This experiment restarts object decomposition from the final V3 segmenter.  It
does not modify V3 and does not turn the historical `region_family_fusion`
control into a new object model.  V3 supplies lawful regions, its native
cartoon/texture measurements, structural ancestry, and exact planar
incidence.  The later fused Gilles--Osher/Bregman decomposition is aligned to
the same raster as an additional evidence layer; it does not alter V3's
regions.  The experiment asks what representation can transport participation
through those regions without hand-written object rules.

The key files are:

- `region_complex.py`: an evidence-only region complex at leaf or compound
  level.  It retains node measurements, region-pair measurements, and exact
  connected boundary arcs separately.
- `incidence_bundle.py`: the relation-lifted state `(region, incoming arc,
  outside region)` and all measured junction continuations.  It selects
  nothing.
- `fused_meyer_evidence.py`: a deterministic fixed-pass adapter for the later
  fused Meyer split.  Cartoon, texture, and exact unresolved residual remain
  separate directed transition coordinates.
- `run_region_audit.py`: the frozen five-image V3 control run.
- `build_pikachu_controls.py`: the exact, non-generative easy/hard Pikachu
  fixture construction; it preserves the complete original image panel.
- `audit_landmarks.py`: a sparse, held-out landmark audit.  The landmarks are
  used only after inference and never enter V3 or a quotient.
- `render_control_atlas.py`: renders source, V3 reconstruction, compound
  regions, and the historical family control side by side.
- `connection_bloom.py`: joint empirical whitening, the first seed-free heat
  control, signed connection nulls, and the all-scale Green control.
- `contour_transport.py`: exact connected one-sided boundary components.
- `relative_enclosure.py`: every frame-bounded relative-complement manifold.
- `participation_algebra.py`: the complete tensor algebra of role, contour,
  and enclosure coordinates, with no selected image-specific combination.
- `junction_depth.py` and `depth_contour_transport.py`: raw three-sector cap
  records, signed transition continuation, and contrast-normalized focus
  ownership lifted onto one-sided contour runs.
- `contour_cycle_nesting.py`: exact mod-2 winding supports and their
  covariance-normalized centered participation coordinate.
- `compositional_bloom.py`: spectral all-order and complete typed order-two
  controls used to test—and reject—generic transitive completion.
- `amodal_contour_transport.py` and `depth_hodge.py`: ternary T-port
  continuations that keep the occluder as context, plus the exact gradient /
  cyclic decomposition of all local depth arrows.
- `support_manifold_transport.py`: a second-level assembly coordinate between
  every bounded manifold and every possible spatial support.
- `proposal_topology_transport.py`: the explicit off-diagonal
  support--manifold incidence and its closed-form unit heat bloom, avoiding
  the support-identity diagonal of the earlier proposal Gram.
- `multiscale_proposal_transport.py`: the exact common-pixel overlap multiplex
  used to test typed paths through independently inferred V3 resolutions.
- `wavelet_leader_evidence.py`: the non-iterative, scale-causal content chart
  adapted from arXiv:2501.08694; raw leader coordinates and affine scale laws
  are retained without importing Potts labels or Gibbs inference.
- `run_wavelet_incidence_transport.py`: transition-only versus complete
  ordered endpoint leader state inside the directed incidence fibre, including
  matched region-correspondence shuffles.
- `wavelet_split_transport.py`, `run_wavelet_split_transport.py`, and
  `run_wavelet_gated_transport.py`: analytical negative controls for treating
  content as a dense transport generator, both directly and after exact
  boundary-role Schur gating.
- `run_object_packet_algebra.py`: the first all-region soft-packet experiment;
  it tests—and rejects—collapsing structural parts, two content payloads, and
  ordered incidence role into one scalar complete kernel.
- `audit_resolution_stability.py`: evaluation-only exact-point and aperture
  scale-space over the frozen 128/256/384 V3 runs.
- `render_support_manifold_atlas.py`, `render_scale_curvature_atlas.py`,
  `render_proposal_topology_atlas.py`, and
  `render_wavelet_leader_atlas.py`: diagnostic field atlases for assembly,
  scale curvature, explicit proposals, and leader content.
- `render_wavelet_incidence_atlas.py`: source, transition and ordered-endpoint
  incidence fields, final proposal transport, and the ordered shuffle null.
- `render_research_atlas.py`: the current five-control participation atlas.
- `RESULTS.md`: the first-principles findings and proposed next experiment.
- `RESEARCH_PLAN.md`: frozen images, typed packet construction, matched nulls,
  recomposition outputs, and the promotion gate for a future object quotient.

## Frozen controls

`assets/` contains the byte-exact easy Pikachu, its deterministic hard-frame
derivative, plus the standard coffee, astronaut,
checkerboard, and coins controls.  Their hashes and provenance are in
`assets/README.md`; the held-out evaluation points are in
`assets/landmarks.json`.

## Reproduce

The V3 run is CPU-heavy and should use the repository's M4 Mini mirror:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.v3_object_transport.run_region_audit \
  --out /tmp/v3_object_transport_audit_256_bundle --side 256
```

Copy the remote `/tmp/v3_object_transport_audit_256_bundle` back immediately,
then run the lightweight audits locally:

```sh
.venv-jpeg/bin/python -m experiments.v3_object_transport.audit_landmarks \
  experiments/v3_object_transport/results/v3_object_transport_audit_256_bundle

.venv-jpeg/bin/python -m experiments.v3_object_transport.render_control_atlas \
  experiments/v3_object_transport/results/v3_object_transport_audit_256_bundle

.venv-jpeg/bin/python -m experiments.v3_object_transport.render_research_atlas
```

Run each SciPy participation measurement on the M4 Mini with `m4build`, an
explicit remote `/tmp` output, and immediate copy-back.  The modules are
`run_connection_bloom`, `run_signed_connection`, `run_contour_transport`,
`run_relative_enclosure`, `run_junction_depth`,
`run_depth_contour_transport`, `run_contour_cycle_nesting`,
`run_participation_algebra`, `run_compositional_bloom`,
`run_amodal_contour_transport`, `run_support_manifold_transport`,
`run_proposal_topology_transport`, `run_multiscale_proposal_transport`, and
`run_wavelet_leader_transport`, `run_wavelet_incidence_transport`,
`run_wavelet_split_transport`, `run_wavelet_gated_transport`, and
`run_object_packet_algebra`.

The leaf-level winding falsification is intentionally limited to coffee:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m experiments.v3_object_transport.run_contour_cycle_nesting \
  --level leaf --controls coffee \
  --out /tmp/v3_contour_cycle_nesting_leaf_coffee
```

Run the exact representation tests on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest experiments.v3_object_transport.test_region_complex
```

The canonical checked-in audit location is
`results/v3_object_transport_audit_256_bundle/`.

Inspect every saved region kernel interactively without rerunning V3:

```sh
.venv-jpeg/bin/python viewer/v3_object_transport_results_app.py
```

Click any source, heat, or recomposition panel to change the anchor region.
The operator menu includes canonical parts, proposal transport, raw and
scale-law leader payloads, directed incidence arms, every matched shuffle,
the rejected dense-diffusion controls, and the first scalar packet experiment.
Use `--results experiments/v3_object_transport/results/v3_object_transport_hard_256`
to inspect the corrected hard fixture where a corresponding saved operator is
available.

That historical bundle contains the supplied easy Pikachu under the original
mislabel; `RESULTS.md` preserves and discloses it rather than rewriting prior
measurements.  The corrected hard-frame rerun is
`results/v3_object_transport_hard_256/`, and the exact-complement diagnostic of
the easy raster is `results/v3_object_transport_easy_contrast_invert/`.

The resolution falsification repeats the complete upstream pipeline at sides
128 and 384.  Their checked-in locations are `results/v3_object_transport_128`
and `results/v3_object_transport_384`; regenerate the consolidated audit with:

```sh
.venv-jpeg/bin/python -m \
  experiments.v3_object_transport.audit_resolution_stability
```
