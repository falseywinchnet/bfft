# Denoiser: post-FMMT truth under distortion

FMMT is rejected as a denoising model and is preserved only as a falsified
historical control. The patch-typicality branch was also rejected as the
foundation. The active replacement is a continual state: radiance, residual
noise law, bounded uncertainty, and an eikonal flux metric evolve together
under an intrinsic descent/contractor gate. No post-FMMT estimator is promoted
yet.

The other BFFT denoising, compression, Meyer, and transport projects are
inspected only for mechanisms that survive this new foundation; they are not
collapsed into one objective.

## What is here

- `continual_eikonal_noise_transport_2d.py`: first patch-free fused recurrence.
  A Back-to-Basics quadratic majorizer advances radiance on a V3-derived
  Selling/eikonal flux graph; the same positive operator transports complete
  noise mixture moments, outer radius, and full-band phase sufficient
  statistics. A Z-style joint amplitude/phase contractor selects its own
  stopping point and leaves the clean Cameraman control exactly unchanged.
- `CONTINUAL_EIKONAL_NOISE_RESULT.md`: equations, invariants, 128-pixel
  Cameraman gate, the rejected shared-basis self-confirmation experiment, and
  the next full-phase statistic target.
- `continual_fabada_eikonal_2d.py`: the requested hierarchy test in which
  positive Selling/eikonal nearest-neighbour averaging is the primitive and a
  transported FABADA-style posterior over smoothing depth is the readout. It
  preserves more additive-noise geometry but is not promoted because its
  current posterior under-denoises and moves clean images.
- `FABADA_EIKONAL_AVERAGING_RESULT.md`: equations, invariants, focused and
  six-source results, and the diagnosis separating the useful averaging path
  from the failed posterior depth measure.
- `TRANSPORTED_RESIDUAL_POSTERIOR_RESULT.md`: the follow-up posterior over the
  residual law. It introduces the exact zero/noise mixture, full between-law
  variance, physical-time residual drift, the operator-split readout, matched
  positive additive-noise result, and the remaining sparse-flux problem.
- `residual_posterior_128/results.json` and
  `residual_posterior_six_source_32.json`: focused and 54-case records for the
  residual posterior, including the screened and FMMT controls.
- `fabada_eikonal_compare_128/results.json` and
  `fabada_eikonal_six_source_32.json`: matched screened/FMMT controls for the
  pure-averaging experiment.
- `test_continual_fabada_eikonal_2d.py`: constant, positive conservative
  averaging, frozen Dirichlet descent, contractor descent, and trajectory
  readout invariants.
- `run_continual_eikonal_benchmark.py` and
  `continual_phase_sasaki_128/results.json`: reproducible clean and six-law
  falsification battery, including hair/camera/tripod regions.
- `continual_phase_six_source_32.json`: six-structure, nine-corruption gate
  showing the general MSE/edge trade and the remaining resolution dependence.
- `test_continual_eikonal_noise_transport_2d.py`: constant fixed point, SPD
  metric, positive/conservative Selling flux, mixture-variance, exact
  observation identity, and per-step descent invariants.
- `ZERO_COMPONENT_TRANSPORT_SIMMER.md`: explicit zero/nonzero residual
  component experiment, observation-displacement measurements, positive
  small-grid terminal mixture, decisive 64-pixel scale falsification, and the
  root-resolved diagnosis that the current law has already marginalized
  uncertainty about transport itself.
- `zero_residual_component_2d.py`: experimental component posterior readouts.
  It retains coherent, complete, cavity, and history-resolved coordinates as
  falsification controls; none is a GUI setting or promoted denoiser.
- `probe_terminal_component_visual_2d.py` and
  `terminal_component_visual64/`: reproducible raw-image and montage gate for
  Cameraman hair/tripod structure, truth error, and displacement from the
  observation.
- `conservative_exchange_transport_2d.py`: current two-reservoir experiment.
  Posterior erosion donates its exact signed refusal to the residual; positive
  residual smoothing returns only phase-supported action; a target-excluded
  curvature/phase intersection witnesses joint closure. Every substep
  preserves `posterior + residual = observation` pointwise.
- `CONSERVATIVE_EXCHANGE_RESULT.md`: equations, the rejected wholesale-joint
  cycle, the 27-case first-cycle screen, and the unresolved amplitude-coverage
  and terminal laws. The operator is intentionally not in the GUI.
- `probe_exchange_transfer_laws_2d.py`: matched ablation of complete,
  phase-only, curvature-union, and Hellinger-intersection return/closure laws.
- `zonotopic_edge_flux_2d.py`: the next set-valued state. Exact shed ancestry,
  reciprocal phase, and curvature explanations are separate zonotopes of
  antisymmetric Selling-edge fluxes. A safe bounded-residual contractor
  narrows generator intervals without manufacturing component probabilities.
- `ZONOTOPIC_EDGE_FLUX_SIMMER.md`: equations, the rejected unsafe empirical
  enclosure, clean/mixed coverage measurements, and the continuous-scale
  lineage push-forward now required.
- `continuous_scale_zonotope_transport_2d.py`: exact but falsified global
  coefficient-per-scale push-forward control.
- `continuous_scale_edge_family_transport_2d.py`: sparse local
  `(scale lineage, Selling edge)` generator families. The complete set is
  pushed through the positive resolvent by a factorized edge-response map and
  re-expressed on the evolved graph without losing ancestry.
- `CONTINUOUS_SCALE_EDGE_FAMILY_RESULT.md`: refinement-stable coverage result,
  parent/child branch law, and the diagnosis that only independent joint
  value/jet evidence can contract the current state further.

- `TRUTH_UNDER_DISTORTION.md`: the post-FMMT research charter joining
  one-shot patch typicality, zonotopic mixture falsification, and low-SNR
  orbit-recovery failure theory into one denoising state.
- `typical_orbit_set.py`: rejected first executable post-FMMT checkpoint. The global
  patch-medoid form is rejected; the surviving local form retains an
  observation unless every target-excluded affine orbit chart falsifies it.
- `TYPICAL_ORBIT_FIRST_RESULT.md`: equations, global-patch rejection,
  local-survival result, FMMT subordinate after-pass, and the connected-BV-bond
  next experiment.
- `run_typical_orbit_benchmark.py`, `typical_orbit_allscale_128.json`, and
  `typical_orbit_twoscale_96.json`: the reproducible M4 falsification matrix
  and its pointwise-phase transition.
- `test_typical_orbit_set.py`: constant/step exactness, isolated replacement,
  feasible-readout, and cross-scale orbit-coherence invariants.

- `fmmt_certified.py`: the supplied integrated FMMT, preserved as the matched
  rejected checkpoint. A small injection point lets historical support laws
  run before its otherwise unchanged posterior.
- `transport_support.py`: one continuous support-birth equation for 1-D and
  2-D fields.
- `affine_relation_transport.py`: a separate, J-invariant 1-D relation
  pushforward experiment; it is a falsifiable simmer, not part of FMMT.
- `cross_predictive_transport.py`: the full-scale three-characteristic 1-D
  candidate with debiased covariance equilibrium.
- `1D_CROSS_PREDICTIVE_RESULT.md`: equations, broad-battery result, failure
  audit, and the precise route into joint 2-D particle transport.
- `1d_cross_predictive_battery.json`: the complete 234-case M4 record.
- `1d_cross_predictive_examples.png`: matched truth/observation/candidate/
  control traces for all six structures under mixed corruption.
- `probe_1d_particle_continuation.py` and
  `probe_1d_causal_collision.py`: W1, branch-particle, collision-population,
  and exactly ancestry-excluded 1-D theorem probes.
- `1d_particle_continuation_gate.json`, `1d_joint_collision_gate.json`, and
  `1d_causal_crossfit_gate.json`: matched positive/negative records from the
  current 1-D lineage simmer.
- `probe_1d_lineage_branch.py`: determinant-one joint information-lineage
  transport, scalar/collision readouts, and optional rejected continuation.
- `1d_information_lineage_full_256_3seed.json`: complete 234-case record for
  the first positive one-pass lineage candidate.
- `cross_predictive_transport_2d.py`: a deliberately minimal four-tangent
  image lift used to falsify global covariance continuation.
- `causal_ancestry.py`: exact source-law pushforward through the V3 eikonal
  parent simplex, including overlap-aware collision population.
- `causal_parity_transport.py`: rejected observation-excluded parity value
  readout and its exact shared-label population diagnostic.
- `causal_predictive_geometry.py`: ancestry-weighted quantile laws, the scalar
  horizontal Wasserstein quotient, and self-consistency descent.
- `CAUSAL_PREDICTIVE_SIMMER.md`: complete positive/negative M4 result and the
  continuous source-measure requirement.
- `crossfit_characteristic_transport_2d.py`: strict direction-lane witness
  with exact target-identity exclusion.
- `witnessed_characteristic_transport_2d.py`: dense jet proposals, proper
  CRPS witness action, exact joint signal/residual disintegration, and source
  ancestry continuation experiments.
- `continuous_tangent_transport_2d.py`: target-free off-grid tangent-circle
  quadrature at common physical radii, local parallel jets, and continuous
  lineage-covariance probes.
- `probe_continuous_tangent_information_geometry.py`: quantile-refinement gate
  for scalar horizontal, raw Sasaki, and source-lineage jet support volume.
- `causal_information_lineage_2d.py`: root-resolved positive branch measure on
  the exact Hopf--Lax parent DAG, using a determinant-one joint information
  metric and no denoising duration or corruption branch.
- `probe_causal_information_lineage_2d.py`: angular, population-phase, and
  six-structure/full-corruption gates for that causal law.
- `causal_information_lineage_2d_six_source_full_gate20.json`: the complete
  60-case first gate; the corruption names are report labels only.
- `causal_information_lineage_2d_interface_diagnostic20.json`: evidence that
  the remaining clean-interface failure is integer germ realization of an
  approximately one-unit continuous source law, not a demand for more support.
- `probe_population_phase_integral_2d.py`: nested phase-fibre integration and
  section-order refinement, including the rejected moment/jet projections.
- `population_phase_section_order_refinement_gate20_8_16.json`: the positive
  three-structure record showing that causal section selection must precede
  numerical population marginalization.
- `population_phase_hj_barycenter_gate20_phase8.json`: the twelve-case gate for
  the parameter-free Haar-density HJ collision barycenter.
- `population_phase_hj_barycenter_interface20_8_16.json`: matched interface
  evidence that the integrated HJ endpoint is materially more phase-stable
  than the hard branch section.
- `population_phase_hj_barycenter_six_source_full_gate16_phase4.json`: broad
  60-case falsification gate across the complete external corruption catalogue.
- `population_phase_hj_simplex_gate20_phase8.json`: twelve-case evidence for
  collision order derived from the accepted Hopf--Lax parent simplex.
- `population_phase_hj_simplex_interface20_8_16.json`: matched phase-refinement
  record for the clean and mixed geometric interface.
- `population_phase_hj_simplex_six_source_full_gate16_phase4.json`: full
  60-case simplex-order falsification screen.
- `1d_hj_joint_focused96.json`, `1d_joint_w1_value_jet_focused96.json`,
  `1d_joint_information_field_focused96.json`, and
  `1d_symmetric_parent_focused96.json`: rejected 1-D joint-section experiments
  that expose the need for transported residual ancestry.
- `1d_paired_side_lineage_focused96.json` and
  `1d_nested_midpoint_lineage_focused96.json`: disjoint-source controls that
  respectively expose one-sided extrapolation noise and context-only
  curvature loss.
- `1d_root_context_equal_depth_broad96.json` and
  `1d_root_context_simplex_broad96.json`: broad evidence for equal causal path
  accounting and the provisional two-context-parent/one-root simplex law.
- `1d_root_context_transition_focused96.json`: matched row-stochastic versus
  unnormalized reciprocal action-density lineage control.
- `1d_effective_ancestry_geodesic_focused96.json`,
  `1d_independent_side_joint_collision_focused96.json`, and
  `1d_continuous_curvature_focused96.json`: rejected effective-ancestry,
  independent-side, and local curvature-reconstruction probes.
- `1d_energy_root_broad128_2seed.json`,
  `1d_transport_energy_root_broad128_2seed.json`, and
  `1d_energy_root_monge_broad128_2seed.json`: broad evidence for the useful
  energy-distance root invariant and the rejected deflating scalar endpoints.
- `1d_hilbert_value_jet_broad128_2seed.json`,
  `1d_phase_sasaki_broad128_2seed.json`, and
  `1d_phase_behavior_broad128_2seed.json`: exact field-section evidence that
  joint dynamical phase improves derivative/impulse rejection but must enter
  causal arrival before amplitude is marginalized.
- `1d_phase_collision_focused96_1seed.json`: rejected adaptive collision order
  proving that balanced forward/backward phase marginals are not a substitute
  for a transported coupling of their causal ancestries.
- `JOINT_TRANSPORT_SIMMER.md`: the new joint equation, 18-case result,
  continuation falsifications, and causal-covariance target.
- `crossfit_characteristic_transport.json`: matched strict/witnessed/joint
  18-case M4 gate.
- `lineage_covariance_full.json`: 24-run equilibrium record for the accepted
  lineage-local covariance continuation.
- `angular_convergence_weighted.json`: rejected primitive-direction tangent
  refinement showing radial/angular nonconvergence.
- `lineage_residual_smoke.json` and `transported_lineage_residual_smoke.json`:
  rejected direct and Selling-transported local residual priors.
- `2D_CHARACTERISTIC_LIFT_RESULT.md`: the 108-case M4 gate, structure-specific
  split, rejected repairs, and causal-ancestry diagnosis.
- `2d_denoiser_battery.json`: the complete six-structure, nine-corruption,
  two-seed 2-D record.
- `2d_characteristic_examples.png`: matched visual plate showing both the
  interface failures and the woven/line successes.
- `FOUNDATIONAL_SIMMER.md`: the unified jet-scale transport target, the reason
  smoothing fails replacement corruption, and the retained negative results.
- `V3_FUSED_DENOISING_THEORY.md`: the V3 population/eikonal deep dive and the
  band-free replacement equations for FMMT.
- `fused_transport_geometry.py`: invariant joint-observation, predictive
  Fisher/Wasserstein information-volume, population, and continuous-eikonal
  bridge kernels.
- `v3_support_under_corruption.json`: the M4 probe showing why raw V3 local
  support cannot be applied directly to a noisy observation.
- `predictive_seed_under_corruption.json`: the bin-free Wasserstein probe that
  rejects an untransported leave-one-out relation seed.
- `sample_series.py`: selectable 1-D component compositions and the shared
  1-D/2-D corruption catalogue.
- `fabada_oracle.py`: repaired PyITD/FABADA diffusion-family comparator. It is
  intentionally given the selected corruption law and exact conditional noise
  moments; conservative heat, continuous Cesaro scale, and unbiased affine
  risk replace the original boundary, variance, evidence, and chi-square
  mechanics.
- `FABADA_ORACLE_COMPARISON.md`: derivation, repair audit, matched result, and
  the exact boundary between this useful control and structural transport.
- `probe_1d_fabada_oracle.py` and
  `1d_fabada_oracle_broad128_2seed.json`: reproducible 156-case comparison to
  the current phase-collision transport record.
- `probe_1d_energy_root.py`: lean six-source, thirteen-condition gate for
  target-free energy-distance root participation and Monge sections.
- `terminal_caustic_transport_2d.py`: resumed 2-D theorem probe that transports
  the causal HJ collision through the scalar quantile Jacobian, without a
  branch mode or intensity bandwidth.
- `2D_TERMINAL_CAUSTIC_RESULT.md` and
  `terminal_caustic_full_gate16_phase4.json`: full-gate rejection as a
  universal endpoint and its positive localization to tapered hair.
- `2D_ACCELERATION.md`: estimator-preserving acceleration of the now-rejected
  continuous-support FMMT, retained as a representation benchmark only.
- `benchmark_2d_acceleration.py` and `2d_acceleration_m4.json`: repeatable
  128/256-square kernel and GUI support-reuse timing record.
- `gui.py`: Dear PyGui laboratory with 1-D and 2-D tabs.
- `probes.py`: a smooth 1-D scene and a tapered 2-D hair-edge falsification
  scene.
- `test_transport_support.py`: conservation, maximum-principle, dimension,
  range, and exact action-budget tests.
- `ANALYTIC_UPLIFT.md`: equations, mechanism comparisons, current results,
  and the route to a later native implementation.
- `TRANSPORT_COMPARISON.md`: repository-wide audit of support, propagation,
  conservation, confidence, topology, and stopping principles.
- `CONSTANT_AUDIT.md`: every important empirical constant and test-selected
  branch in the supplied checkpoint, separated from mathematical identities
  and numerical resolution.
- `RESEARCH_REPORT.md`: the supplied report, retained unedited as source
  material rather than repository instructions.
- `UPSTREAM_README.md`: the supplied README, retained verbatim after this
  experiment adopted `README.md` for its own entry point.

## Run

Use the M4 Mini for NumPy/SciPy measurements:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest denoiser.test_sample_series \
  denoiser.test_transport_support denoiser.test_affine_relation_transport \
  denoiser.test_cross_predictive_transport \
  denoiser.test_cross_predictive_transport_2d \
  denoiser.test_causal_ancestry \
  denoiser.test_causal_parity_transport \
  denoiser.test_causal_predictive_geometry \
  denoiser.test_continuous_source_transport \
  denoiser.test_crossfit_characteristic_transport_2d \
  denoiser.test_continuous_tangent_transport_2d \
  denoiser.test_witnessed_characteristic_transport_2d \
  denoiser.test_fmmt_representation \
  denoiser.test_fused_transport_geometry

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser probes --out /tmp/denoiser_transport_probes

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_1d_foundational_simmer \
  --out /tmp/denoiser_1d_foundational_simmer.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_predictive_seed_geometry \
  --out /tmp/predictive_seed_under_corruption.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_1d_cross_predictive_battery \
  --out /tmp/denoiser_cross_predictive_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_1d_fabada_oracle \
  --size 128 --seeds 2 \
  --transport-record \
    denoiser/1d_phase_collision_posterior_harmonic_broad128_2seed.json \
  --out /tmp/1d_fabada_oracle_broad128_2seed.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_1d_particle_continuation \
  --size 256 --seeds 1 --out /tmp/1d_particle_continuation_gate.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_1d_causal_collision \
  --size 256 --seeds 1 --out /tmp/1d_causal_crossfit_gate.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_1d_lineage_branch \
  --size 256 --seeds 3 --skip-equilibrium \
  --out /tmp/1d_information_lineage_full_256_3seed.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.run_2d_denoiser_battery \
  --size 96 --seeds 2 --out /tmp/denoiser_2d_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_causal_predictive_geometry \
  --size 40 --seeds 2 --out /tmp/causal_predictive_geometry.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_causal_fixed_point \
  --size 20 --quantiles 8 --continuations 16 \
  --out /tmp/causal_fixed_point.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_crossfit_characteristic_transport \
  --size 32 --seeds 1 \
  --out /tmp/crossfit_characteristic_transport.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_continuous_tangent_information_geometry \
  --size 20 --quantile-counts 8,16,32 \
  --out /tmp/continuous_tangent_lineage_jet_geometry.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_transported_sasaki_geometry \
  --size 20 --sources 'tapered hair,geometric interfaces,woven chirps' \
  --source-transports 32 --remetricize --strict-joint-bundle-action \
  --out /tmp/post_lineage_prolongation_full_gate.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_post_lineage_residual_forms \
  --size 40 --sources 'cameraman,tapered hair' \
  --out /tmp/branch_forms_cameraman_hair_40.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_branch_posterior_transport \
  --size 20 --transports 32 \
  --out /tmp/branch_posterior_transport_gate.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_causal_information_lineage_2d \
  --size 20 \
  --sources 'cameraman,tapered hair,geometric interfaces,woven chirps,line drawing,multiscale blobs' \
  --angular-count 4 --quantile-count 16 --phases 0 \
  --condition-catalog full \
  --out /tmp/causal_information_lineage_2d_six_source_full_gate20.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_population_phase_integral_2d \
  --size 20 \
  --sources 'tapered hair,geometric interfaces,woven chirps' \
  --conditions 'clean,uniform 0.10,replacement 0.25,mixed 0.25' \
  --phase-counts 8,16 --angular-count 4 --quantile-count 16 \
  --out /tmp/population_phase_hj_barycenter_interface20_8_16.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.probe_terminal_caustic_transport_2d \
  --size 16 --phase-count 4 --angular-count 4 --quantile-count 16 \
  --baseline-record \
    denoiser/population_phase_hj_simplex_six_source_full_gate16_phase4.json \
  --out /tmp/terminal_caustic_full_gate16_phase4.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser.benchmark_2d_acceleration \
  --sizes 128,256 --repeats 3 \
  --out /tmp/denoiser_2d_acceleration_m4.json
```

The V3 corruption-population diagnostic additionally needs the repository's
full native BFFT library in the runtime environment:

```sh
python3 -m denoiser.probe_v3_support_under_corruption \
  --out /tmp/v3_support_under_corruption.json
```

Run the rejected 2-D FMMT control for reproduction:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m denoiser denoise input.png /tmp/transport.png \
  --method transport --diagnostics /tmp/transport.json
```

The GUI uses the repository's vision-viewer dependencies on the MacBook:

```sh
python3 -m pip install -e '.[vision-viewer]'
python3 -m denoiser gui
```

The archived 2-D workflow is `skimage/file source -> explicit corruption ->
selected FMMT form`. The 1-D workflow is `component composition -> explicit corruption
-> selected denoiser`; each stage has its own button, with a full-pipeline
button for rapid comparisons. `PFABADA-Cesaro oracle risk` is the deliberately
unfair known-noise comparison. It receives the GUI's chosen corruption law and
generating moments, uses no duration control, and displays both its global
readout and the MSE of its rejected point-adaptive control in diagnostics.

The legacy 1-D support-flow form has three independent laboratory controls:

- **provisional smoothing scale** sets the Gaussian bootstrap reach;
- **transport action budget x** scales the observation-derived action that one
  conservative flow may spend;
- **continuation rounds** recompute support and spend another measured budget.

The full-scale cross-predictive form uses none of them. Its only exposed limit
is an internal continuation ceiling that turns the run into an explicit
unresolved failure if covariance equilibrium was not reached. The displayed
maximum flux steps remains a numerical safety guard for the legacy form.

The FMMT bootstrap now advances vector-valued recurrences directly. The scalar
Numba recurrence remains as an exact representation oracle in the tests.

## Present status

No denoiser is currently promoted. FMMT and the 1-D candidates are frozen as
negative evidence. The GUI opens on the rejected 2-D controls and identifies
them as an archive. The next active experiment is the typical-orbit feasible
set specified in `TRUTH_UNDER_DISTORTION.md`; it has no GUI readout yet.

The remaining discussion in this section is the chronological FMMT research
record. Its intermediate improvements do not override the final rejection.

The new support layer is intentionally not a final algorithm. It has removed
the checkpoint's fixed support scales, hand-placed evidence ramps, hard
censor/tail bands, and fixed 128-sweep horizon. Its scale integral is resolved
by log quadrature and its conservative flow stops after spending a measured
coarse-residual action budget.

On the current tapered-hair control, support birth preserves the provisional
edge response to four significant figures while slightly lowering MSE. That
is the right conservative behavior, but not yet the desired recovery of the
already-weakened hair edge. The next mathematical step is an oriented
continuation measure transported along the edge tangent. C++ optimization is
explicitly deferred until that state has stabilized.

The 1-D GUI now offers the legacy Gaussian-plus-support flow, the full-scale
cross-predictive equilibrium, the action-contracting research readout, and the
repaired known-noise FABADA control. The cross-predictive and research forms
ignore the legacy smoothing, action-budget, and continuation controls: they
use every topological lag and stop at their intrinsic equilibria. The
cross-predictive form passed the first
broad 1-D gate, but remains a laboratory candidate while conductance leakage,
boundary mass, heavy replacement, and explicit jet transport are unresolved.
The current W1 readout improves heavy replacement dramatically, but scalar W1
continuation loses oscillatory phase. Retaining the lag/path distribution
instead gives excellent clean fidelity and amplifies corruption. An intrinsic
W1 arrival-population term removes isolated zero-action branch monopolies, and
exact ancestry exclusion is now executable, but neither clears the broad gate.
The next 1-D state must therefore carry positive causal lineage mass; it cannot
approximate lineage by subtracting dependency indices from a local score.
That positive lineage state is now executable. Every branch carries value,
jet, exact residual, and mass; midpoint covariance produces a determinant-one
information metric, and bidirectional positive transport precedes the W1
readout. On the full three-seed gate it lowers noisy MSE from `0.001683` for
the local W1 law to `0.001595`, lowers both derivative errors, and raises clean
TV retention from `0.812` to `0.833`. It remains research-only because the
accepted continuation is still more nearly identity on clean inputs and the
exact dense representation is cubic.

The latest 1-D theorem probe now separates the observed root from a target-free
nested midpoint/secant context. Its shell action is affine-exact and contains
no corruption label, horizon, band, or duration. Making root and context pay
equal causal depth fixes a large mass bias; assigning the endpoint measure by
its two contextual parents and one observed root improves six of thirteen
broad condition means, including heavy replacement and mixed corruption.
It is not promoted: broad aggregate error and clean fidelity regress, and the
salt/pepper 25% screen reveals false support. This narrows the next step to a
continuous terminal collision order extracted from transported effective
ancestry, rather than another smoothing coefficient.

The follow-up identifies one usable part of that order. Scalar energy distance
gives an exact, scale-free root-membership coordinate, and collision
concentration supplies an independent transported witness. Together they
repair dense impulses and improve several derivative/TV screens. They are not
promoted because the stable midpoint context lacks enough curvature: every
tested amplitude blend or contextual quantile section either deflates clean
structure or restores excess noisy variation. Local Richardson and continuous
polynomial-order fibres are also rejected because their negative weights
amplify corruption. The next 1-D lift must parallel-transport curvature as a
bundle coordinate rather than reconstruct it at each target.

That lift now exposes dynamical phase as the immediate missing coordinate.
An exact determinant-one midpoint `(value, jet)` section improves broad noisy
derivative errors and cuts the dense impulse failure by more than half, while
clean phase clouds are substantially more anisotropic than corrupted ones.
The same section deflates amplitude, and a local phase-authority blend restores
noise as well as structure. Phase must therefore govern continuous causal
arrival/collision order before scalar readout. This is distinct from the 2-D
population-phase quadrature, which is a raster gauge to be integrated away.

The present FMMT implementation is a rejected historical control. Its
acceleration results remain valid only as representation measurements. The
replacement does not add another
support layer. It derives population and a determinant-one eikonal metric from
the horizontal Wasserstein volume of the transported joint signal/residual/jet
measure. A local leave-one-out particle cloud has now been explicitly rejected;
the conservative fixed-point transport creating the predictive law is the next
experiment.

The first full-scale 2-D lift has sharpened that requirement. It reaches
covariance equilibrium and preserves aggregate variance/range, beating FMMT on
woven chirps and line drawings, but it loses badly on Cameraman hair, sparse
interfaces, and heavy replacement. Its four-direction readout and image-global
covariance dilute sparse causal support. It is therefore preserved as a
falsification seed, not added to the GUI. The next image state must transport
V3 causal parent fractions and measure distinct ancestry before horizontal
Wasserstein population is evaluated.

The first exact joint implementation now improves aggregate MSE over both that
seed and FMMT on its 18-case smoke gate, driven by gains under random
replacement and mixed corruption. Its strict witness and residual prior both
exclude target identity, and signal plus residual equals the observation
particlewise. It still trails FMMT in SSIM and edge retention and therefore is
not in the GUI. Recursive transport proves the missing edges remain in the
residual, but scalar and particle-count authority laws either freeze them or
restore replacement noise. The next state is covariance on conserved causal
source ancestry, as detailed in `JOINT_TRANSPORT_SIMMER.md`.

That covariance state is now executable. A strictly held-out
residual--prediction product is transported by exact positive source lineage,
and only covariance energy above its overlap-aware finite-population variance
may continue. All 24 clean/corrupted runs reach equilibrium. It improves the
joint candidate's MSE from `0.007304` to `0.006520`, SSIM from `0.6656` to
`0.6695`, and edge retention from `0.4358` to `0.4759`, while retaining its
replacement and mixed-corruption advantage over FMMT. FMMT still leads SSIM
and edges. Primitive lattice tangent refinement and direct local residual
pooling were both rejected; continuous common-scale jet/eikonal geometry is
the next experiment.

That common-scale tangent law now converges under projective-angle refinement
and remains affine exact with zero target self-coefficient. Its scalar
horizontal Wasserstein support is comparatively stable under corruption, but
raw vertical jets reproduce V3's false-support pathology; characteristic
source lineage reduces rather than eliminates it. The next experiment is
therefore not another local jet weighting. It must attach Hopf--Lax parent
identity to the joint particles, parallel-transport them, and only then allow
vertical jet variation to command continuous population. The new positive and
negative records are summarized in `JOINT_TRANSPORT_SIMMER.md`.

The latest ordering experiment differentiates the scalar section only after
source identity transport.  It shows that the present smoothing flow destroys
post-lineage curvature much faster than corruption hides it.  A hard
maximum-posterior characteristic branch recovers substantially more edge and
variance than posterior barycenters, including on the 40-square Cameraman/hair
gate, while a spatial Selling average of branch probability is rejected.  The
next experiment is consequently a Hamilton--Jacobi branch section on joint
position/characteristic space—not another amplitude smoother, support band, or
cell transplant.  Code and records are listed in
`JOINT_TRANSPORT_SIMMER.md`; no new path has been promoted to the GUI.

The first causal 2-D lift is now executable. It transports the complete
positive joint branch/root measure through the recorded Hopf--Lax parent
simplexes; roots are not collapsed to scalar confidence. On the 60-case gate,
its refinement-invariant branch-collision section improves its own local
measure in MSE on 55 cases, SSIM on 56, and edge retention on 56. Aggregate
MSE changes from `0.006990` to `0.006203`, SSIM from `0.6780` to `0.7000`, and
edge retention from `0.4855` to `0.5276`. Integrated FMMT remains ahead overall
at MSE `0.004828`, SSIM `0.7863`, and edge retention `0.6449`. The clean
geometric-interface regression and remaining phase/spatial refinement prevent
promotion or C++ specialization.

The first branch-space Hamilton--Jacobi continuation is now executable as
well. It propagates log density relative to branch Haar measure through the
same Hopf--Lax DAG, then integrates the collision density of two coherent paths
instead of selecting a hard branch. On the twelve-case phase gate this smooth
endpoint improves local MSE and edge retention in every case and SSIM in
eleven, while beating the ordinary collision mean on all three measures in
all twelve. Its aggregate variance remains too low, so it is a stronger
theoretical candidate, not a GUI or C++ promotion.

The preliminary full-catalogue screen preserves that direction: the HJ
barycenter improves local MSE and SSIM in 57 of 60 cases and edge retention in
50. FMMT remains stronger overall, and tapered-hair replacement remains a
specific debt. This phase-four record broadens falsification coverage; it does
not turn quadrature count into a user control.

The causal-simplex endpoint improves that result without a quality setting.
Its collision order is the effective number of Hopf--Lax parent witnesses plus
the local witness. On the 60-case gate it improves the fixed-order HJ endpoint
on MSE in 51 cases, SSIM in 50, edge retention in 58, and variance fidelity in
56. Aggregate MSE/SSIM/edge become `0.005764/0.7432/0.5669`. FMMT remains
stronger overall; tapered hair under replacement remains an explicit debt.

The paper-derived continuation is recorded in
`POSTERIOR_INVARIANCE_PAPER_SIMMER.md` and
`COMPLETE_MOMENT_TRANSPORT_SIMMER.md`. Neural architectures were not imported.
Residual reflection, an external variance-coordinate wrapper, locally inferred
anisotropic nuisance, a diagonal backward smoother, and a phase-free bounded
contractor were tested and rejected. The surviving mechanism is simpler: the
zero-referenced complete residual moment must shape transport geometry, because
centered covariance erases nuisance shared by all witnesses. It improves the
old residual posterior on the focused and 54-case gates and improves the newer
causal HJ lineage law on 23/36 MSE cases and 25/36 SSIM cases. The complete HJ
form nearly matches FMMT aggregate MSE and beats it on Poisson, replacement,
and mixed MSE, but still trails aggregate SSIM/edges and is not promoted to the
GUI or native implementation.

The explicit component follow-up confirms that this near-match was not yet a
new image estimator. A direct complete terminal mixture substantially improves
small-grid variance and edge retention, but fails at 64 pixels by preserving a
granular residual field. Marginal Hellinger, target-free cavity, and
root-resolved controls identify the missing coordinate: the current
`root_mass` carries labels after conditional transport laws have nearly
collapsed, so between-history residual variance is effectively zero. The next
law must retain the transport map itself inside the joint posterior before any
branch/component marginal. Full equations and measurements are in
`ZERO_COMPONENT_TRANSPORT_SIMMER.md`.
