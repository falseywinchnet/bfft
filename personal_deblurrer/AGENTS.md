# Personal deblurrer compute notes

Keep deblurring implementation, papers, synthetic generators, tests, and
selected results inside this directory.  The repository's V3 segmenter is only
a chronology marker and must not be imported as deblurring machinery.

Run the invariant suite on the M4 Mini from the repository root:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest personal_deblurrer.test_exposure_transport_deblur \
  personal_deblurrer.test_uncertainty \
  personal_deblurrer.test_shift_mix_decomposition
```

Build the optional exact reflected gather/scatter ABI before native parity and
performance tests:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  make -C personal_deblurrer/native clean all

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest personal_deblurrer.test_native_operator

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.profile_spatial_batch \
  --size 96 --plans 5 --repeats 1000 --out /tmp/spatial_batch_profile.json
```

The shared library in the mirrored tree is a generated artifact. The NumPy
operator remains the exact fallback and representation oracle.

Run the selected synthetic battery with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_benchmark \
  --size 96 --seeds 2 --out /tmp/personal_deblurrer_full
```

The authoritative tree remains on the MacBook. Copy the remote `/tmp` result
back immediately after the run; do not edit the M4 mirror as the primary copy.

Run the reflect-boundary shift/mix battery with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_shift_mix_benchmark \
  --size 96 --out /tmp/personal_deblurrer_shift_mix
```

Copy that `/tmp` directory back immediately after the run.

Run the unified spatial warp/mixing battery with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_spatial_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_spatial
```

Copy the spatial `/tmp` result back immediately after the run.

Run the estimated rotational-consensus battery with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_spatial_estimation_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_spatial_estimation
```

Run all personal-deblurrer invariants with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m unittest discover -s personal_deblurrer -t . -p 'test_*.py'
```

Run the fixed 23-source center/inverse/noise generalization screen and copy its
JSON back immediately:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_center_first_generalization \
  --out /tmp/center_first_generalization_v7.json
```

Run the dense, visibility, and known-measure multi-sheet controls with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_dense_estimation_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_dense_estimation

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_visibility_benchmark \
  --size 96 --passes 64 --out /tmp/personal_deblurrer_visibility

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
```

Launch the Dear PyGui workbench locally with:

```sh
.venv-jpeg/bin/python -m personal_deblurrer.workbench
```

Run an immutable real capture pair on the M4 and copy its `/tmp` output back
before the next mirror sync:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.real_capture_evaluation \
  personal_deblurrer/real_capture_data/koehler_scene1_web_jpeg/blurry_1_1.jpg \
  personal_deblurrer/real_capture_data/koehler_scene1_web_jpeg/blurry_1_2.jpg \
  --passes 64 --out /tmp/personal_deblurrer_koehler_relative_mixing

python3 -m personal_deblurrer.score_koehler_pair_checkpoint
```

Run the exchange-symmetric 12-capture checkpoint and compact-native profile on
the M4, then copy both `/tmp` artifacts back before the next mirror sync:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_koehler_multicapture_benchmark \
  --passes 64 --out /tmp/personal_deblurrer_koehler_multicapture

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.profile_multicapture_native \
  --size 192 --plans 12 --repeats 20 \
  --out /tmp/personal_deblurrer_multicapture_native_profile.json
```

Run the accepted center-first adaptive spatial covariance atlas, continuous
posterior, and ABI v6 batched native path (the current code removes
deterministic center motion before estimating finite local mixing charts) with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_koehler_multicapture_benchmark \
  --passes 64 --mixing-patch-size 192 --mixing-stride 128 \
  --out /tmp/personal_deblurrer_koehler_multicapture_posterior_v9

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.profile_covariance_native \
  --size 192 --repeats 20 \
  --out /tmp/personal_deblurrer_covariance_native_profile.json
```

Run the axis-restricted spatial fourth-cumulant abstention checkpoint by adding
`--quartic-shape` to the accepted atlas command and writing a distinct output:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_koehler_multicapture_benchmark \
  --passes 64 --mixing-patch-size 192 --mixing-stride 128 --quartic-shape \
  --out /tmp/personal_deblurrer_koehler_multicapture_spatial_quartic
```

Run the full symmetric fourth-cumulant controlled battery and estimation-only
real-capture gates with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_full_quartic_battery \
  --size 96 --passes 32 \
  --out /tmp/full_quartic_positive_directional_battery.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.probe_koehler_full_quartic \
  --out /tmp/koehler_full_quartic_estimation_probe.json

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.run_koehler_quartic_posterior \
  --passes 32 --out /tmp/koehler_quartic_image_posterior_v2

/Users/ultimussecundai/.local/bin/m4build -- \
  python3 -m personal_deblurrer.probe_koehler_spatial_full_quartic \
  --patch-size 192 --stride 128 \
  --out /tmp/koehler_spatial_full_quartic_probe.json
```
