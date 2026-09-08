# Standalone CONV* resize demos

This directory is a copy-isolated implementation and demonstration package. It does not import any module, asset, or generated result from the repository around it. Copy the directory anywhere, create a Python environment, and run either GUI.

## Install and run

Python 3.11 or newer and a C compiler are required. On macOS, `xcode-select --install` supplies Clang.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_image_demo.py
python run_signal_demo.py
```

The first import compiles `native/conv_native.c` into `.native_build/`. Apple Silicon and AArch64 builds use ARM NEON; other targets use the same scalar C ABI. A persistent native worker pool defaults to the machine's logical CPU count; set `CONV_NATIVE_THREADS` before launch to override it. Set `CC` to select another compatible compiler.

## What is implemented

- CONV*: the five exact six-tap interior current filters, second-order outer and near-outer closure, ordered sign-lineage scan, exact signed-fibre projection reduced to a scalar threshold, and arbitrary endpoint-aligned quintic synthesis.
- CONV basin transport: exact degree-five integration of the same ordered profile over the requested target-grid Voronoi basins during reduction, followed by direct CONV synthesis during enlargement.
- CONV-C conservative support cycles: exact 2x2 cell-average restriction, native horizontal/vertical/mixed moment prediction, complete face-current admission, and matched zero-detail synthesis for /2, /4, and /8 comparisons.
- Scale-widened, normalized Lanczos-8 polyphase FIR. Its support widens during reduction, so this arm is the explicit antialiased comparator.
- Matched Lanczos-3, bilinear, and FP32 AMD FSR 1.0 EASU comparison arms. EASU excludes RCAS and uses clamped boundary fetches.
- Arbitrary image width and height, with `/2`, `/4`, `/8`, `x2`, `x4`, and `x8` buttons that only populate the editable target-size fields.
- A matched double-pass view that displays each method's own shrink-to-target then enlarge-to-source result, rather than only reporting its cycle MSE.
- A separate long-signal laboratory with analytic target truth, signed residuals, matched-family cycle error, and FFT plots.

All comparison cycles are matched: a Lanczos result is returned with Lanczos, a CONV* result with CONV*, and so on.

## Coordinate and error conventions

Every backend uses endpoint-aligned coordinates: target index `j` queries source coordinate `j*(N-1)/(M-1)`. At nested sizes `M=s*(N-1)+1`, CONV* therefore reproduces every source anchor exactly. Image previews use nearest sample selection only; the numerical outputs are not resized again for display.

The 1-D point MSE is measured against the named analytic function evaluated directly at the target coordinates. Matched-cycle MSE is measured against the original source array after resizing to the requested length and back with the same family. These are different questions, and the GUI reports both.

## Native FIR provenance

The requested local Airspy FIR implementation was inspected. Its useful implementation pattern is fixed tap-count specialization over contiguous data. The local `iqconverter_float.c` selects a scalar `FIR_STANDARD` path on Apple and its license prohibits redistribution outside the Airspy ecosystem, so no Airspy source text is copied here. The bundled C backend is an independent implementation of that fixed-bank pattern, specialized to CONV*'s five six-tap filters and explicitly vectorized with NEON across contiguous line/channel lanes.

## Verification and timing

```sh
python -m unittest -v test_backend.py
python benchmark.py --sizes 128,256,512 --repeats 5
python benchmark.py --sizes 128,256,512 --repeats 5 --threads 1
```

The tests check exact nested-anchor return, affine and constant reproduction, ordered sign-transition noncreation, arbitrary 1-D/2-D shapes, finite output, and absence of imports from the parent repository.
