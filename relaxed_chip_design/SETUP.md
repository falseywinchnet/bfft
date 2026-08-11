# Setup and Reproduction

## Standalone inverse

The capsule requires Python 3.8 or newer and NumPy 1.20 or newer. From the BFFT
repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r relaxed_chip_design/requirements.txt
python -m relaxed_chip_design.check_unrelaxation
python -m relaxed_chip_design.check_basin_walk
python -m relaxed_chip_design.check_preimage
python -m relaxed_chip_design.check_vector_diffusion
python -m relaxed_chip_design.check_interval_connection
python -m relaxed_chip_design.example
```

The folder is deliberately importable from the checkout without changing
BFFT's distributed package list. It is a research project, not yet public API.

## Full standard-cell harness

The authoritative implementation and benchmark artifacts live in the sibling
`HypersphericalCircuitLab` checkout. Point `CHIP_LAB` at that checkout and
enter it before running the commands below:

```sh
export CHIP_LAB=/path/to/HypersphericalCircuitLab
cd "$CHIP_LAB"
```

That repository contains:

- `research/support_sparse_transport.py` — full forward transport, inverse,
  legal emission, metrics, and timing;
- `research/initial_def_far_field.py` — fixed spherical basin walk and axis
  controls;
- `research/transport_krylov_basin_probe.py` — capacity/HPWL/net-graph
  low-rank displacement probe;
- `research/check_transport_unrelaxation.py` — production-kernel checks;
- `native/dimension_sparse_transport.cpp` — native sparse transport kernels;
- `datasets/replace-gcd/` and `datasets/replace-wb/` — frozen inputs and
  baselines;
- `docs/standard-cell-coupling-readout.md` — experiment chronology.

Use one numerical thread when reproducing the frozen timing:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Build and check the native guest kernel using the repository's own helper:

```sh
bash guest/build_native_transport.sh
python3 research/check_native_sparse_transport.py
python3 research/check_transport_unrelaxation.py
```

Run the selected inverse on `gcd`:

```sh
python3 research/support_sparse_transport.py \
  --lef datasets/replace-gcd/source/NangateOpenCellLibrary.lef \
  --def datasets/replace-gcd/source/gcd.def \
  --baseline-def datasets/replace-gcd/baseline/replace.def \
  --output-json results/gcd_quantile.json \
  --output-def results/gcd_quantile.def \
  --native-support-sparse \
  --quantile-unrelaxation \
  --orbit-conditioner \
  --orbit-conditioner-far-field \
  --orbit-conditioner-retransport
```

Run the identical law on `wb_dma_top`:

```sh
python3 research/support_sparse_transport.py \
  --lef datasets/replace-wb/source/contest.lef \
  --def datasets/replace-wb/source/wb_dma_top.def \
  --baseline-def datasets/replace-wb/baseline/replace.def \
  --output-json results/wb_quantile.json \
  --output-def results/wb_quantile.def \
  --native-support-sparse \
  --quantile-unrelaxation \
  --orbit-conditioner \
  --orbit-conditioner-far-field \
  --orbit-conditioner-retransport
```

### Selected vector-connection wrapper

The literature round's selected law is the residual-gated two-pass wrapper
with connection-valued direction diffusion.  In `HypersphericalCircuitLab`,
run it under that repository's guarded guest workflow:

```sh
python3 research/vector_diffusion_phase_transport.py \
  --lef datasets/replace-wb/source/contest.lef \
  --def datasets/replace-wb/source/wb_dma_top.def \
  --baseline-def datasets/replace-wb/baseline/replace.def \
  --work-dir results/raw/vector_diffusion_phase/wb_work \
  --output-json results/raw/vector_diffusion_phase/wb.json \
  --output-def results/raw/vector_diffusion_phase/defs/wb.def
```

Use the corresponding GCD LEF/DEF/baseline paths without changing the law.
`research/connection_lifted_phase.py` contains the checked circular, sparse
connection, deterministic feature, and lifting primitives.  The adjacent
`probe_*` and circular/conductance wrappers are retained falsifications; do
not choose among their DEFs at runtime.

Do not tune a density switch between these runs. Record direct HPWL, move and
blocked counts, transport-plus-readout time, total guarded wall time, peak RSS,
and explicit state bytes. Keep generated datasets, DEF outputs, and profiles in
the circuit lab rather than copying them into BFFT.

## Research reference

The initialization work that informed the relaxation study is Thornton and
Cuturi, *Rethinking Initialization of the Sinkhorn Algorithm*:
[arXiv:2206.07630](https://arxiv.org/abs/2206.07630).
