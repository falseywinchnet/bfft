# BFFT twiddle ancestry / Fresnel plots

This experiment traces the logical normalized-Bruun cells underneath BFFT's
forward DIF, DIT, and DIP kernels at `N=512`. It produces the conventional
complex-DFT Fresnel coefficient lenses and reverse-factor cascades for the
three BFFT walks.

The DFT reference is two lenses, not a phase image:

- the real coefficient field `cos(2*pi*k*n/N)`; and
- the imaginary coefficient field `-sin(2*pi*k*n/N)`.

For each BFFT cascade, a forward prefix `B_t` maps input samples to the
intermediate coordinate system after stage `t`. The plotted reverse factor is
`S_t = F * inverse(B_t)`, so `S_t * B_t = F`. Consequently:

- rows remain fixed final packed output coordinates in every panel;
- columns are the intermediate factor/residue coordinates at that stage; and
- walking from the output boundary back toward the input reveals the repeated
  recursive coefficient fields rather than a sparse ancestry path.

The older ancestry diagnostic remains available as a three-panel plot whose:

- rows are all `N` real spectral coordinates, packed as `DC`, then
  `(Re X[k], -Im X[k])` for `1 <= k < N/2`, then Nyquist;
- columns are logical twiddle operations in logical-level order (with a second
  figure for production execution order);
- colored cells mean that the output coordinate structurally depends on that
  rotation; and
- color is the normalized Givens angle `theta/pi`.

The production code fuses radix-2 levels, broadcasts cells over SIMD lanes, and
uses different transport schedules. The trace deliberately preserves the
logical operation identity beneath those optimizations. One broadcast of a
twiddle across a column/span is one column; repeated numerical angle values at
different tree locations remain distinct columns. At `N=512`, every form has
247 nontrivial rotations, grouped `1, 3, 7, 15, 31, 63, 127` by level.

Run from the repository root:

```sh
python3 experiments/bfft_twiddle_trace/bfft_twiddle_trace.py --n 512
```

NumPy is a BFFT dependency; plot generation additionally requires Matplotlib.

Outputs are written to `experiments/out/bfft_twiddle_trace/`:

- `bfft_dft_lenses_n512.{png,svg}` — real and imaginary DFT lenses;
- `bfft_dif_cascade_n512.{png,svg}` — DIF reverse-factor cascade;
- `bfft_dit_cascade_n512.{png,svg}` — DIT reverse-factor cascade;
- `bfft_dip_cascade_n512.{png,svg}` — DIP reverse-factor cascade;
- `bfft_twiddle_trace_n512.{png,svg}` — DIF/DIT/DIP ancestry comparison;
- `bfft_twiddle_execution_n512.{png,svg}` — the same columns permuted into
  each production kernel's operation order, exposing the scheduling/transport
  distinction hidden by common logical-level ordering;
- `bfft_twiddle_trace_n512.npz` — boolean support matrices, angles, labels,
  level boundaries, and production execution-order permutations; and
- `bfft_twiddle_trace_n512.json` — compact counts and metadata.

The trace is source-matched to:

- DIF: `forward_residues_inplace`, the production forward schedule, and
  `bruun_idx_int` in `bruun_dif_kernel.hpp`;
- DIT: `compute_vwork_norm` / `merge4_v_norm` and the odd-power terminal; and
- DIP: `fwd_ridge` / `fwd_tree`, with cell angle `theta(d,e)=pi*d/e`.

The cascade is a signed coefficient plot with the common range `[-1,1]` and
the `viridis` color map. It is independent of a chosen input signal. Every
reverse factor is checked numerically by reconstructing the complete packed
real DFT matrix.

In logical-level order, DIT and DIP have identical support. This is expected:
they use the same normalized Bruun cell algebra. Their distinction is the
walk—DIT ascends in fused spectral stages while DIP descends packet subtrees—so
it appears in the execution-order figure. DIF has different logical support as
well as a different execution order.
