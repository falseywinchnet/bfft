# Capstone: the BFFT radix-4 design point and why every direction off it is worse

This records the directions explored to make the BFFT real-FFT kernel cheaper,
and the tradeoff that makes each one less optimal than the current design. It is
a map of bifurcations, not a procedure. The conclusion is that the current point
is a local optimum: every single-axis change pays more than it saves.

## Current design point

- Transform: Bruun real-FFT via a power-of-two residue tree.
- Body: depth-first **fused radix-4** node (`norm2_fused`): parent rotation +
  two child rotations in registers.
- Node arithmetic: already **FMA-optimal** — rotations are 2 mul + 2 FMA
  (the minimal 4-real-mul complex multiply), butterflies are pure add/sub with
  no adjacent multiply to fuse. There is no contraction-blocking shared product
  to remove.
- Output: native residue order is the hot path; standard FFT bins follow by a
  **pure O(N) permutation + conjugation** (`residues_to_complex`: zero
  multiplies).

## Invariants established (the floors)

- **Throughput floor:** the radix-4 evaluation node is 8 FMAs (4 even/odd + 4
  output), critical path 2, 4-wide ILP. Sharing the scale product blocks
  contraction and is strictly worse; 8 is minimal for the even/odd factorization.
- **Multiply-complexity floor:** using the Chebyshev identity `u^2 + v^2 = 1`,
  the node drops to 4 real multiplies, giving `0.5 * N * log2(N)` multiplies
  total. 4 is minimal for the nested even/odd structure.
- **Repack floor:** native residues -> FFT bins is O(N), a permutation +
  conjugation, no arithmetic. There is no final-reversal cost left to remove.

## Directions off the design point, and their tradeoffs

Each row is a way to change the kernel. None dominates the current point.

### Radix
- **Radix-2 (breadth-first).** More passes, more load/store traffic, worse
  locality. Produces identical residues (verified bit-exact); the native/standard
  packing and inverse are wired only for the fused radix-4 layout at N >= 32.
  Tradeoff: strictly worse locality, no arithmetic gain. Kept only as a small-N
  fallback / compile option.
- **Radix-8 / radix-16 (more fused levels).** Does **not** lower multiplies per
  node — the `u^2+v^2=1` saving is per-node and the data multiplied differs at
  every level, so there is no cross-level product reuse. Only register/memory
  traffic changes. Tradeoff: register pressure for at-best memory locality;
  no arithmetic win.

### Node arithmetic basis (same transform, same radix)
- **Composed Chebyshev, fully shared (12 "flops").** Fewest scalar operations,
  but precomputing the shared product emits **zero FMAs** and the most
  instructions (12). Tradeoff: lowest flop count, slowest wall-clock on FMA
  hardware.
- **Composed, duplicate-Q (13) / fully duplicated (14).** Recover FMAs by
  duplicating products; converge toward the monomial node. Tradeoff: interior
  points on the same curve, none beats all-FMA.
- **All-FMA even/odd (8 FMA).** The throughput optimum and the current target.
- **Low-multiply (4 mul + 10 add via `u^2+v^2=1`).** Halves multiplies, but adds
  subtractions, forfeits FMA fusion (14 ops), and is ~0.82x the all-FMA node on
  FMA hardware. Mildly less accurate (worst-case ~2-3x relative error, same
  cancellation condition, not catastrophic). Tradeoff: minimal multiplies, but
  add-port-bound and slower except on multiply-bound / no-FMA / fixed-point
  hardware, or when minimizing the theoretical multiply constant.

### Representation (carry something other than four child streams)
- **Deferred / cross-coupled (carry `eu,ou,ev,ov` or sums/diffs, materialize
  `+/-` later).** Blocked by sibling divergence: the four children sit in
  different residue subtrees with different constants (`w0 != w1`), so a deferred
  intermediate cannot propagate without replicating subtrees. Only the **bottom
  level** can be deferred, saving exactly N FMA, and only when the consumer
  accepts the even/odd pair form (a non-bin endpoint). Tradeoff: an O(1/log N)
  fraction, valid only for non-FFT output.

### Tree architecture (reorder the whole computation)
- **Hourglass / composition swap `T_A(T_B) <-> T_B(T_A)`.** The middle conversion
  between the two composition orders is a **dense transform** whose row density
  grows like (factor)/2 with N — a hidden second `N log N` — for any genuine
  split `A != B`. It is a cheap permutation **only** when `A = B`, which is
  exactly the degenerate case where the two compositions coincide and there is
  nothing to switch. Tradeoff: cheap only when vacuous; otherwise it adds a
  transform. (See `experiments/hourglass_middle_probe.py`.)
- **Four-step / Stockham / DIF-DIT permutations.** Genuinely O(N) middles
  (twiddle + transpose; digit-reversal), but they are locality/order tools that
  reduce no arithmetic. The DIF-then-DIT bit-reversal elimination saves an O(N)
  permutation that BFFT already pays as a cheap permutation. Tradeoff: locality
  only, no cost reduction.

### Output endpoint (change what is produced)
- **Bruun residue / FFT-bin endpoint (current).** FFT compatible; pays the O(N)
  pack tax (or skips it via a native-order API).
- **DCT / Chebyshev / native-polynomial endpoint.** Genuinely cheaper *only by
  doing less*: it never builds the imaginary (sine / Chebyshev-U) half. It is not
  a cheaper route to standard complex FFT bins — assembling bins from a cosine
  tree requires the sine companion, which is the same `N log N` work. Tradeoff:
  ~2x cheaper for consumers who do not need complex bins; no help for those who
  do. Worth building as a separate DCT/DST kernel, not as an FFT path.

## Resolution

- **Composition-swap hourglass:** resolved, no win. Cheap only in the degenerate
  `A = B` case; otherwise the conversion is dense / transform-like.
- **Production direction:** radix-4 fused Bruun residue path; FMA-optimized node;
  cheap linear (permutation) pack; optional native-output API to avoid the pack
  entirely. The remaining genuinely useful new work is a dedicated DCT/DST
  Chebyshev kernel for non-bin consumers, not a cheaper FFT.

The repeated structural reason every architectural change fails is the same: the
real Bruun/Chebyshev factorization makes sibling subtrees diverge (distinct real
constants), so there is no cross-subtree sharing to harvest, and any attempt to
create it (deferral, composition swap) reintroduces an equal amount of work
elsewhere. The only free axes left are endpoint (produce less) and locality
(move the same work), both already at their cheap settings.
