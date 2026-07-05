# The twisted DIP: helical phase-packet walks

Companion: `experiments/dip_twisted_walk.py` (every claim verified there),
`experiments/phase_fft.py` (the admissible shear gauge family),
`notes/phase_packet_fft.md` (the diamond), `src/detail/bruun_dip_kernel.hpp`.

Mission (2026-07-05): the DIP walks the phase toroid radially — (d, e) →
(d, 2e), (e−d, 2e), halve span / halve phase, no angular drift. Sought: legal
SPIRALS — stage-dependent shears of the rank coordinate that preserve the
sparse real Bruun cell while re-ordering the global packet sequence, ideally
inheriting the DIF's mirror-merge operator and improving frequency-side
locality.

## 1. The quantization theorem (Q1, verified numerically)

The complex walk tolerates the whole admissible diamond σ·(n/e)² ≡ 0 (mod n).
The REAL walk does not: the Bruun fold pairs conjugate packets, and in a
sheared frame conjugation maps (κ, j) → (−κ + 2σj mod e, j). The partner is
j-independent — i.e. a span-local fold exists — **iff 2σ ≡ 0 (mod e): σ ∈
{0, e/2}**. Verified by direct partner-map scan at every level (n = 64): the
fold-legal set is exactly {0, e/2} at every level.

**The real spiral is quantized to half-turns.** One bit per level; finer
angular drift makes conjugate partners rank-dependent and densifies the cell.
The frame must return to origin at the last level (σ_m ≡ 0), so the total
winding is a half-integer determined by the bit-word — the "non-integer
winding" lives in this binary family and nowhere else.

## 2. The half-turn collapses to sibling orientation (Q2, verified)

A half-turn at level e relabels the folded diagonal d ↦ e/2 − d, which maps
the child set {d, e−d} to itself — the twist is exactly a SWAP of which child
is placed low. So the entire fold-legal gauge family is realized, at zero
arithmetic cost, by per-level (or, more generally, per-node) sibling
orientation words in the span walk. Verified: the twisted real walk computes
`rfft` to machine precision for arbitrary random orientation words.

## 3. What spirals can and cannot buy (Q3)

- **Comb invariance (proven + verified):** the leaf set of any subtree is the
  comb {jE ± d} in EVERY legal gauge. Frequency-band segregation by packet
  ancestry is unreachable — the sets are gauge-invariant; only the ORDER of
  emission moves. (For band-limited consumers, the win must come from the
  egress order, which is already ascending per subtree.)
- Per-level spiral words barely move order metrics (emission travel 255.8 →
  255.5 at n = 1024; low-band clustering invariant).
- **Per-node greedy orientation** (choose each node's placement to continue
  nearest the previously emitted bin — the DIF heapopt DP, DIP edition) cuts
  emission travel 25% (255.8 → 191.8). Available knob: it changes only the
  T-permutation build and the egress rank map, zero arithmetic. Not yet
  cashed in C++ (the boundary is ~10% of runtime; revisit if egress ever
  dominates again).

## 4. The mirror-merge inheritance (Q4, verified + SHIPPED)

Sibling spans (d, 2e) and (e−d, 2e) have complementary angles: their table
indices sum to exactly n/4, so (cos, sin) of one sibling is the SWAP of the
other. This is the DIT's mirrored-lane twiddle sharing, present in the DIP
tree for free — no twist needed.

Shipped in the kernel (2026-07-05):
- `reset()` enforces quarter-wave symmetry bit-exactly on the table
  (sin_[k] = cos_[n/4 − k] by construction), so sharing is bit-identical.
- `span8_fwd/inv`: 4 index computes + 2 twiddle-vector loads + 1 scalar pair
  instead of 7 + 7; sibling pair cells take `V2_SWAP` of the same register
  (`cell_fwd_pair_v` / `cell_inv_pair_v`).
- All four radix-4 `cell4` sites: the high-child twiddle (ch, sh) = (sl, cl)
  — the kh index and its loads deleted.

Measured (hardened 3-way bench, cool): 64K forward 1.37 → 1.27× DIF,
inverse 1.03 → 0.96×; 4K forward 1.10×, inverse 0.84×. Correctness
machine-precision (dip_fe ~1e-11 vs DIF, roundtrip ~2e-15).

## 5. Verdict

The spiral exists, exactly and at zero cost — but its legal winding is the
half-turn word, its reachable effect is packet ORDER only (combs are
gauge-invariant), and its best concrete dividend was the mirror operator,
which needed no twist at all once the complementary-angle structure was
noticed. The hybrid the mission hoped for is realized in the operator sense
(DIT's twiddle-shared mirrored merging + DIT-style gather boundaries + DIF's
in-place segment recursion, all in DIP space); the band-segregation hope is
closed off by the comb-invariance proof.
