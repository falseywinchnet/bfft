# Diagonal packets: size recursion, carry, and movement

This experiment develops the recovered N=64 phase-walk cube into an explicit
family, checks it independently, and follows its cost through a native schedule.
It does not replace the production BFFT kernels.

## Exact family and coordinates

Let m=2^r, N=m², a=2^floor(r/2), b=2^ceil(r/2), so ab=m.
Natural indices are n=t+mf and k=u+mv. Use the cross pairing

    B((t,f),(u,v)) = (tv+fu)/m  (mod 1).

For odd slope s, define

    H = {(aj, saj+bl) mod m : 0<=j<b, 0<=l<a}.

It has m elements. Its internal pairing is integral because m divides 2sa²
and ab=m. Consequently H=H-perp: it is self-annihilating. Representatives
are (p,q), 0<=p<a, 0<=q<b. Every sample has unique coordinates

    t = p+aj,   f = (q+saj+bl) mod m.

Q gathers those samples and takes the normalized character transform
F_b in j and F_a in l. Packet coordinates are (q,p,alpha,beta).
This is a precise family of intermediate states, not merely a diagram.

At odd r, b=2a and the subgroup is oblique. At even r, a=b and the shear
can be absorbed into l: the subgroup is the rectangular a-by-a lattice.
Different odd slopes reparameterize the same subgroup in this family.
It is not a continuously adjustable family of distinct diagonal orientations.

## Torus transition

Let G be the normalized cross-paired torus Fourier transform. P=QGQ* is
monomial. For output packet (q,p,alpha,beta), set

    p' = (-beta) mod a
    q' = (-alpha-s p') mod b
    alpha' = (s p+q) mod b
    beta' = p.

Then

    (Pz)[q,p,alpha,beta]
      = exp[-2πi(p q'+q p')/m] z[q',p',alpha',beta'].

This follows by summing the subgroup characters on both sides of G. The
annihilator condition makes the two character sums Kronecker deltas.
The implementation and its tests construct Q and G separately to check P.

## Cyclic carry: exact transition at every size

The ordinary normalized DFT F has kernel

    F[(u,v),(t,f)] = G[(u,v),(t,f)] exp[-2πi tu/m²].

This is an entrywise identity. Define C=FG*. In natural coordinates,

    C[(u,v),(u',v')]
      = delta(u,u') / m * sum(t=0..m-1)
          exp[-2πi t(v-v')/m] exp[-2πi tu/m²].

Thus C is a controlled fractional circular shift in v. Its packet-coordinate
form Cp=QCQ* preserves p and beta. Write tau=(-beta) mod a. Each fixed
(p,beta) block acts on (q,alpha), both of length b:

    Cp[(q,alpha),(q',alpha')]
      = 1/b² sum(j,h=0..b-1)
        exp[2πi(alpha'-alpha)j/b]
        exp[-2πi(tau+ah)(q-q')/m]
        exp[-2πi(tau+ah)(p+aj)/m²].

Proof: C preserves t, hence p and j. Summing l,l' in the two Q factors
forces beta=beta' and t_dual = tau+ah. The shear cancels because both sides
have the same j. This also explains why the carry rule is independent of s.

Equivalently, with normalized Fourier matrices,

    Cp_block = E (F_b tensor F_b) diag(vec(D))
                 (F_b* tensor F_b*) E*
    E[q,alpha] = exp[-2πi tau q/m]
    D[h,j] = exp[-2πi(tau+ah)(p+aj)/m²].

The factorization is implemented in `carry_blocks`, without computing an
N-point FFT or converting the carry back into natural coordinates.
The complete transform is F=Q* Cp P Q.

## Growth and the recursive obstruction

There are a² invariant carry blocks, each of dimension b². Their dimensions
are m at even r and 2m at odd r. They grow like sqrt(N), rather than staying
at the N=64 example's dimension 16. A dense application costs O(N b²),
or O(N^(3/2)); treating the finite-size blocks as small constants is invalid.

The exact block factorization above costs O(N log b), with O(N) stored
coefficients. It is economical asymptotically, but explicitly uses two
forward and two inverse b-point Fourier transforms per b-by-b block.
It does not establish a new diagonal FFT recursion independent of those
transforms. A dense recursion into generic smaller carry matrices would
miss this distinction.

There is also an exact separation-rank result. Apart from nonzero row/column
phases, D[h,j]=exp[-2πi h j/b²]. This is a b-by-b Vandermonde matrix at b
distinct nodes, hence has rank b. The operator Schmidt rank of diag(vec(D))
across the h/j axes equals rank(D), because the diagonal matrix units on each
axis are independent. Local Fourier conjugations and E preserve that rank.
Therefore the carry block requires b separable operator terms in that fixed
tensor partition. The rank grows without bound. This excludes a fixed-rank
separable closure of this family, not every possible fast diagonal algorithm.

## Exact cancellations and the reduced route

Two b-FFT/inverse-FFT pairs cancel across the carry boundaries. Keeping them
would artificially inflate the cost of the representation.

On the input side, p'=tau makes the q-dependent torus phase cancel E*.
The inverse q transform then cancels the input j' character transform.
In unnormalized notation the remaining intermediate values are

    W[q',tau,h,p] = sum(l) x[tau+ah + m((q'+sah+bl) mod m)]
                              exp[-2πi p l/a]
    V[p,beta,h,alpha] = exp[-2πi p(q'/m+s h/b)] W[q',tau,h,p]
    q' = (-alpha-s tau) mod b.

Then apply an unnormalized inverse b-FFT in alpha, multiply D[h,j], and
apply a forward b-FFT in h. Multiply E[q,j], return to (q,p,j,beta), and
apply an unnormalized inverse a-FFT in beta. Scatter through the original
sample-coordinate map to obtain the ordinary unnormalized DFT output.
The factors from the two cancellations remove the carry normalization.

The output cancellation is particularly transparent: E does not depend on
alpha, so the carry's final alpha FFT cancels the alpha inverse FFT in Q*.
`transform_cancelled` and the native `cancelled` route implement these
identities separately from the uncancelled construction.

The reduced route consists of transforms of lengths a,b,b,a, with opposite
signs on two axes and known intervening phases/permutations. It has the
ordinary radix-2 butterfly count. This construction has exposed a mixed
row/column schedule, not removed that Fourier work. It does not meet the
owner's stronger criterion of a diagonal mechanism that avoids paying for
DIT/DIF-like behavior.

## Native routing experiment and movement model

`native.cpp` supplies DIT, DIF, Stockham, the full packet route, the cancelled
route, and a fused version of the cancelled route. All compute the same
complex-input, natural-order, unnormalized transform; input is immutable and
output is a separate supplied buffer. Plans and allocation are outside timing.
All use the same scalar complex arithmetic and radix-2 kernels. This is a
matched schedule comparison, not a comparison against optimized FFTW or the
production real-input SIMD BFFT kernels.

The fused route folds every local bit reversal into an existing gather or
scatter: input a-DIT, inverse b-DIT, forward b-DIF, output inverse a-DIF.
The carry diagonal is multiplied during the first forward b-DIF reads.
It consequently has no standalone bit-reversal swaps. This is a concrete
autosorting implementation, but its four routing passes still cost data
movement. Removing a named sorting pass does not remove that cost.

For a length L power-of-two FFT, define

    B(L) = L log2(L)/2
    S(L) = [L - 2^ceil(log2(L)/2)]/2
    G(L) = L log2(L)/2 - 3L/2 + 2  (L>=2; G(1)=0)
    J(L) = L/2 - 1                 (L>=2; J(1)=0).

B counts butterflies, S swaps in a standalone bit reversal, G general
complex multiplications after removing unity and quarter-turn twiddles,
and J quarter turns. Every butterfly transfers two complex inputs and two
outputs; a swap likewise transfers four complex elements. A complex element
is 16 bytes. These are logical accesses to/from arithmetic temporaries;
they are not measurements of DRAM traffic, cache misses, or register spills.

| Schedule | Butterflies | Standalone swaps | Additional full read/write passes |
|---|---:|---:|---:|
| DIT / DIF | Nr | S(N) | 1 input copy |
| Stockham | Nr | 0 | 0 |
| Full packet | N(r+2 log2 b) | 2N S(a)/a + 6N S(b)/b | 5 |
| Cancelled | Nr | 2N S(a)/a + 2N S(b)/b | 5 |
| Fused | Nr | 0 | 4 |

For each row, reads = writes = 2*butterflies + 2*swaps + passes*N.
The five packet passes are entry gathering, monomial/grouping, carry diagonal,
return grouping, and exit scattering. Fusing the diagonal removes one pass.
Stockham writes directly to the chosen alternating buffer each stage and
selects initial destination by stage parity so the final output needs no copy.

General multiplications for full packet are 2N G(a)/a + 6N G(b)/b + 3N;
cancelled/fused use 2N G(a)/a + 2N G(b)/b + 3N. Quarter-turn counts follow
the same formulas with J and no added 3N. The full packet also performs
2N real scaling multiplications on exit. The reduced routes require none.
Each counted general multiplication reads one complex coefficient from its
plan (including the three N-element phase tables). These coefficient reads
are additional to the sample-transfer counters. Quarter turns require no
coefficient load; FFT roots with phase 1 are skipped.

Index metadata: DIT/DIF's reversal loop references N reversal entries.
Stockham uses arithmetic indices without index tables. The full packet
references pack twice and source once (3N index entries), plus 8N reversal
entries in its small FFT calls. Cancelled uses the same 3N route entries
plus 4N reversal entries. Fused uses 3N route entries plus 4N reversal
references embedded in the routing loops. These are source-level table
references before compiler hoisting; an index entry is 4 bytes. No claim
equates them to independent hardware loads.

DIT/DIF need no N-element scratch buffer beyond output; Stockham needs one;
these packet implementations need two. Production-minimal packet plans need
two N-element index tables and three N-element complex coefficient tables,
plus small a/b roots and reversal tables. The exploration executable retains
both cancelled and original plans together for comparisons, so its actual
combined allocation is larger. No timing includes plan creation.

The baseline build disables automatic loop and SLP vectorization. No explicit
SIMD lane shuffles are used by any schedule: complex values are scalar pairs.
Quarter turns exchange scalar real/imaginary roles and negate one component;
they are reported separately. This does not establish a zero-shuffle SIMD
implementation. The explicit NEON follow-up is now in `NEON_LEDGER.md`: it specifies the
packing, accounts for lane exchanges and coefficient operands, and inspects
generated code for vector stack accesses. Other SIMD packings remain outside
this experiment, including cross-butterfly SoA and AVX implementations.

## Verification and reproduction

`first_sweep.json` records independent dense Q/F/G checks through N=1024.
`test_packet.py` checks all basis vectors of the carry through N=256 at three
odd slopes, invariant packet labels, and vector actions through N=65536.
Patterns include constant, alternating, a Fourier tone, and seeded complex
random inputs. The native executable checks against DIT at every size and an
independent direct DFT through N=256. `audit_counts.py` independently derives
all seven counter fields and checks every native size/method case.

Run from the authoritative repository using the M4 mirror:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/diagonal_packet_walk/test_packet.py
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/diagonal_packet_walk/probe.py \
  --max-r 5 --out /tmp/diagonal_packet_first.json
/Users/ultimussecundai/.local/bin/m4build -- sh -c \
  'clang++ -O3 -std=c++17 -fno-vectorize -fno-slp-vectorize \
   experiments/diagonal_packet_walk/native.cpp -o /tmp/diagonal_packet_native && \
   /tmp/diagonal_packet_native 8 > /tmp/diagonal_packet_fused.json'
```

Copy each JSON out of /tmp using the host selected by `m4host` immediately.
Run `python3 experiments/diagonal_packet_walk/audit_counts.py PATH_TO_JSON`
on the copied native result. Native timings are medians of nine batches, with
minimum and maximum retained. Methods run in a fixed order; small differences
are not robust speed claims. Scalar counters, not timing ratios, establish
the schedule's exact arithmetic and routing costs.
