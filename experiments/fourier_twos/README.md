# Fourier arithmetic by twos

2026-09-05. This replaces the walk-search objective for the current investigation.
The earlier walk experiments are preserved and have not been extended here.

**Result:** an executable, exact Fourier encoding whose forward/inverse ring
FFT uses shifts, masks, additions, subtractions and sign-changing wraps. There
are no general signal multiplications or numeric twiddle tables in that path.
For bounded integer input, every output can be lifted to the exact algebraic
value of the ordinary negative-exponent DFT. A second, compact encoding retains
all bins in N integer coefficients and implements a whole time-delay phase ramp
with one modular shift per non-DC orbit.

**Remaining requirement:** returning ordinary numerical complex bins is not
free in either encoding. This experiment does not establish a faster replacement
for BFFT's numeric FFT, or claim that a finite-field spectrum alone is the
ordinary complex spectrum. `diagnostic_complex` explicitly pays for projection.

## 1. Make the root a power of two

Let N >= 2 be a power of two, d=N/2, and choose a binary digit width b. Define

    B = 2^b,       Q = B^d + 1 = 2^(bd) + 1.

In the ring Z/QZ,

    B^d = -1,     B^N = 1.

B has exact order N. It is also a **principal** root: for every 0 < k < N,
sum_j B^(jk) = 0. To see the cancellation, write k=2^v*u with u odd and pair
terms half an order apart; their ratio is B^(d*u)=-1. Thus the radix-two FFT
and its inverse are valid even when Q is composite. N is invertible since Q
is odd. The inverse scale 1/N is a negative power-of-two shift in the ring.
It is not truncating integer division of an arbitrary residue.

Every twiddle application is now

    value -> value << (b*exponent), reduced modulo Q.

Reduction uses the identity

    value = low + 2^(bd)*high  ->  low - high,

where low is a mask and high is a right shift. The implementation repeats this
fold until the result is canonical. Negative shift counts wrap around the
period 2bd. This is the requested sign-changing wrap, not saturation/clipping.
Additions/subtractions and carry handling remain necessary.

The full executor is conventional radix-two scheduling **over this different
algebra**; no new butterfly factorization is claimed. Its input permutation is
explicitly paid, and its array of ring outputs is in natural bin order.

## 2. Recover the actual algebraic complex Fourier values

Introduce the symbolic root z in

    R = Z[z] / (z^d + 1),    z -> exp(-2*pi*i/N).

For integer samples x[j], the exact kth Fourier value has the unique form

    P_k(z) = sum_(r=0..d-1) c[k,r] z^r.

The c[k,r] are obtained by reducing sum_j x[j] z^(jk) modulo z^d+1.
Replacing z with B maps this polynomial to the ring FFT output modulo Q.

Suppose |x[j]| <= A. Every coefficient obeys |c[k,r]| <= N*A. Choose

    B > 2*N*A + 1.

Then each coefficient fits strictly inside a balanced base-B digit and

    |P_k(B)| <= N*A*(B^d-1)/(B-1) < Q/2.

The centered modular representative therefore equals P_k(B) as an integer.
Balanced digit extraction recovers every c[k,r] exactly. This gives a bounded
injective lift to the **ordinary DFT's algebraic values**, not merely an
inverse-transform identity in a different ring. The tests compare all retained
coefficients to a separate direct polynomial reduction.

Inputs with an agreed binary fixed-point scale are also covered: transform the
integer numerators and keep that common power-of-two denominator. Arbitrary
float input is deliberately rejected by this integer prototype; any quantization
must be explicit.

For x=[1,2,3,4,5,6,7,8], b=8 gives B=256 and Q=4294967297. The encoded X[1] is
4227595261. Its centered balanced digits are [-4,-4,-4,-4], so

    X[1] = -4 - 4z - 4z^2 - 4z^3
         = -4 + 4i*(1+sqrt(2)).

The irrational value is exact in the first expression. Producing its decimal
real/imaginary components requires an evaluation step.

### Why the bound matters

A small unguarded Fermat transform is not enough. At N=8 and B=2, the two
distinct exact bin values 2 and z both encode as 2. The test constructs these
from different signals with samples in {-1,0,1}. Full modular transforms remain
invertible, but an individual modular bin cannot then be interpreted as an
individual complex bin. Guarded lifting fixes precisely this ambiguity.

There is no unital ring homomorphism Z/QZ -> C: it would send Q*1=0 to the
false complex equality Q=0. Our lift is valid on a proved bounded subset, not
an unrestricted change of complex coordinates. Additional spectral arithmetic
must carry new coefficient bounds. The phase operation below preserves them.

## 3. Avoid expanding all N bins into N/2 coefficients each

Over the rationals,

    t^N - 1 = (t-1)(t+1)(t^2+1)...(t^(N/2)+1).

The factors group frequencies by the exact order of their roots. An order-m
packet has m/2 coefficients and represents the odd powers of exp(-2*pi*i/m).
These frequencies are Galois conjugates; one coefficient polynomial represents
the whole orbit. The total coefficient count is

    1 + 1 + 2 + 4 + ... + N/2 = N.

The compact transform uses this direct factorization. For an m-coefficient
working polynomial split into low/high halves a,b:

    order-m packet = a-b;     recurse on a+b.

There are exactly 2N-2 scalar additions/subtractions, no twiddle constants, and
no numeric complex arithmetic. Its inverse reconstructs (plus+minus)/2 and
(plus-minus)/2 with exact right shifts; invalid integer parity is rejected.
This is a cyclotomic polynomial remainder representation, not an N-bin numeric
FFT mysteriously computed in linear time. Resolving all the conjugate values
is the work it has deferred.

Order-m coefficients obey |c| <= 2*N*A/m. `orbit_lane_bits` uses that tighter
bound independently for each packet. The largest packet thus needs digit width
depending on A, not on N*A. DC is retained as an ordinary signed integer.

### Execute the factorization on packed bits directly

`fold_packed` provides a third executor for the compact endpoint. With a uniform
guard width, pack the input polynomial into the single signed integer P(B).
Split its bits at B^(m/2): word=low+B^(m/2)*high. Then

    order-m packet = low-high modulo B^(m/2)+1;
    next word      = low+high modulo B^(m/2)-1, centered.

The guard makes the second remainder the exact integer evaluation of the folded
polynomial. Repeat on the second word. No coefficient unpacking occurs inside
this kernel: log2(N) stages produce every compact packet using shrinking packed
words. The sum of the widths is O(N*b), so this kernel's bit work is linear in
the guarded input size with ordinary linear-time shifts/additions. This is a
compact polynomial remainder tree, with the same deferred numeric resolution
as the coefficient version. Initial input packing is charged separately; the
reference Python Horner packer repeatedly allocates growing integers. The packed
fold retains uniform guard widths and therefore uses more bits than the tighter
per-orbit-width representation above.

### A useful operation entirely in this representation

A circular time delay by s samples multiplies bin k by exp(-2*pi*i*k*s/N).
Within every order-m packet this is simply multiplication by t^s modulo
t^(m/2)+1: a signed coefficient rotation. After binary packing, it becomes

    packed_packet << (b_m*s), with sign-changing modular wrap.

There are log2(N) non-DC packets. This avoids constructing a different numeric
twiddle for every bin, and never expands the packets into per-bin polynomials.
The test checks positive/negative delays against the exact time-domain circular
shift and reconstructs the original integer signal through the compact inverse.

This is a substantive phase representation identity, but the shifts operate
on long integers. Their bit cost is linear in packet width, not one CPU cycle
per arbitrary-size packet. Also, time-domain circular delay itself has a cheap
index implementation, so this example is evidence for the representation law,
not evidence of a faster way to delay a signal.

Pointwise multiplication of two spectral packets becomes polynomial
multiplication modulo their cyclotomic factor. Arbitrary independent complex
bin gains do not become free shifts. A useful BFFT integration would need a
downstream operation that stays cheap in the packet representation.

## 4. Where the cost has gone

* Expanded exact transform: N residues of about b*N/2 bits each. FFT operation
  count is O(N log N) **long-integer** operations; elementary bit work can reach
  O(b*N^2 log N), and materializing all polynomial bins needs N^2/2 integers.
* Compact transform: N coefficient lanes, or about O(N*(log(A+1)+1)+log N)
  packed bits with the per-orbit guards. It retains conjugate groups rather than
  independent numerical bins. Construction still reads all N samples.
* Numeric projection: currently a clearly separated, direct O(N^2) diagnostic
  over all expanded bins. It uses trigonometry and multiplication. No claim of
  shifts-only **numeric** output includes that function.

For rational-coordinate linear representations that implement a primitive
N-root's action faithfully, the minimal polynomial z^(N/2)+1 has degree N/2.
Consequently such a root action needs at least N/2 rational dimensions. A
two-dimensional real rotation gets its compactness from its irrational
coefficients. This explains the expansion in this construction; it is not a
general lower bound on all possible bounded-input or approximate algorithms.

Likewise, finite additions/subtractions and powers-of-two scalings of rational
inputs cannot return the exact irrational Cartesian component sqrt(2)/2 of
an eight-point impulse transform. Symbolic algebraic output escapes that
restriction, as this implementation does. Finite-precision Cartesian output
can certainly be computed with bit logic, but replacing general multiplication
with a shift/add circuit does not demonstrate that its cost has disappeared.

## Validation and provenance

Run on the selected M4 Mini and copy the /tmp artifacts immediately:

```sh
sh experiments/fourier_twos/run.sh
```

`m4_results/fourier_twos_results.json` records source hashes, runtime versions,
all-bin exact comparisons through N=1024, inverse and compact-delay checks,
storage counts, and reference-Python representation timings. The tests also
cover every eight-sample +/-1 pattern, impulse bases, extreme signs, very large
integer inputs, composite moduli, unsafe-lift collisions, and rejected ranges.
These timings are not a native BFFT comparison: the output contracts differ.

The power-of-two Fermat root construction is established mathematics used by
Schönhage-Strassen, not a new discovery here. GMP's implementation documentation
explicitly describes Fourier evaluations/interpolation using only shifts,
additions and negations. Cyclotomic/virtual-root extensions and binary
segmentation are also established. This experiment supplies a concrete bounded
lift, compact packet implementation, and validation for this repository; it
makes no priority claim for those mathematical ingredients.

Primary references:

* [GMP: FFT Multiplication](https://gmplib.org/manual/FFT-Multiplication),
  including the separate pointwise multiplication cost.
* [Arnold Schönhage: selected topics](https://pages.iai.uni-bonn.de/schoenhage_arnold/topics.html),
  section 1 on Fermat rings and binary segmentation.
* [Robin Larrieu: Fast Finite Fields Arithmetic, doctoral thesis](https://larrieu.perso.math.cnrs.fr/files/Thesis.pdf),
  Fourier transforms, virtual roots, and conjugacy-based representations.
