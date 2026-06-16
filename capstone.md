# N log N in omega 2: a journey on all possible floors

## Abstract

The FFT  is a system of coupled costs: multiplications, additions, FMA contraction, memory locality, representation, output order, and endpoint compatibility. Moving in any one direction reduces one cost while increasing another. The engineering question tempts with whether a lower-looking formula exists, but winds up becoming where the displaced work reappears.

For the FFT-compatible endpoint, the practical floor is the `2 N log N` real-arithmetic regime. Multiplications alone can be driven toward linear count, but additions rise. Local Chebyshev structure can reduce symbolic multiply count, but materializing FFT-compatible child streams restores work through additions, FMA shape, or boundary conversion. Higher radix can exist, but beyond the useful fusion point it becomes overhead. Alternate endpoints can be cheaper, but only by computing a different object.

The apparent optimum FFT design point is a fused radix-4 transform with a cheap pack to standard FFT bins. 

## 1. The object

A real FFT has two different identities.

As a mathematical transform, it is a linear map from real samples to conjugate-symmetric complex frequency bins.

As an engineered object, it is a path through representations:

```
input samples
    -> internal basis
    -> internal ordering
    -> local arithmetic schedule
    -> endpoint representation
    -> standard bins, if required
```

The cost of the FFT is therefore not only the arithmetic inside the recursive body. It also includes the cost of making the endpoint useful. A representation that is cheap internally but expensive to convert to bins is not automatically a cheaper FFT. It may be a cheaper native transform.

The central trade is:

```
FFT-compatible residues:
    more constrained internally
    cheap standard-bin endpoint

Chebyshev/DCT-like residues:
    more freedom internally
    farther from FFT-bin shape
    endpoint conversion may reintroduce the work
```

## 2. The multiplication floor is not the total floor

The multiplication count of an FFT can be made surprisingly small. In multiplication-minimal formulations, the number of nontrivial real multiplications for a power-of-two complex DFT is linear in `N`, asymptotically around `4N`.

Normalized by `N log2 N`, this tends to zero:

```
4N / (N log2 N) = 4 / log2 N -> 0
```

This does not mean the FFT has become sub-`N log N`. The missing work moves into additions. Multiplication-minimal algorithms pay with large addition networks, more constants, more irregular structure, and poor locality.

The total real-arithmetic floor is therefore addition-dominated. In the engineering regime, pushing below a leading multiplier of `2` for total real work is not the same as reducing multiplications. The additions become the floor.

The lesson is:

```
multiplications can be made cheap;
total work cannot be made cheap by that fact alone.
```

## 3. Radix as a direction

Increasing radix fuses more levels into one local map. This exposes more algebraic structure and can reduce the number of multiplications. It also increases additions, temporary values, constant pressure, code size, and register pressure.

For radix 4, the useful structure is already exposed. A radix-4 Chebyshev node naturally sees the sibling relation

```
u^2 + v^2 = 1
```

and can trade multiplication count against additions.

A higher radix continues the same trend. Radix 8, 16, or 32 can be written. A full radix-`N` codelet can be imagined. But the higher the radix, the less it behaves like a reusable FFT kernel and the more it behaves like a giant synthesized linear circuit.

Mechanically:

```
higher radix:
    fewer stage boundaries
    fewer nontrivial multiplications possible
    more additions
    more temporaries
    more pressure
    worse code generation risk
```

The question is not whether radix N can be done. It can. The question is whether it removes total work. For the FFT-compatible path, it does not appear to. It mainly changes the distribution of work.

The radix-4 point is the useful fusion point: enough structure to reduce passes and improve locality, not enough size to drown in pressure.

## 4. Basis as a direction

Changing basis changes which work is local and which work is deferred.

In a Bruun residue basis, the transform follows real polynomial factors that remain close to the FFT endpoint. The final pack to frequency bins is linear and cheap.

In a Chebyshev coefficient or DCT-like basis, the local node can look even cleaner. The even/odd form

```
f(y) = E(y^2) + y O(y^2)
```

or the composed form

```
f(y) = P(T2(y)) + y Q(T2(y))
```

exposes useful identities. These identities reduce one type of local cost.

The costs separate into several currencies.

A scalar symbolic node may reduce apparent flop count. A low-multiply node may reduce real multiplications. An all-FMA node may run faster on modern hardware even when it uses more scalar operations. 

The basis changes the shape of the local map. If the endpoint is still standard FFT bins, any non-bin basis must eventually pay for compatibility.

## 5. FMA as a direction

FMA changes the machine floor.

A shared-product formula can have fewer scalar operations on paper, but it may block contraction. For example, computing a product once and using it in both `+` and `-` outputs can turn two independent FMAs into one multiply and two loose additions. That is fewer symbolic multiplications but often worse on hardware.

The all-FMA form duplicates products so each output can be a direct fused operation. This increases scalar multiplication count but improves throughput, dependency depth, and instruction scheduling.

Mechanically:

```
sharing products:
    lower symbolic multiply count
    fewer FMAs
    longer dependency path
    more loose adds

duplicating products:
    more scalar multiplications
    more FMAs
    shorter path
    better throughput
```

Thus the FMA-optimal point and the multiplication-minimal point are different. Production code on FMA hardware should usually prefer the FMA-optimal point.

## 6. Representation deferral as a direction

One can try to avoid materializing four child streams at every node. Instead of outputting

```
f(+u), f(-u), f(+v), f(-v)
```

one can carry an intermediate representation such as

```
eu, ou, ev, ov
```

or sums and differences:

```
eu + ev, eu - ev, ou + ov, ou - ov
```

This appears promising because it delays work. The obstruction is subtree divergence.

After a radix-4 split, the four children do not share the same future constants. They enter different residue subtrees. A representation that mixes `u` and `v` siblings cannot be consumed by both subtrees without first unpacking back into child streams.

Mechanically:

```
deferral within one terminal node:
    possible
    saves boundary materialization

deferral across internal levels:
    blocked by divergent child constants
    unpacking becomes necessary

result:
    only bottom-level deferral is cheap
    internal deferral does not change the leading cost
```

So representation deferral can save an `O(N)` boundary term for native consumers, but it does not change the `N log N` body for FFT-bin output.

## 7. Tree order as a direction

A more ambitious idea is to compute part of the transform in one basis, change coordinates in the middle, and finish in another basis. This is the hourglass architecture:

```
top half: basis A
middle: cheap repack
bottom half: basis B
```

The Chebyshev identity

```
T_A(T_B(x)) = T_B(T_A(x))
```

makes this tempting. But the middle conversion is cheap only in the degenerate case where the two sides are the same composition. When the two decompositions are genuinely different, the conversion is dense. The hidden cost is another transform.

Mechanically:

```
A = B:
    middle is cheap
    but the swap is vacuous

A != B:
    swap is meaningful
    but the middle conversion grows dense
    work reappears as another transform
```

It remains useful as a way of thinking about locality, but not as a way to remove arithmetic.

Four-step, Stockham, and DIF/DIT hybrids are real locality tools. They move permutations and transposes to more favorable places. They do not erase the arithmetic floor.

## 8. Endpoint as a direction

Changing the endpoint can produce a cheaper transform, but then the transform is no longer the same FFT object.

A DCT endpoint can be cheaper if the consumer needs only the cosine side.
## 9. Output order and the BFFT trade

BFFT moves disorder to the boundary.

During the transform, native residue order gives a locality advantage. The recursive body stays in an order that is natural for the Bruun factorization. Standard FFT bins are produced afterward by a linear pack: permutation, sign/conjugation, and store.

This is the core trade:

```
classic FFT:
    often pays ordering and stride costs inside the transform
    may land directly in standard bin order

BFFT:
    keeps the transform body locality-friendly
    pays an O(N) pack to standard bins
```

The pack is not free, but it is lower order. It is not another transform. Native consumers can skip it entirely.

Thus BFFT does not claim that standard order is free. It claims that moving standard-order compatibility to the boundary is cheaper than forcing the entire `N log N` body to live in standard order.
