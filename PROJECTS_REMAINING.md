MEMORY OPTIMIZATIONS

Working solely in the main pathway of bruun_kernel and bfft:

sequential/local working sets matter more than clever scattered access.
the hot arithmetic is not obviously guilty, but the memory ownership model is only halfway as principled as the math.
introduce a heap-backed array/slab type and use it for plan/workspace-owned scratch.
Ensure that memory alignment instructions/hinting is used sufficiently by validating offsets in assembly.

The current C API is naturally reentrant when callers provide separate buffers; hiding mutable scratch inside the plan would quietly damage that. The uploaded bfft.cpp is mostly a wrapper around caller-provided buffers, so the cleaner extension is a real workspace object, not a hidden global or hidden mutable plan cache.

Eliminate the use of vectors wherever reasonable, write a std::array class that operates entirely on the heap,
And make use of std::array style operators begin, end, size.

The plan should not merely store twiddles and then have generic traversal code rediscover the execution shape every call. The plan should contain a prebuilt schedule: slices, segment order, codelet kind, offsets, m values, twiddle pointers, output slots. Then the runtime just walks that schedule over planned buffers.

Use heap-backed fixed-capacity arrays for plan-owned schedules and workspace-owned temporaries, with .begin(), .end(), .size(), operator[], and .data().

Plan-owned: N, NB, angle tables, packed leaf twiddles, native/standard maps, and a precomputed procedural schedule.
Workspace-owned: large work buffer, native scratch, traversal stacks if still needed, residue buffers, any temporary bins.
Register/local: codelet scalar temps, SIMD vectors, tiny fixed local arrays inside leaf codelets.  should stay close to the arithmetic.
The current kernel already moves in this direction with packed per-leaf metadata: it creates TW entries so the hot transform streams packed twiddle/index data instead of heap-striding through separate C, S, and IDX lookups. The next step is doing the same thing for traversal: pack the control flow.

Add explicit alignment assumptions at function boundaries where the workspace contract guarantees it.

Right now that structure exists implicitly. run_fwd_segments derives child offsets dynamically as v, v + q, v + 2*q, v + 3*q, and pushes them onto a small stack. Then forward_recursive walks the spine by repeatedly working on v + h and applying a binomial pass to v. That is already a memory schedule hiding inside control flow.
I would make it explicit:

struct MemHint {
    uint32_t base;        // offset in doubles/floats from workspace base
    uint32_t span;        // contiguous elements touched
    uint16_t align;       // desired byte alignment, usually 64
    uint16_t stream;      // logical stream id: spine, subtree, pack, convert
    uint32_t next_base;   // optional next likely region
};

struct FwdOp {
    uint32_t base;
    uint32_t q;
    uint32_t m;
    uint16_t kind;
    uint16_t tw;
    MemHint mem;
};

The important thing is to hint in offsets from the workspace base, not raw pointers. Then the same planned procedure works no matter where the heap slab lands. That also makes it compatible with your heap-array idea: the container gives stable .data(), .begin(), .end(), .size(), while the plan gives offset meaning.
The first win is not even prefetch. It is layout validation. You can ask: does this planned walk move mostly forward? Does it revisit a child while it is still hot? Does standard packing scatter reads too much? Does native order keep subtree intervals contiguous? These become measurable properties of the plan.
inside hot entry points use assume_aligned. 
But only do this after workspace actually guarantees alignment. Otherwise it is undefined-behavior bait.

RADIX CLIMBING
For ordinary FFTs, radix-4 wins because z^4 - 1 factors neatly into fourth roots and reduces twiddle/memory traffic. For Bruun/BFFT, the analogous question is:
Can we split the real polynomial tree in larger chunks while preserving real coefficients, locality, and cheap local maps?
Current BFFT is mostly a binary/quadratic real-factor story: it repeatedly descends through quadratic residue planes. The count we just measured is worse than split radix because every normalized split carries a rotation-like cost. A higher-order tree would only matter if it amortizes that cost across a larger block.
The target would be something like:
x^(4m) - 1 -> four real/paired children
or more naturally in Bruun/Chebyshev language:
T_{4m}(x) - a
split through T_m(x) with a quartic relation, so one planned node handles four descendants at once.
The good sign is that your existing norm2_fused already behaves like a radix-4-ish tree step: one parent split plus two child splits fused into a single local operation. But the op count says this fusion improves locality/codegen, not arithmetic enough. So the real discovery would be a different basis for the 4-way split that reduces the number of multiplies/adds, not merely fusing the existing binary splits.
The thing to search for is a local 4-child transform with fewer than:
norm2_fused(q) = 8q mul + 12q add
for the same four-child CRT split.
If you can derive a legal four-way real CRT basis whose node cost is below that, then the asymptotic count changes. If it is just the same two binary splits scheduled together, it helps memory and SIMD but not the mathematical operation count.
I would explore three candidates:
A Chebyshev basis, because Bruun is naturally real-polynomial/Chebyshev-adjacent.
A scaled local-complex basis J = (z-c)/s, because multiplication by roots becomes complex-like and may expose split-radix-style rescalings.
A true radix-4 residue tree where the local node maps directly to four quadratic leaves, then postpones/rescales constants so some multiplications cancel across levels.

merely fusing two binary Bruun levels into a radix-4-shaped node does not help arithmetic. current norm2_fused already is that kind of two-level fusion: it halves memory traffic by keeping the parent and children in registers, but algebraically it still costs 20q flops for the two-level node. The source confirms that structure: parent rotation, child rotations, then stores.

The asymptotic break-even is C4 = 16q. So a real radix-4 discovery would need to cut the current two-level node from 20q flops to about 16q flops just to tie standard split radix. To beat it, it needs to be below 16q.

That means the target is very concrete now:

current radix-4-shaped node: 8q mul + 12q add = 20q flops

needed radix-4 node: <= 16q flops

So the discovery is not “can we fuse two levels?” we already did. The discovery would be “can we find a different 4-child CRT basis or rescaling where the legal four-way map costs at least 20% less arithmetic than the composed binary Bruun split?”

