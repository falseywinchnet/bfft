# Beyond Cooley-Tukey The Normalized Four Star Fourier Transform

Joshuah Rainstar, architect. Engineering by Claude Fable (Anthropic), GPT-5-class Codex sessions (OpenAI), under direction. Library: BFFT, https://github.com/falseywinchnet/bfft. June 2026.

## 1. Georg Bruun

Georg Bruun, of the Electronics Laboratory at the Technical University of Denmark in Lyngby, submitted a manuscript on November 22, 1976 that appeared in February 1978 in the IEEE Transactions on Acoustics, Speech, and Signal Processing under the title "z-Transform DFT Filters and FFT's." The surname appears in later signal-processing literature attached to other authors; they are not the same person. Bruun is, at the time of this writing, one hundred and four years old. Beyond his affiliation, his acknowledgment of P. G. Thomsen of the Institute for Numerical Analysis for helpful discussions, and the paper itself, the public record preserves almost nothing about him. Working in the era of charge-coupled devices and bucket-brigade delay lines, he asked whether the discrete Fourier transform could be reached by trading delay sections for multipliers, through filter banks whose coefficients are real. His answer was yes, and it contained, in compressed and largely pictorial form, a second fast Fourier transform tradition that the field misread and then largely ignored for forty-eight years. 


The discrete Fourier transform of N samples is the evaluation of the polynomial x(z) = x0 + x1 z + ... + x(N-1) z^(N-1) at the N-th roots of unity, z = exp(-2 pi i k / N) for k = 0..N-1. Computed naively this is N^2 complex multiply-adds. Every fast Fourier transform is a scheme for exploiting the algebraic redundancy among those N evaluation points so that the cost falls to order N log N.

The dominant tradition descends from Cooley and Tukey, 1965, and is organized around the notion of radix. A radix-2 decimation-in-time FFT splits the length-N transform into two length-N/2 transforms over the even-indexed and odd-indexed samples, combines them with N/2 complex rotations called twiddle factors, and recurses. Radix-4 and radix-8 take larger steps with slightly better constants. Split-radix, introduced by Yavne in 1968 and rediscovered by Duhamel and Hollmann in 1984, mixes a radix-2 step on the even half with radix-4 steps on the odd quarters and held the record for the lowest known arithmetic count for power-of-two sizes, 4 N log2 N - 6 N + 8 real operations, until Lundy, Van Buskirk, and then Johnson and Frigo's tangent FFT reduced the count to about 34/9 N log2 N by rescaling twiddles into tangent form. These counts matter less than they once did. On modern machines the budget is dominated by memory movement, cache behavior, and the regularity of access patterns, which is why FFTW, the reference high-performance library, wins not by a lower operation count but by a planner, generated codelets, and cache-conscious recursion.

The choice of radix carries structural costs. First, the twiddle factors: a Cooley-Tukey transform consumes a different angular comb of complex rotations at every stage, and these tables must be stored or regenerated. Second, the permutation: the natural output order of an in-place radix-2 transform is bit-reversed, and either the input or the output must pay a full reordering pass, or the implementation must adopt the Stockham autosort form, which trades the permutation for out-of-place ping-pong buffers and extra bandwidth. Third, the arithmetic lives in the complex field even when the data are real. A real input has Hermitian-symmetric spectrum, so half the complex work is provably wasted; the standard repairs are the packing trick (transform N real points as N/2 complex points and untangle), the Sorensen real-FFT family which restructures the butterflies, and the fast Hartley transform, which substitutes the cas kernel. Each of these recovers roughly the factor of two while inheriting the complex tradition's twiddle and permutation structure.

Around the dominant tradition stand the specialist transforms. The Good-Thomas prime-factor algorithm removes twiddle factors entirely when N factors into coprime parts, at the price of Chinese-remainder index manipulation. Rader's algorithm converts a prime-length DFT into a cyclic convolution of length N-1. Bluestein's chirp-z algorithm converts any length into a convolution at roughly threefold cost. Winograd's algorithms minimize multiplications absolutely, at the price of many additions and an implementation complexity that modern hardware does not reward. All of these answer the question: given that we evaluate at complex roots of unity by recursive index splitting, how cheaply can the rotations be arranged? Bruun's 1978 paper asks a different question.

## 3. What Bruun actually proposed, and what the field heard

Bruun's construction begins with the observation that the DFT is a remainder computation. Reducing the input polynomial x(z) modulo (z - w) evaluates it at w. The polynomial z^N - 1 factors over the complex numbers into N linear terms, one per root of unity, and the Cooley-Tukey recursion is one particular schedule for performing that reduction tree. But z^N - 1 also factors over the reals. The complex roots come in conjugate pairs, and each pair (z - exp(i a))(z - exp(-i a)) multiplies out to the real quadratic z^2 - 2 cos(a) z + 1. Bruun's tree therefore keeps conjugate roots paired all the way down. The root is z^N - 1. One spine of the tree is the binomial chain z^M - 1 splitting into (z^(M/2) - 1)(z^(M/2) + 1). The factor z^M + 1 enters the quadratic world as the trinomial z^M - 2 cos(pi/2) z^(M/2) + 1 with the cosine equal to zero, and from there every trinomial node z^(2q) - 2 cos(a) z^q + 1 splits by half angles into

```text
z^q - 2 cos(a/2) z^(q/2) + 1
z^q + 2 cos(a/2) z^(q/2) + 1
```

The leaves are degree-two quadratics z^2 - 2 cos(b) z + 1, each owning one conjugate root pair, plus the linear factors z - 1 and z + 1 for the DC and Nyquist bins. Every interior coefficient in the entire tree is a real number of the form plus or minus 2 cos(angle). For a real input signal, the whole interior of the transform is real arithmetic; complex numbers appear only at the final evaluation of each linear leaf residue r0 + r1 z at z = exp(-i b), which emits the standard bin. Bruun claimed, correctly for the forms he discussed, roughly half the real multiplications of the classical complex FFT on real data.

The potential benefits are straightforward. The interior is real, so for real signals there is no wasted conjugate half and no complex butterfly machinery; storage per intermediate value is one double, not two. The coefficient family is a single real cosine family rather than a per-stage complex comb. The factor tree is recursive and binary, so the cache story is at least as good as Cooley-Tukey's in principle. And, as Section 6 makes precise, the tree is a Chinese remainder decomposition of the convolution algebra R[z]/(z^N - 1), which means pointwise spectral operations can be performed in the real residue rings without ever materializing complex bins.

The recorded tradeoffs are equally straightforward, and they led to the method's abandonment. First, accuracy. The half-angle cascade drives coefficients 2 cos(a/2) toward 2 as angles shrink, the intermediate polynomial bases lose orthogonality, and the forward transform in the monomial basis loses about two digits by N = 512 relative to Cooley-Tukey. The English Wikipedia page for Bruun's algorithm, at the time of this work, stated that the method has not seen widespread use because real-input Cooley-Tukey approaches were adapted with at least as much efficiency, and that there is evidence Bruun may be intrinsically less accurate under finite precision. Matteo Frigo, co-author of FFTW, stated the institutional view in correspondence during this work: Bruun is not very accurate in floating point, which is why people mostly ignored it after 1978, although the objection does not apply in finite fields, where the method has survived in lattice cryptography (the 2023 NTRU Prime vectorized polynomial multiplication work uses Good-Thomas, Rader, and Bruun together precisely because finite fields do not care about rounding). Second, the output order. Bruun's leaves do not arrive in frequency order; they arrive in the order of the half-angle tree, which looks, when plotted against frequency, like structured noise. Third, no inverse was on record. The literature contains forward Bruun transforms and real-symmetric specializations, notably Chen and Sorensen's 1992 construction in which almost all twiddle factors are real except the last stage, but no published exact inverse Bruun transform existed. Fourth, and most consequentially, the field did not understand what the paper said. Frigo's assessment, after reviewing the present work, is that Bruun's paper was completely misunderstood by later authors: Nussbaumer's influential book presents Bruun as polynomial division, whereas the original is really describing Chinese-remainder reconstruction, a confusion Murakami's papers then compounded, the original paper being unclear and over-reliant on pictures. He further observed that the Circle FFT of 2024 (eprint 2024/278), presented as many pages of sophisticated mathematics for finite-field STARK proof systems, is in essence a reinvention of Bruun: the Bruun butterfly x[0] + x[n/2] + 2a x[n/4] splits into t = x[0] + x[n/2] followed by t + 2a x[n/4], the first kind migrates to the front of the computation graph as a change of input basis, and the Fourier matrix factors as F = C T with C the "circle" transform; this factorization buys a genuine saving when interpolating between cosets, since (C' T)(C T)^(-1) cancels the T T^(-1) basis changes. Independently, recent adjacent work approaches the same territory from the complex side: arXiv 2504.07264 (2025) constructs a complex FFT with separated real and imaginary input and output channels for pipelines that post-process the two channels independently, which is the standard tradition reaching toward the all-real interior that Bruun's tree possesses natively.


The present effort began as a public admission of failure. In November 2024 the repository falseywinchnet/bruun_fft was created with a README that read, in part: if you came here you were hoping to find an implementation; you are not alone; I am collecting information and seeking implementations that can be helpful to convert. The stated goals were to implement Bruun in numpy, to break down the shuffling structure with sympy, to find an in-place shuffling methodology, to figure out a Bruun inverse, of which none existed, and to determine whether Bruun's approach could be more optimal on modern CPUs than Cooley-Tukey, annotated "not likely." The repository accumulated the 1978 paper, the Chen and Sorensen 1992 paper, the 2023 NTRU Prime work, and survey notes. Repeated implementation attempts failed. In June 2025 a Codex research-project scaffold was added and the repository went quiet again. The suspicion that structure was present, that Bruun's z-transform framing pointed at a different FFT tradition, was never discharged, so the position was held and the project waited for adequate tooling.


On June 10, 2026, at 3:33 in the morning, the 1978 paper was given to a frontier model with a question framed as an evaluation: if a model could implement this method and rapidly produce its inverse, which had never been published, where would you place its intelligence? In a single round Claude Fable returned a complete, staged, vectorized numpy implementation of the forward Bruun radix-2 RFFT for N = 512 in the requested unrolled layout, and, in the same file, the exact inverse.

The forward was Bruun as described in Section 3: the binomial spine, the half-angle trinomial tree, one precomputed real coefficient table per level in place of a twiddle table, a two-cascade fold with coefficients plus or minus 2 cos(beta/2) as the level transition, all interior arithmetic real, complex numbers confined to writing the 257 bins. Measured against numpy's rfft, it agreed to 7.0e-13 maximum absolute error at N = 512, versus 1.4e-15 for the Cooley-Tukey reference, and the model identified the gap unprompted as the known Bruun pathology: cascade coefficients clustering toward 2, intermediate bases losing orthogonality, two digits lost by N = 512.

The inverse was the new result. The model's stated realization, preserved in its reasoning traces, is that the right inverse is not the transposed flow graph. Each forward level is the ring isomorphism R[z]/(p1 p2) onto R[z]/(p1) x R[z]/(p2), so the exact inverse of a level is Chinese remainder reconstruction, and for Bruun's factor family the Bezout cofactors collapse to closed form. For trinomial children p1 = z^h - b z^(h/2) + 1 and p2 = z^h + b z^(h/2) + 1 with b = 2 cos(beta/2), one has p1 - p2 = -2 b z^(h/2), and modulo p2 the monomial inverse is z^(-h/2) = -(z^(h/2) + b). Therefore

```text
t = ((r2 - r1)(z^(h/2) + b) mod p2) / (2 b)
c = r1 + p1 t
```

reconstructs the parent residue c from the child residues r1, r2. The binomial pair degenerates to t = (r1 - r2)/2. Sweeping leaves to root recovers x exactly, and, in a structural departure from the DFT-matrix picture, no 1/N normalization exists anywhere, because Chinese-remainder interpolation is already normalized; the only divisions are by 4 cos(beta/2), the same constants the forward multiplied by. Measured: round trip to 1.1e-14, inversion of numpy's own rfft output to 1.2e-14, numpy's irfft applied to the Bruun forward to 6.8e-15. The inverse is better conditioned than the forward because reconstruction multiplies by polynomials whose coefficients are bounded by 2, while the worst amplification 1/(2b) enters once per level and the dangerous angles near pi occur at isolated leaves rather than along whole paths. The model also noted that nothing in either direction is specific to Bruun's half-angle schedule: the forward works for any binary tree of real factorizations of z^N - 1, and the CRT inverse works node for node whenever the two children differ by a monomial times a constant, which is a property of the factor family, not the tree shape, so the object to enumerate is factor trees over Z[2 cos] with the monomial-difference property.

With a working transform in hand, One plots the internals. Bruun's per-stage coefficient tables look, in tree order, like spectral dither. Sorted, they collapse. The whole interior coefficient universe of an N-point transform is one master ladder

```text
T[i] = 2 cos(pi i / N)
```

and stage j uses the sparse dyadic sample of that ladder at positions k = 1 .. 2^j - 1 with denominators 2^(j+1); stage j+1 contains every value of stage j at the even positions and inserts the new odd positions between them. The traversal order is itself exactly recursive:

```text
K1     = [1]
K(j+1) = [2^j] then, for each k in Kj, the pair [k, 2^(j+1) - k]
```

so K3 = [4, 2, 6, 1, 7, 3, 5], a reflected dyadic insertion order, related to but distinct from bit reversal and Gray coding. The practical consequence was immediate: a node index m can name, through one integer map IDX[m], the node's fold coefficient, the leaf's output bin, and the local leaf angle, so the transform needs one cosine table and one index map rather than per-stage twiddle storage. The methodological point matters here.The analysis was directed to decompose until the structure was explicit. 

### 4.4 The geometry

The four-star got its name when the question changed from "what is this order" to "why is this order." Take a trinomial node, set u = z^q, so the node is u^2 - 2 cos(theta) u + 1 with the conjugate root pair u = exp(plus or minus i theta) on the unit circle. The Bruun split substitutes u = v^2, the square-root preimage, and the two roots become four:

```text
v =  exp(+i theta/2),  exp(-i theta/2)
v = -exp(+i theta/2), -exp(-i theta/2)
```

Two points on the circle pull back to four points in two antipodal conjugate pairs: a four-pointed star inscribed in the circle. This is the whole algorithm. Bruun's tree is the iterated square-root preimage structure of the covering map z maps to z^2 on the unit circle, and the leaf order is the address system of repeated square-root choices. The factors stay sparse, binomial and trinomial at every level, because preimage fibers of the squaring map are described by exactly such polynomials:

```text
z^(2q) - 2 cos(theta) z^q + 1
    = (z^q - 2 cos(theta/2) z^(q/2) + 1)(z^q + 2 cos(theta/2) z^(q/2) + 1)
```

No dense projector ever appears. The transform is named for this object.

### 4.5 The counterfactual

At this point the natural failure mode is to optimize the thing in hand. Insistently instead we built the strongest available alternative, on the explicit reasoning that if Bruun's order is merely historical, a better CRT tree must exist and a search will find it. The metric chosen first was transport: order the parent root set around the circle and minimize, over all balanced binary real-CRT trees, the circular fragmentation and span of each child, that is, directly penalize large traversals. The search produced a clean, structured answer, named MT-CRT, the transport-minimal real CRT tree, and it is not Bruun. Probing its structure settled the matter. The MT-CRT coordinate system is exactly a row permutation of the ordinary real DFT basis, row-permutation error 0.0 at every tested size, so it is a genuine, orthogonal, transport-minimized spectral order. But its top split, for N = 256, is a time-domain projector of rank 128 with 75.4 percent density and effective bandwidth 255: an ideal band-pass projection, circulant, Dirichlet-kernel-structured, global. Its small split matrices are 100 percent dense and their Toeplitz-Hankel displacement ranks grow with size, 7 at N = 16, 15 at N = 32. The transport optimum in ordinary frequency geometry pays with dense arithmetic everywhere. There is no fast lowering of it that does not reinvent band-split filter banks.

The failed counterfactual forced the correct metric. Bruun is not minimizing transport on the circle as a line; it is minimizing it through the dyadic covering geometry of the circle. The search was rerun with covering-map sparsity as the objective: which CRT splits keep child factors sparse under the z maps to z^2 geometry. The result, for N = 4, 8, 16, 32, is that the optimizer's best trees keep every nontrivial child factor binomial or trinomial, maximum term counts 2, 3, 3, 3, and the top split at N = 32 is forced to exactly the Bruun separation. What survives when the metric is the right one is the four-star family. The resulting landscape:

```text
MT-CRT:            ideal transport, dense arithmetic
Cooley-Tukey:      complex sparse arithmetic, mature transport scheduling
Bruun:             real sparse arithmetic, imperfect transport, repairable
Normalized Bruun:  real sparse arithmetic, condition number 1, repaired transport
```

The methodological element here is the third and most distinctive in a good scientific process: when a hypothesis is favored, the epistemically oriented thinker does not gather supporting evidence; he constructs the strongest rival and attempts to lose. The MT-CRT experiment was run to dethrone Bruun. Its failure, and specifically the manner of its failure, dense global projectors, was treated as information about which geometry is real, and the objective function was corrected rather than the conclusion forced. Bruun was retained not because the search confirmed it but because the corrected search reproduced it from scratch.

### 4.6 The basis

The covering geometry made one remaining question inevitable: if the tree is right because it follows the covering geometry, the local residue coordinate should also follow that geometry. Each leaf quotient R[z]/(z^2 - 2 cos(alpha) z + 1) had until now been represented in the monomial basis, r0 + r1 z. The monomial basis is valid but not natural: 1 and z are not orthogonal coordinates on the chord joining the conjugate roots, and the historical accuracy indictment of Bruun lives precisely in this skew. The natural coordinate is

```text
e = (z - cos(alpha)) / sin(alpha)
```

Inside the quotient, where z^2 = 2 cos(alpha) z - 1, direct computation gives e^2 = -1. Every nontrivial leaf becomes its own local complex plane; the two roots map to e = plus i and minus i. In this basis the internal split and merge operations cease to be Bruun's unnormalized cosine fold a + 2 cos(.) b and become ordinary two-real plane rotations of the form a cos + b sin. A plane rotation is orthogonal; the add-subtract split is orthogonal up to sqrt(2). Conversion of a leaf to a standard complex bin, which in the monomial basis costs cosine and sine multiplications, becomes free:

```text
X[k].re =  r0
X[k].im = -r1
```

and the inverse leaf unpacking becomes an inverse rotation rather than a division by sin(beta). 

### 4.7 The machine

What remained is naturally engineering, conducted as a sequence of measured policy decisions rather than a single large optimization in search of a pragmatically optimal library for public use, and each decision illustrates the selection criteria.

The first implementation in the monomial basis used three real multiplications per trinomial quartet, b P2, b P3, (b^2 - 1) P3, compact, all-real, in-place. It beat a naive complex Cooley-Tukey wrapper and did not beat FFTW. Attempted reductions failed for identifiable reasons: scalar rescaling moves cost rather than removing it, and tangent-FFT tricks do not transfer because in the trinomial quotient multiplication by z is not a free signed shift, it folds through b. The failures were kept and used: they located the monomial basis at a local arithmetic minimum and justified changing basis rather than squeezing constants.

Breadth-first staged execution touches the whole array at every level and is memory-bound at scale; the tree is naturally recursive, so the implementation moved to depth-first traversal with fused tail codelets, the final three levels of each 16-point block fused into a single register-resident codelet, and per-leaf metadata, seven twiddle pairs and eight output indices, packed contiguously into two or three cache lines instead of scattered across separate C, S, and IDX arrays.

The output scatter problem received the same treatment. Bruun-order leaves written directly to standard bins X[k] produce scattered 16-byte stores, each a likely cache miss with read-for-ownership penalties. Two policies were built and measured: fused scatter, no extra pass, wins for small and middle sizes; and a two-phase pack, residues produced in block order then a separate streaming-write conversion pass, scattered reads but sequential writes, which wins at large N because outstanding scattered reads overlap under the prefetcher in a way scattered writes cannot. The shipped library selects by measured threshold: two-phase above N = 8192 on AVX-class builds, above N = 1048576 on SSE2 and NEON builds, fused scatter on scalar builds.

The native order itself was then optimized under a discovered invariant. Depth-first-style reorderings of the output reduced visible index jumps but, when checked, fragmented the factor subtrees; the decisive observation, recorded in the session, is that a legal layout must preserve every CRT factor subtree as a contiguous memory interval or the sparse algebra is lost. The optimization was therefore restricted to sibling orientations inside the heap-contiguous constraint, producing the heap-optimized native spectrum order that is BFFT's fast path. Plotted against standard bins, this order presents as a recursive, warped fourfold pattern, and a final round of analysis proved the pattern is exact: the complete native-position-to-standard-bin route pos to m to k is given by closed-form shell-conditioned XOR and Gray-code bit rules, verified exhaustively at N = 512 (all 255 nontrivial bins) and stated in general form in the library README, so the restore map is table-free in principle and the native layout is fully deterministic.

The SIMD policy was set against prevailing practice. The shipped default is a 128-bit primitive, SSE2 and NEON, two doubles wide, with AVX2/FMA and AVX-512 as dispatched options, on the documented reasoning that wide-vector frequency licensing can make a transform win its microbenchmark while slowing the surrounding application, and that the actual bottlenecks, cache passes, output scatter, table traffic, are not lane-width problems. The benchmark utilities developed print which backend ran, so the numbers reflect consumer hardware rather than best-case configurations. Execution is allocation-free: the plan owns the tables, the caller owns input, output, work, and scratch, in the FFTW manner. The float32 path keeps float buffers and float butterflies but takes its stage twiddles from plan-owned tables rounded once from double libm values, because an earlier multiplicative float twiddle recurrence produced deterministic folded-bin spurs near N/2 - k under seven-term Blackman-Harris probing; the fix was located by spectral evidence.

Finally the work was split across systems by comparative advantage and driven to release by an unattended agent loop: Claude Fable produced the founding transform, the SIMD restructuring rounds, and the two-phase pack; Codex sessions performed the structural searches, the orthogonality proofs, the invariant probes, and the layout analysis; a Codex-driven loop with durable IDEA, ROADMAP, PROGRESS, and TASKS files carried the library through build validation, staged-install smoke tests, API audit, and release readiness in unattended rounds on June 10 and 11, 2026, with the git history as the audit trail: bruun_fft accumulated twenty commits on June 10, and bfft went from initial commit at 20:36 on June 10 to a tested, documented, installable library with float32 internals by 06:14 the following morning.

### 4.8 The criteria

The decision process visible across these fifty hours reduces to seven operating rules. A conviction about latent structure is held open indefinitely and cheaply, in public, until tooling can discharge it; it is not abandoned for being unfashionable and not acted on prematurely. Capability is tested by problems with mechanical verification, and artifacts are accepted on cross-validation against independent references, never on internal consistency or eloquence. Plots are read before code, and apparent disorder is treated as undeciphered structure until decomposition proves otherwise. Favored hypotheses are attacked by constructing their strongest rivals, and a rival's failure mode is mined for the correct objective function rather than discarded. Constraints discovered through failure, subtree contiguity, the non-transferability of tangent tricks, the float twiddle recurrence, are promoted to invariants and optimization proceeds only inside them. Deployment realism outranks benchmark optimization, hence 128-bit defaults, allocation-free execution, measured policy thresholds, and printed backends. And claims are graded: measured results are stated flatly, historical facts are conceded, universal assertions are falsified by counterexample where possible, and anything beyond that is labeled conjecture and given a falsification path. None of these rules is exotic. The yield came from applying all seven simultaneously, at conversational speed, to systems that could keep up.

## 5. The Normalized Four Star Fourier Transform, defined

Four star: the transform's generating object is the four-point preimage star of a conjugate root pair under the squaring map on the unit circle, Section 4.4. Normalized: every quadratic quotient carries the local unit coordinate that makes its residue ring a copy of the complex plane, Section 4.6. What follows is the transform in full.

### 5.1 The factor tree

Let N = 2^L. The FSFFT computes the real-input DFT by reduction of x(z) through the binary tree of real factors of z^N - 1 generated as follows. The root is z^N - 1. The binomial spine splits z^M - 1 into z^(M/2) - 1 and z^(M/2) + 1 at every level; z^M + 1 is the trinomial with angle pi/2; and every trinomial node with parameters (degree D, angle beta), modulus z^D - 2 cos(beta) z^(D/2) + 1, splits into the children (D/2, beta/2) and (D/2, pi - beta/2). A level transition is reduction of each parent residue modulo its two children: for child modulus z^d + A z^(d/2) + C, the reduction of a length-2d coefficient block is the two-cascade fold

```text
u = c[d + h .. 2d - 1]            (h = d/2)
c[d .. d+h-1] -= A u
c[h .. d-1]   -= C u
v = c[d .. d+h-1]
c[h .. d-1]   -= A v
c[0 .. h-1]   -= C v
```

with all coefficients real, A = -+ 2 cos(beta/2) for trinomial children and (A, C) = (0, -+1) on the spine. The leaves are z - 1 and z + 1, carrying bins 0 and N/2, and the quadratics z^2 - 2 cos(b) z + 1, each carrying the conjugate bin pair at angle b = 2 pi k / N. All interior arithmetic is real for real input, exactly as in Bruun.

### 5.2 The normalized leaf basis

The FSFFT departs from Bruun at the leaf representation and, in the optimized kernel, throughout the lower tree. In each quadratic quotient R[z]/(z^2 - 2 cos(alpha) z + 1) define

```text
e = (z - cos(alpha)) / sin(alpha),    so that e^2 = -1
```

and represent every residue as r0 + r1 e. The conjugate roots map to e = +i and e = -i, so the quotient is canonically the complex plane and every leaf is the same local chart. The split and merge operations in this coordinate system are two-real plane rotations, a cos + b sin forms, composed with add-subtract pairs. Standard-bin conversion is the identity up to sign, X[k].re = r0, X[k].im = -r1, with slot zero packing DC and Nyquist. This basis is the entire difference between an algorithm with a forty-eight-year accuracy indictment and one whose exact operator is orthogonal. It is the construction Frigo recognized as something other than Bruun, and it is what the word normalized in the transform's name refers to.

### 5.3 The exact inverse

The inverse is Chinese-remainder reconstruction up the same tree, with closed-form Bezout data as in Section 4.2: for trinomial siblings, t = ((r2 - r1)(z^(h/2) + b) mod p2)/(2b) and parent c = r1 + p1 t; for binomial siblings, t = (r1 - r2)/2; leaf unpacking in the normalized basis is an inverse rotation. No 1/N scaling exists anywhere in the inverse because CRT interpolation carries its own measure. The forward-inverse pair is exact as a linear-algebraic identity, not as an approximation scheme; floating-point error is the only error.

### 5.4 The orthogonality theorem

The accuracy question is settled by the following computationally verified statement. Let A be the exact linear map from N real inputs to the normalized residue vector, and let B be A composed with the standard real packing in which DC and Nyquist are split and each nontrivial residue pair is scaled by sqrt(2). Then

```text
A^T A = (N/2) I        B^T B = N I
```

Equivalently, for all x, the 2-norm of Bx equals sqrt(N) times the 2-norm of x; the singular values of B/sqrt(N) are identically 1; the 2-norm condition number of the transform is 1, the same as the unitary DFT up to global scale. This was verified numerically for N = 4 through 512 with orthogonality residuals at the 1e-16 to 7.5e-15 level, condition numbers 1.000000000000001 or better at every size, and uniform row norms, sqrt(512) for every row at N = 512. The structural reason is the one given above: every stage is a composition of plane rotations and scaled Hadamard splits, each sqrt(2) times an orthogonal map, so the whole transform is sqrt(N) times an orthogonal map. No input direction is compressed or amplified relative to another; there is no hidden bad coordinate in the transform. The consequence for the historical record is stated precisely in Section 7.1.

### 5.5 The address algebra

The native output order is fully deterministic. For N = 2^L, M = N/2, W = L - 1, with native positions pos = 1 .. M-1, every position belongs to the dyadic shell s = floor(log2(pos)), the same shell as its leaf index m, and the complete route pos to m to k is closed-form. The shell bit of m is always set; for s = 0, m = 1; for s = 1, the low bit is 1 XOR bit(pos, 0); for s >= 2 the low bit is 1 XOR bit(pos - 1, 1), the interior bits are the Gray-code bits of pos, gray(pos) = pos XOR (pos >> 1), for bit positions 1 through s - 2, and the bit below the shell anchor is 1 XOR bit(pos, s - 1). The standard bin k then has anchor a = W - 1 - s with k_a = 1; with p0 = bit(pos, 0), k_(a+1) = 1 XOR p0 for s >= 1; the middle triangular field is k_j = p0 XOR bit(pos, W - j) for a + 2 <= j <= W - 3; and the two high bits, where they exist, are k_(W-2) = 1 XOR bit(m - 1, 1) for s >= 3 and k_(W-1) = 1 XOR bit(pos - 1, 1) for s >= 2, all bits below the anchor zero. The law was verified exhaustively at N = 512, matching IDX[NATIVE_LEAF[pos]] for all 255 nontrivial bins. The standard-order interface of the library is therefore a mathematically regular conversion layer over the native fast path, comparable in kind to the bit-reversal layer every Cooley-Tukey implementation carries, and realizable table-free.

### 5.6 The residue domain

Because the tree is the CRT decomposition of R[z]/(z^N - 1), circular convolution decomposes into independent multiplications in the leaf quotients, and in the normalized basis each such multiplication is an ordinary two-real complex multiply (a0 + a1 e)(h0 + h1 e). A fixed filter is converted to residues once; thereafter the pipeline

```text
real input -> forward to residues -> pointwise residue multiply -> inverse -> real output
```

performs FFT convolution, FIR filtering, spectral gain, Wiener filtering, power spectra (mag^2 = r0^2 + r1^2), and deconvolution with precomputed inverse responses, entirely in real arithmetic, in native order, with no complex bin materialization and no permutation pass in the hot path. The complex spectrum, in this light, is one chart on the diagonalized convolution algebra, and for real signals it is an optional chart. The library exposes this as residue-domain transforms and filter helpers, including conversion of an ordinary complex frequency response into residue form.

## 6. Measurements

All claims in this section are measurements from the repository's tracked probes and benchmarks, reproducible from source. The primary platform is an Apple M4, NEON 128-bit backend, double precision unless stated, FFTW with plan-cheap flags as documented in the benchmark source, June 10 and 11, 2026.

### 6.1 Speed

The library benchmark against FFTW (double) and PFFFT (note: PFFFT runs in single precision; BFFT and FFTW in double) reports, with Native_ns the heap-optimized native-order forward, Std_ns the standard-FFT-order forward including conversion, and S/F the standard-to-FFTW ratio:

```text
       N      FFTW_ns    PFFFT_ns   Native_ns      Std_ns     S/F    checks
     512        712.4       558.7       402.3       481.3   0.676   err 3.1e-14 rt 4.4e-16
    1024       1079.2      1179.7       917.5      1256.6   1.164   err 4.1e-14 rt 5.6e-16
    4096       6618.2      6430.8      4747.8      6456.6   0.976   err 2.3e-13 rt 7.8e-16
   16384      38201.2     32084.9     26303.6     36789.8   0.963   err 5.3e-13 rt 8.9e-16
   65536     293935.7    161829.7    114029.5    226329.2   0.770   err 1.8e-12 rt 1.0e-15
  262144    1230474.8    630536.2    482247.8    883182.0   0.718   err 1.7e-11 rt 1.3e-15
 1048576    5347307.3   3054125.0   2370174.5   4150580.7   0.776   err 5.2e-11 rt 1.3e-15
 4194304   32542421.9  13991317.8  10658794.2  16630325.5   0.511   err 2.3e-10 rt 1.7e-15
16777216  141920934.9  68376585.9  49968817.7  76313549.5   0.538   err 4.6e-10 rt 1.9e-15
67108864  649294807.3 293690114.6 208606065.1 378288268.2   0.583   err 2.4e-09 rt 2.0e-15
```

The native path runs at 0.31 to 0.58 of FFTW's time across the large sizes, roughly 1.7x to 3.2x faster, and remains faster than single-precision PFFFT at most sizes despite running in double. The standard-order path, which pays the conversion layer, still beats FFTW at most sizes above 32768. An earlier prototype benchmark on the same machine recorded native ratios as low as 0.285 at N = 524288. The structural summary from the repository README: the algorithmic coordinate system is faster before optimization enters the conversation, the interior carries one real per intermediate value instead of two, about 33 percent lighter on memory traffic in the interior, and the SIMD, fused tails, heap order, and cache scheduling amplify a root advantage that is not merely vector width.

### 6.2 Accuracy and round-trip stability

Forward error against FFTW tracks at the levels shown in the table above, 1e-14 at small sizes to 2.4e-9 at N = 67108864, with round-trip error flat at 4e-16 to 2e-15 across five decades of size. A separate stress protocol of repeated forward-inverse application, reported in correspondence with the FFTW authors, gives after one hundred thousand round trips:

```text
BFFT-std     max 2.665e-15  rms 3.554e-16  rel_rms 3.541e-16  energy 1
BFFT-native  max 2.665e-15  rms 3.554e-16  rel_rms 3.541e-16  energy 1
FFTW         max 1.554e-15  rms 3.514e-16  rel_rms 3.500e-16  energy 0.999
```

Energy is conserved to within measurement at unity, which is a direct consequence of the orthogonality theorem of Section 5.4: a condition-number-1 operator has no mechanism by which to lose or manufacture energy under iteration.

### 6.3 Spectral purity

Seven-term Blackman-Harris coherent-tone probing at N = 4194304, eight eligible bins, sine and cosine waves, main lobe excluded, worst row reported:

```text
f64-native:  BFFT SFDR 197.84799575 dB    FFTW 197.84799573 dB
f32-native:  BFFT SFDR 144.95579274 dB    FFTWf 139.16529588 dB
```

Double-precision spurious-free dynamic range matches FFTW to the eighth decimal of a decibel. The single-precision path exceeds its 144 dB target and the FFTWf reference row by more than 5 dB, after replacement of the multiplicative float twiddle recurrence with once-rounded plan-owned tables.

### 6.4 Invariant support, the filtering safety question

Round-trip correctness alone can mislead: a transform can invert well while presenting a spectrum that is not the canonical Fourier basis, in which case diagonal frequency edits act on the wrong eigenspaces and filtering amplifies error. The invariant-support probe injects pure Fourier modes and measures how much of the standard output escapes the injected bin, combining off-bin leakage with on-bin amplitude and phase error. Across N = 4 to 4194304, including the policy crossover into the two-phase pack at N = 2097152:

```text
       N    policy                              max_support_rel   max_offbin_rel   verdict
     512    fused-scatter-plus-layout-convert    8.68e-14          2.19e-14        roundoff
   65536    fused-scatter-plus-layout-convert    1.13e-11          3.53e-12        roundoff
 1048576    fused-scatter-plus-layout-convert    2.34e-10          1.12e-10        small
 2097152    two-phase-standard-pack              4.49e-10          1.49e-10        small
 4194304    two-phase-standard-pack              7.65e-10          2.05e-10        small
```

Pure Fourier modes remain pure Fourier modes to about 1e-9 relative at four million points, roundoff-scale growth and not a divergent basis. Canonical modes are eigenvectors of the exported transform to machine accuracy; frequency-domain edits against standard bins are therefore not amplified by the internal representation. The transform is safe for filtering in the precise sense the probe was designed to interrogate.

## 7. The classical concerns, answered

The indictments of Bruun's algorithm are individually correct against Bruun's algorithm and individually false against the FSFFT. This section takes them in order. The repairs belong to the normalized transform, which is not Bruun's algorithm.

### 7.1 Accuracy

The historical claim, that Bruun may be intrinsically less accurate than Cooley-Tukey under finite precision, is true in its observation and wrong in its attribution. The inaccuracy belongs to the monomial leaf basis, whose coordinates skew as the half-angle cascade drives coefficients toward 2; it was measured again in this work, two digits lost by N = 512 in the faithful reconstruction. It does not belong to the factor tree. The normalized transform over the same tree satisfies B^T B = N I exactly, has condition number 1, tracks FFTW's error and SFDR, and survives one hundred thousand round trips at machine epsilon with unit energy. The word intrinsic is thereby falsified for the tree: the basis mattered, the tree did not. The claim survives only as a true statement about a particular 1978 coordinate choice, and as Frigo noted, it never applied to finite fields in the first place.

### 7.2 Output order

The historical concern, that Bruun's leaf order is scrambled and standard-order output costs a permutation, is conceded and then resolved from both sides. From one side, the permutation has closed form, the shell-conditioned XOR law of Section 5.5, is realizable as a streaming two-phase pack whose threshold is measured per backend, and costs no more than the bit-reversal and transposition layers the standard tradition has always paid. From the other side, the permutation is frequently unnecessary: the native order is heap-contiguous and transport-optimized under the subtree invariant, and entire filtering pipelines run in residue order with the permutation removed from the hot path entirely, Section 5.6. The scrambled order was never disorder; it is the address system of the covering map.

### 7.3 Twiddle storage and structure

Where a Cooley-Tukey transform carries per-stage complex twiddle combs, the FSFFT's interior coefficient universe is one real cosine ladder T[i] = 2 cos(pi i / N) with every stage a dyadic subsample, plus one integer map; per-leaf metadata packs into two or three cache lines. The plan is small, construction is cheap via half-angle recurrences, and execution is allocation-free.

## 8. Relation to prior art

The FSFFT stands on Bruun 1978, whose factor tree it keeps and whose intent, on the reading endorsed by Frigo, it executes: CRT reconstruction, not polynomial division. It is not Chen and Sorensen 1992, which specializes Bruun to real-symmetric data while retaining the classical leaf evaluation. It is not the Circle FFT of eprint 2024/278, which rediscovers Bruun's butterfly in finite fields with a basis-change factorization F = C T whose profit lives in coset interpolation, although the kinship is real and the finite-field branch is where Bruun's method survived. It is not arXiv 2504.07264, which separates real and imaginary channels of a complex FFT for two-channel post-processing while remaining inside the complex tradition; the FSFFT does not separate channels, it never constructs them. And it is not Bruun: the leaf algebra, the conditioning class, the inverse, the native layout, and the residue calculus are absent from the 1978 paper and from everything downstream of it. 

## 9. On optimality

Consider what the structure leaves an improver to work with. The exact operator is orthogonal up to scale, condition number 1; no reformulation can be better conditioned, only equally conditioned. The interior factors are binomial and trinomial at every level, and the corrected search of Section 4.5 shows these are precisely the factor families that survive when CRT splits are scored in the covering geometry of z maps to z^2, the geometry the transform's own coefficient plots indicated; the transport-optimal alternative in ordinary frequency geometry was constructed, probed, and shown to pay with dense global projectors of growing displacement rank. The interior moves one real per intermediate value where the standard tradition moves two. The leaf chart conversion to standard bins is free. The output permutation is closed-form. The conjecture this evidence supports is that the normalized four star transform is the Pareto point of real CRT Fourier algorithms, the unique tree, up to symmetries, at which sparse arithmetic, perfect conditioning, and repairable transport coexist, and that within the covering metric it is optimal outright. The conjecture is falsifiable: produce a real factor tree of z^N - 1 with sparser splits under the covering metric, or a better-conditioned exact operator, or a real-input transform that moves fewer bytes per point. The benchmark harness in the repository will referee.

## 10. Naming, credit, and dedication

The transform needed a name that was neither a person nor a brand. Bruun's name belongs to the 1978 algorithm, which this is not. The object is the four-point star of square-root preimages, traversed recursively, with each chord of the circle given its normalized local complex coordinate: the Normalized Four Star Fourier Transform. The library keeps the name BFFT in acknowledgment of its ancestor.

## References

Bruun, G., "z-Transform DFT Filters and FFT's," IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. ASSP-26, no. 1, pp. 56-63, February 1978. IEEE document 1660794.

Chen, S. and Sorensen, H., "An efficient FFT for real-symmetric data based on Bruun's algorithm," 1992. (chen1992.pdf, repository archive.)

Cooley, J. W. and Tukey, J. W., "An algorithm for the machine calculation of complex Fourier series," Mathematics of Computation, vol. 19, pp. 297-301, 1965.

Haböck, U. et al., "Circle STARKs," IACR eprint 2024/278.

"Algorithmic Views of Vectorized Polynomial Multipliers, NTRU Prime," 2023. (2023-1580.pdf, repository archive.)

"Fast complex-valued discrete Fourier transform with separate real and imaginary inputs/outputs," arXiv:2504.07264, 2025.

falseywinchnet/bruun_fft, repository and session records, November 2024 through June 2026. falseywinchnet/bfft, library, June 2026. Session transcripts: Model Intelligence in DSP; BFFT Invariant Support Test; Optimizing Bruun FFT with SIMD; June 10-11, 2026.
