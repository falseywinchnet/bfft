# Recovered phase-walk cube discussions

2026-09-04. Recovered from the owner's ChatGPT history, read alongside the current BFFT implementation. ChatGPT responses are research records, not independent proofs or novelty certificates.

## Conversation map

- [DIT vs DIF Performance](https://chatgpt.com/c/6a4848c4-daf8-83e8-95cf-299a359d9332): July 3 diagonal phase-cell exploration; N=16 parity partitions, N=64 diagonal subgroup packets, torus/cyclic distinction, and four 16-point carry components.
- [DIP Core Rewrite](https://chatgpt.com/c/6a499acb-1cd4-83e8-b6ff-79bd2f8f8d98): July 6 cube/walk account; distribution versus collection, boundary ordering, retained packets, and downstream physical/spectral exploration.
- [FFT ODFT Relation Explained](https://chatgpt.com/c/6a3a642f-13a4-83ea-b97b-644a5093687b): June 23 half-bin and half-hop interstitial evidence, constrained spectrogram fusion, and failed/successful display experiments. This discussion already refers to a native real-only ODFT implementation.
- [ODFT IODFT Benchmark](https://chatgpt.com/c/6a3e5437-1ff4-83ea-8a4c-b47c87269a8b): June 26 accuracy, timing and windowed-tone tests; includes owner-supplied small-size timing results.
- [ODFT Hann Window Issue](https://chatgpt.com/c/6a3d275b-ed70-83ea-a794-00c9c14dee8a): anti-periodic wrap/sign convention.
- [BODFT Output and Order](https://chatgpt.com/c/6a3ea796-5f10-83ea-a37b-66561e4d3c08): natural half-bin output order, without DC/Nyquist special endpoints.

These dates establish that the located ODFT discussions precede the located July cube discussion. They do not recover the earliest cube exploration or establish the full causal discovery chronology.

## Independent N=64 recheck

Run on M4 system Python/NumPy, using explicitly constructed normalized 64-by-64 matrices rather than copying numerical claims from the old chat.

Represent n=t+8f and k=u+8v. Then

    nk/64 = (tv+fu)/8 + tu/64 + fv.

Let F be the ordinary cyclic DFT64 and G the cross-paired Fourier transform on Z8 × Z8. Their kernels satisfy

    F[k,n] = G[(u,v),(t,f)] exp(-2πi tu/64).

This is an **entrywise kernel identity**, not an assertion that one input-only or output-only diagonal multiplication converts G into F.

Use the eight-element subgroup

    H = {(2j, 2j+4l) mod 8 : j=0,...,3; l=0,1}.

Its eight cosets partition the 64 coordinates. In that order:

| Quantity | Independently measured result |
|---|---:|
| Kernel phase-split maximum error | 5.1231e-15 |
| Every torus H-coset block rank | 1 |
| Every cyclic H-coset block rank | 4 |
| Nonzeros in the locally character-transformed torus matrix | 64 |
| Nonzeros in the residual cyclic carry operator | 1024 |
| Carry connected component sizes | 16, 16, 16, 16 |
| Carry unitarity maximum entry error | 3.8822e-15 |

Ranks/nonzeros use threshold 1e-10. The local character matrix is F4 tensor F2 on the (j,l) subgroup coordinates. If Q applies those transforms after coset ordering, define Gp=QGQ*, Fp=QFQ*, and carry=Fp Gp*. Connectivity is taken over the union of row and column support.

These tests reproduce the principal finite-size support result. They do not prove a new asymptotic factorization, low arithmetic cost, numerical stability for arbitrary sizes, or a speed advantage. The old chat's operator-Schmidt-rank classification was not rechecked here.

## ODFT distinction

The operator H[k]=sum_n x[n] exp(-2πi(k+1/2)n/N) is established prior art. See Martucci, *Symmetric convolution and the discrete sine and cosine transforms* (1994), which explicitly identifies the odd-frequency DFT:
https://www.ee.columbia.edu/~marios/symmetry/papers/martucci94symmetric.pdf

The repository's implementation-specific object is a native paired radix-4 real-input half-bin kernel, including its conjugate partner sharing, inverse, SIMD schedule, and output layout. See `src/detail/bodft_kernel.hpp` and `paper/on_bruun_revisited/sections/09_odft.tex`. Novelty of that particular algorithm requires comparison to earlier specialized odd-frequency FFT algorithms, not merely the existence of the half-bin operator.

The useful closure identity for radix r, n=rq+s and N=rM, is

    H_N[k] = sum_s exp(-2πi(k+1/2)s/N) H_M^(s)[k mod M].

Thus the half-bin shift survives interleaved radix decimation. This differs from recursively splitting a frame into contiguous halves, where parent half-bin outputs introduce quarter-bin and three-quarter-bin child channels. The latter obstruction is already recorded in `notes/thle_transform.md` and `src/detail/fct_kernel.hpp`.

## Current assessment

The recovered work contains specific mathematical structure worth developing. Its strongest current themes are normalized real local coordinates, exact retained intermediate states, phase placement freedom, a native half-bin kernel, and constructive physical representations. A claim of a new universally superior FFT or a wholly new odd-frequency transform is not established by these records.

The small-size ODFT timings in the June chat compare a different output operator with ordinary RFFT, include Python/copy overhead, and use favorable on-grid tones for the windowed spectral test. They are useful historical measurements, not a matched universal speed or spectral-resolution claim. FFT plus half-bin samples of the same windowed record equals interleaved samples of its zero-padded 2N-point spectrum; it does not add independent measurements to that record.
