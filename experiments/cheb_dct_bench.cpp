// Ballpark throughput for the Chebyshev DCT kernels and the DCT/DST-assembled
// real FFT, against the shipped Bruun real FFT (ROADMAP Phase 5 / Phase 7 probe).
//
// Three measurements:
//   1. Node microbench: the radix-4 all-FMA even/odd node, scalar (no restrict,
//      as the shared composed kernel emits) vs restrict (as ChebDCT now uses).
//      Shows the vectorization win from the one-line memory-aliasing fix.
//   2. Transform throughput: ChebDCT DCT-III (the cosine tree) vs the shipped
//      Bruun real FFT, same N. Tests the capstone claim that a single Chebyshev
//      tree is ~half an FFT.
//   3. Assembled rFFT (DCT-I + DST-I folds) vs the Bruun real FFT. Shows the
//      cost of the endpoint-route assembly relative to the fused kernel.
//
// Build: see Makefile target $(DCT_BENCH).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <bfft/bfft.hpp>

#include "chebyshev_dct_radix4.hpp"
#include "cheb_dct_assemble.hpp"
#include "bruun_dct.hpp"

static double frand(unsigned& s) {
    s = s * 1103515245u + 12345u;
    return (static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
}

#define R __restrict__

// Scalar node (mirrors the shared composed monomial_node_fma: no restrict).
static void node_scalar(double* p, int q, double u, double v) {
    const double u2 = u * u, v2 = v * v;
    double* a0 = p; double* a1 = p + q; double* a2 = p + 2 * q; double* a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        const double eu = A0 + A2 * u2, ou = A1 + A3 * u2;
        const double ev = A0 + A2 * v2, ov = A1 + A3 * v2;
        a0[n] = eu + u * ou; a1[n] = eu - u * ou;
        a2[n] = ev + v * ov; a3[n] = ev - v * ov;
    }
}
// Restrict node (mirrors ChebDCT::forward_node).
static void node_restrict(double* R p, int q, double u, double v) {
    const double u2 = u * u, v2 = v * v;
    double* R a0 = p; double* R a1 = p + q; double* R a2 = p + 2 * q; double* R a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        const double eu = A0 + A2 * u2, ou = A1 + A3 * u2;
        const double ev = A0 + A2 * v2, ov = A1 + A3 * v2;
        a0[n] = eu + u * ou; a1[n] = eu - u * ou;
        a2[n] = ev + v * ov; a3[n] = ev - v * ov;
    }
}

template <typename F>
static double best_ns(F&& once, int iters, int trials) {
    double best = 1e300;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) once();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best = std::min(best, ns / iters);
    }
    return best;
}

static void node_microbench() {
    const int q = 256, blocks = 4;
    const long total = static_cast<long>(blocks) * 4 * q;
    std::vector<double> base(total), buf(total);
    unsigned s = 7u;
    for (long i = 0; i < total; ++i) base[i] = frand(s);
    const double u = 0.97, v = 0.21;
    const int iters = 20000, trials = 7;

    // Reset ONCE per trial (the node is in place); the memcpy is then amortized
    // over `iters` and does not pollute the per-lane timing.
    auto run = [&](auto&& node) {
        double best = 1e300;
        for (int t = 0; t < trials; ++t) {
            std::memcpy(buf.data(), base.data(), sizeof(double) * total);
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
                for (int b = 0; b < blocks; ++b)
                    node(buf.data() + static_cast<long>(b) * 4 * q, q, u, v);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            best = std::min(best, ns / (static_cast<double>(iters) * blocks * q));
        }
        return best;
    };
    const double sc = run(node_scalar);
    const double ve = run(node_restrict);
    std::printf("  node scalar  (no restrict): %.4f ns/lane\n", sc);
    std::printf("  node restrict (vectorized): %.4f ns/lane   %.2fx faster\n", ve, sc / ve);
}

static void transform_bench() {
    std::printf("  %-7s %11s %12s %12s %11s %9s\n",
                "N", "Bruun rFFT", "ChebDCT-III", "BruunDCT-III", "asm rFFT", "Bruun/FFT");
    for (int N : {64, 256, 1024, 4096, 16384}) {
        const int iters = std::max(50, (1 << 22) / N), trials = 5;

        // Bruun real FFT (preallocated buffers, raw API, no per-call alloc).
        bfft::plan fft(static_cast<std::size_t>(N));
        std::vector<double> xin(N);
        unsigned s = 123u + N;
        for (int i = 0; i < N; ++i) xin[i] = frand(s);
        std::vector<bfft::complex> fout(fft.bins()), fscratch(fft.native_scratch_size());
        std::vector<double> fwork(fft.work_size());
        const double ns_fft = best_ns([&]() {
            fft.forward(xin.data(), fout.data(), fwork.data(), fscratch.data());
        }, iters, trials);

        // ChebDCT DCT-III (tower, power-of-T_q, with the input repack).
        cheb_dct_r4::ChebDCT dct;
        dct.init(N);
        std::vector<double> dout(N), dwork(N);
        const double ns_dct = best_ns([&]() {
            dct.dct3_forward(xin.data(), dout.data(), dwork.data());
        }, iters, trials);

        // BruunDCT DCT-III (repack-free, the sparsified Bruun fold).
        bruun_dct::BruunDCT bd;
        bd.init(N);
        std::vector<double> bout(N);
        const double ns_bd = best_ns([&]() {
            bd.dct3_forward(xin.data(), bout.data());
        }, iters, trials);

        // Assembled rFFT from DCT-I/DST-I.
        double ns_asm = 0.0;
        if (cheb_dct_assemble::AssembledRFFT::supported(N)) {
            cheb_dct_assemble::AssembledRFFT asm_(N);
            std::vector<bfft::complex> aout(asm_.bins());
            ns_asm = best_ns([&]() { asm_.forward(xin.data(), aout.data()); }, iters, trials);
        }
        std::printf("  %-7d %9.1f ns %10.1f ns %10.1f ns %9.1f ns %8.2fx\n",
                    N, ns_fft, ns_dct, ns_bd, ns_asm, ns_bd / ns_fft);
    }
    std::printf("  (Bruun/FFT: repack-free DCT-III vs the full complex real-FFT; ~0.5 = \"half an FFT\".)\n");
}

int main() {
    std::printf("=== Node microbench: restrict / vectorization win ===\n");
    node_microbench();
    std::printf("\n=== Transform throughput (best of trials, warm cache) ===\n");
    transform_bench();
    return 0;
}
