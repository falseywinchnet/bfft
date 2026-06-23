// BODFT (odd-frequency DFT) benchmark and validation.
//
// Validates the iterative paired radix-4 BODFT kernel in both double and single
// precision, without any long double (there is no extended precision on the NEON
// target). Correctness is checked three ways:
//
//   1. analytic exact: the BODFT of a unit impulse has |H[k]| == 1 with a
//      closed-form phase;
//   2. independent cross-check against a direct same-precision DFT;
//   3. exact inverse roundtrip.
//
// Build (from repo root):
//   c++ -O3 -std=c++17 -Iinclude examples/bodft_benchmark.cpp -lm -o build/examples/bodft_benchmark

#include "../src/detail/bodft_kernel.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

template <class RT>
std::vector<RT> make_input(int n, std::mt19937_64& rng) {
    std::vector<RT> x(static_cast<std::size_t>(n));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = static_cast<RT>(dist(rng));
    return x;
}

template <class Plan, class CT, class RT>
void run_correctness(int N, std::mt19937_64& rng, const char* tag) {
    Plan p(N);
    const int bins = N / 2;

    // (1) BODFT of a unit impulse at n0: H[k] = exp(-2*pi*i*(k+1/2)*n0/N).
    std::vector<RT> imp(static_cast<std::size_t>(N), static_cast<RT>(0));
    const int n0 = 1;
    imp[static_cast<std::size_t>(n0)] = static_cast<RT>(1);
    std::vector<CT> Himp(static_cast<std::size_t>(bins));
    p.forward(imp.data(), Himp.data());
    double imp_err = 0.0;
    for (int k = 0; k < bins; ++k) {
        const double ang = -2.0 * PI * (k + 0.5) * n0 / N;
        const double dr = static_cast<double>(Himp[static_cast<std::size_t>(k)].re) - std::cos(ang);
        const double di = static_cast<double>(Himp[static_cast<std::size_t>(k)].im) - std::sin(ang);
        imp_err = std::max(imp_err, std::sqrt(dr * dr + di * di));
    }

    // (2) random signal vs direct same-precision DFT, and (3) roundtrip.
    std::vector<RT> x = make_input<RT>(N, rng);
    std::vector<CT> H(static_cast<std::size_t>(bins));
    p.forward(x.data(), H.data());
    double xcheck = 0.0, ref_mag = 0.0;
    for (int k = 0; k < bins; ++k) {
        double sr = 0.0, si = 0.0;
        for (int n = 0; n < N; ++n) {
            const double ang = -2.0 * PI * (k + 0.5) * n / N;
            sr += static_cast<double>(x[static_cast<std::size_t>(n)]) * std::cos(ang);
            si += static_cast<double>(x[static_cast<std::size_t>(n)]) * std::sin(ang);
        }
        const double dr = static_cast<double>(H[static_cast<std::size_t>(k)].re) - sr;
        const double di = static_cast<double>(H[static_cast<std::size_t>(k)].im) - si;
        xcheck = std::max(xcheck, std::sqrt(dr * dr + di * di));
        ref_mag = std::max(ref_mag, std::sqrt(sr * sr + si * si));
    }

    std::vector<RT> xr(static_cast<std::size_t>(N));
    p.inverse(H.data(), xr.data());
    double round_err = 0.0;
    for (int n = 0; n < N; ++n) {
        round_err = std::max(round_err,
                             std::fabs(static_cast<double>(xr[static_cast<std::size_t>(n)] -
                                                           x[static_cast<std::size_t>(n)])));
    }

    std::printf("%-6s N %7d  impulse %.3e  vs_dft(rel) %.3e  roundtrip %.3e\n",
                tag, N, imp_err, ref_mag > 0 ? xcheck / ref_mag : 0.0, round_err);
}

template <class Plan, class CT, class RT>
void run_timing(int N, std::mt19937_64& rng, const char* tag) {
    std::vector<RT> x = make_input<RT>(N, rng);
    Plan p(N);
    std::vector<CT> H(static_cast<std::size_t>(N / 2));
    std::vector<RT> xr(static_cast<std::size_t>(N));

    long iters = 1;
    {
        const double per = static_cast<double>(N) * std::log2(static_cast<double>(N));
        iters = static_cast<long>(2.0e8 / per);
        if (iters < 1) iters = 1;
    }

    using clock = std::chrono::steady_clock;
    p.forward(x.data(), H.data());
    p.inverse(H.data(), xr.data());

    auto t0 = clock::now();
    for (long i = 0; i < iters; ++i) p.forward(x.data(), H.data());
    auto t1 = clock::now();
    for (long i = 0; i < iters; ++i) p.inverse(H.data(), xr.data());
    auto t2 = clock::now();

    const double fwd_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    const double inv_ns = std::chrono::duration<double, std::nano>(t2 - t1).count() / iters;
    std::printf("%-6s N %7d  forward %10.1f ns  inverse %10.1f ns  (%ld iters)\n",
                tag, N, fwd_ns, inv_ns, iters);
}

} // namespace

int main(int argc, char** argv) {
    std::mt19937_64 rng(12345);

    std::printf("backend: %s\n", bruun::simd_backend_name());

    std::printf("\n== correctness (double) ==\n");
    for (int N = 2; N <= 8192; N <<= 1)
        run_correctness<bodft::plan, bodft::complex_t, double>(N, rng, "f64");

    std::printf("\n== correctness (float) ==\n");
    for (int N = 2; N <= 8192; N <<= 1)
        run_correctness<bodft::plan_f32, bodft::complex_f32_t, float>(N, rng, "f32");

    std::printf("\n== timing (double) ==\n");
    for (int N = 64; N <= (1 << 20); N <<= 1)
        run_timing<bodft::plan, bodft::complex_t, double>(N, rng, "f64");

    std::printf("\n== timing (float) ==\n");
    for (int N = 64; N <= (1 << 20); N <<= 1)
        run_timing<bodft::plan_f32, bodft::complex_f32_t, float>(N, rng, "f32");

    (void)argc;
    (void)argv;
    return 0;
}
