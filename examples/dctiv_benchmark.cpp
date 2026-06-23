// DCT-IV / half-shift (BDCT) benchmark and validation.
//
// Validates the paired radix-4 half-shift kernel and the DCT-IV built on it,
// in both double and single precision, without any long double (there is no
// extended precision on the NEON target). Correctness is checked three ways:
//
//   1. analytic exact: half-shift of a unit impulse has |H[k]| == 1 with a
//      closed-form phase, and DCT-IV of a single basis vector is a clean spike;
//   2. independent cross-check against a direct same-precision DFT;
//   3. exact inverse roundtrip.
//
// Build (from repo root):
//   c++ -O3 -std=c++17 -Iinclude examples/dctiv_benchmark.cpp -lm -o build/examples/dctiv_benchmark

#include "../src/detail/bdct_kernel.hpp"

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

// --- correctness for one precision -----------------------------------------

template <class HS, class DCT, class CT, class RT>
void run_correctness(int N, std::mt19937_64& rng, const char* tag) {
    HS hs(N);
    DCT dct(N);
    const int bins = N / 2;

    // (1a) half-shift of a unit impulse at n0: H[k] = exp(-2*pi*i*(k+1/2)*n0/N).
    std::vector<RT> imp(static_cast<std::size_t>(N), static_cast<RT>(0));
    const int n0 = 1;
    imp[static_cast<std::size_t>(n0)] = static_cast<RT>(1);
    std::vector<CT> Himp(static_cast<std::size_t>(bins));
    hs.forward(imp.data(), Himp.data());
    double imp_err = 0.0;
    for (int k = 0; k < bins; ++k) {
        const double ang = -2.0 * PI * (static_cast<double>(k) + 0.5) *
                           static_cast<double>(n0) / static_cast<double>(N);
        const double dr = static_cast<double>(Himp[static_cast<std::size_t>(k)].re) - std::cos(ang);
        const double di = static_cast<double>(Himp[static_cast<std::size_t>(k)].im) - std::sin(ang);
        imp_err = std::max(imp_err, std::sqrt(dr * dr + di * di));
    }

    // (1b) DCT-IV of basis vector m: x[n]=cos(pi*(m+1/2)*(n+1/2)/N) -> (N/2)*delta_km.
    const int m = (N >= 8) ? 3 : 1;
    std::vector<RT> basis(static_cast<std::size_t>(N));
    for (int n = 0; n < N; ++n) {
        basis[static_cast<std::size_t>(n)] = static_cast<RT>(
            std::cos(PI * (m + 0.5) * (n + 0.5) / N));
    }
    std::vector<RT> c(static_cast<std::size_t>(N));
    dct.forward(basis.data(), c.data());
    double dct_err = 0.0;
    for (int k = 0; k < N; ++k) {
        const double expect = (k == m) ? 0.5 * N : 0.0;
        dct_err = std::max(dct_err, std::fabs(static_cast<double>(c[static_cast<std::size_t>(k)]) - expect));
    }

    // (2) random signal vs direct same-precision DFT, and (3) roundtrip.
    std::vector<RT> x = make_input<RT>(N, rng);
    std::vector<CT> H(static_cast<std::size_t>(bins));
    hs.forward(x.data(), H.data());
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
    hs.inverse(H.data(), xr.data());
    double round_err = 0.0;
    for (int n = 0; n < N; ++n) {
        round_err = std::max(round_err,
                             std::fabs(static_cast<double>(xr[static_cast<std::size_t>(n)] -
                                                           x[static_cast<std::size_t>(n)])));
    }

    std::printf("%-6s N %7d  hs_impulse %.3e  dctiv_basis %.3e  "
                "hs_vs_dft(rel) %.3e  roundtrip %.3e\n",
                tag, N, imp_err, dct_err,
                ref_mag > 0 ? xcheck / ref_mag : 0.0, round_err);
}

// --- timing ----------------------------------------------------------------

template <class HS, class DCT, class CT, class RT>
void run_timing(int N, std::mt19937_64& rng, const char* tag) {
    std::vector<RT> x = make_input<RT>(N, rng);
    HS hs(N);
    DCT dct(N);
    std::vector<CT> H(static_cast<std::size_t>(N / 2));
    std::vector<RT> c(static_cast<std::size_t>(N));

    long iters = 1;
    {
        const double per = static_cast<double>(N) * std::log2(static_cast<double>(N));
        iters = static_cast<long>(2.0e8 / per);
        if (iters < 1) iters = 1;
    }

    using clock = std::chrono::steady_clock;
    hs.forward(x.data(), H.data());
    dct.forward(x.data(), c.data());

    auto t0 = clock::now();
    for (long i = 0; i < iters; ++i) hs.forward(x.data(), H.data());
    auto t1 = clock::now();
    for (long i = 0; i < iters; ++i) dct.forward(x.data(), c.data());
    auto t2 = clock::now();

    const double hs_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    const double dct_ns = std::chrono::duration<double, std::nano>(t2 - t1).count() / iters;
    std::printf("%-6s N %7d  halfshift %10.1f ns  dctiv %10.1f ns  (%ld iters)\n",
                tag, N, hs_ns, dct_ns, iters);
}

} // namespace

int main(int argc, char** argv) {
    std::mt19937_64 rng(12345);

    std::printf("backend: %s\n", bruun::simd_backend_name());

    std::printf("\n== correctness (double) ==\n");
    for (int N = 4; N <= 8192; N <<= 1)
        run_correctness<bdct::half_shift_plan, bdct::dctiv_plan, bdct::complex_t, double>(N, rng, "f64");

    std::printf("\n== correctness (float) ==\n");
    for (int N = 4; N <= 8192; N <<= 1)
        run_correctness<bdct::half_shift_plan_f32, bdct::dctiv_plan_f32, bdct::complex_f32_t, float>(N, rng, "f32");

    std::printf("\n== timing (double) ==\n");
    for (int N = 64; N <= (1 << 20); N <<= 1)
        run_timing<bdct::half_shift_plan, bdct::dctiv_plan, bdct::complex_t, double>(N, rng, "f64");

    std::printf("\n== timing (float) ==\n");
    for (int N = 64; N <= (1 << 20); N <<= 1)
        run_timing<bdct::half_shift_plan_f32, bdct::dctiv_plan_f32, bdct::complex_f32_t, float>(N, rng, "f32");

    (void)argc;
    (void)argv;
    return 0;
}
