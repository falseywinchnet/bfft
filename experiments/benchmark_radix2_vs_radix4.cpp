// Radix-2 vs radix-4 Bruun forward-residue benchmark.
//
// This is a stripped-down benchmark that compares ONLY the two forward-residue
// paths exposed by the experimental Bruun radix-4 kernel:
//
//   radix-2  : RFFT_Radix4::forward_residues()        breadth-first,
//              one norm_q (radix-2 normalized split) per tree node.
//   radix-4  : RFFT_Radix4::forward_residues_radix4()  depth-first,
//              fused 2-level radix-4 nodes (norm2_fused) where possible.
//
// Both produce the identical N-element residue vector, so the benchmark also
// reports the max-abs agreement between the two paths and a round-trip error
// against BFFT's inverse_residues as a correctness anchor.
//
// Build (after `make` has produced build/libbfft.a):
//   c++ -O2 -std=c++17 -Iinclude \
//       experiments/benchmark_radix2_vs_radix4.cpp build/libbfft.a -lm \
//       -o build/experiments/benchmark_radix2_vs_radix4
//
// Run:
//   ./build/experiments/benchmark_radix2_vs_radix4            # size sweep
//   ./build/experiments/benchmark_radix2_vs_radix4 N [iters]  # single size

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "bfft/bfft.hpp"
#include "bruun_radix4_kernel.hpp"

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

bool is_pow2(std::size_t n) { return n > 0 && ((n & (n - 1)) == 0); }

int ilog2_pow2(std::size_t n) {
    int l = 0;
    while (n > 1) { n >>= 1; ++l; }
    return l;
}

// Match benchmark.cpp's adaptive iteration budget so timings are comparable.
int default_iters(std::size_t n) {
    const long long target = 180000000LL;
    const int l = std::max(1, ilog2_pow2(n));
    long long it = target / (static_cast<long long>(n) * l);
    if (it < 16) it = 16;
    if (it > 200000) it = 200000;
    return static_cast<int>(it);
}

std::vector<double> make_input(std::size_t n) {
    std::vector<double> input(n);
    unsigned seed = 1234567u + static_cast<unsigned>(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        seed = seed * 1103515245u + 12345u;
        const double noise =
            (static_cast<double>(seed & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
        input[i] = 0.70 * std::sin(2.0 * pi * 3.0 * t)
                 + 0.20 * std::cos(2.0 * pi * 11.0 * t)
                 + 0.07 * std::sin(2.0 * pi * 29.0 * t)
                 + 0.03 * noise;
    }
    return input;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}

template <class Fn>
double bench_ns(int iters, Fn&& fn) {
    for (int i = 0; i < 3; ++i) fn();  // warm up
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count()
         / static_cast<double>(iters);
}

void bench_one(std::size_t n, int forced_iters) {
    const int iters = forced_iters > 0 ? forced_iters : default_iters(n);

    bfft::plan oracle(static_cast<int>(n));

    bruun_radix4::RFFT_Radix4 r4;
    if (!r4.init(static_cast<int>(n))) {
        std::printf("%9zu  RFFT_Radix4::init failed (needs power-of-two N >= 16)\n", n);
        return;
    }

    std::vector<double> input = make_input(n);
    std::vector<double> ref(n);     // BFFT oracle residues (correctness anchor)
    std::vector<double> res2(n);    // radix-2 path output
    std::vector<double> res4(n);    // radix-4 path output

    oracle.forward_residues(input.data(), ref.data());
    r4.forward_residues(input.data(), res2.data());
    r4.forward_residues_radix4(input.data(), res4.data());

    // Correctness: both paths vs each other and vs the BFFT oracle.
    const double err_2_vs_oracle = max_abs_diff(res2, ref);
    const double err_4_vs_oracle = max_abs_diff(res4, ref);
    const double err_4_vs_2 = max_abs_diff(res4, res2);

    // Round-trip anchor: radix-4 forward then BFFT (unit-scaled) inverse.
    std::vector<double> rt = res4;
    oracle.inverse_residues(rt.data());
    const double rt_err = max_abs_diff(rt, input);

    // Volatile sink so the optimizer cannot elide the transforms.
    double sink = 0.0;
    double t2 = bench_ns(iters, [&] {
        r4.forward_residues(input.data(), res2.data());
        sink += res2[(static_cast<std::size_t>(sink) * 17) % n];
    });
    double t4 = bench_ns(iters, [&] {
        r4.forward_residues_radix4(input.data(), res4.data());
        sink += res4[(static_cast<std::size_t>(sink) * 17) % n];
    });

    const double ratio = (t2 > 0.0) ? t4 / t2 : NAN;       // <1.0 => radix-4 faster
    const double speedup = (t4 > 0.0) ? t2 / t4 : NAN;     // >1.0 => radix-4 faster
    const bool engaged = n >= 64;  // forward_residues_radix4 falls back below 64

    std::printf("%9zu %8d %12.1f %12.1f %8.3f %8.3f  %3s  "
                "r4/orc %.1e r2/orc %.1e r4/r2 %.1e rt %.1e  sink %.3g\n",
                n, iters, t2, t4, ratio, speedup,
                engaged ? "r4" : "r2*",
                err_4_vs_oracle, err_2_vs_oracle, err_4_vs_2, rt_err, sink);
}

std::size_t parse_size(const char* s) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || v == 0) throw std::runtime_error("bad size");
    return static_cast<std::size_t>(v);
}

int parse_iters(const char* s) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 0) throw std::runtime_error("bad iters");
    return static_cast<int>(v);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t forced_n = 0;
        int forced_iters = 0;
        if (argc >= 2) forced_n = parse_size(argv[1]);
        if (argc >= 3) forced_iters = parse_iters(argv[2]);

        std::printf("BFFT radix-2 vs radix-4 forward-residue benchmark, version: %s\n",
                    bfft::version_string().c_str());
        std::printf("radix-2 = breadth-first norm_q; radix-4 = depth-first fused norm2.\n");
        std::printf("ratio = r4_ns/r2_ns (<1 means radix-4 wins); speedup = r2_ns/r4_ns.\n");
        std::printf("path 'r4' = fused radix-4 engaged; 'r2*' = N<64 fallback to radix-2.\n\n");
        std::printf("%9s %8s %12s %12s %8s %8s  %3s  %s\n",
                    "N", "iters", "radix2_ns", "radix4_ns", "ratio", "speedup",
                    "pth", "checks (max-abs errors)");

        if (forced_n > 0) {
            if (!is_pow2(forced_n) || forced_n < 16) {
                std::fprintf(stderr, "N must be a power of two >= 16.\n");
                return 2;
            }
            bench_one(forced_n, forced_iters);
            return 0;
        }

        const std::size_t sizes[] = {
            16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304
        };
        for (std::size_t n : sizes) bench_one(n, forced_iters);
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "benchmark failed: %s\n", exc.what());
        return 1;
    }
}
