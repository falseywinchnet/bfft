// Smoke test and local timing harness for GenBruun non-power-of-two SIMD
// primitives.
//
// Build:
//   g++ -O3 -std=c++17 -Isrc benchmarks/genbruun_nonpower2_simd_benchmark.cpp -lm -o genbruun_nonpower2_simd_bench
// Optional AVX2/FMA:
//   g++ -O3 -std=c++17 -mavx2 -mfma -Isrc benchmarks/genbruun_nonpower2_simd_benchmark.cpp -lm -o genbruun_nonpower2_simd_bench

#include "detail/genbruun_nonpower2_simd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out = std::max(out, std::abs(a[i] - b[i]));
    }
    return out;
}

bool require_close(const char* label, const std::vector<double>& a, const std::vector<double>& b) {
    const double err = max_abs_diff(a, b);
    if (err >= 1e-12) {
        std::fprintf(stderr, "%s mismatch %.3e\n", label, err);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const int n = 19;
    std::vector<double> a0(n), b0(n), a1(n), b1(n), c(n), s(n);
    std::vector<double> ref_a0(n), ref_b0(n), ref_a1(n), ref_b1(n);
    std::vector<double> orig_a0(n), orig_b0(n), orig_a1(n), orig_b1(n);
    for (int i = 0; i < n; ++i) {
        a0[i] = 0.25 + 0.07 * i;
        b0[i] = -0.5 + 0.03 * i;
        a1[i] = 0.75 - 0.02 * i;
        b1[i] = -0.125 + 0.05 * i;
        const double angle = 0.11 * (i + 1);
        c[i] = std::cos(angle);
        s[i] = std::sin(angle);
    }
    ref_a0 = a0;
    ref_b0 = b0;
    ref_a1 = a1;
    ref_b1 = b1;
    orig_a0 = a0;
    orig_b0 = b0;
    orig_a1 = a1;
    orig_b1 = b1;

    bruun_nonpower2::QuadraticBlock block{a0.data(), b0.data(), a1.data(), b1.data(), c.data(), s.data(), n};
    bruun_nonpower2::quadratic_fwd_variable(block);

    for (int i = 0; i < n; ++i) {
        const double old_a0 = 0.25 + 0.07 * i;
        const double old_a1 = 0.75 - 0.02 * i;
        const double old_b0 = -0.5 + 0.03 * i;
        const double old_b1 = -0.125 + 0.05 * i;
        const double R = c[i] * old_b0 - s[i] * old_b1;
        const double I = s[i] * old_b0 + c[i] * old_b1;
        ref_a0[i] = old_a0 + R;
        ref_b0[i] = old_a1 + I;
        ref_a1[i] = old_a0 - R;
        ref_b1[i] = I - old_a1;
    }

    if (!require_close("forward a0", a0, ref_a0)) return 1;
    if (!require_close("forward b0", b0, ref_b0)) return 1;
    if (!require_close("forward a1", a1, ref_a1)) return 1;
    if (!require_close("forward b1", b1, ref_b1)) return 1;

    bruun_nonpower2::quadratic_inv_variable(block);
    if (!require_close("inverse a0", a0, orig_a0)) return 1;
    if (!require_close("inverse b0", b0, orig_b0)) return 1;
    if (!require_close("inverse a1", a1, orig_a1)) return 1;
    if (!require_close("inverse b1", b1, orig_b1)) return 1;

    const int block_count = 384;
    std::vector<int> lengths(block_count);
    int total_lanes = 0;
    for (int block_index = 0; block_index < block_count; ++block_index) {
        const int lane_mod = (block_index * 17 + 5) % 29;
        lengths[block_index] = 5 + lane_mod;
        total_lanes += lengths[block_index];
    }

    std::vector<double> bench_a0(total_lanes), bench_b0(total_lanes);
    std::vector<double> bench_a1(total_lanes), bench_b1(total_lanes);
    std::vector<double> bench_c(total_lanes), bench_s(total_lanes);
    for (int i = 0; i < total_lanes; ++i) {
        bench_a0[i] = 0.001 * (i % 97);
        bench_b0[i] = -0.002 * (i % 89);
        bench_a1[i] = 0.003 * (i % 83);
        bench_b1[i] = -0.004 * (i % 79);
        const double theta = 0.0007 * (i + 3);
        bench_c[i] = std::cos(theta);
        bench_s[i] = std::sin(theta);
    }

    bruun_nonpower2::QuadraticScheduler scheduler;
    int offset = 0;
    for (int length : lengths) {
        scheduler.add_block(offset, offset, length);
        offset += length;
    }
    scheduler.sort_for_simd();
    if (scheduler.block_count() != static_cast<std::size_t>(block_count)) return 1;
    if (scheduler.lane_count() != static_cast<std::size_t>(total_lanes)) return 1;

    std::vector<double> sched_orig_a0 = bench_a0;
    std::vector<double> sched_orig_b0 = bench_b0;
    std::vector<double> sched_orig_a1 = bench_a1;
    std::vector<double> sched_orig_b1 = bench_b1;
    scheduler.execute(bench_a0.data(), bench_b0.data(), bench_a1.data(), bench_b1.data(),
                      bench_c.data(), bench_s.data());
    scheduler.execute_inverse(bench_a0.data(), bench_b0.data(), bench_a1.data(), bench_b1.data(),
                              bench_c.data(), bench_s.data());
    if (!require_close("scheduler a0", bench_a0, sched_orig_a0)) return 1;
    if (!require_close("scheduler b0", bench_b0, sched_orig_b0)) return 1;
    if (!require_close("scheduler a1", bench_a1, sched_orig_a1)) return 1;
    if (!require_close("scheduler b1", bench_b1, sched_orig_b1)) return 1;

    const int repeats = 1200;
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        scheduler.execute(bench_a0.data(), bench_b0.data(), bench_a1.data(), bench_b1.data(),
                          bench_c.data(), bench_s.data());
    }
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    const double lane_visits = static_cast<double>(total_lanes) * static_cast<double>(repeats);
    const double ns_per_lane = elapsed.count() * 1000000000.0 / lane_visits;

    std::printf("genbruun_nonpower2_simd backend=%s ok blocks=%zu lanes=%zu ns_per_lane=%.3f\n",
                bruun_nonpower2::simd_backend_name(), scheduler.block_count(),
                scheduler.lane_count(), ns_per_lane);
    return 0;
}
