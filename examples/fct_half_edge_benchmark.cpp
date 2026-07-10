#include "../src/detail/fct_half_edge_simd.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr std::size_t lanes = 4096;
    constexpr int repeats = 4000;
    std::mt19937_64 rng(0x53494d444449534bULL);
    std::uniform_real_distribution<double> dist(-8.0, 8.0);
    std::uniform_real_distribution<double> score(0.1, 30.0);
    std::vector<double> br(lanes), bi(lanes), vr(lanes), vi(lanes), s(lanes);
    std::vector<std::uint8_t> mask(lanes);
    std::vector<std::uint8_t> intrinsic_mask(lanes);
    std::vector<std::uint8_t> scalar_mask(lanes);
    for (std::size_t i = 0; i < lanes; ++i) {
        br[i] = dist(rng);
        bi[i] = dist(rng);
        vr[i] = dist(rng);
        vi[i] = dist(rng);
        s[i] = score(rng);
    }

    std::uint64_t scalar_sum = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        for (std::size_t i = 0; i < lanes; ++i)
            scalar_mask[i] = static_cast<std::uint8_t>(
                fct::half_edge::prunable_scalar(
                br[i], bi[i], vr[i], vi[i], s[i], 64.0, 64, 127));
        for (std::uint8_t v : scalar_mask) scalar_sum += v;
    }
    const auto t1 = std::chrono::steady_clock::now();

    std::uint64_t simd_sum = 0;
    const auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        fct::half_edge::prune_same_node(
            br.data(), bi.data(), vr.data(), vi.data(), s.data(), lanes,
            64.0, 64, 127, mask.data());
        for (std::uint8_t v : mask) simd_sum += v;
    }
    const auto t3 = std::chrono::steady_clock::now();

    std::uint64_t intrinsic_sum = 0;
    const auto t4 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        fct::half_edge::prune_same_node_intrinsic(
            br.data(), bi.data(), vr.data(), vi.data(), s.data(), lanes,
            64.0, 64, 127, intrinsic_mask.data());
        for (std::uint8_t v : intrinsic_mask) intrinsic_sum += v;
    }
    const auto t5 = std::chrono::steady_clock::now();

    const double count = static_cast<double>(lanes) * repeats;
    const double scalar_ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / count;
    const double simd_ns =
        std::chrono::duration<double, std::nano>(t3 - t2).count() / count;
    const double intrinsic_ns =
        std::chrono::duration<double, std::nano>(t5 - t4).count() / count;
    std::cout << "half-edge backend=" << bruun::simd_backend_name() << '\n'
              << std::fixed << std::setprecision(3)
              << "scalar " << scalar_ns << " ns/lane, SIMD " << simd_ns
              << " ns/lane, speedup " << scalar_ns / simd_ns << "x\n"
              << "intrinsic " << intrinsic_ns << " ns/lane\n"
              << "checksum " << scalar_sum << '/' << simd_sum << '/'
              << intrinsic_sum << '\n';
    return scalar_sum == simd_sum && simd_sum == intrinsic_sum ? 0 : 1;
}
