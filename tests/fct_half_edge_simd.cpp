#include "../src/detail/fct_half_edge_simd.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::mt19937_64 rng(0x48414c4645444745ULL);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (int length : {1, 2, 4, 8, 16, 64, 256}) {
        constexpr std::size_t lanes = 37; // exercises SIMD plus odd tail
        std::vector<double> br(lanes), bi(lanes), vr(lanes), vi(lanes);
        std::vector<double> incumbent(lanes);
        std::vector<std::uint8_t> mask(lanes);
        std::vector<std::vector<std::complex<double>>> signal(lanes);
        double energy = 0.0;

        // One time node has one energy.  Use unit-modulus samples so every
        // lane (frequency) shares exactly E=L, as in the transform.
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            signal[lane].resize(static_cast<std::size_t>(length));
            std::complex<double> total{};
            for (int n = 0; n < length; ++n) {
                const double phase = 6.2831853071795864769 * unit(rng);
                const std::complex<double> y(std::cos(phase), std::sin(phase));
                signal[lane][static_cast<std::size_t>(n)] = y;
                total += y;
            }
            const std::complex<double> before(normal(rng), normal(rng));
            br[lane] = before.real();
            bi[lane] = before.imag();
            vr[lane] = total.real();
            vi[lane] = total.imag();

            // Mix certifiably prunable and non-prunable thresholds.
            const double h = std::abs(before + 0.5 * total);
            const double radius = 0.5 * std::sqrt(static_cast<double>(length) *
                                                  static_cast<double>(length));
            const int lo = 13;
            const double edge = (h + radius) * (h + radius) / (lo + 1);
            incumbent[lane] = edge * (lane % 3 == 0 ? 1.2 : 0.8);
        }
        energy = static_cast<double>(length);
        const int lo = 13;
        fct::half_edge::prune_same_node(
            br.data(), bi.data(), vr.data(), vi.data(), incumbent.data(),
            lanes, energy, length, lo, mask.data());

        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const bool scalar = fct::half_edge::prunable_scalar(
                br[lane], bi[lane], vr[lane], vi[lane], incumbent[lane],
                energy, length, lo);
            assert(static_cast<bool>(mask[lane]) == scalar);
            if (!mask[lane]) continue;

            // A positive SIMD decision is a theorem, not a heuristic: every
            // actual prefix score must be below the incumbent.
            std::complex<double> prefix(br[lane], bi[lane]);
            double actual_max = 0.0;
            for (int j = 0; j < length; ++j) {
                prefix += signal[lane][static_cast<std::size_t>(j)];
                actual_max = std::max(
                    actual_max,
                    std::norm(prefix) / static_cast<double>(lo + j + 1));
            }
            assert(actual_max <= incumbent[lane] * (1.0 + 1e-12));
        }
    }

    std::cout << "PASS half-edge SIMD certificate ("
              << bruun::simd_backend_name() << ")\n";
    return 0;
}
