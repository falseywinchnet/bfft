#include "../src/detail/fct_intrinsic_kernel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using fct::fractional::value;
    constexpr double twopi = 6.283185307179586476925286766559005768;
    std::mt19937_64 rng(0x494e5452494e5349ULL);
    std::normal_distribution<double> normal(0.0, 1.0);
    double worst = 0.0;

    for (int n : {16, 32, 64, 128}) {
        for (int trial = 0; trial < 5; ++trial) {
            std::vector<value> x(static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j) {
                const double burst = (trial > 0 && j > n / 5 && j < 3 * n / 5)
                    ? std::cos(twopi * (7.3 + trial) * j / n)
                    : 0.0;
                x[static_cast<std::size_t>(j)] =
                    value{0.2 * normal(rng) + burst, 0.2 * normal(rng)};
            }
            fct::intrinsic::plan plan(n);
            std::vector<value> out(static_cast<std::size_t>(n));
            std::vector<std::int64_t> tau(static_cast<std::size_t>(n));
            assert(plan.forward(x.data(), out.data(), tau.data(), 0.0));

            for (int k = 0; k < n; ++k) {
                std::complex<double> prefix{};
                double best_score = -1.0;
                int best_tau = 0;
                std::complex<double> best{};
                for (int j = 0; j < n; ++j) {
                    const double a = -twopi * static_cast<double>(k) * j / n;
                    prefix += std::complex<double>(x[j].re, x[j].im) *
                              std::complex<double>(std::cos(a), std::sin(a));
                    const double score = std::norm(prefix) / (j + 1.0);
                    if (score > best_score) {
                        best_score = score;
                        best_tau = j + 1;
                        best = prefix;
                    }
                }
                assert(tau[static_cast<std::size_t>(k)] == best_tau);
                const std::complex<double> got(
                    out[static_cast<std::size_t>(k)].re,
                    out[static_cast<std::size_t>(k)].im);
                worst = std::max(worst, std::abs(got - best));
                assert(std::abs(got - best) <= 3e-9 * (1.0 + std::abs(best)));
            }
            assert(plan.stats().mask_lanes == plan.stats().nodes);
            assert(plan.stats().max_packet > 0);
            assert(plan.stats().provider_cells > 0);
        }
        std::cout << "joined intrinsic N=" << n << " exact\n";
    }

    // A declared minimum support restricts the feasible set without changing
    // the objective or certification. Check several non-dyadic boundaries so
    // nodes that straddle min_tau must still be descended exactly.
    for (int n : {32, 64}) {
        std::vector<value> x(static_cast<std::size_t>(n));
        for (value& z : x) z = value{normal(rng), normal(rng)};
        for (int min_tau : {2, 7, 17}) {
            if (min_tau > n) continue;
            fct::intrinsic::plan plan(n);
            std::vector<value> out(static_cast<std::size_t>(n));
            std::vector<std::int64_t> tau(static_cast<std::size_t>(n));
            assert(plan.forward(x.data(), out.data(), tau.data(), 0.0,
                                min_tau));
            for (int k = 0; k < n; ++k) {
                std::complex<double> prefix{};
                double best_score = -1.0;
                int best_tau = n;
                for (int j = 0; j < n; ++j) {
                    const double a = -twopi * static_cast<double>(k) * j / n;
                    prefix += std::complex<double>(x[j].re, x[j].im) *
                              std::complex<double>(std::cos(a), std::sin(a));
                    if (j + 1 < min_tau) continue;
                    const double score = std::norm(prefix) / (j + 1.0);
                    if (score > best_score) {
                        best_score = score;
                        best_tau = j + 1;
                    }
                }
                assert(tau[static_cast<std::size_t>(k)] == best_tau);
            }
        }
    }
    std::cout << "PASS provider + SIMD-mask joined kernel; worst abs error "
              << worst << '\n';
    return 0;
}
