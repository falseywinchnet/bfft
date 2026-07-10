#include "../src/detail/fct_fractional_provider.hpp"

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
    constexpr double tau = 6.283185307179586476925286766559005768;
    std::mt19937_64 rng(0x50524f5649444552ULL);
    std::normal_distribution<double> normal(0.0, 1.0);

    double worst = 0.0;
    for (int n : {16, 32, 64, 128, 256}) {
        std::vector<value> x(static_cast<std::size_t>(n));
        for (value& z : x) z = value{normal(rng), normal(rng)};
        fct::fractional::provider provider(x.data(), n);
        for (int q = 0; q < 4000; ++q) {
            int log_l = static_cast<int>(rng() %
                static_cast<unsigned>(std::log2(n) + 1));
            const int length = 1 << log_l;
            const int lo = static_cast<int>(rng() % (n / length)) * length;
            const int k = static_cast<int>(rng() % n);
            const value got = provider.get(lo, length, k);
            std::complex<double> ref{};
            for (int j = 0; j < length; ++j) {
                const double a = -tau * static_cast<double>(k) * (lo + j) / n;
                const std::complex<double> w(std::cos(a), std::sin(a));
                ref += std::complex<double>(x[lo + j].re, x[lo + j].im) * w;
            }
            worst = std::max(worst, std::abs(
                std::complex<double>(got.re, got.im) - ref));
            assert(std::abs(std::complex<double>(got.re, got.im) - ref) <=
                   2e-10 * (1.0 + std::abs(ref)));
        }
        std::cout << "provider N=" << n
                  << " values=" << provider.value_count()
                  << " channels=" << provider.channel_count()
                  << " cells=" << provider.butterfly_cells() << '\n';
    }
    std::cout << "PASS incremental fractional provider; worst abs error "
              << worst << '\n';
    return 0;
}
