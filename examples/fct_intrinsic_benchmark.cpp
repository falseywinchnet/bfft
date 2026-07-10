#include "../src/detail/fct_intrinsic_kernel.hpp"
#include "../src/detail/fct_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

template <typename Fn>
double best_us(Fn&& fn, int repeats) {
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        const auto a = std::chrono::steady_clock::now();
        fn();
        const auto b = std::chrono::steady_clock::now();
        best = std::min(best,
            std::chrono::duration<double, std::micro>(b - a).count());
    }
    return best;
}

int main() {
    constexpr double twopi = 6.283185307179586476925286766559005768;
    std::cout << "N   intrinsic_us heuristic_us ratio nodes/bin avg_packet "
                 "max_packet rounds cells channels\n";
    for (int n : {128, 256, 512, 1024}) {
        std::vector<fct::fractional::value> x(static_cast<std::size_t>(n));
        std::vector<fct::cplx> xh(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            const double phase = twopi * (17.0 * j / n + 0.31 * j * j / n);
            const double gate = (j > n / 5 && j < 4 * n / 5) ? 1.0 : 0.15;
            x[static_cast<std::size_t>(j)] =
                fct::fractional::value{gate * std::cos(phase),
                                       gate * std::sin(phase)};
            xh[static_cast<std::size_t>(j)] =
                fct::cplx{x[static_cast<std::size_t>(j)].re,
                           x[static_cast<std::size_t>(j)].im};
        }

        fct::intrinsic::plan intrinsic(n);
        fct::plan heuristic(n);
        std::vector<fct::fractional::value> out(static_cast<std::size_t>(n));
        std::vector<fct::cplx> out_h(static_cast<std::size_t>(n));
        std::vector<std::int64_t> tau(static_cast<std::size_t>(n));
        std::vector<std::int64_t> tau_h(static_cast<std::size_t>(n));
        const int repeats = n <= 256 ? 5 : n <= 512 ? 3 : 2;

        // Warm instruction/data pages; each intrinsic call intentionally owns
        // a fresh per-frame sparse provider cache.
        intrinsic.forward(x.data(), out.data(), tau.data());
        heuristic.forward_complex(xh.data(), out_h.data(), tau_h.data());
        const double ti = best_us([&] {
            intrinsic.forward(x.data(), out.data(), tau.data());
        }, repeats);
        const double th = best_us([&] {
            heuristic.forward_complex(xh.data(), out_h.data(), tau_h.data());
        }, repeats);
        const auto& s = intrinsic.stats();
        std::cout << std::setw(4) << n << ' '
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << ti << ' '
                  << std::setw(12) << th << ' '
                  << std::setprecision(2) << std::setw(5) << ti / th << ' '
                  << std::setw(9) << static_cast<double>(s.nodes) / n << ' '
                  << std::setw(10) << static_cast<double>(s.mask_lanes) /
                                             std::max<std::size_t>(s.mask_packets, 1)
                  << ' ' << std::setw(10) << s.max_packet
                  << ' ' << std::setw(6) << s.rounds
                  << ' ' << std::setw(8) << s.provider_cells
                  << ' ' << std::setw(8) << s.provider_channels << '\n';
    }
    return 0;
}
