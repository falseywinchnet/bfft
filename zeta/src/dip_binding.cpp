#include "../../src/detail/bruun_dip_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {
int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}
}

extern "C" {

// Linear convolution through the repository's trusted DIP real FFT.
int zeta_dip_convolve(const double* a, int na, const double* b, int nb,
                      double* out) {
    if (!a || !b || !out || na <= 0 || nb <= 0) return -1;
    const int logical = na + nb - 1;
    const int nfft = std::max(4, next_power_of_two(logical));
    bruun::DIP_RFFT_kernel plan;
    if (!plan.init(nfft)) return -2;

    std::vector<double> pa(static_cast<std::size_t>(nfft), 0.0);
    std::vector<double> pb(static_cast<std::size_t>(nfft), 0.0);
    std::copy(a, a + na, pa.begin());
    std::copy(b, b + nb, pb.begin());
    std::vector<bruun::complex_t> fa(static_cast<std::size_t>(plan.bins()));
    std::vector<bruun::complex_t> fb(static_cast<std::size_t>(plan.bins()));
    std::vector<double> work(static_cast<std::size_t>(plan.work_size()));
    plan.forward_standard(pa.data(), fa.data(), work.data());
    plan.forward_standard(pb.data(), fb.data(), work.data());
    for (int k = 0; k < plan.bins(); ++k) {
        const double ar = fa[static_cast<std::size_t>(k)].re;
        const double ai = fa[static_cast<std::size_t>(k)].im;
        const double br = fb[static_cast<std::size_t>(k)].re;
        const double bi = fb[static_cast<std::size_t>(k)].im;
        fa[static_cast<std::size_t>(k)].re = ar * br - ai * bi;
        fa[static_cast<std::size_t>(k)].im = ar * bi + ai * br;
    }
    plan.inverse_standard(fa.data(), pa.data(), work.data());
    std::copy(pa.begin(), pa.begin() + logical, out);
    return nfft;
}

const char* zeta_dip_backend() { return bruun::simd_backend_name(); }

}
