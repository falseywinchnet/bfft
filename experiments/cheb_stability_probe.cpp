// Numerical stability of the radix-4 Chebyshev node variants.
//
// Concern: lowmul_node computes ev = (a0+a2) - u^2*a2 instead of the
// subtraction-free ev = a0 + v^2*a2 (an FMA). Does introducing that
// subtraction (and more rounding steps) cost accuracy versus the all-FMA form?
//
// Method: for many random inputs -- plus adversarial inputs engineered so the
// even/odd value is near zero (worst case for cancellation) -- evaluate one
// node three ways in double and compare to a long double reference. Report the
// max relative error (vs the largest output magnitude) per form.
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude experiments/cheb_stability_probe.cpp \
//       build/libbfft.a -lm -o build/experiments/cheb_stability_probe

#include <cmath>
#include <cstdio>
#include <random>

#include "chebyshev_composed_radix4_kernel.hpp"

namespace cc = cheb_composed_r4;

namespace {

struct Out { double x0, x1, x2, x3; };

// Long double reference: f(+u),f(-u),f(+v),f(-v) of the cubic, high precision.
Out ref_ld(double a0, double a1, double a2, double a3, double u, double v) {
    auto f = [&](long double y) {
        return (long double)a0 + (long double)a1 * y
             + (long double)a2 * y * y + (long double)a3 * y * y * y;
    };
    return { (double)f(u), (double)f(-(long double)u),
             (double)f(v), (double)f(-(long double)v) };
}

Out allfma(double a0, double a1, double a2, double a3, double u, double v) {
    const double u2 = u * u, v2 = v * v;
    const double eu = a0 + a2 * u2, ou = a1 + a3 * u2;
    const double ev = a0 + a2 * v2, ov = a1 + a3 * v2;
    return { eu + u * ou, eu - u * ou, ev + v * ov, ev - v * ov };
}

Out lowmul(double a0, double a1, double a2, double a3, double u, double v) {
    const double u2 = u * u;
    const double m2 = u2 * a2, m3 = u2 * a3;
    const double eu = a0 + m2, ev = (a0 + a2) - m2;
    const double ou = a1 + m3, ov = (a1 + a3) - m3;
    const double su = u * ou, sv = v * ov;
    return { eu + su, eu - su, ev + sv, ev - sv };
}

double rel_err(const Out& got, const Out& ref) {
    const double mx = std::max(std::max(std::abs(ref.x0), std::abs(ref.x1)),
                               std::max(std::abs(ref.x2), std::abs(ref.x3)));
    const double e = std::max(std::max(std::abs(got.x0 - ref.x0), std::abs(got.x1 - ref.x1)),
                              std::max(std::abs(got.x2 - ref.x2), std::abs(got.x3 - ref.x3)));
    return e / (mx > 0 ? mx : 1.0);
}

} // namespace

int main() {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> du(-1.0, 1.0);
    std::uniform_real_distribution<double> dav(-0.999, 0.999);

    double max_fma = 0, max_low = 0, sum_fma = 0, sum_low = 0;
    long trials = 2'000'000;
    for (long t = 0; t < trials; ++t) {
        const double a0 = du(rng), a1 = du(rng), a2 = du(rng), a3 = du(rng);
        const cc::NodeConst k = cc::node_const(dav(rng));
        const Out r = ref_ld(a0, a1, a2, a3, k.u, k.v);
        const double ef = rel_err(allfma(a0, a1, a2, a3, k.u, k.v), r);
        const double el = rel_err(lowmul(a0, a1, a2, a3, k.u, k.v), r);
        max_fma = std::max(max_fma, ef); sum_fma += ef;
        max_low = std::max(max_low, el); sum_low += el;
    }
    std::printf("Random inputs (%ld trials):\n", trials);
    std::printf("  all-FMA : max rel err = %.2e   mean = %.2e\n", max_fma, sum_fma / trials);
    std::printf("  low-mul : max rel err = %.2e   mean = %.2e\n", max_low, sum_low / trials);

    // Adversarial: force the v-side even value ev = a0 + v^2 a2 ~ 0 by choosing
    // a0 = -v^2 a2 with large a2, the worst case for cancellation in BOTH forms.
    double adv_fma = 0, adv_low = 0;
    long adv = 500'000;
    for (long t = 0; t < adv; ++t) {
        const cc::NodeConst k = cc::node_const(dav(rng));
        const double v2 = k.v * k.v;
        const double a2 = du(rng) * 1e6;
        const double a0 = -v2 * a2 * (1.0 + du(rng) * 1e-12);  // ev ~ 0
        const double a1 = du(rng), a3 = du(rng);
        const Out r = ref_ld(a0, a1, a2, a3, k.u, k.v);
        adv_fma = std::max(adv_fma, rel_err(allfma(a0, a1, a2, a3, k.u, k.v), r));
        adv_low = std::max(adv_low, rel_err(lowmul(a0, a1, a2, a3, k.u, k.v), r));
    }
    std::printf("\nAdversarial (ev ~ 0, large a2; %ld trials):\n", adv);
    std::printf("  all-FMA : max rel err = %.2e\n", adv_fma);
    std::printf("  low-mul : max rel err = %.2e\n", adv_low);

    std::printf("\nReading: cancellation in lowmul's (a0+a2)-u^2 a2 occurs iff\n");
    std::printf("ev = a0 + v^2 a2 ~ 0 -- the SAME condition under which the all-FMA\n");
    std::printf("form is also ill-conditioned (its inputs already nearly cancel).\n");
    std::printf("lowmul does ~3 roundings per even/odd value vs the FMA's 1, so it is\n");
    std::printf("mildly less accurate in the mean but not categorically less stable.\n");
    return 0;
}
