#include <bfft/meyer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

void scene(std::vector<double>& image, std::size_t h, std::size_t w) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    unsigned state = 0x6d2b79f5u;
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            state = 1664525u * state + 1013904223u;
            const double noise = (double(state >> 8) / 16777216.0 - 0.5) * 4.0;
            double value = 45.0 + 125.0 * double(x) / double(w);
            value += y > h / 2 ? 38.0 : 0.0;
            value += 21.0 * std::cos(2.0 * pi * double(x + y) / 7.0);
            if (x < w / 2)
                value += 13.0 * std::cos(2.0 * pi * double(y) / 5.0);
            image[y * w + x] = value + noise;
        }
    }
}

double objective(const std::vector<double>& u,
                 const std::vector<double>& g,
                 std::size_t h, std::size_t w, double c) {
    double value = 0.0;
    for (std::size_t y = 0; y < h; ++y) {
        const std::size_t yn = y + 1 == h ? 0 : y + 1;
        for (std::size_t x = 0; x < w; ++x) {
            const std::size_t xn = x + 1 == w ? 0 : x + 1;
            const std::size_t i = y * w + x;
            const double gx = u[y * w + xn] - u[i];
            const double gy = u[yn * w + x] - u[i];
            const double residual = u[i] - g[i];
            value += std::sqrt(gx * gx + gy * gy) +
                0.5 * c * residual * residual;
        }
    }
    return value;
}

double relative_error(const std::vector<double>& a,
                      const std::vector<double>& b) {
    double difference = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        difference += d * d;
        scale += b[i] * b[i];
    }
    return std::sqrt(difference / std::max(scale, 1e-30));
}

double max_error(const std::vector<double>& a,
                 const std::vector<double>& b) {
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        value = std::max(value, std::abs(a[i] - b[i]));
    return value;
}

}  // namespace

int main() {
    constexpr std::size_t H = 128, W = 128, N = H * W;
    constexpr double c = 0.05, eta = 0.10;
    std::vector<double> image(N), plain8(N), hodge8(N), plain256(N),
        hodge256(N), reference(N);
    scene(image, H, W);

    bfft_meyer_plan* plan = nullptr;
    if (bfft_meyer_plan_create(H, W, c, 40.0, 1, 1, 0.0, 1, &plan) !=
        BFFT_OK)
        return 1;
    if (bfft_meyer_rof(plan, image.data(), plain8.data(), c, eta, 8, 0.0) !=
            BFFT_OK ||
        bfft_meyer_rof_accelerated(
            plan, image.data(), hodge8.data(), c, eta, 8, 0.0, 4) !=
            BFFT_OK ||
        !bfft_meyer_plan_last_rof_hodge_applied(plan) ||
        bfft_meyer_plan_last_rof_sweeps(plan) != 8)
        return 2;

    if (bfft_meyer_rof(
            plan, image.data(), plain256.data(), c, eta, 256, 0.0) !=
            BFFT_OK ||
        bfft_meyer_rof_accelerated(
            plan, image.data(), hodge256.data(), c, eta, 256, 0.0, 4) !=
            BFFT_OK ||
        bfft_meyer_rof(
            plan, image.data(), reference.data(), c, eta, 1024, 0.0) !=
            BFFT_OK)
        return 3;

    const double plain8_objective = objective(plain8, image, H, W, c);
    const double hodge8_objective = objective(hodge8, image, H, W, c);
    const double plain8_error = relative_error(plain8, reference);
    const double hodge8_error = relative_error(hodge8, reference);
    const double plain256_error = relative_error(plain256, reference);
    const double hodge256_error = relative_error(hodge256, reference);
    std::printf(
        "early objective plain %.9e hodge %.9e; errors %.3e %.3e\n",
        plain8_objective, hodge8_objective, plain8_error, hodge8_error);
    std::printf("deep errors plain %.3e hodge %.3e\n",
                plain256_error, hodge256_error);
    if (!(hodge8_objective < plain8_objective) ||
        !(hodge8_error < plain8_error) ||
        !(hodge256_error <= 1.05 * plain256_error))
        return 4;

    // Every parallel stage writes disjoint rows or spectral ranges, while
    // the acceptance reduction is deliberately serial.  The accelerated
    // answer therefore retains the engine's thread-count invariance.
    bfft_meyer_plan* parallel = nullptr;
    std::vector<double> parallel8(N);
    if (bfft_meyer_plan_create(H, W, c, 40.0, 1, 1, 0.0, 4, &parallel) !=
            BFFT_OK ||
        bfft_meyer_rof_accelerated(
            parallel, image.data(), parallel8.data(), c, eta, 8, 0.0, 4) !=
            BFFT_OK ||
        max_error(hodge8, parallel8) != 0.0)
        return 5;
    bfft_meyer_plan_destroy(parallel);

    // A constant image has no accepted Hodge direction.  Rejection must not
    // perturb the live Split-Bregman state or its output by even one bit.
    std::vector<double> constant(N, 73.0), ordinary(N), rejected(N);
    if (bfft_meyer_rof(
            plan, constant.data(), ordinary.data(), c, eta, 32, 0.0) !=
            BFFT_OK ||
        bfft_meyer_rof_accelerated(
            plan, constant.data(), rejected.data(), c, eta, 32, 0.0, 4) !=
            BFFT_OK ||
        bfft_meyer_plan_last_rof_hodge_applied(plan) ||
        max_error(ordinary, rejected) != 0.0)
        return 6;

    // If tolerance stops before the requested insertion point, no closure
    // is attempted and the diagnostics must say so.
    if (bfft_meyer_rof_accelerated(
            plan, constant.data(), rejected.data(), c, eta, 32, 1.0, 4) !=
            BFFT_OK ||
        bfft_meyer_plan_last_rof_sweeps(plan) >= 4 ||
        bfft_meyer_plan_last_rof_hodge_applied(plan))
        return 7;

    bfft_meyer_plan_destroy(plan);

    // Full Hodge acceleration deliberately does not pretend that one-axis
    // FACR or Neumann boundaries implement the same Poisson projector.
    bfft_meyer_plan* facr = nullptr;
    if (bfft_meyer_plan_create(
            95, 128, c, 40.0, 1, 1, 0.0, 1, &facr) != BFFT_OK ||
        bfft_meyer_plan_set_solver(facr, 1) != BFFT_OK)
        return 8;
    std::vector<double> facr_image(95 * 128, 1.0);
    std::vector<double> facr_out(95 * 128);
    const bfft_status unsupported = bfft_meyer_rof_accelerated(
        facr, facr_image.data(), facr_out.data(), c, eta, 8, 0.0, 4);
    bfft_meyer_plan_destroy(facr);
    return unsupported == BFFT_ERROR_INVALID_ARGUMENT ? 0 : 9;
}
