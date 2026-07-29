#include "../src/detail/meyer_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

double inside_fraction(const std::vector<double>& x,
                       const std::vector<double>& y, double threshold) {
    std::size_t inside = 0;
    const double threshold2 = threshold * threshold;
    for (std::size_t i = 0; i < x.size(); ++i)
        inside += x[i] * x[i] + y[i] * y[i] <= threshold2;
    return double(inside) / double(x.size());
}

void run(const char* name, const std::vector<double>& image) {
    constexpr std::size_t H = 256, W = 256;
    meyer::engine e;
    if (e.init(H, W, 0.05, 40.0, 24, 1, 0.0, 4) != BFFT_OK)
        return;
    const std::size_t bytes = H * W * sizeof(double);
    for (auto* p : {&e.u, &e.w, &e.bux, &e.buy, &e.bvx, &e.bvy})
        std::memset(p->data(), 0, bytes);
    e.u_spec.zero();
    e.w_spec.zero();
    e.fwd2d(image.data(), e.f_spec);
    const double c_u = e.lam, eta_u = 2.0 * e.lam;
    const double c_v = 1.0 / e.mu, eta_v = 10.0 / e.mu;
    std::printf("%s\n", name);
    for (int pass = 0; pass < e.passes; ++pass) {
        if (pass == 0) {
            e.solve_meyer_triangle_first(c_u, c_v);
        } else {
            if (pass == 1 || pass == 2 || pass == 4 || pass == 8 ||
                pass == 16 || pass == 23)
                std::printf("  pass %2d inside: cartoon %.3f texture %.3f\n",
                            pass,
                            inside_fraction(e.bux, e.buy, 1.0 / eta_u),
                            inside_fraction(e.bvx, e.bvy, 1.0 / eta_v));
            e.reflection_divergence_pair(
                e.bux, e.buy, eta_u, e.u.data(),
                e.bvx, e.bvy, eta_v, e.w.data());
            e.fwd2d(e.u.data(), e.d_spec);
            e.fwd2d(e.w.data(), e.q_spec);
            e.solve_meyer_triangle(e.d_spec, e.q_spec, c_u, eta_u, c_v,
                                   eta_v);
        }
        e.inv2d(e.u_spec, e.u.data());
        e.inv2d(e.w_spec, e.w.data());
        e.update_reflected_dual_pair(
            e.u, e.bux, e.buy, eta_u, e.w, e.bvx, e.bvy, eta_v);
    }
}

}  // namespace

int main() {
    constexpr std::size_t H = 256, W = 256, N = H * W;
    std::vector<double> sine(N), blocks(N), noise(N);
    unsigned state = 1;
    for (std::size_t y = 0; y < H; ++y) {
        for (std::size_t x = 0; x < W; ++x) {
            const std::size_t i = y * W + x;
            sine[i] = 127.0 + 60.0 * std::sin(0.013 * double(i));
            blocks[i] = (x < W / 2 ? 60.0 : 190.0) +
                (y < H / 2 ? 20.0 : -20.0) +
                18.0 * std::sin(0.31 * double(x) + 0.17 * double(y));
            state = 1664525u * state + 1013904223u;
            noise[i] = 255.0 * double(state >> 8) / double(1u << 24);
        }
    }
    run("sine benchmark", sine);
    run("blocks + oscillation", blocks);
    run("white noise", noise);
}
