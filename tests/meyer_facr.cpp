#include <bfft/meyer.h>

#include "../src/detail/meyer_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

void fill(std::vector<double>& x) {
    unsigned state = 0x12345678u;
    for (double& v : x) {
        state = 1664525u * state + 1013904223u;
        v = double(state >> 8) * (255.0 / 16777216.0);
    }
}

bool finite(const std::vector<double>& x) {
    for (double v : x)
        if (!std::isfinite(v)) return false;
    return true;
}

double max_error(const std::vector<double>& a,
                 const std::vector<double>& b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

double operator_residual(std::size_t h, std::size_t w, int solver) {
    meyer::engine e;
    if (e.init(h, w, 0.05, 40.0, 2, 2, 0.0, 3) != BFFT_OK ||
        !e.set_solver(solver))
        return 1.0;
    const std::size_t n = h * w;
    std::vector<double> g(n), u(n);
    fill(g);
    e.facr_fwd(g.data(), e.ff_spec);
    e.facr_scale(e.ff_spec, 0.05, e.t_u, e.fu_spec);
    e.facr_inv(e.fu_spec, u.data());
    double error2 = 0.0, rhs2 = 0.0;
    constexpr double c = 0.05, eta = 0.10;
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            const std::size_t i = y * w + x;
            const std::size_t xl = x == 0 ? w - 1 : x - 1;
            const std::size_t xr = x + 1 == w ? 0 : x + 1;
            const std::size_t yu = y == 0 ? h - 1 : y - 1;
            const std::size_t yd = y + 1 == h ? 0 : y + 1;
            double lapx = u[y * w + xl] + u[y * w + xr] - 2.0 * u[i];
            double lapy = u[yu * w + x] + u[yd * w + x] - 2.0 * u[i];
            if (solver == 2 && e.sweep_height) {
                if (y == 0) lapy = u[w + x] - u[i];
                if (y + 1 == h) lapy = u[(h - 2) * w + x] - u[i];
            }
            if (solver == 2 && !e.sweep_height) {
                if (x == 0) lapx = u[i + 1] - u[i];
                if (x + 1 == w) lapx = u[i - 1] - u[i];
            }
            const double rhs = c * g[i];
            const double err = c * u[i] - eta * (lapx + lapy) - rhs;
            error2 += err * err;
            rhs2 += rhs * rhs;
        }
    }
    return std::sqrt(error2 / rhs2);
}

double reduced_state_error(std::size_t h, std::size_t w, int solver) {
    meyer::engine reduced, full;
    if (reduced.init(h, w, 0.05, 40.0, 5, 2, 0.0, 2) != BFFT_OK ||
        full.init(h, w, 0.05, 40.0, 5, 2, 0.0, 2) != BFFT_OK)
        return 1.0;
    if (solver != 0 &&
        (!reduced.set_solver(solver) || !full.set_solver(solver)))
        return 1.0;
    if (!reduced.xit.empty() || !reduced.dbux.empty() ||
        !reduced.vplane.empty() || !reduced.v_spec.a.empty() ||
        !reduced.fv_spec.a.empty() || !reduced.s_r0.empty() ||
        !reduced.t_r0.pivot.empty())
        return 1.0;
    const std::size_t n = h * w;
    std::vector<double> image(n), cartoon(n), texture(n);
    fill(image);
    reduced.split(image.data(), cartoon.data(), texture.data());
    if (!reduced.xit.empty() || !reduced.dbux.empty() ||
        !reduced.vplane.empty() || !reduced.v_spec.a.empty() ||
        !reduced.fv_spec.a.empty())
        return 1.0;
    full.ensure_decompose_storage();
    full.run_passes(image.data());
    double error = max_error(cartoon, full.u);
    error = std::max(error, max_error(texture, full.vplane));
    return error;
}

}  // namespace

int main() {
    constexpr std::size_t H = 32, W = 64, N = H * W;
    std::vector<double> image(N), c0(N), v0(N), c1(N), v1(N);
    fill(image);
    bfft_meyer_plan *p0 = nullptr, *p1 = nullptr;
    if (bfft_meyer_plan_create(H, W, 0.05, 40.0, 5, 4, 0.0, 2,
                               &p0) != BFFT_OK ||
        bfft_meyer_plan_create(H, W, 0.05, 40.0, 5, 4, 0.0, 2,
                               &p1) != BFFT_OK ||
        bfft_meyer_plan_set_solver(p1, 1) != BFFT_OK ||
        bfft_meyer_split(p0, image.data(), c0.data(), v0.data()) != BFFT_OK ||
        bfft_meyer_split(p1, image.data(), c1.data(), v1.data()) != BFFT_OK)
        return 1;
    const double periodic_error =
        std::max(max_error(c0, c1), max_error(v0, v1));
    std::printf("spectral/FACR max error: %.3e\n", periodic_error);
    bfft_meyer_plan_destroy(p0);
    bfft_meyer_plan_destroy(p1);
    if (periodic_error > 2e-10) return 2;

    const double periodic_h_residual = operator_residual(37, 64, 1);
    const double periodic_w_residual = operator_residual(64, 45, 1);
    const double periodic_two_residual = operator_residual(2, 8, 1);
    const double neumann_residual = operator_residual(37, 64, 2);
    std::printf("operator residuals: periodic-h %.3e, periodic-w %.3e, "
                "periodic-n2 %.3e, Neumann %.3e\n", periodic_h_residual,
                periodic_w_residual, periodic_two_residual, neumann_residual);
    if (std::max({periodic_h_residual, periodic_w_residual,
                  periodic_two_residual, neumann_residual}) > 2e-12)
        return 7;

    const double reduced_spectral = reduced_state_error(32, 64, 0);
    const double reduced_facr = reduced_state_error(37, 64, 1);
    std::printf("reduced/full state errors: spectral %.3e, FACR %.3e\n",
                reduced_spectral, reduced_facr);
    if (std::max(reduced_spectral, reduced_facr) > 2e-12) return 8;

    // Non-power-of-two swept dimensions and both sweep orientations.
    for (const auto shape : {std::pair<std::size_t, std::size_t>{37, 64},
                             {64, 45},
                             // Exact 16:9 work grid used by the OBS filter.
                             {288, 512}}) {
        const std::size_t n = shape.first * shape.second;
        std::vector<double> in(n), ca(n), va(n), cb(n), vb(n);
        fill(in);
        bfft_meyer_plan *a = nullptr, *b = nullptr;
        if (bfft_meyer_plan_create(shape.first, shape.second, 0.05, 40.0,
                                   3, 2, 0.0, 1, &a) != BFFT_OK ||
            bfft_meyer_plan_create(shape.first, shape.second, 0.05, 40.0,
                                   3, 2, 0.0, 4, &b) != BFFT_OK ||
            bfft_meyer_plan_set_solver(a, 1) != BFFT_OK ||
            bfft_meyer_plan_set_solver(b, 1) != BFFT_OK ||
            bfft_meyer_split(a, in.data(), ca.data(), va.data()) != BFFT_OK ||
            bfft_meyer_split(b, in.data(), cb.data(), vb.data()) != BFFT_OK)
            return 3;
        const double thread_error =
            std::max(max_error(ca, cb), max_error(va, vb));
        bfft_meyer_plan_destroy(a);
        bfft_meyer_plan_destroy(b);
        if (!finite(ca) || !finite(va) || thread_error != 0.0) return 4;
    }

    // Neumann mode is a deliberate behavior change but must remain finite.
    bfft_meyer_plan* pn = nullptr;
    std::vector<double> cn(N), vn(N);
    if (bfft_meyer_plan_create(H, W, 0.05, 40.0, 3, 2, 0.0, 2,
                               &pn) != BFFT_OK ||
        bfft_meyer_plan_set_solver(pn, 2) != BFFT_OK ||
        bfft_meyer_split(pn, image.data(), cn.data(), vn.data()) != BFFT_OK)
        return 5;
    bfft_meyer_plan_destroy(pn);
    return finite(cn) && finite(vn) ? 0 : 6;
}
