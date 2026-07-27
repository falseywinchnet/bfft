#include <bfft/meyer.h>

#include "detail/meyer_kernel.hpp"

#include <new>

struct bfft_meyer_plan {
    meyer::engine eng;
    bool configured = true;
};

namespace {

bool pow2_ge8(size_t n) { return n >= 8 && (n & (n - 1)) == 0; }

}  // namespace

bfft_status bfft_meyer_plan_create(size_t height, size_t width, double lam,
                                   double mu, int passes, int rung_sweeps,
                                   double rung_tol, int threads,
                                   bfft_meyer_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (height < 2 || width < 2 ||
        (!pow2_ge8(height) && !pow2_ge8(width)))
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(lam > 0.0) || !(mu > 0.0) || passes < 1 || rung_sweeps < 1 ||
        !(rung_tol >= 0.0) || threads < 0 || threads > 64)
        return BFFT_ERROR_INVALID_ARGUMENT;
    bfft_meyer_plan* p = new (std::nothrow) bfft_meyer_plan();
    if (p == nullptr) return BFFT_ERROR_ALLOCATION;
    const bfft_status st =
        p->eng.init(height, width, lam, mu, passes, rung_sweeps, rung_tol,
                    threads);
    if (st != BFFT_OK) {
        delete p;
        return st;
    }
    p->configured = pow2_ge8(height) && pow2_ge8(width);
    *plan = p;
    return BFFT_OK;
}

void bfft_meyer_plan_destroy(bfft_meyer_plan* plan) { delete plan; }

size_t bfft_meyer_plan_height(const bfft_meyer_plan* plan) {
    return plan ? plan->eng.H : 0;
}

size_t bfft_meyer_plan_width(const bfft_meyer_plan* plan) {
    return plan ? plan->eng.W : 0;
}

bfft_status bfft_meyer_plan_set_passes(bfft_meyer_plan* plan, int passes) {
    if (plan == nullptr || passes < 1) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.passes = passes;
    return BFFT_OK;
}

bfft_status bfft_meyer_plan_set_solver(bfft_meyer_plan* plan, int mode) {
    if (plan == nullptr || !plan->eng.set_solver(mode))
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->configured = true;
    return BFFT_OK;
}

int bfft_meyer_plan_solver(const bfft_meyer_plan* plan) {
    return plan ? plan->eng.solver : 0;
}

bfft_status bfft_meyer_split(bfft_meyer_plan* plan, const double* image,
                             double* cartoon, double* texture) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr ||
        texture == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split(image, cartoon, texture);
    return BFFT_OK;
}

bfft_status bfft_meyer_split_trace(bfft_meyer_plan* plan,
                                   const double* image,
                                   double* cartoon_trace,
                                   double* texture_trace) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon_trace == nullptr ||
        texture_trace == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split_trace(image, cartoon_trace, texture_trace);
    return BFFT_OK;
}

bfft_status bfft_meyer_split_visit(bfft_meyer_plan* plan,
                                   const double* image,
                                   bfft_meyer_trace_visitor visitor,
                                   void* user) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        visitor == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split_visit(image, visitor, user);
    return BFFT_OK;
}

bfft_status bfft_meyer_decompose(bfft_meyer_plan* plan, const double* image,
                                 double* cartoon, double* texture,
                                 double* band_coarse, double* band_mid,
                                 double* band_fine) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr ||
        texture == nullptr || band_coarse == nullptr || band_mid == nullptr ||
        band_fine == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.decompose(image, cartoon, texture, band_coarse, band_mid,
                        band_fine);
    return BFFT_OK;
}

bfft_status bfft_meyer_rof(bfft_meyer_plan* plan, const double* image,
                           double* smooth, double c, double eta, int sweeps,
                           double tol) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        smooth == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(c > 0.0) || sweeps < 1 || !(tol >= 0.0))
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(eta > 0.0)) eta = 10.0 * c;
    plan->eng.rof(image, smooth, c, eta, sweeps, tol);
    return BFFT_OK;
}
