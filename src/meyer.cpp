#include <bfft/meyer.h>

#include "detail/meyer_kernel.hpp"

#include <new>

struct bfft_meyer_plan {
    meyer::engine eng;
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
    if (!pow2_ge8(height) || !pow2_ge8(width)) return BFFT_ERROR_INVALID_ARGUMENT;
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

bfft_status bfft_meyer_split(bfft_meyer_plan* plan, const double* image,
                             double* cartoon, double* texture) {
    if (plan == nullptr || image == nullptr || cartoon == nullptr ||
        texture == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split(image, cartoon, texture);
    return BFFT_OK;
}

bfft_status bfft_meyer_decompose(bfft_meyer_plan* plan, const double* image,
                                 double* cartoon, double* texture,
                                 double* band_coarse, double* band_mid,
                                 double* band_fine) {
    if (plan == nullptr || image == nullptr || cartoon == nullptr ||
        texture == nullptr || band_coarse == nullptr || band_mid == nullptr ||
        band_fine == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.decompose(image, cartoon, texture, band_coarse, band_mid,
                        band_fine);
    return BFFT_OK;
}
