#include <bfft/fct.h>

#include "detail/fct_kernel.hpp"

#include <cstddef>
#include <new>

static_assert(sizeof(bfft_complex) == sizeof(fct::cplx),
              "bfft_complex / fct::cplx size mismatch");
static_assert(offsetof(bfft_complex, re) == offsetof(fct::cplx, re),
              "bfft_complex / fct::cplx re offset mismatch");
static_assert(offsetof(bfft_complex, im) == offsetof(fct::cplx, im),
              "bfft_complex / fct::cplx im offset mismatch");

struct fct_plan {
    fct::plan p;
    fct_plan(int n, int t_min, double rel, double act)
        : p(n, t_min, rel, act) {}
};

namespace {

fct::cplx* as_c(bfft_complex* v) { return reinterpret_cast<fct::cplx*>(v); }
const fct::cplx* as_c(const bfft_complex* v) {
    return reinterpret_cast<const fct::cplx*>(v);
}

bool valid_size(size_t n) {
    if (n > static_cast<size_t>(2147483647)) return false;
    const int ni = static_cast<int>(n);
    return ni >= 16 && (ni & (ni - 1)) == 0;
}

} // namespace

bfft_status fct_plan_create_ex(size_t n, int t_min, double rel, double act,
                               fct_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (!valid_size(n)) return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(rel > 0.0 && rel <= 1.0) || !(act >= 0.0)) return BFFT_ERROR_INVALID_ARGUMENT;
    fct_plan* p = new (std::nothrow)
        fct_plan(static_cast<int>(n), t_min, rel, act);
    if (!p) return BFFT_ERROR_ALLOCATION;
    if (!p->p.valid()) {
        delete p;
        return BFFT_ERROR_ALLOCATION;
    }
    *plan = p;
    return BFFT_OK;
}

bfft_status fct_plan_create(size_t n, fct_plan** plan) {
    return fct_plan_create_ex(n, 4, 0.5, 1.5, plan);
}

void fct_plan_destroy(fct_plan* plan) { delete plan; }

size_t fct_plan_size(const fct_plan* plan) {
    return plan ? static_cast<size_t>(plan->p.size()) : 0;
}

size_t fct_plan_bins(const fct_plan* plan) {
    return plan ? static_cast<size_t>(plan->p.bins()) : 0;
}

bfft_status fct_forward(fct_plan* plan, const double* input,
                        bfft_complex* output, int64_t* tau) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->p.forward(input, as_c(output), tau)) return BFFT_ERROR_INTERNAL;
    return BFFT_OK;
}

bfft_status fct_forward_complex(fct_plan* plan, const bfft_complex* input,
                                bfft_complex* output, int64_t* tau) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->p.forward_complex(as_c(input), as_c(output), tau))
        return BFFT_ERROR_INTERNAL;
    return BFFT_OK;
}

bfft_status fct_forward_numba(fct_plan* plan, const double* input,
                              bfft_complex* output, double* /*work*/,
                              bfft_complex* /*native_scratch*/) {
    return fct_forward(plan, input, output, nullptr);
}
