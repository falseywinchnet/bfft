#include <bfft/bdct.h>

#include "detail/bdct_kernel.hpp"

#include <cstddef>
#include <new>

static_assert(sizeof(bfft_complex) == sizeof(bdct::complex_t),
              "bfft_complex / bdct::complex_t size mismatch");
static_assert(offsetof(bfft_complex, re) == offsetof(bdct::complex_t, re),
              "bfft_complex / bdct::complex_t re offset mismatch");
static_assert(offsetof(bfft_complex, im) == offsetof(bdct::complex_t, im),
              "bfft_complex / bdct::complex_t im offset mismatch");
static_assert(sizeof(bfft_complex_f32) == sizeof(bdct::complex_f32_t),
              "bfft_complex_f32 / bdct::complex_f32_t size mismatch");
static_assert(offsetof(bfft_complex_f32, re) == offsetof(bdct::complex_f32_t, re),
              "bfft_complex_f32 / bdct::complex_f32_t re offset mismatch");
static_assert(offsetof(bfft_complex_f32, im) == offsetof(bdct::complex_f32_t, im),
              "bfft_complex_f32 / bdct::complex_f32_t im offset mismatch");

// Each public plan carries a double and a single precision kernel so callers
// can mix precisions through one handle, mirroring the BFFT real plan.
struct bdct_plan {
    bdct::half_shift_plan f64;
    bdct::half_shift_plan_f32 f32;
    explicit bdct_plan(int n) : f64(n), f32(n) {}
};

struct bdct_dctiv_plan {
    bdct::dctiv_plan f64;
    bdct::dctiv_plan_f32 f32;
    explicit bdct_dctiv_plan(int n) : f64(n), f32(n) {}
};

namespace {

bdct::complex_t* as_c(bfft_complex* v) { return reinterpret_cast<bdct::complex_t*>(v); }
const bdct::complex_t* as_c(const bfft_complex* v) {
    return reinterpret_cast<const bdct::complex_t*>(v);
}
bdct::complex_f32_t* as_c(bfft_complex_f32* v) {
    return reinterpret_cast<bdct::complex_f32_t*>(v);
}
const bdct::complex_f32_t* as_c(const bfft_complex_f32* v) {
    return reinterpret_cast<const bdct::complex_f32_t*>(v);
}

bool valid_size(size_t n) {
    if (n > static_cast<size_t>(2147483647)) return false;
    const int ni = static_cast<int>(n);
    return ni >= 2 && (ni & (ni - 1)) == 0;
}

} // namespace

const char* bdct_backend_name(void) {
    return bruun::simd_backend_name();
}

// -- half-shift -------------------------------------------------------------

bfft_status bdct_plan_create(size_t n, bdct_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (!valid_size(n)) return BFFT_ERROR_INVALID_ARGUMENT;
    bdct_plan* p = new (std::nothrow) bdct_plan(static_cast<int>(n));
    if (!p) return BFFT_ERROR_ALLOCATION;
    *plan = p;
    return BFFT_OK;
}

void bdct_plan_destroy(bdct_plan* plan) { delete plan; }

size_t bdct_plan_size(const bdct_plan* plan) {
    return plan ? static_cast<size_t>(plan->f64.size()) : 0;
}

size_t bdct_plan_bins(const bdct_plan* plan) {
    return plan ? static_cast<size_t>(plan->f64.bins()) : 0;
}

bfft_status bdct_forward(const bdct_plan* plan, const double* input, bfft_complex* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f64.forward(input, as_c(output));
    return BFFT_OK;
}

bfft_status bdct_inverse(const bdct_plan* plan, const bfft_complex* input, double* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f64.inverse(as_c(input), output);
    return BFFT_OK;
}

bfft_status bdct_forward_f32(const bdct_plan* plan, const float* input, bfft_complex_f32* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f32.forward(input, as_c(output));
    return BFFT_OK;
}

bfft_status bdct_inverse_f32(const bdct_plan* plan, const bfft_complex_f32* input, float* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f32.inverse(as_c(input), output);
    return BFFT_OK;
}

// -- DCT-IV -----------------------------------------------------------------

bfft_status bdct_dctiv_plan_create(size_t n, bdct_dctiv_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (!valid_size(n)) return BFFT_ERROR_INVALID_ARGUMENT;
    bdct_dctiv_plan* p = new (std::nothrow) bdct_dctiv_plan(static_cast<int>(n));
    if (!p) return BFFT_ERROR_ALLOCATION;
    *plan = p;
    return BFFT_OK;
}

void bdct_dctiv_plan_destroy(bdct_dctiv_plan* plan) { delete plan; }

size_t bdct_dctiv_plan_size(const bdct_dctiv_plan* plan) {
    return plan ? static_cast<size_t>(plan->f64.size()) : 0;
}

bfft_status bdct_dctiv_forward(const bdct_dctiv_plan* plan, const double* input, double* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f64.forward(input, output);
    return BFFT_OK;
}

bfft_status bdct_dctiv_inverse(const bdct_dctiv_plan* plan, const double* input, double* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f64.inverse(input, output);
    return BFFT_OK;
}

bfft_status bdct_dctiv_forward_f32(const bdct_dctiv_plan* plan, const float* input, float* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f32.forward(input, output);
    return BFFT_OK;
}

bfft_status bdct_dctiv_inverse_f32(const bdct_dctiv_plan* plan, const float* input, float* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f32.inverse(input, output);
    return BFFT_OK;
}
