#include <bfft/bfft.h>

#include "detail/bruun_kernel.hpp"
#include <cmath>
#include <cstddef>
#include <new>

static_assert(sizeof(bfft_complex) == sizeof(bruun::complex_t),
              "bfft_complex / bruun::complex_t size mismatch");
static_assert(alignof(bfft_complex) == alignof(bruun::complex_t),
              "bfft_complex / bruun::complex_t alignment mismatch");
static_assert(offsetof(bfft_complex, re) == offsetof(bruun::complex_t, re),
              "bfft_complex / bruun::complex_t re offset mismatch");
static_assert(offsetof(bfft_complex, im) == offsetof(bruun::complex_t, im),
              "bfft_complex / bruun::complex_t im offset mismatch");
static_assert(sizeof(bfft_complex_f32) == sizeof(bruun::complex_f32_t),
              "bfft_complex_f32 / bruun::complex_f32_t size mismatch");
static_assert(alignof(bfft_complex_f32) == alignof(bruun::complex_f32_t),
              "bfft_complex_f32 / bruun::complex_f32_t alignment mismatch");
static_assert(offsetof(bfft_complex_f32, re) == offsetof(bruun::complex_f32_t, re),
              "bfft_complex_f32 / bruun::complex_f32_t re offset mismatch");
static_assert(offsetof(bfft_complex_f32, im) == offsetof(bruun::complex_f32_t, im),
              "bfft_complex_f32 / bruun::complex_f32_t im offset mismatch");

struct bfft_plan {
    bruun::RFFT impl;
};

struct bfft_workspace {
    size_t n;
    bruun::heap_array<double> work;
};

namespace {

bruun::complex_t* as_bruun_complex(bfft_complex* value) {
    return reinterpret_cast<bruun::complex_t*>(value);
}

const bruun::complex_t* as_bruun_complex(const bfft_complex* value) {
    return reinterpret_cast<const bruun::complex_t*>(value);
}

bruun::complex_f32_t* as_bruun_complex_f32(bfft_complex_f32* value) {
    return reinterpret_cast<bruun::complex_f32_t*>(value);
}

const bruun::complex_f32_t* as_bruun_complex_f32(const bfft_complex_f32* value) {
    return reinterpret_cast<const bruun::complex_f32_t*>(value);
}

bool missing_plan(const bfft_plan* plan) {
    return plan == nullptr;
}

bool not_pow2_plan(const bfft_plan* plan) {
    return plan == nullptr;
}

bool missing_ptr(const void* ptr) {
    return ptr == nullptr;
}

bfft_status guard_binary(const bfft_plan* plan, const void* a, const void* b) {
    if (missing_plan(plan) || missing_ptr(a) || missing_ptr(b)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    return BFFT_OK;
}





} // namespace

const char* bfft_version_string(void) {
    return "1.0";
}
