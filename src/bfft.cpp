#include <bfft/bfft.h>

#include "detail/bruun_kernel.hpp"
#include "detail/experimental_radix4_rfft_kernel.hpp"
#include <cmath>
#include <cstddef>
#include <new>

#if defined(BFFT_USE_DIT_RFFT) || defined(BFFT_USE_DIT_RFFT_FORWARD)
#define BFFT_ACTIVE_RFFT_DIT_FORWARD 1
#else
#define BFFT_ACTIVE_RFFT_DIT_FORWARD 0
#endif

#if defined(BFFT_USE_DIT_RFFT_INVERSE)
#define BFFT_ACTIVE_RFFT_DIT_INVERSE 1
#else
#define BFFT_ACTIVE_RFFT_DIT_INVERSE 0
#endif

#if BFFT_ACTIVE_RFFT_DIT_FORWARD || BFFT_ACTIVE_RFFT_DIT_INVERSE
#define BFFT_ACTIVE_RFFT_DIT 1
#else
#define BFFT_ACTIVE_RFFT_DIT 0
#endif

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
    bruun::DIF_RFFT_kernel dif;
#if BFFT_ACTIVE_RFFT_DIT
    bruun::DIT_RFFT_kernel dit;
#endif
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

const char* bfft_backend_name(void) {
    return bruun::simd_backend_name();
}

const char* bfft_status_string(bfft_status status) {
    if (status == BFFT_OK) {
        return "ok";
    }
    if (status == BFFT_ERROR_INVALID_ARGUMENT) {
        return "invalid argument";
    }
    if (status == BFFT_ERROR_ALLOCATION) {
        return "allocation failed";
    }
    return "internal error";
}

bfft_status bfft_plan_create(size_t n, bfft_plan** plan) {
    if (plan == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    *plan = nullptr;
    if (n > static_cast<size_t>(2147483647)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    const auto ni = static_cast<int>(n);
    if (ni < 4 || (ni & (ni - 1)) != 0) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    bfft_plan* p = new (std::nothrow) bfft_plan;
    if (!p) {
        return BFFT_ERROR_ALLOCATION;
    }
    if (!p->dif.init(ni)) {
        delete p;
        return BFFT_ERROR_ALLOCATION;
    }
#if BFFT_ACTIVE_RFFT_DIT
    if (!p->dit.init(ni)) {
        delete p;
        return BFFT_ERROR_ALLOCATION;
    }
#endif
    *plan = p;
    return BFFT_OK;
}

void bfft_plan_destroy(bfft_plan* plan) {
    delete plan;
}

size_t bfft_plan_size(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return 0;
    }
    return static_cast<size_t>(plan->dif.size());
}

size_t bfft_plan_bins(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return 0;
    }
    return static_cast<size_t>(plan->dif.bins());
}

size_t bfft_plan_work_size(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return 0;
    }
#if BFFT_ACTIVE_RFFT_DIT
    const size_t dif_work = static_cast<size_t>(plan->dif.work_size());
    const size_t dit_work = static_cast<size_t>(plan->dit.work_size());
    return dif_work > dit_work ? dif_work : dit_work;
#else
    return static_cast<size_t>(plan->dif.work_size());
#endif
}

size_t bfft_plan_work_size_f32(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return 0;
    }
    return static_cast<size_t>(plan->dif.work_size_f32());
}

size_t bfft_plan_native_scratch_size(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return 0;
    }
    return static_cast<size_t>(plan->dif.native_scratch_size());
}

bfft_status bfft_workspace_create(const bfft_plan* plan, bfft_workspace** workspace) {
    if (missing_plan(plan) || workspace == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    *workspace = nullptr;
    bfft_workspace* ws = new (std::nothrow) bfft_workspace;
    if (!ws) {
        return BFFT_ERROR_ALLOCATION;
    }
    ws->n = bfft_plan_size(plan);
    if (!ws->work.resize(bfft_plan_work_size(plan))) {
        delete ws;
        return BFFT_ERROR_ALLOCATION;
    }
    *workspace = ws;
    return BFFT_OK;
}

void bfft_workspace_destroy(bfft_workspace* workspace) {
    delete workspace;
}

const char* bfft_plan_standard_policy(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return "invalid-plan";
    }
#if BFFT_ACTIVE_RFFT_DIT
    if (BFFT_ACTIVE_RFFT_DIT_FORWARD && BFFT_ACTIVE_RFFT_DIT_INVERSE) {
        return "dit-forward-dit-inverse";
    }
    if (BFFT_ACTIVE_RFFT_DIT_FORWARD) {
        return "dit-forward-dif-inverse";
    }
    return "dif-forward-dit-inverse";
#else
    return plan->dif.standard_output_policy_name();
#endif
}

bfft_status bfft_forward(const bfft_plan* plan,
                         const double* input,
                         bfft_complex* output,
                         double* work,
                         bfft_complex* native_scratch) {
    if (missing_plan(plan) || missing_ptr(input) || missing_ptr(output)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
#if BFFT_ACTIVE_RFFT_DIT_FORWARD
    (void)native_scratch;
    if (plan->dit.size() >= 16) {
        plan->dit.forward_simd(input, as_bruun_complex(output), work);
    } else {
        if (!plan->dif.standard_output_uses_two_phase() && missing_ptr(native_scratch)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        plan->dif.forward_standard(input, as_bruun_complex(output), work, as_bruun_complex(native_scratch));
    }
#else
    if (!plan->dif.standard_output_uses_two_phase() && missing_ptr(native_scratch)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_standard(input, as_bruun_complex(output), work, as_bruun_complex(native_scratch));
#endif
    return BFFT_OK;
}

bfft_status bfft_forward_native(const bfft_plan* plan,
                                const double* input,
                                bfft_complex* output,
                                double* work) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_native(input, as_bruun_complex(output), work);
    return BFFT_OK;
}

bfft_status bfft_forward_native_workspace(const bfft_plan* plan,
                                          bfft_workspace* workspace,
                                          const double* input,
                                          bfft_complex* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK || workspace == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (workspace->n != bfft_plan_size(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_native_aligned_workspace(input, as_bruun_complex(output), workspace->work.data());
    return BFFT_OK;
}

bfft_status bfft_forward_f32(const bfft_plan* plan,
                             const float* input,
                             bfft_complex_f32* output,
                             float* work,
                             bfft_complex_f32* native_scratch) {
    if (missing_plan(plan) || missing_ptr(input) || missing_ptr(output)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_standard_f32(input,
                                    as_bruun_complex_f32(output),
                                    work,
                                    as_bruun_complex_f32(native_scratch));
    return BFFT_OK;
}

bfft_status bfft_forward_native_f32(const bfft_plan* plan,
                                    const float* input,
                                    bfft_complex_f32* output,
                                    float* work) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_native_f32(input, as_bruun_complex_f32(output), work);
    return BFFT_OK;
}

bfft_status bfft_forward_magnitude(const bfft_plan* plan,
                                   const double* input,
                                   double* magnitudes,
                                   double* work) {
    bfft_status status = guard_binary(plan, input, magnitudes);
    if (status != BFFT_OK) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_magnitude(input, magnitudes, work);
    return BFFT_OK;
}

bfft_status bfft_forward_magnitude_f32(const bfft_plan* plan,
                                       const float* input,
                                       float* magnitudes,
                                       float* work) {
    bfft_status status = guard_binary(plan, input, magnitudes);
    if (status != BFFT_OK) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_magnitude_f32(input, magnitudes, work);
    return BFFT_OK;
}

bfft_status bfft_forward_mag_phase(const bfft_plan* plan,
                                   const double* input,
                                   bfft_complex* output,
                                   double* work) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_mag_phase(input, as_bruun_complex(output), work);
    return BFFT_OK;
}

bfft_status bfft_forward_mag_phase_f32(const bfft_plan* plan,
                                       const float* input,
                                       bfft_complex_f32* output,
                                       float* work) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_ptr(work)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.forward_mag_phase_f32(input, as_bruun_complex_f32(output), work);
    return BFFT_OK;
}

bfft_status bfft_inverse(const bfft_plan* plan,
                         const bfft_complex* input,
                         double* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return status;
    }
#if BFFT_ACTIVE_RFFT_DIT_INVERSE
    if (plan->dit.size() >= 16) {
        bruun::heap_array<double> work;
        if (!work.resize(static_cast<size_t>(plan->dit.work_size()))) {
            return BFFT_ERROR_ALLOCATION;
        }
        plan->dit.inverse_simd(as_bruun_complex(input), output, work.data());
    } else {
        plan->dif.inverse(as_bruun_complex(input), output);
    }
#else
    plan->dif.inverse(as_bruun_complex(input), output);
#endif
    return BFFT_OK;
}

bfft_status bfft_inverse_f32(const bfft_plan* plan,
                             const bfft_complex_f32* input,
                             float* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.inverse_f32(as_bruun_complex_f32(input), output);
    return BFFT_OK;
}

bfft_status bfft_inverse_mag_phase(const bfft_plan* plan,
                                   const bfft_complex* input,
                                   double* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.inverse_mag_phase(as_bruun_complex(input), output);
    return BFFT_OK;
}

bfft_status bfft_inverse_mag_phase_f32(const bfft_plan* plan,
                                       const bfft_complex_f32* input,
                                       float* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.inverse_mag_phase_f32(as_bruun_complex_f32(input), output);
    return BFFT_OK;
}

bfft_status bfft_inverse_native(const bfft_plan* plan,
                                const bfft_complex* input,
                                double* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.inverse_native(as_bruun_complex(input), output);
    return BFFT_OK;
}

bfft_status bfft_inverse_native_f32(const bfft_plan* plan,
                                    const bfft_complex_f32* input,
                                    float* output) {
    bfft_status status = guard_binary(plan, input, output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.inverse_native_f32(as_bruun_complex_f32(input), output);
    return BFFT_OK;
}

bfft_status bfft_forward_residues(const bfft_plan* plan,
                                  const double* input,
                                  double* residues) {
    if (not_pow2_plan(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    bfft_status status = guard_binary(plan, input, residues);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.forward_residues(input, residues);
    return BFFT_OK;
}

bfft_status bfft_inverse_residues(const bfft_plan* plan,
                                  double* residues_signal) {
    if (not_pow2_plan(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_plan(plan) || missing_ptr(residues_signal)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.inverse_residues(residues_signal);
    return BFFT_OK;
}

bfft_status bfft_native_to_standard(const bfft_plan* plan,
                                    const bfft_complex* native_input,
                                    bfft_complex* standard_output) {
    bfft_status status = guard_binary(plan, native_input, standard_output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.native_to_standard_complex(as_bruun_complex(native_input), as_bruun_complex(standard_output));
    return BFFT_OK;
}

bfft_status bfft_standard_to_native(const bfft_plan* plan,
                                    const bfft_complex* standard_input,
                                    bfft_complex* native_output) {
    bfft_status status = guard_binary(plan, standard_input, native_output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.standard_to_native_complex(as_bruun_complex(standard_input), as_bruun_complex(native_output));
    return BFFT_OK;
}

bfft_status bfft_native_to_standard_f32(const bfft_plan* plan,
                                        const bfft_complex_f32* native_input,
                                        bfft_complex_f32* standard_output) {
    bfft_status status = guard_binary(plan, native_input, standard_output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.native_to_standard_complex_f32(as_bruun_complex_f32(native_input),
                                             as_bruun_complex_f32(standard_output));
    return BFFT_OK;
}

bfft_status bfft_standard_to_native_f32(const bfft_plan* plan,
                                        const bfft_complex_f32* standard_input,
                                        bfft_complex_f32* native_output) {
    bfft_status status = guard_binary(plan, standard_input, native_output);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.standard_to_native_complex_f32(as_bruun_complex_f32(standard_input),
                                             as_bruun_complex_f32(native_output));
    return BFFT_OK;
}

size_t bfft_filter_size(const bfft_plan* plan) {
    if (missing_plan(plan)) {
        return 0;
    }
    return static_cast<size_t>(plan->dif.filter_size());
}

bfft_status bfft_residue_filter_from_standard(const bfft_plan* plan,
                                              const bfft_complex* standard_response,
                                              double* residue_filter) {
    if (not_pow2_plan(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    bfft_status status = guard_binary(plan, standard_response, residue_filter);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.residue_filter_from_standard(as_bruun_complex(standard_response), residue_filter);
    return BFFT_OK;
}

bfft_status bfft_residue_filter_from_real(const bfft_plan* plan,
                                          const double* real_response,
                                          double* residue_filter) {
    if (not_pow2_plan(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    bfft_status status = guard_binary(plan, real_response, residue_filter);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.residue_filter_from_real(real_response, residue_filter);
    return BFFT_OK;
}

bfft_status bfft_apply_residue_filter(const bfft_plan* plan,
                                      double* residues,
                                      const double* residue_filter) {
    if (not_pow2_plan(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    bfft_status status = guard_binary(plan, residues, residue_filter);
    if (status != BFFT_OK) {
        return status;
    }
    plan->dif.apply_residue_filter(residues, residue_filter);
    return BFFT_OK;
}

bfft_status bfft_filter_signal(const bfft_plan* plan,
                               const double* input,
                               const double* residue_filter,
                               double* output) {
    if (not_pow2_plan(plan)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (missing_plan(plan) || missing_ptr(input) || missing_ptr(residue_filter) || missing_ptr(output)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    plan->dif.filter_signal(input, residue_filter, output);
    return BFFT_OK;
}
