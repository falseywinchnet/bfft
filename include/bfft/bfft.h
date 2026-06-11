#ifndef BFFT_BFFT_H
#define BFFT_BFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BFFT_VERSION_MAJOR 0
#define BFFT_VERSION_MINOR 1
#define BFFT_VERSION_PATCH 0

/* Opaque transform plan. Create with bfft_plan_create and destroy with
   bfft_plan_destroy. */
typedef struct bfft_plan bfft_plan;

/* Complex value used by the C ABI. Real-to-complex output uses N / 2 + 1
   values in standard or native spectrum order. */
typedef struct bfft_complex {
    double re;
    double im;
} bfft_complex;

/* Return status for C API calls. */
typedef enum bfft_status {
    BFFT_OK = 0,
    BFFT_ERROR_INVALID_ARGUMENT = 1,
    BFFT_ERROR_ALLOCATION = 2,
    BFFT_ERROR_INTERNAL = 3
} bfft_status;

/* Public spectrum representations. This enum is descriptive; transform calls
   select the layout by function name. */
typedef enum bfft_layout {
    BFFT_LAYOUT_STANDARD = 0,
    BFFT_LAYOUT_NATIVE = 1,
    BFFT_LAYOUT_RESIDUES = 2
} bfft_layout;

/* Library version, selected SIMD backend, and diagnostic strings. */
const char* bfft_version_string(void);
const char* bfft_backend_name(void);
const char* bfft_status_string(bfft_status status);

/* Create a reusable plan for a power-of-two transform size N >= 4. */
bfft_status bfft_plan_create(size_t n, bfft_plan** plan);

/* Destroy a plan. Passing NULL is allowed. */
void bfft_plan_destroy(bfft_plan* plan);

/* Plan metadata. Size query functions return 0 for a NULL plan. */
size_t bfft_plan_size(const bfft_plan* plan);
size_t bfft_plan_bins(const bfft_plan* plan);
size_t bfft_plan_work_size(const bfft_plan* plan);
size_t bfft_plan_native_scratch_size(const bfft_plan* plan);

/* Returns the standard-output packing policy for this plan, or "invalid-plan"
   for a NULL plan. */
const char* bfft_plan_standard_policy(const bfft_plan* plan);

/* Standard FFT-order real-to-complex forward transform.
   input has N doubles, output has bfft_plan_bins(plan) complex values, work has
   bfft_plan_work_size(plan) doubles, and native_scratch should have
   bfft_plan_native_scratch_size(plan) complex values. native_scratch may be
   NULL only when bfft_plan_standard_policy returns "two-phase-standard-pack". */
bfft_status bfft_forward(const bfft_plan* plan,
                         const double* input,
                         bfft_complex* output,
                         double* work,
                         bfft_complex* native_scratch);

/* Native-order real-to-complex forward transform. Output has
   bfft_plan_bins(plan) complex values and work has bfft_plan_work_size(plan)
   doubles. */
bfft_status bfft_forward_native(const bfft_plan* plan,
                                const double* input,
                                bfft_complex* output,
                                double* work);

/* Standard FFT-order inverse transform. input has bfft_plan_bins(plan) complex
   values and output has N doubles. */
bfft_status bfft_inverse(const bfft_plan* plan,
                         const bfft_complex* input,
                         double* output);

/* Native-order inverse transform. */
bfft_status bfft_inverse_native(const bfft_plan* plan,
                                const bfft_complex* input,
                                double* output);

/* Residue-domain transform. input has N doubles and residues has N doubles. */
bfft_status bfft_forward_residues(const bfft_plan* plan,
                                  const double* input,
                                  double* residues);

/* In-place inverse from residue coordinates back to time samples. */
bfft_status bfft_inverse_residues(const bfft_plan* plan,
                                  double* residues_signal);

/* Convert between native spectrum order and standard FFT-order bins. Both
   arrays have bfft_plan_bins(plan) complex values. */
bfft_status bfft_native_to_standard(const bfft_plan* plan,
                                    const bfft_complex* native_input,
                                    bfft_complex* standard_output);

bfft_status bfft_standard_to_native(const bfft_plan* plan,
                                    const bfft_complex* standard_input,
                                    bfft_complex* native_output);

/* Residue-domain filter size in doubles. */
size_t bfft_filter_size(const bfft_plan* plan);

/* Convert a standard FFT-order complex frequency response into a residue-domain
   filter. standard_response has bfft_plan_bins(plan) complex values and
   residue_filter has bfft_filter_size(plan) doubles. */
bfft_status bfft_residue_filter_from_standard(const bfft_plan* plan,
                                              const bfft_complex* standard_response,
                                              double* residue_filter);

/* Convert a real zero-phase response with bfft_plan_bins(plan) doubles into a
   residue-domain filter. */
bfft_status bfft_residue_filter_from_real(const bfft_plan* plan,
                                          const double* real_response,
                                          double* residue_filter);

/* Apply a residue-domain filter in place to a residue vector. Both arrays have
   bfft_filter_size(plan) doubles. */
bfft_status bfft_apply_residue_filter(const bfft_plan* plan,
                                      double* residues,
                                      const double* residue_filter);

/* Transform input to residues, apply residue_filter, and invert to output.
   input and output have N doubles and must not alias. */
bfft_status bfft_filter_signal(const bfft_plan* plan,
                               const double* input,
                               const double* residue_filter,
                               double* output);

#ifdef __cplusplus
}
#endif

#endif
