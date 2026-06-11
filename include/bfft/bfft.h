#ifndef BFFT_BFFT_H
#define BFFT_BFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BFFT_VERSION_MAJOR 0
#define BFFT_VERSION_MINOR 1
#define BFFT_VERSION_PATCH 0

typedef struct bfft_plan bfft_plan;

typedef struct bfft_complex {
    double re;
    double im;
} bfft_complex;

typedef enum bfft_status {
    BFFT_OK = 0,
    BFFT_ERROR_INVALID_ARGUMENT = 1,
    BFFT_ERROR_ALLOCATION = 2,
    BFFT_ERROR_INTERNAL = 3
} bfft_status;

typedef enum bfft_layout {
    BFFT_LAYOUT_STANDARD = 0,
    BFFT_LAYOUT_NATIVE = 1,
    BFFT_LAYOUT_RESIDUES = 2
} bfft_layout;

const char* bfft_version_string(void);
const char* bfft_backend_name(void);
const char* bfft_status_string(bfft_status status);

bfft_status bfft_plan_create(size_t n, bfft_plan** plan);
void bfft_plan_destroy(bfft_plan* plan);

size_t bfft_plan_size(const bfft_plan* plan);
size_t bfft_plan_bins(const bfft_plan* plan);
size_t bfft_plan_work_size(const bfft_plan* plan);
size_t bfft_plan_native_scratch_size(const bfft_plan* plan);
const char* bfft_plan_standard_policy(const bfft_plan* plan);

bfft_status bfft_forward(const bfft_plan* plan,
                         const double* input,
                         bfft_complex* output,
                         double* work,
                         bfft_complex* native_scratch);

bfft_status bfft_forward_native(const bfft_plan* plan,
                                const double* input,
                                bfft_complex* output,
                                double* work);

bfft_status bfft_inverse(const bfft_plan* plan,
                         const bfft_complex* input,
                         double* output);

bfft_status bfft_inverse_native(const bfft_plan* plan,
                                const bfft_complex* input,
                                double* output);

bfft_status bfft_forward_residues(const bfft_plan* plan,
                                  const double* input,
                                  double* residues);

bfft_status bfft_inverse_residues(const bfft_plan* plan,
                                  double* residues_signal);

bfft_status bfft_native_to_standard(const bfft_plan* plan,
                                    const bfft_complex* native_input,
                                    bfft_complex* standard_output);

bfft_status bfft_standard_to_native(const bfft_plan* plan,
                                    const bfft_complex* standard_input,
                                    bfft_complex* native_output);

size_t bfft_filter_size(const bfft_plan* plan);

bfft_status bfft_residue_filter_from_standard(const bfft_plan* plan,
                                              const bfft_complex* standard_response,
                                              double* residue_filter);

bfft_status bfft_residue_filter_from_real(const bfft_plan* plan,
                                          const double* real_response,
                                          double* residue_filter);

bfft_status bfft_apply_residue_filter(const bfft_plan* plan,
                                      double* residues,
                                      const double* residue_filter);

bfft_status bfft_filter_signal(const bfft_plan* plan,
                               const double* input,
                               const double* residue_filter,
                               double* output);

#ifdef __cplusplus
}
#endif

#endif
