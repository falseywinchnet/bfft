#ifndef BFFT_BDCT_H
#define BFFT_BDCT_H

/* Public C ABI for the BDCT kernel: the native half-bin-shifted real transform
   and the DCT-IV built on it. This header lives alongside <bfft/bfft.h> and
   reuses its complex and status types. */

#include <bfft/bfft.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque half-shift transform plan. Create with bdct_plan_create and destroy
   with bdct_plan_destroy. */
typedef struct bdct_plan bdct_plan;

/* Opaque DCT-IV plan. */
typedef struct bdct_dctiv_plan bdct_dctiv_plan;

/* SIMD backend selected by the build target (shared with the BFFT kernel). */
const char* bdct_backend_name(void);

/* -- Half-shift transform --------------------------------------------------

   Forward maps N real samples to N/2 packed complex bins:

       H[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*(k+1/2)*n/N),   k = 0..N/2-1.

   The upper half is recovered by H[N-1-k] = conj(H[k]). Inverse maps the N/2
   packed bins back to the N real samples exactly. */

/* Create a half-shift plan for a power-of-two transform size N >= 2. */
bfft_status bdct_plan_create(size_t n, bdct_plan** plan);

/* Destroy a plan. Passing NULL is allowed. */
void bdct_plan_destroy(bdct_plan* plan);

/* Transform size N, and the packed bin count N/2. Return 0 for a NULL plan. */
size_t bdct_plan_size(const bdct_plan* plan);
size_t bdct_plan_bins(const bdct_plan* plan);

/* Double-precision forward and inverse. input/output have N doubles; the packed
   spectrum has bdct_plan_bins(plan) complex values. */
bfft_status bdct_forward(const bdct_plan* plan,
                         const double* input,
                         bfft_complex* output);
bfft_status bdct_inverse(const bdct_plan* plan,
                         const bfft_complex* input,
                         double* output);

/* Single-precision forward and inverse. */
bfft_status bdct_forward_f32(const bdct_plan* plan,
                             const float* input,
                             bfft_complex_f32* output);
bfft_status bdct_inverse_f32(const bdct_plan* plan,
                             const bfft_complex_f32* input,
                             float* output);

/* -- DCT-IV ----------------------------------------------------------------

   Unnormalized orthogonal DCT-IV:

       C[k] = sum_{n=0}^{N-1} x[n] * cos( (pi/N) * (k+1/2) * (n+1/2) ).

   Forward and inverse are real-to-real of length N. DCT-IV is its own inverse
   up to a 2/N scale, which bdct_dctiv_inverse applies. */

/* Create a DCT-IV plan for a power-of-two transform size N >= 2. */
bfft_status bdct_dctiv_plan_create(size_t n, bdct_dctiv_plan** plan);

/* Destroy a DCT-IV plan. Passing NULL is allowed. */
void bdct_dctiv_plan_destroy(bdct_dctiv_plan* plan);

/* Transform size N, or 0 for a NULL plan. */
size_t bdct_dctiv_plan_size(const bdct_dctiv_plan* plan);

/* Double-precision forward (unnormalized) and inverse (scaled by 2/N). input
   and output have N doubles. */
bfft_status bdct_dctiv_forward(const bdct_dctiv_plan* plan,
                               const double* input,
                               double* output);
bfft_status bdct_dctiv_inverse(const bdct_dctiv_plan* plan,
                               const double* input,
                               double* output);

/* Single-precision forward and inverse. */
bfft_status bdct_dctiv_forward_f32(const bdct_dctiv_plan* plan,
                                   const float* input,
                                   float* output);
bfft_status bdct_dctiv_inverse_f32(const bdct_dctiv_plan* plan,
                                   const float* input,
                                   float* output);

#ifdef __cplusplus
}
#endif

#endif
