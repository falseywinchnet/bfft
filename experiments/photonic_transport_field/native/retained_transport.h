#ifndef PHOTONIC_RETAINED_TRANSPORT_H
#define PHOTONIC_RETAINED_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define PFT_RESTRICT __restrict
#define PFT_EXPORT __declspec(dllexport)
#else
#define PFT_RESTRICT __restrict__
#define PFT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    PFT_BLOCK_SOURCE_CONTIGUOUS = 1u,
    PFT_BLOCK_RECEIVER_CONTIGUOUS = 2u
};

typedef struct pft_retained_block_f64 {
    uint32_t source_offset;
    uint32_t source_count;
    uint32_t receiver_offset;
    uint32_t receiver_count;
    uint32_t source_base;
    uint32_t receiver_base;
    uint32_t flags;
    uint32_t reserved;
    double optical_depth;
    double extinction;
    double linear_diffusive_extinction;
    double circular_diffusive_extinction;
} pft_retained_block_f64;

typedef struct pft_retained_stats {
    uint64_t block_applications;
    uint64_t gathered_source_coefficients;
    uint64_t scattered_receiver_coefficients;
    uint64_t contiguous_source_blocks;
    uint64_t contiguous_receiver_blocks;
} pft_retained_stats;

typedef struct pft_retained_plan pft_retained_plan;
typedef struct pft_retained_mode_plan pft_retained_mode_plan;

/* Copy and validate immutable topology once. */
PFT_EXPORT int pft_retained_plan_create_f64(
    size_t node_count,
    size_t block_count,
    const pft_retained_block_f64* PFT_RESTRICT blocks,
    size_t source_coefficient_count,
    const uint32_t* PFT_RESTRICT source_nodes,
    const double* PFT_RESTRICT source_factors,
    size_t receiver_coefficient_count,
    const uint32_t* PFT_RESTRICT receiver_nodes,
    const double* PFT_RESTRICT receiver_factors,
    pft_retained_plan** PFT_RESTRICT plan);

PFT_EXPORT void pft_retained_plan_destroy(pft_retained_plan* plan);

PFT_EXPORT int pft_retained_plan_apply_f64(
    const pft_retained_plan* plan,
    size_t mode_count,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats);

/* Precompute block-by-mode extinction for a stable optical basis. */
PFT_EXPORT int pft_retained_mode_plan_create_f64(
    const pft_retained_plan* plan,
    size_t mode_count,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    pft_retained_mode_plan** PFT_RESTRICT mode_plan);

PFT_EXPORT void pft_retained_mode_plan_destroy(pft_retained_mode_plan* mode_plan);

PFT_EXPORT int pft_retained_mode_plan_apply_f64(
    const pft_retained_mode_plan* mode_plan,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats);

/*
 * Apply one complete retained-rank wave.
 *
 * Powers and output are mode-major arrays of shape [mode_count, node_count].
 * The block operator is v (u^T power), followed by one mode-dependent
 * homogeneous-medium extinction per gathered mode. The function never
 * materializes a source-by-receiver matrix and performs no heap allocation.
 * Scratch must contain at least mode_count doubles. Input and output may not
 * alias; every public pointer is restrict-qualified to expose that promise to
 * the compiler.
 */
PFT_EXPORT int pft_retained_apply_f64(
    size_t node_count,
    size_t mode_count,
    size_t block_count,
    const pft_retained_block_f64* PFT_RESTRICT blocks,
    const uint32_t* PFT_RESTRICT source_nodes,
    const double* PFT_RESTRICT source_factors,
    const uint32_t* PFT_RESTRICT receiver_nodes,
    const double* PFT_RESTRICT receiver_factors,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats);

PFT_EXPORT const char* pft_retained_backend(void);

#ifdef __cplusplus
}
#endif

#endif
