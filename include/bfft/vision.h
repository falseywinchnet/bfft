#ifndef BFFT_VISION_H
#define BFFT_VISION_H

#include <bfft/bfft.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
   Exact kernels for a measured two-owner partition-of-unity image model.

   These routines do not search, generate, or reuse candidate graphs.  The
   caller supplies the owner/runner assignment measured for the current image
   state and the block slots of that one graph.

   Arrays are contiguous and row-major.  The image model has a caller-selected
   basis width and exactly three output channels.
*/

/*
   Assemble G = A^T A and h = A^T target directly, without materializing A.

   For pixel p, let

       u = owner_weight[p]  * owner_basis[p, :]
       v = runner_weight[p] * runner_basis[p, :].

   The pixel contributes uu^T to its owner diagonal block, and, when
   has_runner[p] is nonzero, vv^T to the runner diagonal block plus uv^T and
   vu^T to the supplied directed off-diagonal block slots.  The right-hand
   side receives u*target and v*target for each of three channels.

   owner and runner contain cell indices in [0, cell_count).  runner must be a
   valid fallback index even when has_runner is zero.  diagonal_block has
   cell_count entries.  owner_runner_block and runner_owner_block have
   pixel_count entries; their values are read only where has_runner is nonzero.

   normal_blocks is overwritten and has
       normal_block_count * basis_width * basis_width doubles.
   rhs is overwritten and has
       cell_count * basis_width * 3 doubles.
*/
bfft_status bfft_vision_assemble_normal(
    size_t pixel_count,
    size_t cell_count,
    size_t basis_width,
    size_t normal_block_count,
    const int32_t* owner,
    const int32_t* runner,
    const uint8_t* has_runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* target,
    const int64_t* diagonal_block,
    const int64_t* owner_runner_block,
    const int64_t* runner_owner_block,
    double* normal_blocks,
    double* rhs);

/*
   Render the two affine predictions and their partition-of-unity blend:

       owner_prediction[p, c] =
           dot(coeff[owner[p], :, c], owner_basis[p, :])
       runner_prediction[p, c] =
           dot(coeff[runner[p], :, c], runner_basis[p, :])
       field[p, c] =
           owner_weight[p] * owner_prediction[p, c]
         + runner_weight[p] * runner_prediction[p, c].

   coeff has cell_count * basis_width * 3 doubles.  Each output is overwritten
   and has pixel_count * 3 doubles.  Output arrays must not overlap inputs or
   one another.
*/
bfft_status bfft_vision_render_affine(
    size_t pixel_count,
    size_t cell_count,
    size_t basis_width,
    const int32_t* owner,
    const int32_t* runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* coeff,
    double* owner_prediction,
    double* runner_prediction,
    double* field);

/*
   Find the best bounded residual ridge independently inside every measured
   owner cell.  For each supplied angle and threshold bin, the score is the
   channel-weighted squared signed residual sum divided by cell mass.  The
   first angle and first bin win exact ties.

   residual is pixel_count x 3.  dx and dy are pixel offsets from their
   measured owner site.  angle_cos and angle_sin have angle_count entries.
   channel_weight has three entries.  score, best_angle, and best_bin each
   have cell_count entries and are overwritten.
*/
bfft_status bfft_vision_scan_residual_ridges(
    size_t pixel_count,
    size_t cell_count,
    size_t angle_count,
    size_t bin_count,
    double spacing,
    double span,
    const int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* dx,
    const double* dy,
    const double* angle_cos,
    const double* angle_sin,
    const double* channel_weight,
    double* score,
    int32_t* best_angle,
    int32_t* best_bin);

/* Compact multi-support affine operator used by the HD matrix-free solver.
   Each sample contributes

       weight * (c0 + basis_x*c1 + basis_y*c2)

   from one cell to one pixel.  rows and sites have sample_count entries.
   The normal application overwrites both pixel_scratch[pixel_count] and
   output[3*cell_count]. */
bfft_status bfft_vision_support_forward(
    size_t sample_count, size_t pixel_count, size_t cell_count,
    const int32_t* rows, const int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* coefficient,
    double* pixel);

bfft_status bfft_vision_support_transpose(
    size_t sample_count, size_t pixel_count, size_t cell_count,
    const int32_t* rows, const int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* pixel,
    double* coefficient);

bfft_status bfft_vision_support_normal_apply(
    size_t sample_count, size_t pixel_count, size_t cell_count,
    const int32_t* rows, const int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* coefficient,
    double* pixel_scratch, double* output);

#ifdef __cplusplus
}
#endif

#endif
