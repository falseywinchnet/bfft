#ifndef BFFT_MEYER_H
#define BFFT_MEYER_H

#include <bfft/bfft.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Meyer G-norm cartoon + texture decomposer (transport geometry fusion
   descent).  Solves the Aujol/Gilles-Osher two-projector alternation

       u <- ROF(f - v, lambda)        (one warm Split Bregman sweep)
       v <- (f - u) - ROF(f - u, 1/mu)  (one warm Split Bregman sweep)

   for a fixed number of passes against persistent Bregman states, then
   splits the texture layer into three scale bands along the ratio-4 rung
   ladder {mu, mu/4, mu/16} by independent ROF solves (fresh states per
   rung; Bregman states are eta- and c-scaled and must never cross rungs).

   Geometry: periodic boundaries, forward-difference TV, FFT-diagonal
   u-solve.  Height and width must each be a power of two >= 8.  All
   transforms run through the library's own real-FFT plans (row plan of
   size width, column plan of size height); spectra of f, u and the
   texture-side ROF survivor w are maintained across passes so each sweep
   costs exactly one forward and one inverse 2-D transform.

   Output: five height*width arrays with

       cartoon + band_coarse + band_mid + band_fine = u + v
       cartoon = u + s0   (s0 = coarsest rung survivor of v)
       texture = v        (= band sum + s0; f - u - v is the model residual)
*/

typedef struct bfft_meyer_plan bfft_meyer_plan;

/* Create a decomposer plan.  height, width: powers of two >= 8.
   lam: cartoon fidelity (Gilles lambda, e.g. 0.05 for [0,255] images).
   mu: texture G-ball radius (e.g. 40).  passes: outer TGFD passes
   (e.g. 64).  rung_sweeps: max Split Bregman sweeps per ladder rung; each
   rung also stops early when the relative iterate change drops below
   rung_tol (pass 0 to disable the early stop).  threads: worker lanes for
   the parallel stages (rows, columns, shrink, solves); 0 selects a
   hardware default.  Every lane owns its own transform plans and work
   buffers, and outputs are bit-identical for every thread count. */
bfft_status bfft_meyer_plan_create(size_t height, size_t width,
                                   double lam, double mu,
                                   int passes, int rung_sweeps,
                                   double rung_tol, int threads,
                                   bfft_meyer_plan** plan);

/* Destroy a plan.  Passing NULL is allowed. */
void bfft_meyer_plan_destroy(bfft_meyer_plan* plan);

/* Plan metadata.  Return 0 for a NULL plan. */
size_t bfft_meyer_plan_height(const bfft_meyer_plan* plan);
size_t bfft_meyer_plan_width(const bfft_meyer_plan* plan);

/* Run the model decomposition alone: cartoon = u, texture = v, exactly
   the pair produced by the Gilles-Osher alternation, with no ladder.
   image and both outputs are height*width doubles, row-major,
   non-aliasing.  Note that bfft_meyer_decompose reports a different
   cartoon (u plus the ladder's coarsest rung survivor) so that its
   cartoon and three bands sum to u + v. */
bfft_status bfft_meyer_split(bfft_meyer_plan* plan,
                             const double* image,
                             double* cartoon,
                             double* texture);

/* Run the decomposition.  image and the five outputs are height*width
   doubles, row-major, non-aliasing.  The plan's internal state is reset on
   every call; a plan may be reused for any number of images of its size. */
bfft_status bfft_meyer_decompose(bfft_meyer_plan* plan,
                                 const double* image,
                                 double* cartoon,
                                 double* texture,
                                 double* band_coarse,
                                 double* band_mid,
                                 double* band_fine);

#ifdef __cplusplus
}
#endif

#endif
