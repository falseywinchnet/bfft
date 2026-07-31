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

   Geometry: periodic boundaries by default, forward-difference TV,
   FFT-diagonal u-solve.  Height and width must each be a power of two >= 8
   for solver 0.  A one-axis FACR solver may instead leave its swept axis
   unpadded.  All
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
typedef void (*bfft_meyer_trace_visitor)(int pass,
                                         const double* cartoon,
                                         const double* texture,
                                         size_t count,
                                         void* user);

/* Create a decomposer plan.  For the default solver, height and width are
   powers of two >= 8.  To select a FACR solver immediately after creation,
   one dimension may instead be any value >= 2, provided the other remains
   a power of two >= 8.
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

/* Change only the number of outer TGFD passes.  This controls
   bfft_meyer_split_legacy, trace, and decomposition; the default fixed-cost
   spectral bfft_meyer_split does not consume it.  This does not rebuild the
   transform plans, worker pool, symbols, or image-sized scratch buffers, so
   realtime callers may adjust quality without allocating.  The next split
   or decomposition uses the new value. */
bfft_status bfft_meyer_plan_set_passes(bfft_meyer_plan* plan, int passes);

/* Select the screened-Poisson solver:
     0 = full 2-D spectral solve (default; both axes must be powers of two)
     1 = periodically sweep the worse-padded axis with FACR; if neither
         axis needs padding, retain the faster full spectral path
     2 = FACR with Neumann boundaries on the swept axis (output changes)
   Modes 1 and 2 automatically sweep the non-power-of-two axis, or height
   on a power-of-two shape.  Changing the solver may rebuild plan-owned
   factors and scratch storage. */
bfft_status bfft_meyer_plan_set_solver(bfft_meyer_plan* plan, int mode);
int bfft_meyer_plan_solver(const bfft_meyer_plan* plan);

/* Default two-product split. Uses the fixed-cost jump-measure construction
   with validated virtual depth 8 on the full periodic spectral solver and
   its one-axis periodic FACR realization.  The FACR realization uses the
   same oriented jump/Hodge construction, an O(N) four-direction structural
   gate, and a two-pole approximation of the virtual-depth resolvent. Neumann
   solver mode 2 retains the legacy alternation because its boundary
   functional is deliberately different. cartoon + texture == image up to
   floating-point roundoff.

   image may alias cartoon or texture, which lets realtime callers reuse the
   input plane after the split. cartoon and texture must not alias each other. */
bfft_status bfft_meyer_split(bfft_meyer_plan* plan,
                             const double* image,
                             double* cartoon,
                             double* texture);

/* Explicit legacy Gilles-Osher alternation using the plan's configured pass
   count. This is retained for traces, FACR research, and reproducibility. */
bfft_status bfft_meyer_split_legacy(bfft_meyer_plan* plan,
                                    const double* image,
                                    double* cartoon,
                                    double* texture);

/* Opt-in noniterative first-pass structural conditioner.

   Reuses the source spectrum to form a four-direction symmetric-variation
   gate, predicts the first reflected Split-Bregman flux directly from the
   source gradient, and solves

       (lambda I - 2 lambda Delta) u
         = lambda f - 2 lambda div(conditioned_flux).

   The texture-side first solve is unchanged.  This always emits one pass,
   independent of the plan's configured pass count.  strength=1.5 is the
   validated default; strength must be nonnegative.  The method requires the
   full periodic spectral path.  Active FACR plans return
   BFFT_ERROR_INVALID_ARGUMENT. */
bfft_status bfft_meyer_split_conditioned_first(
    bfft_meyer_plan* plan, const double* image,
    double* cartoon, double* texture, double strength);

/* Opt-in finite-state Meyer preconditioner.

   Takes virtual_passes linear texture-interior cartoon steps as one spectral
   multiplier, blocks that jump at the symmetric-variation structural gate,
   performs one conditioned cartoon solve, then Hodge-lifts the proposed
   texture through a scalar Poisson solve.  One deterministic transverse
   tangent-frame route redistributes longitudinal capacity overload without
   changing divergence.  The routed field is projected pointwise onto the
   plan's radius-mu disk before its divergence is returned, so texture is
   constructively G_mu-feasible and
   cartoon + texture == image up to floating-point roundoff.

   virtual_passes=8 and gate_power=8 are the synthetic-truth research
   defaults.  Both integer arguments must be in [1,64].  This is a finite
   preconditioner, not an outer iteration count or runtime candidate scan.
   The method requires the full periodic spectral path; active FACR plans
   return BFFT_ERROR_INVALID_ARGUMENT. */
bfft_status bfft_meyer_split_preconditioned(
    bfft_meyer_plan* plan, const double* image,
    double* cartoon, double* texture, double strength,
    int virtual_passes, int gate_power);

/* Fixed-cost jump-measure split.

   Estimates discontinuities as an oriented Hodge measure, removes one
   feed-forward carrier estimate before rebuilding that measure, and routes
   the remaining oscillation through one transverse G_mu capacity correction.
   The public result remains exactly two products:

       texture = (I-H_u) jump_potential + routed_oscillation
       cartoon = image - texture.

   Thus cartoon retains objects with continuous first-resolvent boundaries;
   texture owns the complementary transition and material energy. There is no
   convergence loop or runtime candidate scan. virtual_passes is a spectral
   integer exponent in [1,64] (the validated default is 8). Full spectral and
   periodic FACR plans are supported; Neumann FACR plans return
   BFFT_ERROR_INVALID_ARGUMENT. image may alias either output, but the two
   outputs must not alias each other. */
bfft_status bfft_meyer_split_jump_measure(
    bfft_meyer_plan* plan, const double* image,
    double* cartoon, double* texture, int virtual_passes);

/* Run the model once and retain every intermediate outer-pass state.
   cartoon_trace and texture_trace are passes*height*width doubles in
   pass-major order.  This is equivalent to separately requesting split
   results for pass counts 1..passes, but costs only passes total sweeps
   instead of passes*(passes+1)/2. */
bfft_status bfft_meyer_split_trace(bfft_meyer_plan* plan,
                                   const double* image,
                                   double* cartoon_trace,
                                   double* texture_trace);

/* Visit every intermediate state without retaining a passes-deep output
   volume.  The cartoon and texture pointers remain valid only for the
   duration of the callback. */
bfft_status bfft_meyer_split_visit(bfft_meyer_plan* plan,
                                   const double* image,
                                   bfft_meyer_trace_visitor visitor,
                                   void* user);

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

/* Run a plain ROF (Rudin-Osher-Fatemi) solve on its own:

       smooth <- argmin_x TV(x) + (c/2) |x - image|^2

   by Split Bregman sweeps from a fresh state, with Bregman penalty eta
   (pass eta <= 0 for the ladder's convention, eta = 10*c).  Sweeps stop
   early once the relative iterate change falls below tol; pass tol = 0 to
   run all of them.  image - smooth is the ROF residual, which is what the
   G-ball projection identity turns into the texture layer.

   This is the same solver the ladder rungs use, exposed because
   recomposition effects need it: subtracting a ROF solve of the cartoon
   layer isolates the smooth illumination the flat cartoon discards.  The
   symbol table is cached, so repeated calls at fixed (c, eta) -- the video
   case -- rebuild nothing.  image and smooth are height*width doubles,
   row-major, non-aliasing. */
bfft_status bfft_meyer_rof(bfft_meyer_plan* plan,
                           const double* image,
                           double* smooth,
                           double c, double eta,
                           int sweeps, double tol);

/* Static-ROF-only one-shot Fourier/Hodge accelerator.

   Runs the same Split Bregman problem and state equations as bfft_meyer_rof,
   but after hodge_after ordinary sweeps performs one objective-checked
   longitudinal Hodge closure, projects its flux onto the Euclidean unit
   disk, and re-seats (d,b) before continuing.  This changes the trajectory,
   not the ROF target.  It is opt-in and currently requires solver mode 0
   (the full periodic spectral path); FACR and Neumann plans return
   BFFT_ERROR_INVALID_ARGUMENT. */
bfft_status bfft_meyer_rof_accelerated(bfft_meyer_plan* plan,
                                       const double* image,
                                       double* smooth,
                                       double c, double eta,
                                       int sweeps, double tol,
                                       int hodge_after);

/* Diagnostics for the most recent plain or accelerated ROF call. */
int bfft_meyer_plan_last_rof_sweeps(const bfft_meyer_plan* plan);
int bfft_meyer_plan_last_rof_hodge_applied(const bfft_meyer_plan* plan);

#ifdef __cplusplus
}
#endif

#endif
