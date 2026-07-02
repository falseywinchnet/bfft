#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal radix-4 real DIT FFT experiment kernel.
//
// This header keeps the standalone NEON prototype as a reusable kernel object:
// construction owns all planning tables, callers provide input/output/work
// buffers, and the SIMD path is selected through the shared BFFT two-lane
// backend abstraction instead of including one architecture directly.
//
// ---------------------------------------------------------------------------
// The mirrored-lane walk (the SIMD path below)
// ---------------------------------------------------------------------------
// The transform is the usual real-Bruun DIT tree: seed nodes are gathered from
// the input in bit-reversed order, radix-4 merges reduce children modulo the
// real quadratic factors of z^N - 1, and each spectral bin k finally leaves the
// tree as a residue pair (a, b) with X_k = a + b * e^{-i 2 pi k / N}.
//
// The walk through that tree is chosen differently from the obvious one.
// Observe two facts about the node-index space [0, N/2):
//
//   1. Node b and node b + N/4 (top index bit) run *identical* butterfly
//      sequences with *identical* twiddles through every stage below the
//      final merge; they only meet at the top, where they sit in the
//      (q0,q2) / (q1,q3) child quarters of the last radix-4 reduction.
//   2. Their seed reads are input-adjacent: rev(b + N/4) == rev(b) + 1.
//
// So instead of packing SIMD lanes across neighbouring spectral nodes inside
// one block - which forces a deinterleave/interleave permute pair around every
// butterfly plus scalar DC/Nyquist/midpoint special cases - the two lanes of
// every vector are the two mirror trees. Work layout ("slot" = node pair):
//
//   slot s in [0, N/4):  vwork[4s+0..1] = { a(node s), a(node s + N/4) }
//                        vwork[4s+2..3] = { b(node s), b(node s + N/4) }
//
// Consequences:
//   - Seed loads are contiguous V2 loads at rev(b): the intake shuffle stays
//     frontloaded (ordered -> out-of-order reads) but is now vector-width and
//     table-halved. Seed blocks are visited in bit-reversed block order so the
//     gather decays into ~2*SB sequential comb streams instead of full-array
//     sweeps that refetch every input line; large N stops thrashing.
//   - Every merge stage below the top is pure vertical SIMD: no permutes at
//     all, twiddles broadcast with V2_SET1, and the DC/Nyquist/sqrt2 chains
//     vectorize across the mirror lanes instead of running scalar.
//   - The single unavoidable lane crossing happens exactly once, in the
//     terminal merge, where it fuses with the (a,b) -> complex projection.
//     The projection therefore costs no separate pass at any size, and its
//     four output bins per step share one {cos,sin} table vector through the
//     quarter-turn symmetries cos(pi/2 - t) = sin(t), cos(pi - t) = -cos(t).
//
// The inverse runs the same walk backwards and keeps the same single
// lane-crossing point. Standard bins deproject straight into mirrored-lane
// layout inside the terminal split (the reverse of terminal_v, fused with
// (re,im) -> (a,b) deprojection), and every split below it is lane-vertical
// with no per-butterfly permutes. The seed output transport consumes each
// completed vwork block with linear V2 loads; the permutation cost stays on
// the store side, where mirror lanes still land as adjacent time samples
// because rev(b + N/4) = rev(b) + 1. Each inverse primitive is the exact
// algebraic inverse of its forward counterpart, so the distributed 0.5 factors
// compose to exactly 1/N and inverse(forward(x)) == x, matching the bfft
// convention. The in-place inverse merge is clobber-free by the same (i, M-i)
// pairing as the forward: the parent read set of a pair equals its child write
// set.
//
// The scalar path below keeps the plain single-lane walk for reference and
// for BRUUN_LEVEL == 0 builds.

#include "bruun_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

// Interpolation-inverse fused tile depths (doubles per L1 tile), on the
// radix-4 chains 8*4^k (even-parity areas) and 16*4^k (odd-parity areas).
#ifndef BRUUN_ITP_TILE_EVEN
#define BRUUN_ITP_TILE_EVEN 128
#endif
#ifndef BRUUN_ITP_TILE_ODD
#define BRUUN_ITP_TILE_ODD 256
#endif
// Upper-sweep cache tiers (doubles): consecutive merge levels are run
// chunk-wise inside tier-sized windows so a chunk is loaded once per tier
// instead of once per level.
#ifndef BRUUN_ITP_TIER2
#define BRUUN_ITP_TIER2 32768
#endif
#ifndef BRUUN_ITP_TIER3
#define BRUUN_ITP_TIER3 0
#endif

namespace bruun {

class radix4_rfft_kernel {
public:
    radix4_rfft_kernel() noexcept
        : n_(0), half_(0), twiddle_stage_count_(0), radix2_terminal_(false) {}

    bool reset(int n) {
        if (!valid_size(n)) {
            clear();
            return false;
        }

        n_ = n;
        half_ = n >> 1;

        const int logn = ilog2_pow2(n);
        radix2_terminal_ = (logn & 1) != 0;
        const int rev_bits = logn - 1;
        if (!rev_.resize(static_cast<std::size_t>(half_))) {
            clear();
            return false;
        }
        for (int b = 0; b < half_; ++b) {
            rev_[static_cast<std::size_t>(b)] = bitrev_int(b, rev_bits);
        }

        if (!cos_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }
        if (!neg_sin_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }
        if (!inv_neg_sin_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }
        for (int k = 0; k <= half_; ++k) {
            const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(n_);
            cos_[static_cast<std::size_t>(k)] = std::cos(theta);
            neg_sin_[static_cast<std::size_t>(k)] = -std::sin(theta);
            inv_neg_sin_[static_cast<std::size_t>(k)] =
                (k == 0 || k == half_) ? 0.0 : 1.0 / neg_sin_[static_cast<std::size_t>(k)];
        }

        twiddle_stage_count_ = 0;
        int twiddle_count = 0;
        for (int s = 4; s <= n_; s <<= 1) {
            ++twiddle_stage_count_;
            const int q = s >> 2;
            if (q > 1) {
                twiddle_count += q - 1;
            }
        }

        if (!twiddle_offset_.resize(static_cast<std::size_t>(twiddle_stage_count_))) {
            clear();
            return false;
        }
        int alloc_count = twiddle_count;
        if (alloc_count == 0) {
            alloc_count = 1;
        }
        if (!twiddle_.resize(static_cast<std::size_t>(alloc_count))) {
            clear();
            return false;
        }

        int offset = 0;
        int stage = 0;
        for (int s = 4; s <= n_; s <<= 1) {
            twiddle_offset_[static_cast<std::size_t>(stage)] = offset;
            const int q = s >> 2;
            for (int k = 1; k < q; ++k) {
                const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(s);
                twiddle_[static_cast<std::size_t>(offset)] = 2.0 * std::cos(theta);
                ++offset;
            }
            ++stage;
        }

        // Inverse twiddles, parallel to twiddle_: per entry { 0.5/c, 1 - c^2 }
        // with c = 2*cos(theta), the constants of the exact pair_reduce
        // inverse. Element offsets are 2 * twiddle_offset_.
        if (!inv_twiddle_.resize(static_cast<std::size_t>(2 * alloc_count))) {
            clear();
            return false;
        }
        for (int t = 0; t < twiddle_count; ++t) {
            const double c = twiddle_[static_cast<std::size_t>(t)];
            inv_twiddle_[static_cast<std::size_t>(2 * t)] = 0.5 / c;
            inv_twiddle_[static_cast<std::size_t>(2 * t + 1)] = 1.0 - c * c;
        }

        // Terminal-stage table: for each i in [1, n/8) one 6-double record
        //   { t1[i], t1[n/4 - i], cos(th_i), sin(th_i), -sin(th_i), -cos(th_i) }
        // covering both level-B twiddles of the lane-crossing merge and the
        // projection coefficients of all four bins {i, n/4-i, n/2-i, n/4+i}.
        const int tm = n_ >> 3;
        int term_count = tm > 1 ? 6 * (tm - 1) : 1;
        if (!term_tw_.resize(static_cast<std::size_t>(term_count))) {
            clear();
            return false;
        }
        for (int i = 1; i < tm; ++i) {
            double* dst = term_tw_.data() + 6 * (i - 1);
            const double theta = bruun_tau * static_cast<double>(i) / static_cast<double>(n_);
            const double theta_hi = bruun_tau * static_cast<double>(2 * tm - i) / static_cast<double>(n_);
            dst[0] = 2.0 * std::cos(theta);
            dst[1] = 2.0 * std::cos(theta_hi);
            dst[2] = std::cos(theta);
            dst[3] = std::sin(theta);
            dst[4] = -dst[3];
            dst[5] = -dst[2];
        }

        // Inverse terminal table: for each i in [1, n/8) one 8-double record
        //   { ic_lo, ic_hi, w_lo, w_hi, cos, sin, -1/sin, -1/cos }
        // where lo/hi are the level-B twiddle constants at angles th_i and
        // th_{n/4 - i}, and the last four drive the fused deprojection.
        int iterm_count = tm > 1 ? 8 * (tm - 1) : 1;
        if (!iterm_tw_.resize(static_cast<std::size_t>(iterm_count))) {
            clear();
            return false;
        }
        for (int i = 1; i < tm; ++i) {
            double* dst = iterm_tw_.data() + 8 * (i - 1);
            const double theta = bruun_tau * static_cast<double>(i) / static_cast<double>(n_);
            const double theta_hi = bruun_tau * static_cast<double>(2 * tm - i) / static_cast<double>(n_);
            const double c_lo = 2.0 * std::cos(theta);
            const double c_hi = 2.0 * std::cos(theta_hi);
            dst[0] = 0.5 / c_lo;
            dst[1] = 0.5 / c_hi;
            dst[2] = 1.0 - c_lo * c_lo;
            dst[3] = 1.0 - c_hi * c_hi;
            dst[4] = std::cos(theta);
            dst[5] = std::sin(theta);
            dst[6] = -1.0 / dst[5];
            dst[7] = -1.0 / dst[4];
        }

        if (!build_interp_plan()) {
            clear();
            return false;
        }
        return true;
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return half_ + 1; }
    int work_size() const noexcept { return n_; }

    void forward_scalar(const double* input, double* output_re, double* output_im, double* work) const {
        forward_split_scalar(input, output_re, output_im, work);
    }

    void forward_simd(const double* input, double* output_re, double* output_im, double* work) const {
#if BRUUN_LEVEL >= 1
        if (n_ >= 16) {
            compute_vwork(input, work);
            terminal_split_writer w{output_re, output_im};
            if (radix2_terminal_) {
                terminal_radix2_v(work, w);
            } else {
                terminal_v(work, w);
            }
            return;
        }
#endif
        forward_split_scalar(input, output_re, output_im, work);
    }

    template <typename Complex>
    void forward_complex_scalar(const double* input, Complex* output, double* work) const {
        forward_complex_scalar_impl(input, output, work);
    }

    template <typename Complex>
    void forward_complex_simd(const double* input, Complex* output, double* work) const {
#if BRUUN_LEVEL >= 1
        if (n_ >= 16) {
            compute_vwork(input, work);
            terminal_complex_writer<Complex> w{output};
            if (radix2_terminal_) {
                terminal_radix2_v(work, w);
            } else {
                terminal_v(work, w);
            }
            return;
        }
#endif
        forward_complex_scalar_impl(input, output, work);
    }

    // Normalized inverse from standard bins: inverse(forward(x)) == x.
    template <typename Complex>
    void inverse_complex_scalar(const Complex* input, double* output, double* work) const {
        inverse_complex_scalar_impl(input, output, work);
    }

    template <typename Complex>
    void inverse_complex_simd(const Complex* input, double* output, double* work) const {
#if BRUUN_LEVEL >= 1
        if (n_ >= 16) {
            if (radix2_terminal_) {
                terminal_radix2_inv_v(input, work);
            } else {
                terminal_inv_v(input, work);
            }
            compute_vwork_inv(work, output);
            return;
        }
#endif
        inverse_complex_scalar_impl(input, output, work);
    }

    // Experimental DIT-shaped inverse: standard bins -> compact Bruun residues,
    // forward-flow residue split stages -> ordered real output. This is an
    // internal development target for the no-tail-transport inverse.
    template <typename Complex>
    void inverse_complex_dit_compact_simd(const Complex* input, double* output, double* work) const {
        inverse_complex_dit_compact_impl(input, output, work);
    }

    template <typename Complex>
    void inverse_complex_dit_compact_fused_simd(const Complex* input, double* output, double* work) const {
        inverse_complex_dit_compact_fused_impl(input, output, work);
    }

    // DIT-shaped interpolation inverse: the frequency-decimated factorization
    // of the inverse (the accompanying basis of the mirrored-lane forward).
    // Standard bins are gathered in Bruun factor-tree leaf order (the head
    // shuffle, fused with (re,im) -> (a,b) deprojection - the single
    // lane-crossing zone), CRT interpolation merges run forward-flow with one
    // broadcast constant pair per block, and the root merge streams ordered
    // real samples straight into the output. See the commentary above
    // itp_prologue below.
    template <typename Complex>
    void inverse_complex_interp_scalar(const Complex* input, double* output, double* work) const {
        if (n_ < 16) {
            inverse_complex_scalar_impl(input, output, work);
            return;
        }
        inverse_interp_scalar_impl(input, output, work);
    }

    template <typename Complex>
    void inverse_complex_interp_simd(const Complex* input, double* output, double* work) const {
#if BRUUN_LEVEL >= 1
        if (n_ >= 16) {
            inverse_interp_simd_impl(input, output, work);
            return;
        }
#endif
        inverse_complex_interp_scalar(input, output, work);
    }

private:
    static bool valid_size(int n) {
        if (!is_power2(n)) {
            return false;
        }
        if (n < 4) {
            return false;
        }
        return true;
    }

    void clear() noexcept {
        n_ = 0;
        half_ = 0;
        twiddle_stage_count_ = 0;
        radix2_terminal_ = false;
        rev_.clear();
        cos_.clear();
        neg_sin_.clear();
        inv_neg_sin_.clear();
        twiddle_.clear();
        twiddle_offset_.clear();
        term_tw_.clear();
        inv_twiddle_.clear();
        iterm_tw_.clear();
        ihead_j_.clear();
        ihead_tw_.clear();
        ir2_tw_.clear();
        isw_tw_.clear();
        itile_ord_.clear();
        itile_meta_.clear();
        itile_count_ = 0;
    }

    // Bruun radix-2 residue reduction: child pair at angle 2*theta splits into
    // parent pairs at theta and pi - theta, c2 = 2*cos(theta).
    static BRUUN_ALWAYS_INLINE void pair_reduce(double ea, double eb, double oa, double ob, double c2,
                                                double& la, double& lb, double& ha, double& hb) {
        const double a = ea - eb;
        const double c2sq = c2 * c2;
        const double b = (oa - ob) + c2sq * ob;   // fma-friendly: no separate p
        la = a - c2 * ob;   // fma
        ha = a + c2 * ob;   // fma
        lb = b + c2 * eb;   // fma
        hb = b - c2 * eb;   // fma
    }

    // ------------------------------------------------------------------
    // Scalar reference walk (also serves BRUUN_LEVEL == 0 and n == 4).
    // ------------------------------------------------------------------

    void butterfly4_scalar(const double* q0, const double* q1, const double* q2, const double* q3,
                           int i, double t2, double t1, double t1q, double* out) const {
        double h0a, h0b, h0pa, h0pb, h1a, h1b, h1pa, h1pb;
        pair_reduce(q0[2 * i], q0[2 * i + 1], q1[2 * i], q1[2 * i + 1], t2, h0a, h0b, h0pa, h0pb);
        pair_reduce(q2[2 * i], q2[2 * i + 1], q3[2 * i], q3[2 * i + 1], t2, h1a, h1b, h1pa, h1pb);
        pair_reduce(h0a, h0b, h1a, h1b, t1, out[0], out[1], out[2], out[3]);
        pair_reduce(h0pa, h0pb, h1pa, h1pb, t1q, out[4], out[5], out[6], out[7]);
    }

    void write_butterfly4(double* block, int hs, int q4, int i, const double* out) const {
        block[2 * i] = out[0];
        block[2 * i + 1] = out[1];
        block[2 * (hs - i)] = out[2];
        block[2 * (hs - i) + 1] = out[3];
        block[2 * (q4 - i)] = out[4];
        block[2 * (q4 - i) + 1] = out[5];
        block[2 * (q4 + i)] = out[6];
        block[2 * (q4 + i) + 1] = out[7];
    }

    void merge4_scalar(double* block, int s, const double* t2, const double* t1) const {
        const int q4 = s >> 2;
        const int hs = s >> 1;
        double* q0 = block;
        double* q1 = block + q4;
        double* q2 = block + 2 * q4;
        double* q3 = block + 3 * q4;

        const double q0dc = q0[0];
        const double q0ny = q0[1];
        const double q1dc = q1[0];
        const double q1ny = q1[1];
        const double q2dc = q2[0];
        const double q2ny = q2[1];
        const double q3dc = q3[0];
        const double q3ny = q3[1];
        const double h0dc = q0dc + q1dc;
        const double h0ny = q0dc - q1dc;
        const double h1dc = q2dc + q3dc;
        const double h1ny = q2dc - q3dc;
        double b2a0, b2b0, b2a1, b2b1;
        pair_reduce(q0ny, q1ny, q2ny, q3ny, t1[q4 / 2 - 1], b2a0, b2b0, b2a1, b2b1);
        block[0] = h0dc + h1dc;
        block[1] = h0dc - h1dc;
        block[2 * q4] = h0ny;
        block[2 * q4 + 1] = h1ny;
        block[2 * (q4 / 2)] = b2a0;
        block[2 * (q4 / 2) + 1] = b2b0;
        block[2 * (hs - q4 / 2)] = b2a1;
        block[2 * (hs - q4 / 2) + 1] = b2b1;

        for (int i = 1; i < q4 / 2 - i; ++i) {
            const int m = q4 / 2 - i;
            double oi[8];
            double om[8];
            butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
            butterfly4_scalar(q0, q1, q2, q3, m, t2[m - 1], t1[m - 1], t1[q4 - m - 1], om);
            write_butterfly4(block, hs, q4, i, oi);
            write_butterfly4(block, hs, q4, m, om);
        }

        if (((q4 / 2) & 1) == 0 && q4 / 2 >= 2) {
            const int i = q4 >> 2;
            double oi[8];
            butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
            write_butterfly4(block, hs, q4, i, oi);
        }
    }

    BRUUN_ALWAYS_INLINE void seed_merge2_pair(const double* input, double* work, int b) const {
        const int j0 = rev_[static_cast<std::size_t>(b)];
        const int j1 = rev_[static_cast<std::size_t>(b + 1)];
        const double a0 = input[j0];
        const double c0 = input[half_ + j0];
        const double a1 = input[j1];
        const double c1 = input[half_ + j1];
        const double e0 = a0 + c0;
        const double e1 = a1 + c1;
        work[2 * b] = e0 + e1;
        work[2 * b + 1] = e0 - e1;
        work[2 * b + 2] = a0 - c0;
        work[2 * b + 3] = a1 - c1;
    }

    void compute_work_scalar(const double* input, double* work) const {
        BRUUN_ASSERT(n_ >= 4);
        for (int b = 0; b < half_; b += 2) {
            seed_merge2_pair(input, work, b);
        }
        int stage = 2;
        for (int s = 16; s <= n_; s <<= 2, stage += 2) {
            const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                merge4_scalar(work + off, s, t2, t1);
            }
        }
    }

    void forward_split_scalar(const double* input, double* output_re, double* output_im, double* work) const {
        compute_work_scalar(input, work);
        if (radix2_terminal_) {
            terminal_radix2_split_scalar(work, output_re, output_im);
            return;
        }
        output_re[0] = work[0];
        output_im[0] = 0.0;
        output_re[half_] = work[1];
        output_im[half_] = 0.0;
        for (int k = 1; k < half_; ++k) {
            const double a = work[2 * k];
            const double b = work[2 * k + 1];
            output_re[k] = a + b * cos_[static_cast<std::size_t>(k)];
            output_im[k] = b * neg_sin_[static_cast<std::size_t>(k)];
        }
    }

    template <typename Complex>
    void forward_complex_scalar_impl(const double* input, Complex* output, double* work) const {
        compute_work_scalar(input, work);
        if (radix2_terminal_) {
            terminal_radix2_complex_scalar(work, output);
            return;
        }
        output[0].re = work[0];
        output[0].im = 0.0;
        output[half_].re = work[1];
        output[half_].im = 0.0;
        for (int k = 1; k < half_; ++k) {
            const double a = work[2 * k];
            const double b = work[2 * k + 1];
            output[k].re = a + b * cos_[static_cast<std::size_t>(k)];
            output[k].im = b * neg_sin_[static_cast<std::size_t>(k)];
        }
    }

    template <typename Store>
    void terminal_radix2_scalar_impl(const double* work, const Store& store) const {
        const int q = half_ >> 1;
        store(0, work[0] + work[2 * q], 0.0);
        store(half_, work[0] - work[2 * q], 0.0);
        store(q, work[1], -work[2 * q + 1]);
        for (int i = 1; i < q; ++i) {
            const int hi = half_ - i;
            double la, lb, ha, hb;
            pair_reduce(work[2 * i], work[2 * i + 1],
                        work[2 * (i + q)], work[2 * (i + q) + 1],
                        twiddle_[static_cast<std::size_t>(twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)] + i - 1)],
                        la, lb, ha, hb);
            store(i,
                  la + lb * cos_[static_cast<std::size_t>(i)],
                  lb * neg_sin_[static_cast<std::size_t>(i)]);
            store(hi,
                  ha + hb * cos_[static_cast<std::size_t>(hi)],
                  hb * neg_sin_[static_cast<std::size_t>(hi)]);
        }
    }

    void terminal_radix2_split_scalar(const double* work, double* output_re, double* output_im) const {
        terminal_radix2_scalar_impl(work, [&](int k, double re, double im) {
            output_re[k] = re;
            output_im[k] = im;
        });
    }

    template <typename Complex>
    void terminal_radix2_complex_scalar(const double* work, Complex* output) const {
        terminal_radix2_scalar_impl(work, [&](int k, double re, double im) {
            output[k].re = re;
            output[k].im = im;
        });
    }

    // Exact inverse of pair_reduce. ic = 0.5/c2, w = 1 - c2^2 (precomputed).
    static BRUUN_ALWAYS_INLINE void pair_expand(double la, double lb, double ha, double hb,
                                                double ic, double w,
                                                double& ea, double& eb, double& oa, double& ob) {
        ob = (ha - la) * ic;
        eb = (lb - hb) * ic;
        ea = 0.5 * (la + ha) + eb;
        oa = 0.5 * (lb + hb) + w * ob;
    }

    // Exact inverse of butterfly4_scalar: parent nodes {i, hs-i, q4-i, q4+i}
    // back to child node i of the four children.
    void inv_butterfly4_scalar(const double* block, int q4, int i,
                               const double* it2, const double* it1, double* o) const {
        const int hs = q4 << 1;
        double h0a, h0b, h1a, h1b, h0pa, h0pb, h1pa, h1pb;
        pair_expand(block[2 * i], block[2 * i + 1],
                    block[2 * (hs - i)], block[2 * (hs - i) + 1],
                    it1[2 * (i - 1)], it1[2 * (i - 1) + 1], h0a, h0b, h1a, h1b);
        pair_expand(block[2 * (q4 - i)], block[2 * (q4 - i) + 1],
                    block[2 * (q4 + i)], block[2 * (q4 + i) + 1],
                    it1[2 * (q4 - i - 1)], it1[2 * (q4 - i - 1) + 1], h0pa, h0pb, h1pa, h1pb);
        pair_expand(h0a, h0b, h0pa, h0pb, it2[2 * (i - 1)], it2[2 * (i - 1) + 1],
                    o[0], o[1], o[2], o[3]);
        pair_expand(h1a, h1b, h1pa, h1pb, it2[2 * (i - 1)], it2[2 * (i - 1) + 1],
                    o[4], o[5], o[6], o[7]);
    }

    static void write_inv_butterfly4(double* q0, double* q1, double* q2, double* q3,
                                     int i, const double* o) {
        q0[2 * i] = o[0];
        q0[2 * i + 1] = o[1];
        q1[2 * i] = o[2];
        q1[2 * i + 1] = o[3];
        q2[2 * i] = o[4];
        q2[2 * i + 1] = o[5];
        q3[2 * i] = o[6];
        q3[2 * i + 1] = o[7];
    }

    void split4_scalar(double* block, int s, const double* it2, const double* it1) const {
        const int q4 = s >> 2;
        const int M = q4 >> 1;
        double* q0 = block;
        double* q1 = block + q4;
        double* q2 = block + 2 * q4;
        double* q3 = block + 3 * q4;

        const double n0a = block[0];
        const double n0b = block[1];
        const double nMa = block[q4];
        const double nMb = block[q4 + 1];
        const double n2a = block[2 * q4];
        const double n2b = block[2 * q4 + 1];
        const double n3a = block[3 * q4];
        const double n3b = block[3 * q4 + 1];
        const double h0dc = 0.5 * (n0a + n0b);
        const double h1dc = 0.5 * (n0a - n0b);
        double ny0, ny1, ny2, ny3;
        pair_expand(nMa, nMb, n3a, n3b,
                    it1[2 * (M - 1)], it1[2 * (M - 1) + 1], ny0, ny1, ny2, ny3);
        q0[0] = 0.5 * (h0dc + n2a);
        q0[1] = ny0;
        q1[0] = 0.5 * (h0dc - n2a);
        q1[1] = ny1;
        q2[0] = 0.5 * (h1dc + n2b);
        q2[1] = ny2;
        q3[0] = 0.5 * (h1dc - n2b);
        q3[1] = ny3;

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            double oi[8];
            double om[8];
            inv_butterfly4_scalar(block, q4, i, it2, it1, oi);
            inv_butterfly4_scalar(block, q4, m, it2, it1, om);
            write_inv_butterfly4(q0, q1, q2, q3, i, oi);
            write_inv_butterfly4(q0, q1, q2, q3, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            double oi[8];
            inv_butterfly4_scalar(block, q4, i, it2, it1, oi);
            write_inv_butterfly4(q0, q1, q2, q3, i, oi);
        }
    }

    BRUUN_ALWAYS_INLINE void seed_split2_pair(double* work, double* output, int b) const {
        const int j0 = rev_[static_cast<std::size_t>(b)];
        const int j1 = rev_[static_cast<std::size_t>(b + 1)];
        const double e0 = 0.5 * (work[2 * b] + work[2 * b + 1]);
        const double e1 = 0.5 * (work[2 * b] - work[2 * b + 1]);
        const double d0 = work[2 * b + 2];
        const double d1 = work[2 * b + 3];
        output[j0] = 0.5 * (e0 + d0);
        output[half_ + j0] = 0.5 * (e0 - d0);
        output[j1] = 0.5 * (e1 + d1);
        output[half_ + j1] = 0.5 * (e1 - d1);
    }

    template <typename Complex>
    void inverse_complex_scalar_impl(const Complex* input, double* output, double* work) const {
        if (radix2_terminal_) {
            terminal_radix2_inv_scalar(input, work);
            int stage = twiddle_stage_count_ - 2;
            for (int s = n_ >> 1; s >= 16; s >>= 2, stage -= 2) {
                const double* it2 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
                const double* it1 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
                for (int off = 0; off < n_; off += s) {
                    split4_scalar(work + off, s, it2, it1);
                }
            }
            for (int b = 0; b < half_; b += 2) {
                seed_split2_pair(work, output, b);
            }
            return;
        }
        work[0] = input[0].re;
        work[1] = input[half_].re;
        for (int k = 1; k < half_; ++k) {
            const double b = input[k].im * inv_neg_sin_[static_cast<std::size_t>(k)];
            work[2 * k] = input[k].re - b * cos_[static_cast<std::size_t>(k)];
            work[2 * k + 1] = b;
        }
        int stage = twiddle_stage_count_ - 1;
        for (int s = n_; s >= 16; s >>= 2, stage -= 2) {
            const double* it2 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* it1 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                split4_scalar(work + off, s, it2, it1);
            }
        }
        for (int b = 0; b < half_; b += 2) {
            seed_split2_pair(work, output, b);
        }
    }

    template <typename Complex>
    void terminal_radix2_inv_scalar(const Complex* input, double* work) const {
        const int q = half_ >> 1;
        work[0] = 0.5 * (input[0].re + input[half_].re);
        work[2 * q] = 0.5 * (input[0].re - input[half_].re);
        work[1] = input[q].re;
        work[2 * q + 1] = -input[q].im;
        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
        for (int i = 1; i < q; ++i) {
            const int hi = half_ - i;
            const double lb = input[i].im * inv_neg_sin_[static_cast<std::size_t>(i)];
            const double la = input[i].re - lb * cos_[static_cast<std::size_t>(i)];
            const double hb = input[hi].im * inv_neg_sin_[static_cast<std::size_t>(hi)];
            const double ha = input[hi].re - hb * cos_[static_cast<std::size_t>(hi)];
            double ea, eb, oa, ob;
            pair_expand(la, lb, ha, hb, it[2 * (i - 1)], it[2 * (i - 1) + 1], ea, eb, oa, ob);
            work[2 * i] = ea;
            work[2 * i + 1] = eb;
            work[2 * (i + q)] = oa;
            work[2 * (i + q) + 1] = ob;
        }
    }

    template <typename Complex>
    void compact_residue_ingest(const Complex* input, double* work) const {
        work[0] = input[0].re;
        work[1] = input[half_].re;
        for (int k = 1; k < half_; ++k) {
            const double b = input[k].im * inv_neg_sin_[static_cast<std::size_t>(k)];
            work[2 * k] = input[k].re - b * cos_[static_cast<std::size_t>(k)];
            work[2 * k + 1] = b;
        }
    }

#if BRUUN_LEVEL >= 1
    template <typename Complex>
    void compact_residue_ingest_v2(const Complex* input, double* work) const {
        work[0] = input[0].re;
        work[1] = input[half_].re;
        int k = 1;
        for (; k + 1 < half_; k += 2) {
            const bruun_v2 x0 = V2_LD(&input[k].re);
            const bruun_v2 x1 = V2_LD(&input[k + 1].re);
            const bruun_v2 re = V2_UNPLO(x0, x1);
            const bruun_v2 im = V2_UNPHI(x0, x1);
            const bruun_v2 b = V2_MUL(im, V2_LD(inv_neg_sin_.data() + k));
            const bruun_v2 a = V2_MSUB(re, b, V2_LD(cos_.data() + k));
            V2_ST(work + 2 * k, V2_UNPLO(a, b));
            V2_ST(work + 2 * k + 2, V2_UNPHI(a, b));
        }
        for (; k < half_; ++k) {
            const double b = input[k].im * inv_neg_sin_[static_cast<std::size_t>(k)];
            work[2 * k] = input[k].re - b * cos_[static_cast<std::size_t>(k)];
            work[2 * k + 1] = b;
        }
    }
#endif

    BRUUN_ALWAYS_INLINE void compact_split_node_scalar(const double* parent, int len, int tw_stage,
                                                       double* even, double* odd) const {
        const int half = len >> 1;
        const int quarter = len >> 2;
        even[0] = 0.5 * (parent[0] + parent[1]);
        odd[0] = 0.5 * (parent[0] - parent[1]);
        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(tw_stage)];
        for (int r = 1; r < quarter; ++r) {
            const double a = parent[2 * r];
            const double b = parent[2 * r + 1];
            const double A = parent[2 * (half - r)];
            const double B = parent[2 * (half - r) + 1];
            const double inv4c = it[2 * (r - 1)];
            const double w = -it[2 * (r - 1) + 1] * inv4c;
            const double db = b - B;
            const double da = a - A;
            const double eb = db * inv4c;
            even[2 * r] = 0.5 * (a + A) + eb;
            even[2 * r + 1] = eb;
            odd[2 * r] = 0.5 * (b + B) + da * w;
            odd[2 * r + 1] = -da * inv4c;
        }
        even[1] = parent[2 * quarter];
        odd[1] = parent[2 * quarter + 1];
    }

    static BRUUN_ALWAYS_INLINE void compact_split_pair_scalar(double a, double b, double A, double B,
                                                              double inv4c, double w,
                                                              double& ea, double& eb,
                                                              double& oa, double& ob) {
        const double db = b - B;
        const double da = a - A;
        eb = db * inv4c;
        ea = 0.5 * (a + A) + eb;
        oa = 0.5 * (b + B) + da * w;
        ob = -da * inv4c;
    }

    BRUUN_ALWAYS_INLINE void compact_split_first_pair_scalar(const double* parent, int len, int tw_stage, int r,
                                                             double& ea, double& eb,
                                                             double& oa, double& ob) const {
        const int half = len >> 1;
        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(tw_stage)];
        const double inv4c = it[2 * (r - 1)];
        const double w = -it[2 * (r - 1) + 1] * inv4c;
        compact_split_pair_scalar(parent[2 * r],
                                  parent[2 * r + 1],
                                  parent[2 * (half - r)],
                                  parent[2 * (half - r) + 1],
                                  inv4c,
                                  w,
                                  ea,
                                  eb,
                                  oa,
                                  ob);
    }

    BRUUN_ALWAYS_INLINE void compact_split2_node_scalar(const double* parent, int len, int tw_stage,
                                                        double* even_even, double* odd_even,
                                                        double* even_odd, double* odd_odd) const {
        const int quarter = len >> 2;
        const int eighth = len >> 3;

        const double e0 = 0.5 * (parent[0] + parent[1]);
        const double o0 = 0.5 * (parent[0] - parent[1]);
        const double eq = parent[2 * quarter];
        const double oq = parent[2 * quarter + 1];
        even_even[0] = 0.5 * (e0 + eq);
        even_odd[0] = 0.5 * (e0 - eq);
        odd_even[0] = 0.5 * (o0 + oq);
        odd_odd[0] = 0.5 * (o0 - oq);

        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(tw_stage - 1)];
        for (int r = 1; r < eighth; ++r) {
            double ea, eb, oa, ob;
            double eA, eB, oA, oB;
            compact_split_first_pair_scalar(parent, len, tw_stage, r, ea, eb, oa, ob);
            compact_split_first_pair_scalar(parent, len, tw_stage, quarter - r, eA, eB, oA, oB);

            const double inv4c = it[2 * (r - 1)];
            const double w = -it[2 * (r - 1) + 1] * inv4c;
            double gaa, gab, gba, gbb;
            compact_split_pair_scalar(ea, eb, eA, eB, inv4c, w, gaa, gab, gba, gbb);
            even_even[2 * r] = gaa;
            even_even[2 * r + 1] = gab;
            even_odd[2 * r] = gba;
            even_odd[2 * r + 1] = gbb;
            compact_split_pair_scalar(oa, ob, oA, oB, inv4c, w, gaa, gab, gba, gbb);
            odd_even[2 * r] = gaa;
            odd_even[2 * r + 1] = gab;
            odd_odd[2 * r] = gba;
            odd_odd[2 * r + 1] = gbb;
        }

        double ea, eb, oa, ob;
        compact_split_first_pair_scalar(parent, len, tw_stage, eighth, ea, eb, oa, ob);
        even_even[1] = ea;
        even_odd[1] = eb;
        odd_even[1] = oa;
        odd_odd[1] = ob;
    }

    BRUUN_ALWAYS_INLINE void compact_store_real_leaf(double* output, int leaf, double dc, double ny) const {
        output[leaf] = 0.5 * (dc + ny);
        output[leaf + half_] = 0.5 * (dc - ny);
    }

    BRUUN_ALWAYS_INLINE void compact_final4_node_to_real(const double* parent, int node, int nodes,
                                                         double* output) const {
        const double e0 = 0.5 * (parent[0] + parent[1]);
        const double o0 = 0.5 * (parent[0] - parent[1]);
        compact_store_real_leaf(output, node, e0, parent[2]);
        compact_store_real_leaf(output, node + nodes, o0, parent[3]);
    }

    BRUUN_ALWAYS_INLINE void compact_final8_node_to_real(const double* parent, int node, int nodes,
                                                         double* output) const {
        const double e0 = 0.5 * (parent[0] + parent[1]);
        const double o0 = 0.5 * (parent[0] - parent[1]);
        const double eq = parent[4];
        const double oq = parent[5];

        double ea, eb, oa, ob;
        compact_split_first_pair_scalar(parent, 8, 1, 1, ea, eb, oa, ob);
        compact_store_real_leaf(output, node, 0.5 * (e0 + eq), ea);
        compact_store_real_leaf(output, node + nodes, 0.5 * (o0 + oq), oa);
        compact_store_real_leaf(output, node + 2 * nodes, 0.5 * (e0 - eq), eb);
        compact_store_real_leaf(output, node + 3 * nodes, 0.5 * (o0 - oq), ob);
    }

#if BRUUN_LEVEL >= 1
    static BRUUN_ALWAYS_INLINE bruun_v2 compact_swap_v2(bruun_v2 v) {
        return V2_UNPLO(V2_DUP1(v), V2_DUP0(v));
    }

    static BRUUN_ALWAYS_INLINE void compact_split_pair_v2(bruun_v2 a, bruun_v2 b,
                                                          bruun_v2 A, bruun_v2 B,
                                                          bruun_v2 inv4c, bruun_v2 w,
                                                          bruun_v2& ea, bruun_v2& eb,
                                                          bruun_v2& oa, bruun_v2& ob) {
        const bruun_v2 hv = V2_SET1(0.5);
        const bruun_v2 zv = V2_SET1(0.0);
        const bruun_v2 db = V2_SUB(b, B);
        const bruun_v2 da = V2_SUB(a, A);
        eb = V2_MUL(db, inv4c);
        ea = V2_MADD(eb, V2_ADD(a, A), hv);
        oa = V2_MADD(V2_MUL(V2_ADD(b, B), hv), da, w);
        ob = V2_MUL(V2_SUB(zv, da), inv4c);
    }

    BRUUN_ALWAYS_INLINE void compact_split_first_pair_v2(const double* parent, int len, int tw_stage, int r,
                                                         bruun_v2& ea, bruun_v2& eb,
                                                         bruun_v2& oa, bruun_v2& ob) const {
        const int half = len >> 1;
        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(tw_stage)];

        const bruun_v2 lo0 = V2_LD(parent + 2 * r);
        const bruun_v2 lo1 = V2_LD(parent + 2 * r + 2);
        const bruun_v2 a = V2_UNPLO(lo0, lo1);
        const bruun_v2 b = V2_UNPHI(lo0, lo1);

        const bruun_v2 hi0 = V2_LD(parent + 2 * (half - r));
        const bruun_v2 hi1 = V2_LD(parent + 2 * (half - r - 1));
        const bruun_v2 A = V2_UNPLO(hi0, hi1);
        const bruun_v2 B = V2_UNPHI(hi0, hi1);

        const bruun_v2 inv4c = V2_SETLH(it[2 * (r - 1)], it[2 * r]);
        const bruun_v2 oldw = V2_SETLH(it[2 * (r - 1) + 1], it[2 * r + 1]);
        compact_split_pair_v2(a, b, A, B, inv4c, V2_MUL(V2_SUB(V2_SET1(0.0), oldw), inv4c), ea, eb, oa, ob);
    }

    BRUUN_ALWAYS_INLINE void compact_split2_node_v2(const double* parent, int len, int tw_stage,
                                                    double* even_even, double* odd_even,
                                                    double* even_odd, double* odd_odd) const {
        const int quarter = len >> 2;
        const int eighth = len >> 3;

        const double e0 = 0.5 * (parent[0] + parent[1]);
        const double o0 = 0.5 * (parent[0] - parent[1]);
        const double eq = parent[2 * quarter];
        const double oq = parent[2 * quarter + 1];
        even_even[0] = 0.5 * (e0 + eq);
        even_odd[0] = 0.5 * (e0 - eq);
        odd_even[0] = 0.5 * (o0 + oq);
        odd_odd[0] = 0.5 * (o0 - oq);

        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(tw_stage - 1)];
        int r = 1;
        for (; r + 1 < eighth; r += 2) {
            bruun_v2 ea, eb, oa, ob;
            bruun_v2 eA, eB, oA, oB;
            compact_split_first_pair_v2(parent, len, tw_stage, r, ea, eb, oa, ob);
            compact_split_first_pair_v2(parent, len, tw_stage, quarter - r - 1, eA, eB, oA, oB);
            eA = compact_swap_v2(eA);
            eB = compact_swap_v2(eB);
            oA = compact_swap_v2(oA);
            oB = compact_swap_v2(oB);

            const bruun_v2 inv4c = V2_SETLH(it[2 * (r - 1)], it[2 * r]);
            const bruun_v2 oldw = V2_SETLH(it[2 * (r - 1) + 1], it[2 * r + 1]);
            const bruun_v2 w = V2_MUL(V2_SUB(V2_SET1(0.0), oldw), inv4c);
            bruun_v2 gaa, gab, gba, gbb;
            compact_split_pair_v2(ea, eb, eA, eB, inv4c, w, gaa, gab, gba, gbb);
            V2_ST(even_even + 2 * r, V2_UNPLO(gaa, gab));
            V2_ST(even_even + 2 * r + 2, V2_UNPHI(gaa, gab));
            V2_ST(even_odd + 2 * r, V2_UNPLO(gba, gbb));
            V2_ST(even_odd + 2 * r + 2, V2_UNPHI(gba, gbb));
            compact_split_pair_v2(oa, ob, oA, oB, inv4c, w, gaa, gab, gba, gbb);
            V2_ST(odd_even + 2 * r, V2_UNPLO(gaa, gab));
            V2_ST(odd_even + 2 * r + 2, V2_UNPHI(gaa, gab));
            V2_ST(odd_odd + 2 * r, V2_UNPLO(gba, gbb));
            V2_ST(odd_odd + 2 * r + 2, V2_UNPHI(gba, gbb));
        }

        for (; r < eighth; ++r) {
            double ea, eb, oa, ob;
            double eA, eB, oA, oB;
            compact_split_first_pair_scalar(parent, len, tw_stage, r, ea, eb, oa, ob);
            compact_split_first_pair_scalar(parent, len, tw_stage, quarter - r, eA, eB, oA, oB);
            const double inv4c = it[2 * (r - 1)];
            const double w = -it[2 * (r - 1) + 1] * inv4c;
            double gaa, gab, gba, gbb;
            compact_split_pair_scalar(ea, eb, eA, eB, inv4c, w, gaa, gab, gba, gbb);
            even_even[2 * r] = gaa;
            even_even[2 * r + 1] = gab;
            even_odd[2 * r] = gba;
            even_odd[2 * r + 1] = gbb;
            compact_split_pair_scalar(oa, ob, oA, oB, inv4c, w, gaa, gab, gba, gbb);
            odd_even[2 * r] = gaa;
            odd_even[2 * r + 1] = gab;
            odd_odd[2 * r] = gba;
            odd_odd[2 * r + 1] = gbb;
        }

        double ea, eb, oa, ob;
        compact_split_first_pair_scalar(parent, len, tw_stage, eighth, ea, eb, oa, ob);
        even_even[1] = ea;
        even_odd[1] = eb;
        odd_even[1] = oa;
        odd_odd[1] = ob;
    }

    BRUUN_ALWAYS_INLINE void compact_split_node_v2(const double* parent, int len, int tw_stage,
                                                   double* even, double* odd) const {
        const int half = len >> 1;
        const int quarter = len >> 2;
        even[0] = 0.5 * (parent[0] + parent[1]);
        odd[0] = 0.5 * (parent[0] - parent[1]);
        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(tw_stage)];
        const bruun_v2 hv = V2_SET1(0.5);
        const bruun_v2 zv = V2_SET1(0.0);
        int r = 1;
        for (; r + 1 < quarter; r += 2) {
            const bruun_v2 lo0 = V2_LD(parent + 2 * r);
            const bruun_v2 lo1 = V2_LD(parent + 2 * r + 2);
            const bruun_v2 a = V2_UNPLO(lo0, lo1);
            const bruun_v2 b = V2_UNPHI(lo0, lo1);

            const bruun_v2 hi0 = V2_LD(parent + 2 * (half - r));
            const bruun_v2 hi1 = V2_LD(parent + 2 * (half - r - 1));
            const bruun_v2 A = V2_UNPLO(hi0, hi1);
            const bruun_v2 B = V2_UNPHI(hi0, hi1);

            const bruun_v2 inv4c = V2_SETLH(it[2 * (r - 1)], it[2 * r]);
            const bruun_v2 oldw = V2_SETLH(it[2 * (r - 1) + 1], it[2 * r + 1]);
            const bruun_v2 w = V2_MUL(V2_SUB(zv, oldw), inv4c);
            const bruun_v2 db = V2_SUB(b, B);
            const bruun_v2 da = V2_SUB(a, A);
            const bruun_v2 eb = V2_MUL(db, inv4c);
            const bruun_v2 ea = V2_MADD(eb, V2_ADD(a, A), hv);
            const bruun_v2 oa = V2_MADD(V2_MUL(V2_ADD(b, B), hv), da, w);
            const bruun_v2 ob = V2_MUL(V2_SUB(zv, da), inv4c);

            V2_ST(even + 2 * r, V2_UNPLO(ea, eb));
            V2_ST(even + 2 * r + 2, V2_UNPHI(ea, eb));
            V2_ST(odd + 2 * r, V2_UNPLO(oa, ob));
            V2_ST(odd + 2 * r + 2, V2_UNPHI(oa, ob));
        }
        for (; r < quarter; ++r) {
            const double a = parent[2 * r];
            const double b = parent[2 * r + 1];
            const double A = parent[2 * (half - r)];
            const double B = parent[2 * (half - r) + 1];
            const double inv4c = it[2 * (r - 1)];
            const double w = -it[2 * (r - 1) + 1] * inv4c;
            const double db = b - B;
            const double da = a - A;
            const double eb = db * inv4c;
            even[2 * r] = 0.5 * (a + A) + eb;
            even[2 * r + 1] = eb;
            odd[2 * r] = 0.5 * (b + B) + da * w;
            odd[2 * r + 1] = -da * inv4c;
        }
        even[1] = parent[2 * quarter];
        odd[1] = parent[2 * quarter + 1];
    }
#endif

    template <typename Complex>
    void inverse_complex_dit_compact_impl(const Complex* input, double* output, double* work) const {
#if BRUUN_LEVEL >= 1
        compact_residue_ingest_v2(input, work);
#else
        compact_residue_ingest(input, work);
#endif

        const double* cur = work;
        double* next = output;
        int nodes = 1;
        int len = n_;
        for (int depth = 0; depth < twiddle_stage_count_; ++depth) {
            const int child_len = len >> 1;
            const int tw_stage = ilog2_pow2(len) - 2;
            for (int node = 0; node < nodes; ++node) {
                const double* parent = cur + node * len;
                double* even = next + node * child_len;
                double* odd = next + (node + nodes) * child_len;
#if BRUUN_LEVEL >= 1
                compact_split_node_v2(parent, len, tw_stage, even, odd);
#else
                compact_split_node_scalar(parent, len, tw_stage, even, odd);
#endif
            }
            cur = next;
            next = (next == output) ? work : output;
            nodes <<= 1;
            len = child_len;
        }

        double* dst = (cur == output) ? work : output;
        for (int node = 0; node < nodes; ++node) {
            const double dc = cur[2 * node];
            const double ny = cur[2 * node + 1];
            dst[node] = 0.5 * (dc + ny);
            dst[node + nodes] = 0.5 * (dc - ny);
        }
        if (dst != output) {
            std::copy(dst, dst + n_, output);
        }
    }

    template <typename Complex>
    void inverse_complex_dit_compact_fused_impl(const Complex* input, double* output, double* work) const {
#if BRUUN_LEVEL >= 1
        compact_residue_ingest_v2(input, work);
#else
        compact_residue_ingest(input, work);
#endif

        const double* cur = work;
        double* next = output;
        int nodes = 1;
        int len = n_;
        int remaining = twiddle_stage_count_;
        while (remaining >= 2) {
            if (remaining == 2 && cur != output) {
                for (int node = 0; node < nodes; ++node) {
                    compact_final8_node_to_real(cur + node * len, node, nodes, output);
                }
                return;
            }
            const int child_len = len >> 2;
            const int tw_stage = ilog2_pow2(len) - 2;
            for (int node = 0; node < nodes; ++node) {
                const double* parent = cur + node * len;
                double* even_even = next + node * child_len;
                double* odd_even = next + (node + nodes) * child_len;
                double* even_odd = next + (node + 2 * nodes) * child_len;
                double* odd_odd = next + (node + 3 * nodes) * child_len;
#if BRUUN_LEVEL >= 1
                compact_split2_node_v2(parent, len, tw_stage, even_even, odd_even, even_odd, odd_odd);
#else
                compact_split2_node_scalar(parent, len, tw_stage, even_even, odd_even, even_odd, odd_odd);
#endif
            }
            cur = next;
            next = (next == output) ? work : output;
            nodes <<= 2;
            len = child_len;
            remaining -= 2;
        }

        if (remaining != 0) {
            if (cur != output) {
                for (int node = 0; node < nodes; ++node) {
                    compact_final4_node_to_real(cur + node * len, node, nodes, output);
                }
                return;
            }
            const int child_len = len >> 1;
            const int tw_stage = ilog2_pow2(len) - 2;
            for (int node = 0; node < nodes; ++node) {
                const double* parent = cur + node * len;
                double* even = next + node * child_len;
                double* odd = next + (node + nodes) * child_len;
#if BRUUN_LEVEL >= 1
                compact_split_node_v2(parent, len, tw_stage, even, odd);
#else
                compact_split_node_scalar(parent, len, tw_stage, even, odd);
#endif
            }
            cur = next;
            nodes <<= 1;
        }

        double* dst = (cur == output) ? work : output;
        for (int node = 0; node < nodes; ++node) {
            const double dc = cur[2 * node];
            const double ny = cur[2 * node + 1];
            dst[node] = 0.5 * (dc + ny);
            dst[node + nodes] = 0.5 * (dc - ny);
        }
        if (dst != output) {
            std::copy(dst, dst + n_, output);
        }
    }

    // ------------------------------------------------------------------
    // Interpolation inverse (DIT-of-the-inverse). Basis:
    //
    // The factor tree of z^N - 1 over the reals: quad node Q(K, S) is the
    // factor z^S - 2*cos(pi*K/n) * z^{S/2} + 1 with children Q(K/2, S/2) and
    // Q(n/2 - K/2, S/2); the real-root chain R(S) = z^S - 1 has children
    // R(S/2) and Q(n/4, S/2) (= z^{S/2} + 1). Leaves are the standard bins:
    // Q(k, 2) = z^2 - 2*cos(theta_k) z + 1 holds bin k's residue (a, b) with
    // X_k = a + b e^{-i theta_k}, and z -/+ 1 hold bins 0 and n/2.
    //
    // The inverse is CRT interpolation up this tree. The merge of two sibling
    // residues r1 (lo, angle psi) and r2 (hi, angle pi - psi) into their
    // parent at angle 2*psi is, per coefficient position i in [0, S/4) with
    // the ONE constant c = 2*cos(psi):
    //
    //   pair_expand(r1[i], r1[i+S/4], r2[i], r2[i+S/4], 0.5/c, 1 - c*c)
    //     -> parent[i], parent[i+S/2], parent[i+S/4], parent[i+3S/4]
    //
    // i.e. exactly the primitive of the reverse walk, but applied per-BLOCK
    // with a broadcast constant instead of per-position twiddles. The read
    // set equals the write set (in-place), positions are adjacent (pure
    // vertical SIMD, zero permutes), and the R-chain merge is a half-scaled
    // add/sub. This is the exact dual of the forward mirrored-lane DIT:
    // there per-position twiddles forced the mirror-lane packing and one
    // lane crossing at the spectral terminal; here per-block constants make
    // every interior merge trivially vertical and the one lane-crossing zone
    // is the spectral HEAD, fused with deprojection. Transport mirrors too:
    // the head gathers bins {j, n/4-j, n/4+j, n/2-j} (scrambled loads, which
    // absorb the shuffle for free), all interior traffic is contiguous and
    // in-place in work[n], and the root merge R(n) streams the ordered real
    // signal into the output with 4 sequential store streams. CRT is exact,
    // so the distributed 0.5s compose to 1/N with no normalization pass.
    //
    // Work layout: P(S) := Q(n/4, S) subtree occupies work[S : 2S) (so
    // work[0:2] = bins {0, n/2}, P(2) = bin n/4 at [2:4), P(4) at [4:8),
    // ..., P(n/2) at [n/2 : n)), each subtree laid out [lo | hi]
    // recursively; the R-chain accumulates in the prefix. Per area the
    // level count from 8-blocks is log2(S/8): odd areas take one radix-2
    // pass at 16, everything else runs fused radix-4 merges, tiled to stay
    // L1-resident before the per-area upper sweeps.
    // ------------------------------------------------------------------

    // Angle index of the node at `idx` blocks of size s into area P(S):
    // walk the subtree address bits from the area root Q(n/4, S).
    int itp_node_K(int S, int s, int idx) const {
        const int bits = ilog2_pow2(S / s);
        int K = n_ >> 2;
        for (int b = bits - 1; b >= 0; --b) {
            K >>= 1;
            if ((idx >> b) & 1) {
                K = (n_ >> 1) - K;
            }
        }
        return K;
    }

    static BRUUN_ALWAYS_INLINE bool itp_area_odd(int S) {
        return (ilog2_pow2(S >> 3) & 1) != 0;
    }

    static BRUUN_ALWAYS_INLINE int itp_area_tile(int S, bool odd) {
        const int tmax = odd ? BRUUN_ITP_TILE_ODD : BRUUN_ITP_TILE_EVEN;
        return S < tmax ? S : tmax;
    }

    // Upper sweeps of area P(S) (node sizes 4T..S), cache-tiered: levels are
    // grouped into windows of at most tier-cap doubles and each window's
    // chunks run all their levels back to back. f(o, s) is invoked once per
    // fused radix-4 node in a fixed order shared by planning and execution.
    template <typename F>
    void itp_upper_walk(int S, int T, F&& f) const {
        static const int caps[2] = {BRUUN_ITP_TIER2, BRUUN_ITP_TIER3};
        int prev = T;
        for (int c = 0; c < 2; ++c) {
            int tier = prev;
            while (4 * tier <= S && 4 * tier <= caps[c]) {
                tier <<= 2;
            }
            if (tier > prev) {
                for (int chunk = S; chunk < 2 * S; chunk += tier) {
                    for (int s = 4 * prev; s <= tier; s <<= 2) {
                        for (int o = chunk; o < chunk + tier; o += s) {
                            f(o, s);
                        }
                    }
                }
                prev = tier;
            }
        }
        for (int s = 4 * prev; s <= S; s <<= 2) {
            for (int o = S; o < 2 * S; o += s) {
                f(o, s);
            }
        }
    }

    bool build_interp_plan() {
        if (n_ < 16) {
            if (!ihead_j_.resize(1) || !ihead_tw_.resize(1) ||
                !ir2_tw_.resize(1) || !isw_tw_.resize(1) ||
                !itile_ord_.resize(1) || !itile_meta_.resize(1)) {
                return false;
            }
            itile_count_ = 0;
            return true;
        }

        // Pass 0: counts (tile order does not change them).
        int gh = 0;
        int g2 = 0;
        int gs = 0;
        int nt = 0;
        for (int S = 8; S <= half_; S <<= 1) {
            const bool odd = itp_area_odd(S);
            const int s0 = odd ? 16 : 8;
            const int T = itp_area_tile(S, odd);
            gh += S >> 3;
            if (odd) {
                g2 += S >> 4;
            }
            for (int s = 4 * s0; s <= T; s <<= 2) {
                gs += S / s;
            }
            for (int s = 4 * T; s <= S; s <<= 2) {
                gs += S / s;
            }
            nt += S / T;
        }
        if (!ihead_j_.resize(static_cast<std::size_t>(gh)) ||
            !ihead_tw_.resize(static_cast<std::size_t>(10 * gh)) ||
            !ir2_tw_.resize(static_cast<std::size_t>(g2 > 0 ? 2 * g2 : 1)) ||
            !isw_tw_.resize(static_cast<std::size_t>(gs > 0 ? 6 * gs : 1)) ||
            !itile_ord_.resize(static_cast<std::size_t>(nt))) {
            return false;
        }

        // Global tile order: tiles of ALL areas, sorted by comb phase (their
        // minimum leaf bin). The head then sweeps X exactly once - every
        // input line is consumed by all its consumers (across areas) while
        // still hot, instead of being refetched once per area whose comb
        // touches it (mirror of the forward's bit-reversed seed-block order).
        if (!itile_meta_.resize(static_cast<std::size_t>(nt))) {
            return false;
        }
        {
            heap_array<int> minj;
            if (!minj.resize(static_cast<std::size_t>(nt))) {
                return false;
            }
            int tb = 0;
            for (int S = 8; S <= half_; S <<= 1) {
                const bool odd = itp_area_odd(S);
                const int T = itp_area_tile(S, odd);
                const int ntiles = S / T;
                const int gpt = T >> 3;
                for (int t = 0; t < ntiles; ++t) {
                    int mj = n_;
                    for (int g = 0; g < gpt; ++g) {
                        const int j = itp_node_K(S, 8, t * gpt + g) >> 2;
                        mj = j < mj ? j : mj;
                    }
                    itile_ord_[static_cast<std::size_t>(tb + t)] = S + t * T;
                    itile_meta_[static_cast<std::size_t>(tb + t)] = (S << 1) | (odd ? 1 : 0);
                    minj[static_cast<std::size_t>(tb + t)] = mj;
                }
                tb += ntiles;
            }
            heap_array<int> idx;
            if (!idx.resize(static_cast<std::size_t>(nt))) {
                return false;
            }
            for (int t = 0; t < nt; ++t) {
                idx[static_cast<std::size_t>(t)] = t;
            }
            std::sort(idx.data(), idx.data() + nt, [&](int a, int b) {
                return minj[static_cast<std::size_t>(a)] < minj[static_cast<std::size_t>(b)];
            });
            heap_array<int> tmp_off;
            heap_array<int> tmp_meta;
            if (!tmp_off.resize(static_cast<std::size_t>(nt)) ||
                !tmp_meta.resize(static_cast<std::size_t>(nt))) {
                return false;
            }
            for (int t = 0; t < nt; ++t) {
                tmp_off[static_cast<std::size_t>(t)] = itile_ord_[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])];
                tmp_meta[static_cast<std::size_t>(t)] = itile_meta_[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])];
            }
            for (int t = 0; t < nt; ++t) {
                itile_ord_[static_cast<std::size_t>(t)] = tmp_off[static_cast<std::size_t>(t)];
                itile_meta_[static_cast<std::size_t>(t)] = tmp_meta[static_cast<std::size_t>(t)];
            }
        }
        itile_count_ = nt;

        // Pass 1: emit constant streams in exact execution order: the global
        // head/tile phase first, then the per-area upper sweeps.
        gh = 0;
        g2 = 0;
        gs = 0;
        for (int t = 0; t < nt; ++t) {
            const int toff = itile_ord_[static_cast<std::size_t>(t)];
            const int S = itile_meta_[static_cast<std::size_t>(t)] >> 1;
            const bool odd = (itile_meta_[static_cast<std::size_t>(t)] & 1) != 0;
            const int s0 = odd ? 16 : 8;
            const int T = itp_area_tile(S, odd);
            for (int o = toff; o < toff + T; o += 8) {
                fill_interp_head(gh, itp_node_K(S, 8, (o - S) >> 3) >> 2);
                ++gh;
            }
            if (odd) {
                for (int o = toff; o < toff + T; o += 16) {
                    const int K = itp_node_K(S, 16, (o - S) >> 4);
                    const double c = 2.0 * cos_[static_cast<std::size_t>(K >> 1)];
                    ir2_tw_[static_cast<std::size_t>(2 * g2)] = 0.5 / c;
                    ir2_tw_[static_cast<std::size_t>(2 * g2 + 1)] = 1.0 - c * c;
                    ++g2;
                }
            }
            for (int s = 4 * s0; s <= T; s <<= 2) {
                for (int o = toff; o < toff + T; o += s) {
                    fill_interp_sweep(gs, itp_node_K(S, s, (o - S) / s));
                    ++gs;
                }
            }
        }
        for (int S = 8; S <= half_; S <<= 1) {
            const bool odd = itp_area_odd(S);
            const int T = itp_area_tile(S, odd);
            itp_upper_walk(S, T, [&](int o, int s) {
                fill_interp_sweep(gs, itp_node_K(S, s, (o - S) / s));
                ++gs;
            });
        }
        return true;
    }

    // Head record, 10 doubles per size-8 group covering bins
    // {j, n/4-j, n/4+j, n/2-j}:
    //   { cos, sin, -1/sin, -1/cos, ic_lo, ic_hi, w_lo, w_hi, ic_p, w_p }
    // The first four drive the fused deprojection through the quarter-turn
    // symmetries (mirror of the forward terminal table); the rest are the
    // two level-1 merges (c = 2cos(theta), 2sin(theta)) lane-paired plus the
    // level-2 merge (c = 2cos(2theta)). j = n/8 never appears here: bins
    // n/8 and 3n/8 live in the prologue's P(4), so cos(2theta) != 0.
    void fill_interp_head(int g, int j) {
        double* r = ihead_tw_.data() + 10 * g;
        const double c = cos_[static_cast<std::size_t>(j)];
        const double s = -neg_sin_[static_cast<std::size_t>(j)];
        const double c1 = 2.0 * c;
        const double c2 = 2.0 * s;
        const double cp = 2.0 * cos_[static_cast<std::size_t>(2 * j)];
        r[0] = c;
        r[1] = s;
        r[2] = -1.0 / s;
        r[3] = -1.0 / c;
        r[4] = 0.5 / c1;
        r[5] = 0.5 / c2;
        r[6] = 1.0 - c1 * c1;
        r[7] = 1.0 - c2 * c2;
        r[8] = 0.5 / cp;
        r[9] = 1.0 - cp * cp;
        ihead_j_[static_cast<std::size_t>(g)] = j;
    }

    // Fused radix-4 node record: { ic_lo, w_lo, ic_hi, w_hi, ic_p, w_p } for
    // the two level-1 merges (angles K/2 and n/2 - K/2 halved) and the
    // parent merge at angle K halved. All angles stay in (0, pi/2), so no
    // constant is ever singular.
    void fill_interp_sweep(int g, int K) {
        double* r = isw_tw_.data() + 6 * g;
        const double clo = 2.0 * cos_[static_cast<std::size_t>(K >> 2)];
        const double chi = 2.0 * cos_[static_cast<std::size_t>((n_ >> 2) - (K >> 2))];
        const double cp = 2.0 * cos_[static_cast<std::size_t>(K >> 1)];
        r[0] = 0.5 / clo;
        r[1] = 1.0 - clo * clo;
        r[2] = 0.5 / chi;
        r[3] = 1.0 - chi * chi;
        r[4] = 0.5 / cp;
        r[5] = 1.0 - cp * cp;
    }

    // Prologue: bins {0, n/2} -> R(2), bin n/4 -> P(2), R(4) merge, bins
    // {n/8, 3n/8} (theta = pi/4: the sqrt2 specials) -> P(4), and R(8) once
    // n >= 32. Fixed cost, all singular-angle bins handled here.
    template <typename Complex>
    void itp_prologue(const Complex* X, double* w) const {
        const int n8 = n_ >> 3;
        const double u0 = 0.5 * (X[0].re + X[half_].re);
        const double u1 = 0.5 * (X[0].re - X[half_].re);
        const double aq = X[n_ >> 2].re;
        const double bq = -X[n_ >> 2].im;
        w[0] = 0.5 * (u0 + aq);
        w[1] = 0.5 * (u1 + bq);
        w[2] = 0.5 * (u0 - aq);
        w[3] = 0.5 * (u1 - bq);
        const double rt2 = 1.4142135623730950488;
        const double blo = -X[n8].im * rt2;
        const double alo = X[n8].re - blo * (0.5 * rt2);
        const double bhi = -X[3 * n8].im * rt2;
        const double ahi = X[3 * n8].re + bhi * (0.5 * rt2);
        pair_expand(alo, blo, ahi, bhi, 0.5 / rt2, -1.0, w[4], w[6], w[5], w[7]);
        if (n_ >= 32) {
            for (int i = 0; i < 4; ++i) {
                const double a = w[i];
                const double b = w[4 + i];
                w[i] = 0.5 * (a + b);
                w[4 + i] = 0.5 * (a - b);
            }
        }
    }

    // Scalar head: deproject bins {j, n/4-j, n/4+j, n/2-j} and run both
    // level-1 merges plus the level-2 merge into one contiguous 8-block.
    template <typename Complex>
    BRUUN_ALWAYS_INLINE void itp_head8_scalar(const Complex* X, double* blk, int j, const double* r) const {
        const int q4 = n_ >> 2;
        const double blo = X[j].im * r[2];
        const double alo = X[j].re - blo * r[0];
        const double bhi = X[half_ - j].im * r[2];
        const double ahi = X[half_ - j].re + bhi * r[0];
        const double bql = X[q4 - j].im * r[3];
        const double aql = X[q4 - j].re - bql * r[1];
        const double bqh = X[q4 + j].im * r[3];
        const double aqh = X[q4 + j].re + bqh * r[1];
        double lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
        pair_expand(alo, blo, ahi, bhi, r[4], r[6], lo0, lo2, lo1, lo3);
        pair_expand(aql, bql, aqh, bqh, r[5], r[7], hi0, hi2, hi1, hi3);
        pair_expand(lo0, lo2, hi0, hi2, r[8], r[9], blk[0], blk[4], blk[2], blk[6]);
        pair_expand(lo1, lo3, hi1, hi3, r[8], r[9], blk[1], blk[5], blk[3], blk[7]);
    }

    // One radix-2 interpolation merge of a block of s coefficients.
    void itp_merge2_scalar(double* blk, int s, double ic, double w) const {
        const int q = s >> 2;
        for (int i = 0; i < q; ++i) {
            pair_expand(blk[i], blk[i + q], blk[i + 2 * q], blk[i + 3 * q], ic, w,
                        blk[i], blk[i + 2 * q], blk[i + q], blk[i + 3 * q]);
        }
    }

    // Fused radix-4 interpolation merge: four s/4 children -> parent of s,
    // in place; per position the read set equals the write set.
    void itp_fused4_scalar(double* blk, int s, const double* r) const {
        const int q = s >> 3;
        const double* g0 = blk;
        const double* g1 = blk + 2 * q;
        const double* g2 = blk + 4 * q;
        const double* g3 = blk + 6 * q;
        for (int i = 0; i < q; ++i) {
            double cl0, cl1, cl2, cl3, ch0, ch1, ch2, ch3;
            pair_expand(g0[i], g0[i + q], g1[i], g1[i + q], r[0], r[1], cl0, cl2, cl1, cl3);
            pair_expand(g2[i], g2[i + q], g3[i], g3[i + q], r[2], r[3], ch0, ch2, ch1, ch3);
            pair_expand(cl0, cl2, ch0, ch2, r[4], r[5],
                        blk[i], blk[i + 4 * q], blk[i + 2 * q], blk[i + 6 * q]);
            pair_expand(cl1, cl3, ch1, ch3, r[4], r[5],
                        blk[i + q], blk[i + 5 * q], blk[i + 3 * q], blk[i + 7 * q]);
        }
    }

    // R-chain catch-up: R(2S) at [0, 2S) from R(S) = [0, S) and P(S) = [S, 2S).
    static void itp_rmerge_scalar(double* w, int S) {
        for (int i = 0; i < S; ++i) {
            const double a = w[i];
            const double b = w[S + i];
            w[i] = 0.5 * (a + b);
            w[S + i] = 0.5 * (a - b);
        }
    }

    // Root: R(n) from R(n/4), P(n/4), P(n/2), fused two R levels, writing
    // the ordered time signal as four sequential output streams.
    void itp_terminal_scalar(const double* w, double* out) const {
        const int q = n_ >> 2;
        const double* u = w;
        const double* v = w + q;
        const double* pl = w + 2 * q;
        const double* ph = w + 3 * q;
        for (int i = 0; i < q; ++i) {
            const double rlo = 0.5 * (u[i] + v[i]);
            const double rhi = 0.5 * (u[i] - v[i]);
            out[i] = 0.5 * (rlo + pl[i]);
            out[i + q] = 0.5 * (rhi + ph[i]);
            out[i + 2 * q] = 0.5 * (rlo - pl[i]);
            out[i + 3 * q] = 0.5 * (rhi - ph[i]);
        }
    }

    template <typename Complex>
    void inverse_interp_scalar_impl(const Complex* X, double* out, double* w) const {
        itp_prologue(X, w);
        const int* jp = ihead_j_.data();
        const double* hr = ihead_tw_.data();
        const double* r2 = ir2_tw_.data();
        const double* sw = isw_tw_.data();
        for (int t = 0; t < itile_count_; ++t) {
            const int toff = itile_ord_[static_cast<std::size_t>(t)];
            const int S = itile_meta_[static_cast<std::size_t>(t)] >> 1;
            const bool odd = (itile_meta_[static_cast<std::size_t>(t)] & 1) != 0;
            const int s0 = odd ? 16 : 8;
            const int T = itp_area_tile(S, odd);
            for (int o = toff; o < toff + T; o += 8) {
                itp_head8_scalar(X, w + o, *jp++, hr);
                hr += 10;
            }
            if (odd) {
                for (int o = toff; o < toff + T; o += 16) {
                    itp_merge2_scalar(w + o, 16, r2[0], r2[1]);
                    r2 += 2;
                }
            }
            for (int s = 4 * s0; s <= T; s <<= 2) {
                for (int o = toff; o < toff + T; o += s) {
                    itp_fused4_scalar(w + o, s, sw);
                    sw += 6;
                }
            }
        }
        for (int S = 8; S <= half_; S <<= 1) {
            const bool odd = itp_area_odd(S);
            const int T = itp_area_tile(S, odd);
            itp_upper_walk(S, T, [&](int o, int s) {
                itp_fused4_scalar(w + o, s, sw);
                sw += 6;
            });
            if (S <= (n_ >> 3)) {
                itp_rmerge_scalar(w, S);
            }
        }
        itp_terminal_scalar(w, out);
    }

#if BRUUN_LEVEL >= 1
    // ------------------------------------------------------------------
    // Mirrored-lane SIMD walk. Lanes of every vector hold node s (lane 0)
    // and node s + N/4 (lane 1); see the header commentary.
    // ------------------------------------------------------------------

    static BRUUN_ALWAYS_INLINE void pair_reduce_v2(bruun_v2 ea, bruun_v2 eb, bruun_v2 oa, bruun_v2 ob, bruun_v2 c2,
                                                   bruun_v2& la, bruun_v2& lb, bruun_v2& ha, bruun_v2& hb) {
        const bruun_v2 a = V2_SUB(ea, eb);
        const bruun_v2 c2sq = V2_MUL(c2, c2);
        const bruun_v2 b = V2_MADD(V2_SUB(oa, ob), c2sq, ob);  // (oa-ob) + c2^2*ob
        la = V2_MSUB(a, c2, ob);   // a - c2*ob
        ha = V2_MADD(a, c2, ob);   // a + c2*ob
        lb = V2_MADD(b, c2, eb);   // b + c2*eb
        hb = V2_MSUB(b, c2, eb);   // b - c2*eb
    }

    static BRUUN_ALWAYS_INLINE double v2_lane0(bruun_v2 v) {
#if defined(BRUUN_NEON_128)
        return vgetq_lane_f64(v, 0);
#elif defined(BRUUN_X86_128)
        return _mm_cvtsd_f64(v);
#else
        double lane[2];
        V2_ST(lane, v);
        return lane[0];
#endif
    }

    static BRUUN_ALWAYS_INLINE double v2_lane1(bruun_v2 v) {
#if defined(BRUUN_NEON_128)
        return vgetq_lane_f64(v, 1);
#elif defined(BRUUN_X86_128)
        return _mm_cvtsd_f64(V2_UNPHI(v, v));
#else
        double lane[2];
        V2_ST(lane, v);
        return lane[1];
#endif
    }

    // Seed: fused half-split + radix-2 merge into standard s=4 blocks, both
    // mirror lanes at once. rev(b + N/4) == rev(b) + 1 makes both input reads
    // contiguous V2 loads; rev(b + 1) == rev(b) + N/8 removes the second
    // table lookup.
    void seed_block_v(const double* input, double* vw, int base, int count) const {
        const int hq = half_ >> 1;
        for (int b = base; b < base + count; b += 2) {
            const int j0 = rev_[static_cast<std::size_t>(b)];
            const int j1 = j0 + hq;
            const bruun_v2 a0 = V2_LD(input + j0);
            const bruun_v2 c0 = V2_LD(input + j0 + half_);
            const bruun_v2 a1 = V2_LD(input + j1);
            const bruun_v2 c1 = V2_LD(input + j1 + half_);
            const bruun_v2 e0 = V2_ADD(a0, c0);
            const bruun_v2 e1 = V2_ADD(a1, c1);
            double* dst = vw + 4 * b;
            V2_ST(dst, V2_ADD(e0, e1));
            V2_ST(dst + 2, V2_SUB(e0, e1));
            V2_ST(dst + 4, V2_SUB(a0, c0));
            V2_ST(dst + 6, V2_SUB(a1, c1));
        }
    }

    // One radix-4 butterfly on mirrored lanes: child node i of all four
    // children -> parent nodes {i, 4M-i, 2M-i, 2M+i}. Pure vertical SIMD.
    BRUUN_ALWAYS_INLINE void butterfly4_v(bruun_v2 a0, bruun_v2 b0, bruun_v2 a1, bruun_v2 b1,
                                          bruun_v2 a2, bruun_v2 b2, bruun_v2 a3, bruun_v2 b3,
                                          double t2, double t1, double t1q, bruun_v2* o) const {
        const bruun_v2 t2v = V2_SET1(t2);
        bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
        pair_reduce_v2(a0, b0, a1, b1, t2v, h0la, h0lb, h0ha, h0hb);
        pair_reduce_v2(a2, b2, a3, b3, t2v, h1la, h1lb, h1ha, h1hb);
        pair_reduce_v2(h0la, h0lb, h1la, h1lb, V2_SET1(t1), o[0], o[1], o[2], o[3]);
        pair_reduce_v2(h0ha, h0hb, h1ha, h1hb, V2_SET1(t1q), o[4], o[5], o[6], o[7]);
    }

    static BRUUN_ALWAYS_INLINE void store_butterfly4_v(double* vblk, int M, int i, const bruun_v2* o) {
        V2_ST(vblk + 4 * i, o[0]);
        V2_ST(vblk + 4 * i + 2, o[1]);
        V2_ST(vblk + 4 * (4 * M - i), o[2]);
        V2_ST(vblk + 4 * (4 * M - i) + 2, o[3]);
        V2_ST(vblk + 4 * (2 * M - i), o[4]);
        V2_ST(vblk + 4 * (2 * M - i) + 2, o[5]);
        V2_ST(vblk + 4 * (2 * M + i), o[6]);
        V2_ST(vblk + 4 * (2 * M + i) + 2, o[7]);
    }

    // In-place radix-4 merge of four child blocks of M nodes each, both
    // mirror lanes at once. The (i, M-i) pairing exists purely so all reads
    // of a slot land before the palindromic writes that overwrite it.
    void merge4_v(double* vblk, int M, const double* t2, const double* t1) const {
        double* c0 = vblk;
        double* c1 = vblk + 4 * M;
        double* c2 = vblk + 8 * M;
        double* c3 = vblk + 12 * M;

        const bruun_v2 dc0 = V2_LD(c0);
        const bruun_v2 ny0 = V2_LD(c0 + 2);
        const bruun_v2 dc1 = V2_LD(c1);
        const bruun_v2 ny1 = V2_LD(c1 + 2);
        const bruun_v2 dc2 = V2_LD(c2);
        const bruun_v2 ny2 = V2_LD(c2 + 2);
        const bruun_v2 dc3 = V2_LD(c3);
        const bruun_v2 ny3 = V2_LD(c3 + 2);
        bruun_v2 ma, mb, mha, mhb;
        pair_reduce_v2(ny0, ny1, ny2, ny3, V2_SET1(t1[M - 1]), ma, mb, mha, mhb);
        const bruun_v2 h0dc = V2_ADD(dc0, dc1);
        const bruun_v2 h0ny = V2_SUB(dc0, dc1);
        const bruun_v2 h1dc = V2_ADD(dc2, dc3);
        const bruun_v2 h1ny = V2_SUB(dc2, dc3);
        V2_ST(c0, V2_ADD(h0dc, h1dc));          // node 0: (dc, ny)
        V2_ST(c0 + 2, V2_SUB(h0dc, h1dc));
        V2_ST(c1, ma);                          // node M (angle pi/4 of block)
        V2_ST(c1 + 2, mb);
        V2_ST(c2, h0ny);                        // node 2M (angle pi/2)
        V2_ST(c2 + 2, h1ny);
        V2_ST(c3, mha);                         // node 3M
        V2_ST(c3 + 2, mhb);

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            const bruun_v2 a0i = V2_LD(c0 + 4 * i);
            const bruun_v2 b0i = V2_LD(c0 + 4 * i + 2);
            const bruun_v2 a1i = V2_LD(c1 + 4 * i);
            const bruun_v2 b1i = V2_LD(c1 + 4 * i + 2);
            const bruun_v2 a2i = V2_LD(c2 + 4 * i);
            const bruun_v2 b2i = V2_LD(c2 + 4 * i + 2);
            const bruun_v2 a3i = V2_LD(c3 + 4 * i);
            const bruun_v2 b3i = V2_LD(c3 + 4 * i + 2);
            const bruun_v2 a0m = V2_LD(c0 + 4 * m);
            const bruun_v2 b0m = V2_LD(c0 + 4 * m + 2);
            const bruun_v2 a1m = V2_LD(c1 + 4 * m);
            const bruun_v2 b1m = V2_LD(c1 + 4 * m + 2);
            const bruun_v2 a2m = V2_LD(c2 + 4 * m);
            const bruun_v2 b2m = V2_LD(c2 + 4 * m + 2);
            const bruun_v2 a3m = V2_LD(c3 + 4 * m);
            const bruun_v2 b3m = V2_LD(c3 + 4 * m + 2);
            bruun_v2 oi[8];
            bruun_v2 om[8];
            butterfly4_v(a0i, b0i, a1i, b1i, a2i, b2i, a3i, b3i,
                         t2[i - 1], t1[i - 1], t1[2 * M - i - 1], oi);
            butterfly4_v(a0m, b0m, a1m, b1m, a2m, b2m, a3m, b3m,
                         t2[m - 1], t1[m - 1], t1[2 * M - m - 1], om);
            store_butterfly4_v(vblk, M, i, oi);
            store_butterfly4_v(vblk, M, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            const bruun_v2 a0i = V2_LD(c0 + 4 * i);
            const bruun_v2 b0i = V2_LD(c0 + 4 * i + 2);
            const bruun_v2 a1i = V2_LD(c1 + 4 * i);
            const bruun_v2 b1i = V2_LD(c1 + 4 * i + 2);
            const bruun_v2 a2i = V2_LD(c2 + 4 * i);
            const bruun_v2 b2i = V2_LD(c2 + 4 * i + 2);
            const bruun_v2 a3i = V2_LD(c3 + 4 * i);
            const bruun_v2 b3i = V2_LD(c3 + 4 * i + 2);
            bruun_v2 oi[8];
            butterfly4_v(a0i, b0i, a1i, b1i, a2i, b2i, a3i, b3i,
                         t2[i - 1], t1[i - 1], t1[2 * M - i - 1], oi);
            store_butterfly4_v(vblk, M, i, oi);
        }
    }

    // Seed one fused slot block: vector seed plus in-cache radix-4 merges up
    // to `count` slots (standard block size 2*count per lane).
    void seed_fused_block_v(const double* input, double* vw, int base, int count) const {
        seed_block_v(input, vw, base, count);
        int stage = 2;
        for (int s = 16; (s >> 1) <= count; s <<= 2, stage += 2) {
            const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int sub = base; sub < base + count; sub += (s >> 1)) {
                merge4_v(vw + 4 * sub, M, t2, t1);
            }
        }
    }

    // Everything below the terminal merge. Seed blocks run in bit-reversed
    // block order so the bit-reversed gather reads the input as sequential
    // comb streams; the contiguous block writes tolerate any order.
    void compute_vwork(const double* input, double* vw) const {
        BRUUN_ASSERT(n_ >= 16);
        const int q = half_ >> 1;
        const int chain_limit = radix2_terminal_ ? (n_ >> 1) : (n_ >> 2);
        // Fused seed depth: one whole L1-resident transform when n == 4096;
        // 256 for larger n, where a deeper seed block widens the gather comb
        // past what the load streams tolerate (measured regression).
        const int s_target = n_ > 4096 ? 256 : chain_limit;
        const int block_slots = s_target >> 1;
        const int nb = q / block_slots;
        const int nb_bits = ilog2_pow2(nb);
        for (int p = 0; p < nb; ++p) {
            const int base = bitrev_int(p, nb_bits) * block_slots;
            seed_fused_block_v(input, vw, base, block_slots);
        }

        // Remaining stages as plain per-stage sweeps. A radix-16 paired sweep
        // was tried here and measured slower at n=16384, neutral elsewhere.
        int s = s_target << 2;
        int stage = ilog2_pow2(s) - 2;
        for (; s <= chain_limit; s <<= 2, stage += 2) {
            const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int off = 0; off < n_; off += 2 * s) {
                merge4_v(vw + off, M, t2, t1);
            }
        }
    }

    template <typename Complex>
    struct terminal_complex_writer {
        Complex* X;
        BRUUN_ALWAYS_INLINE void store_bin(int k, double re, double im) const {
            X[k].re = re;
            X[k].im = im;
        }
        BRUUN_ALWAYS_INLINE void store_pair(int k0, int k1, bruun_v2 re, bruun_v2 im) const {
            V2_ST(&X[k0].re, V2_UNPLO(re, im));
            V2_ST(&X[k1].re, V2_UNPHI(re, im));
        }
    };

    struct terminal_split_writer {
        double* re_;
        double* im_;
        BRUUN_ALWAYS_INLINE void store_bin(int k, double re, double im) const {
            re_[k] = re;
            im_[k] = im;
        }
        BRUUN_ALWAYS_INLINE void store_pair(int k0, int k1, bruun_v2 re, bruun_v2 im) const {
            re_[k0] = v2_lane0(re);
            im_[k0] = v2_lane0(im);
            re_[k1] = v2_lane1(re);
            im_[k1] = v2_lane1(im);
        }
    };

    // Terminal merge: the single lane crossing of the walk, fused with the
    // residue -> complex projection. Children: lane 0 holds {C0 | C1}, lane 1
    // holds {C2 | C3}; level A reduces (C0,C1) and (C2,C3) vertically, one
    // trn pair rotates the lane axis, level B and the projection stay vector.
    template <typename Writer>
    void terminal_v(const double* vw, const Writer& w) const {
        const int M = half_ >> 2;   // child node count = n/8
        const double* cA = vw;
        const double* cB = vw + 4 * M;
        const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 2)];
        const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];

        const bruun_v2 dcA = V2_LD(cA);
        const bruun_v2 nyA = V2_LD(cA + 2);
        const bruun_v2 dcB = V2_LD(cB);
        const bruun_v2 nyB = V2_LD(cB + 2);
        const bruun_v2 sum = V2_ADD(dcA, dcB);   // {h0dc, h1dc}
        const bruun_v2 dif = V2_SUB(dcA, dcB);   // {h0ny, h1ny}
        w.store_bin(0, v2_lane0(sum) + v2_lane1(sum), 0.0);
        w.store_bin(half_, v2_lane0(sum) - v2_lane1(sum), 0.0);
        w.store_bin(2 * M, v2_lane0(dif), -v2_lane1(dif));   // bin n/4: cos=0, -sin=-1
        {
            double la, lb, ha, hb;
            pair_reduce(v2_lane0(nyA), v2_lane0(nyB), v2_lane1(nyA), v2_lane1(nyB),
                        t1[M - 1], la, lb, ha, hb);
            w.store_bin(M, la + lb * cos_[static_cast<std::size_t>(M)],
                        lb * neg_sin_[static_cast<std::size_t>(M)]);
            w.store_bin(half_ - M, ha + hb * cos_[static_cast<std::size_t>(half_ - M)],
                        hb * neg_sin_[static_cast<std::size_t>(half_ - M)]);
        }

        const double* tw = term_tw_.data();
        for (int i = 1; i < M; ++i, tw += 6) {
            const bruun_v2 aA = V2_LD(cA + 4 * i);
            const bruun_v2 bA = V2_LD(cA + 4 * i + 2);
            const bruun_v2 aB = V2_LD(cB + 4 * i);
            const bruun_v2 bB = V2_LD(cB + 4 * i + 2);
            bruun_v2 la, lb, ha, hb;
            pair_reduce_v2(aA, bA, aB, bB, V2_SET1(t2[i - 1]), la, lb, ha, hb);
            const bruun_v2 ua = V2_UNPLO(la, ha);   // {h0 low, h0 high}
            const bruun_v2 ub = V2_UNPLO(lb, hb);
            const bruun_v2 va = V2_UNPHI(la, ha);   // {h1 low, h1 high}
            const bruun_v2 vb = V2_UNPHI(lb, hb);
            bruun_v2 fa, fb, ga, gb;
            pair_reduce_v2(ua, ub, va, vb, V2_LD(tw), fa, fb, ga, gb);
            // fa/fb lanes = bins {i, n/4 - i}; ga/gb lanes = bins {n/2 - i, n/4 + i}.
            const bruun_v2 cs = V2_LD(tw + 2);    // { cos, sin }
            const bruun_v2 msc = V2_LD(tw + 4);   // { -sin, -cos }
            w.store_pair(i, 2 * M - i, V2_MADD(fa, fb, cs), V2_MUL(fb, msc));
            w.store_pair(half_ - i, 2 * M + i, V2_MSUB(ga, gb, cs), V2_MUL(gb, msc));
        }
    }

    // Odd powers of two end with a radix-2 lane crossing. Below this point the
    // mirrored lanes have already run the full radix-4 chain to N/2.
    template <typename Writer>
    void terminal_radix2_v(const double* vw, const Writer& w) const {
        const int q = half_ >> 1;
        const double* t = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];

        const bruun_v2 dc = V2_LD(vw);
        const bruun_v2 ny = V2_LD(vw + 2);
        w.store_bin(0, v2_lane0(dc) + v2_lane1(dc), 0.0);
        w.store_bin(half_, v2_lane0(dc) - v2_lane1(dc), 0.0);
        w.store_bin(q, v2_lane0(ny), -v2_lane1(ny));

        int i = 1;
        for (; i + 1 < q; i += 2) {
            const bruun_v2 a0 = V2_LD(vw + 4 * i);
            const bruun_v2 b0 = V2_LD(vw + 4 * i + 2);
            const bruun_v2 a1 = V2_LD(vw + 4 * (i + 1));
            const bruun_v2 b1 = V2_LD(vw + 4 * (i + 1) + 2);
            const bruun_v2 ea = V2_UNPLO(a0, a1);
            const bruun_v2 oa = V2_UNPHI(a0, a1);
            const bruun_v2 eb = V2_UNPLO(b0, b1);
            const bruun_v2 ob = V2_UNPHI(b0, b1);
            bruun_v2 la, lb, ha, hb;
            pair_reduce_v2(ea, eb, oa, ob, V2_LD(t + i - 1), la, lb, ha, hb);

            const bruun_v2 cs_lo = V2_LD(cos_.data() + i);
            const bruun_v2 ns_lo = V2_LD(neg_sin_.data() + i);
            w.store_pair(i, i + 1, V2_MADD(la, lb, cs_lo), V2_MUL(lb, ns_lo));

            const int hi0 = half_ - i;
            const int hi1 = half_ - i - 1;
            const bruun_v2 cs_hi = V2_SETLH(cos_[static_cast<std::size_t>(hi0)],
                                            cos_[static_cast<std::size_t>(hi1)]);
            const bruun_v2 ns_hi = V2_SETLH(neg_sin_[static_cast<std::size_t>(hi0)],
                                            neg_sin_[static_cast<std::size_t>(hi1)]);
            w.store_pair(hi0, hi1, V2_MADD(ha, hb, cs_hi), V2_MUL(hb, ns_hi));
        }

        if (i < q) {
            const bruun_v2 a = V2_LD(vw + 4 * i);
            const bruun_v2 b = V2_LD(vw + 4 * i + 2);
            double la, lb, ha, hb;
            pair_reduce(v2_lane0(a), v2_lane0(b), v2_lane1(a), v2_lane1(b),
                        t[i - 1], la, lb, ha, hb);
            const int hi = half_ - i;
            w.store_bin(i,
                        la + lb * cos_[static_cast<std::size_t>(i)],
                        lb * neg_sin_[static_cast<std::size_t>(i)]);
            w.store_bin(hi,
                        ha + hb * cos_[static_cast<std::size_t>(hi)],
                        hb * neg_sin_[static_cast<std::size_t>(hi)]);
        }
    }

    // ------------------------------------------------------------------
    // Mirrored-lane inverse: the same walk backwards. Standard bins enter
    // through the terminal split (the only lane crossing, fused with
    // deprojection), every split below is lane-vertical, and the seed
    // scatter leaves as contiguous V2 stores at rev(b).
    // ------------------------------------------------------------------

    static BRUUN_ALWAYS_INLINE void pair_expand_v2(bruun_v2 la, bruun_v2 lb, bruun_v2 ha, bruun_v2 hb,
                                                   bruun_v2 ic, bruun_v2 w, bruun_v2 hv,
                                                   bruun_v2& ea, bruun_v2& eb, bruun_v2& oa, bruun_v2& ob) {
        ob = V2_MUL(V2_SUB(ha, la), ic);
        eb = V2_MUL(V2_SUB(lb, hb), ic);
        ea = V2_MADD(eb, V2_ADD(la, ha), hv);            // eb + 0.5*(la+ha)
        oa = V2_MADD(V2_MUL(V2_ADD(lb, hb), hv), ob, w); // 0.5*(lb+hb) + w*ob
    }

    // Exact inverse of butterfly4_v: parent nodes {i, 4M-i, 2M-i, 2M+i} back
    // to child node i of the four children. o[] = {C0a,C0b,C1a,C1b,C2a,...}.
    BRUUN_ALWAYS_INLINE void inv_butterfly4_v(bruun_v2 nia, bruun_v2 nib, bruun_v2 nha, bruun_v2 nhb,
                                              bruun_v2 nqla, bruun_v2 nqlb, bruun_v2 nqha, bruun_v2 nqhb,
                                              const double* it1i, const double* it1q, const double* it2i,
                                              bruun_v2 hv, bruun_v2* o) const {
        bruun_v2 h0la, h0lb, h1la, h1lb, h0ha, h0hb, h1ha, h1hb;
        pair_expand_v2(nia, nib, nha, nhb, V2_SET1(it1i[0]), V2_SET1(it1i[1]), hv,
                       h0la, h0lb, h1la, h1lb);
        pair_expand_v2(nqla, nqlb, nqha, nqhb, V2_SET1(it1q[0]), V2_SET1(it1q[1]), hv,
                       h0ha, h0hb, h1ha, h1hb);
        const bruun_v2 ic2 = V2_SET1(it2i[0]);
        const bruun_v2 w2 = V2_SET1(it2i[1]);
        pair_expand_v2(h0la, h0lb, h0ha, h0hb, ic2, w2, hv, o[0], o[1], o[2], o[3]);
        pair_expand_v2(h1la, h1lb, h1ha, h1hb, ic2, w2, hv, o[4], o[5], o[6], o[7]);
    }

    static BRUUN_ALWAYS_INLINE void store_inv_butterfly4_v(double* vblk, int M, int i, const bruun_v2* o) {
        V2_ST(vblk + 4 * i, o[0]);
        V2_ST(vblk + 4 * i + 2, o[1]);
        V2_ST(vblk + 4 * (M + i), o[2]);
        V2_ST(vblk + 4 * (M + i) + 2, o[3]);
        V2_ST(vblk + 4 * (2 * M + i), o[4]);
        V2_ST(vblk + 4 * (2 * M + i) + 2, o[5]);
        V2_ST(vblk + 4 * (3 * M + i), o[6]);
        V2_ST(vblk + 4 * (3 * M + i) + 2, o[7]);
    }

    // In-place inverse of merge4_v. The (i, M-i) pairing makes each
    // iteration closed: the parent read set equals the child write set.
    void split4_v(double* vblk, int M, const double* it2, const double* it1) const {
        const bruun_v2 hv = V2_SET1(0.5);
        double* c0 = vblk;
        double* c1 = vblk + 4 * M;
        double* c2 = vblk + 8 * M;
        double* c3 = vblk + 12 * M;

        const bruun_v2 n0a = V2_LD(c0);
        const bruun_v2 n0b = V2_LD(c0 + 2);
        const bruun_v2 nMa = V2_LD(c1);
        const bruun_v2 nMb = V2_LD(c1 + 2);
        const bruun_v2 n2a = V2_LD(c2);
        const bruun_v2 n2b = V2_LD(c2 + 2);
        const bruun_v2 n3a = V2_LD(c3);
        const bruun_v2 n3b = V2_LD(c3 + 2);
        const bruun_v2 h0dc = V2_MUL(V2_ADD(n0a, n0b), hv);
        const bruun_v2 h1dc = V2_MUL(V2_SUB(n0a, n0b), hv);
        bruun_v2 ny0, ny1, ny2, ny3;
        pair_expand_v2(nMa, nMb, n3a, n3b,
                       V2_SET1(it1[2 * (M - 1)]), V2_SET1(it1[2 * (M - 1) + 1]), hv,
                       ny0, ny1, ny2, ny3);
        V2_ST(c0, V2_MUL(V2_ADD(h0dc, n2a), hv));
        V2_ST(c0 + 2, ny0);
        V2_ST(c1, V2_MUL(V2_SUB(h0dc, n2a), hv));
        V2_ST(c1 + 2, ny1);
        V2_ST(c2, V2_MUL(V2_ADD(h1dc, n2b), hv));
        V2_ST(c2 + 2, ny2);
        V2_ST(c3, V2_MUL(V2_SUB(h1dc, n2b), hv));
        V2_ST(c3 + 2, ny3);

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            const bruun_v2 ia = V2_LD(vblk + 4 * i);
            const bruun_v2 ib = V2_LD(vblk + 4 * i + 2);
            const bruun_v2 iha = V2_LD(vblk + 4 * (4 * M - i));
            const bruun_v2 ihb = V2_LD(vblk + 4 * (4 * M - i) + 2);
            const bruun_v2 iqla = V2_LD(vblk + 4 * (2 * M - i));
            const bruun_v2 iqlb = V2_LD(vblk + 4 * (2 * M - i) + 2);
            const bruun_v2 iqha = V2_LD(vblk + 4 * (2 * M + i));
            const bruun_v2 iqhb = V2_LD(vblk + 4 * (2 * M + i) + 2);
            const bruun_v2 ma = V2_LD(vblk + 4 * m);
            const bruun_v2 mb = V2_LD(vblk + 4 * m + 2);
            const bruun_v2 mha = V2_LD(vblk + 4 * (4 * M - m));
            const bruun_v2 mhb = V2_LD(vblk + 4 * (4 * M - m) + 2);
            const bruun_v2 mqla = V2_LD(vblk + 4 * (2 * M - m));
            const bruun_v2 mqlb = V2_LD(vblk + 4 * (2 * M - m) + 2);
            const bruun_v2 mqha = V2_LD(vblk + 4 * (2 * M + m));
            const bruun_v2 mqhb = V2_LD(vblk + 4 * (2 * M + m) + 2);
            bruun_v2 oi[8];
            bruun_v2 om[8];
            inv_butterfly4_v(ia, ib, iha, ihb, iqla, iqlb, iqha, iqhb,
                             it1 + 2 * (i - 1), it1 + 2 * (2 * M - i - 1), it2 + 2 * (i - 1), hv, oi);
            inv_butterfly4_v(ma, mb, mha, mhb, mqla, mqlb, mqha, mqhb,
                             it1 + 2 * (m - 1), it1 + 2 * (2 * M - m - 1), it2 + 2 * (m - 1), hv, om);
            store_inv_butterfly4_v(vblk, M, i, oi);
            store_inv_butterfly4_v(vblk, M, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            const bruun_v2 ia = V2_LD(vblk + 4 * i);
            const bruun_v2 ib = V2_LD(vblk + 4 * i + 2);
            const bruun_v2 iha = V2_LD(vblk + 4 * (4 * M - i));
            const bruun_v2 ihb = V2_LD(vblk + 4 * (4 * M - i) + 2);
            const bruun_v2 iqla = V2_LD(vblk + 4 * (2 * M - i));
            const bruun_v2 iqlb = V2_LD(vblk + 4 * (2 * M - i) + 2);
            const bruun_v2 iqha = V2_LD(vblk + 4 * (2 * M + i));
            const bruun_v2 iqhb = V2_LD(vblk + 4 * (2 * M + i) + 2);
            bruun_v2 oi[8];
            inv_butterfly4_v(ia, ib, iha, ihb, iqla, iqlb, iqha, iqhb,
                             it1 + 2 * (i - 1), it1 + 2 * (2 * M - i - 1), it2 + 2 * (i - 1), hv, oi);
            store_inv_butterfly4_v(vblk, M, i, oi);
        }
    }

    // Inverse seed: read one completed mirrored-lane seed block linearly from
    // vwork. Mirror lanes are adjacent time samples, so each destination is a
    // contiguous V2 store at rev(b); block order carries the permutation cost.
    void seed_scatter_block_v(const double* vw, double* output, int base, int count) const {
        const bruun_v2 hv = V2_SET1(0.5);
        const int hq = half_ >> 1;
        for (int b = base; b < base + count; b += 2) {
            const int j0 = rev_[static_cast<std::size_t>(b)];
            const int j1 = j0 + hq;
            const double* src = vw + 4 * b;
            const bruun_v2 va = V2_LD(src);
            const bruun_v2 vb = V2_LD(src + 2);
            const bruun_v2 vc = V2_LD(src + 4);
            const bruun_v2 vd = V2_LD(src + 6);
            const bruun_v2 e0 = V2_MUL(V2_ADD(va, vb), hv);
            const bruun_v2 e1 = V2_MUL(V2_SUB(va, vb), hv);
            V2_ST(output + j0, V2_MUL(V2_ADD(e0, vc), hv));
            V2_ST(output + j0 + half_, V2_MUL(V2_SUB(e0, vc), hv));
            V2_ST(output + j1, V2_MUL(V2_ADD(e1, vd), hv));
            V2_ST(output + j1 + half_, V2_MUL(V2_SUB(e1, vd), hv));
        }
    }

    void seed_scatter_row_tiles_v(const double* vw, double* output,
                                  int block_slots, int nb, int nb_bits) const {
        const bruun_v2 hv = V2_SET1(0.5);
        const int hq = half_ >> 1;
        const int pairs = block_slots >> 1;
        const int row_bits = ilog2_pow2(pairs);

        for (int r = 0; r < pairs; ++r) {
            const int row = bitrev_int(r, row_bits);
            const int row_base = 2 * row * nb;
            double* row0 = output + row_base;
            double* row1 = output + row_base + half_;
            double* row2 = output + row_base + hq;
            double* row3 = output + row_base + hq + half_;
            for (int h = 0; h < nb; ++h) {
                const int col = bitrev_int(h, nb_bits);
                const int b = h * block_slots + 2 * r;
                const double* src = vw + 4 * b;
                const bruun_v2 va = V2_LD(src);
                const bruun_v2 vb = V2_LD(src + 2);
                const bruun_v2 vc = V2_LD(src + 4);
                const bruun_v2 vd = V2_LD(src + 6);
                const bruun_v2 e0 = V2_MUL(V2_ADD(va, vb), hv);
                const bruun_v2 e1 = V2_MUL(V2_SUB(va, vb), hv);
                V2_ST(row0 + 2 * col, V2_MUL(V2_ADD(e0, vc), hv));
                V2_ST(row1 + 2 * col, V2_MUL(V2_SUB(e0, vc), hv));
                V2_ST(row2 + 2 * col, V2_MUL(V2_ADD(e1, vd), hv));
                V2_ST(row3 + 2 * col, V2_MUL(V2_SUB(e1, vd), hv));
            }
        }
    }

    // Everything below the terminal split, in reverse: full-array split
    // sweeps down to the fused block size, then the tail blocks split into
    // seed state. Small sizes consume each block immediately with linear vwork
    // loads; larger sizes use the final output rows as cache-local tiles so
    // the bit-reversal transport never needs a global scrambled load pass or
    // extra intermediate storage.
    void compute_vwork_inv(double* vw, double* output) const {
        BRUUN_ASSERT(n_ >= 16);
        const int q = half_ >> 1;
        const int chain_limit = radix2_terminal_ ? (n_ >> 1) : (n_ >> 2);
        const int s_target = n_ > 4096 ? 256 : chain_limit;
        const int block_slots = s_target >> 1;
        const int nb = q / block_slots;
        const int nb_bits = ilog2_pow2(nb);

        int s = chain_limit;
        int stage = ilog2_pow2(s) - 2;
        for (; s >= (s_target << 2); s >>= 2, stage -= 2) {
            const double* it2 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* it1 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int off = 0; off < n_; off += 2 * s) {
                split4_v(vw + off, M, it2, it1);
            }
        }

        if (nb >= 64) {
            for (int p = 0; p < nb; ++p) {
                const int base = p * block_slots;
                int ss = s_target;
                int st = ilog2_pow2(ss) - 2;
                for (; ss >= 16; ss >>= 2, st -= 2) {
                    const double* it2 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st - 1)];
                    const double* it1 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st)];
                    const int M = ss >> 3;
                    for (int sub = base; sub < base + block_slots; sub += (ss >> 1)) {
                        split4_v(vw + 4 * sub, M, it2, it1);
                    }
                }
            }
            seed_scatter_row_tiles_v(vw, output, block_slots, nb, nb_bits);
            return;
        }

        for (int p = 0; p < nb; ++p) {
            const int base = bitrev_int(p, nb_bits) * block_slots;
            int ss = s_target;
            int st = ilog2_pow2(ss) - 2;
            for (; ss >= 16; ss >>= 2, st -= 2) {
                const double* it2 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st - 1)];
                const double* it1 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st)];
                const int M = ss >> 3;
                for (int sub = base; sub < base + block_slots; sub += (ss >> 1)) {
                    split4_v(vw + 4 * sub, M, it2, it1);
                }
            }
            seed_scatter_block_v(vw, output, base, block_slots);
        }
    }

    // Inverse terminal: standard bins -> mirrored-lane child slots, the
    // reverse of terminal_v fused with (re,im) -> (a,b) deprojection. The
    // only lane-crossing point of the inverse walk.
    template <typename Complex>
    void terminal_inv_v(const Complex* X, double* vw) const {
        const int M = half_ >> 2;   // child node count = n/8
        double* cA = vw;
        double* cB = vw + 4 * M;
        const double* it2 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 2)];
        const double* it1 = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
        const bruun_v2 hv = V2_SET1(0.5);

        const double h0dc = 0.5 * (X[0].re + X[half_].re);
        const double h1dc = 0.5 * (X[0].re - X[half_].re);
        const double h0ny = X[2 * M].re;
        const double h1ny = -X[2 * M].im;
        {
            double la, lb, ha, hb;
            lb = X[M].im * inv_neg_sin_[static_cast<std::size_t>(M)];
            la = X[M].re - lb * cos_[static_cast<std::size_t>(M)];
            hb = X[half_ - M].im * inv_neg_sin_[static_cast<std::size_t>(half_ - M)];
            ha = X[half_ - M].re - hb * cos_[static_cast<std::size_t>(half_ - M)];
            double ny0, ny1, ny2, ny3;
            pair_expand(la, lb, ha, hb, it1[2 * (M - 1)], it1[2 * (M - 1) + 1], ny0, ny1, ny2, ny3);
            V2_ST(cA, V2_SETLH(0.5 * (h0dc + h0ny), 0.5 * (h1dc + h1ny)));   // {C0dc, C2dc}
            V2_ST(cA + 2, V2_SETLH(ny0, ny2));                               // {C0ny, C2ny}
            V2_ST(cB, V2_SETLH(0.5 * (h0dc - h0ny), 0.5 * (h1dc - h1ny)));   // {C1dc, C3dc}
            V2_ST(cB + 2, V2_SETLH(ny1, ny3));                               // {C1ny, C3ny}
        }

        const double* tw = iterm_tw_.data();
        for (int i = 1; i < M; ++i, tw += 8) {
            const bruun_v2 xl0 = V2_LD(&X[i].re);
            const bruun_v2 xl1 = V2_LD(&X[2 * M - i].re);
            const bruun_v2 xh0 = V2_LD(&X[half_ - i].re);
            const bruun_v2 xh1 = V2_LD(&X[2 * M + i].re);
            const bruun_v2 re_l = V2_UNPLO(xl0, xl1);
            const bruun_v2 im_l = V2_UNPHI(xl0, xl1);
            const bruun_v2 re_h = V2_UNPLO(xh0, xh1);
            const bruun_v2 im_h = V2_UNPHI(xh0, xh1);
            const bruun_v2 cs = V2_LD(tw + 4);     // { cos, sin }
            const bruun_v2 minv = V2_LD(tw + 6);   // { -1/sin, -1/cos }
            const bruun_v2 fb = V2_MUL(im_l, minv);
            const bruun_v2 fa = V2_MSUB(re_l, fb, cs);
            const bruun_v2 gb = V2_MUL(im_h, minv);
            const bruun_v2 ga = V2_MADD(re_h, gb, cs);
            bruun_v2 ua, ub, va, vb;
            pair_expand_v2(fa, fb, ga, gb, V2_LD(tw), V2_LD(tw + 2), hv, ua, ub, va, vb);
            const bruun_v2 la = V2_UNPLO(ua, va);   // {h0 low, h1 low}
            const bruun_v2 lb = V2_UNPLO(ub, vb);
            const bruun_v2 ha = V2_UNPHI(ua, va);   // {h0 high, h1 high}
            const bruun_v2 hb = V2_UNPHI(ub, vb);
            bruun_v2 aA, bA, aB, bB;
            pair_expand_v2(la, lb, ha, hb,
                           V2_SET1(it2[2 * (i - 1)]), V2_SET1(it2[2 * (i - 1) + 1]), hv,
                           aA, bA, aB, bB);
            V2_ST(cA + 4 * i, aA);
            V2_ST(cA + 4 * i + 2, bA);
            V2_ST(cB + 4 * i, aB);
            V2_ST(cB + 4 * i + 2, bB);
        }
    }

    template <typename Complex>
    void terminal_radix2_inv_v(const Complex* X, double* vw) const {
        const int q = half_ >> 1;
        const double* it = inv_twiddle_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];

        V2_ST(vw, V2_SETLH(0.5 * (X[0].re + X[half_].re),
                           0.5 * (X[0].re - X[half_].re)));
        V2_ST(vw + 2, V2_SETLH(X[q].re, -X[q].im));

        int i = 1;
        for (; i + 1 < q; i += 2) {
            const bruun_v2 xl0 = V2_LD(&X[i].re);
            const bruun_v2 xl1 = V2_LD(&X[i + 1].re);
            const bruun_v2 re_l = V2_UNPLO(xl0, xl1);
            const bruun_v2 im_l = V2_UNPHI(xl0, xl1);

            const int hi0 = half_ - i;
            const int hi1 = half_ - i - 1;
            const bruun_v2 xh0 = V2_LD(&X[hi0].re);
            const bruun_v2 xh1 = V2_LD(&X[hi1].re);
            const bruun_v2 re_h = V2_UNPLO(xh0, xh1);
            const bruun_v2 im_h = V2_UNPHI(xh0, xh1);

            const bruun_v2 cs_l = V2_LD(cos_.data() + i);
            const bruun_v2 ins_l = V2_LD(inv_neg_sin_.data() + i);
            const bruun_v2 lb = V2_MUL(im_l, ins_l);
            const bruun_v2 la = V2_MSUB(re_l, lb, cs_l);

            const bruun_v2 cs_h = V2_SETLH(cos_[static_cast<std::size_t>(hi0)],
                                           cos_[static_cast<std::size_t>(hi1)]);
            const bruun_v2 ins_h = V2_SETLH(inv_neg_sin_[static_cast<std::size_t>(hi0)],
                                            inv_neg_sin_[static_cast<std::size_t>(hi1)]);
            const bruun_v2 hb = V2_MUL(im_h, ins_h);
            const bruun_v2 ha = V2_MSUB(re_h, hb, cs_h);

            bruun_v2 ea, eb, oa, ob;
            pair_expand_v2(la, lb, ha, hb,
                           V2_SETLH(it[2 * (i - 1)], it[2 * i]),
                           V2_SETLH(it[2 * (i - 1) + 1], it[2 * i + 1]),
                           V2_SET1(0.5),
                           ea, eb, oa, ob);

            V2_ST(vw + 4 * i, V2_UNPLO(ea, oa));
            V2_ST(vw + 4 * i + 2, V2_UNPLO(eb, ob));
            V2_ST(vw + 4 * (i + 1), V2_UNPHI(ea, oa));
            V2_ST(vw + 4 * (i + 1) + 2, V2_UNPHI(eb, ob));
        }

        if (i < q) {
            const int hi = half_ - i;
            const double lb = X[i].im * inv_neg_sin_[static_cast<std::size_t>(i)];
            const double la = X[i].re - lb * cos_[static_cast<std::size_t>(i)];
            const double hb = X[hi].im * inv_neg_sin_[static_cast<std::size_t>(hi)];
            const double ha = X[hi].re - hb * cos_[static_cast<std::size_t>(hi)];
            double ea, eb, oa, ob;
            pair_expand(la, lb, ha, hb, it[2 * (i - 1)], it[2 * (i - 1) + 1], ea, eb, oa, ob);
            V2_ST(vw + 4 * i, V2_SETLH(ea, oa));
            V2_ST(vw + 4 * i + 2, V2_SETLH(eb, ob));
        }
    }

    // ------------------------------------------------------------------
    // Interpolation inverse, SIMD path. The head is the only shuffle zone
    // (mirror of terminal_inv_v); every merge above it is pure vertical
    // V2 with broadcast constants.
    // ------------------------------------------------------------------

    // Fused head: gather bins {j, n/4-j, n/4+j, n/2-j}, deproject through
    // the shared {cos,sin}/{-1/sin,-1/cos} vectors, run the two level-1
    // merges lane-paired and the level-2 merge after one unpack rotation,
    // and store the completed 8-block contiguously.
    template <typename Complex>
    BRUUN_ALWAYS_INLINE void itp_head8_core_v(const Complex* X, int j, const double* r,
                                              bruun_v2& p0, bruun_v2& p2,
                                              bruun_v2& p4, bruun_v2& p6) const {
        const int q4 = n_ >> 2;
        const bruun_v2 hv = V2_SET1(0.5);
        const bruun_v2 xl0 = V2_LD(&X[j].re);
        const bruun_v2 xl1 = V2_LD(&X[q4 - j].re);
        const bruun_v2 xh0 = V2_LD(&X[half_ - j].re);
        const bruun_v2 xh1 = V2_LD(&X[q4 + j].re);
        const bruun_v2 re_l = V2_UNPLO(xl0, xl1);
        const bruun_v2 im_l = V2_UNPHI(xl0, xl1);
        const bruun_v2 re_h = V2_UNPLO(xh0, xh1);
        const bruun_v2 im_h = V2_UNPHI(xh0, xh1);
        const bruun_v2 cs = V2_LD(r);         // { cos, sin }
        const bruun_v2 minv = V2_LD(r + 2);   // { -1/sin, -1/cos }
        const bruun_v2 lb = V2_MUL(im_l, minv);
        const bruun_v2 la = V2_MSUB(re_l, lb, cs);
        const bruun_v2 hb = V2_MUL(im_h, minv);
        const bruun_v2 ha = V2_MADD(re_h, hb, cs);
        bruun_v2 ea, eb, oa, ob;
        pair_expand_v2(la, lb, ha, hb, V2_LD(r + 4), V2_LD(r + 6), hv, ea, eb, oa, ob);
        // lane 0 = lo child (bins j, n/2-j), lane 1 = hi child (n/4-/+j).
        const bruun_v2 la2 = V2_UNPLO(ea, oa);
        const bruun_v2 lb2 = V2_UNPLO(eb, ob);
        const bruun_v2 ha2 = V2_UNPHI(ea, oa);
        const bruun_v2 hb2 = V2_UNPHI(eb, ob);
        pair_expand_v2(la2, lb2, ha2, hb2, V2_SET1(r[8]), V2_SET1(r[9]), hv, p0, p4, p2, p6);
    }

    template <typename Complex>
    BRUUN_ALWAYS_INLINE void itp_head8_v(const Complex* X, double* blk, int j, const double* r) const {
        bruun_v2 p0, p2, p4, p6;
        itp_head8_core_v(X, j, r, p0, p2, p4, p6);
        V2_ST(blk, p0);
        V2_ST(blk + 2, p2);
        V2_ST(blk + 4, p4);
        V2_ST(blk + 6, p6);
    }

    // Odd-parity areas: two head groups plus their radix-2 16-merge, fused in
    // registers (still pure vertical - the 16-merge positions are exactly the
    // vectors the two head cores produce).
    template <typename Complex>
    BRUUN_ALWAYS_INLINE void itp_head16_v(const Complex* X, double* blk, int j0, int j1,
                                          const double* r0, const double* r1,
                                          double ic, double w) const {
        bruun_v2 a0, b0, c0, d0, a1, b1, c1, d1;
        itp_head8_core_v(X, j0, r0, a0, b0, c0, d0);
        itp_head8_core_v(X, j1, r1, a1, b1, c1, d1);
        const bruun_v2 hv = V2_SET1(0.5);
        const bruun_v2 icv = V2_SET1(ic);
        const bruun_v2 wv = V2_SET1(w);
        bruun_v2 ea, eb, oa, ob;
        pair_expand_v2(a0, c0, a1, c1, icv, wv, hv, ea, eb, oa, ob);
        V2_ST(blk, ea);
        V2_ST(blk + 4, oa);
        V2_ST(blk + 8, eb);
        V2_ST(blk + 12, ob);
        pair_expand_v2(b0, d0, b1, d1, icv, wv, hv, ea, eb, oa, ob);
        V2_ST(blk + 2, ea);
        V2_ST(blk + 6, oa);
        V2_ST(blk + 10, eb);
        V2_ST(blk + 14, ob);
    }

    void itp_merge2_v(double* blk, int s, double ic, double w) const {
        const bruun_v2 hv = V2_SET1(0.5);
        const bruun_v2 icv = V2_SET1(ic);
        const bruun_v2 wv = V2_SET1(w);
        const int q = s >> 2;
        for (int i = 0; i < q; i += 2) {
            bruun_v2 ea, eb, oa, ob;
            pair_expand_v2(V2_LD(blk + i), V2_LD(blk + i + q),
                           V2_LD(blk + i + 2 * q), V2_LD(blk + i + 3 * q),
                           icv, wv, hv, ea, eb, oa, ob);
            V2_ST(blk + i, ea);
            V2_ST(blk + i + q, oa);
            V2_ST(blk + i + 2 * q, eb);
            V2_ST(blk + i + 3 * q, ob);
        }
    }

    void itp_fused4_v(double* blk, int s, const double* r) const {
        const bruun_v2 hv = V2_SET1(0.5);
        const bruun_v2 icl = V2_SET1(r[0]);
        const bruun_v2 wl = V2_SET1(r[1]);
        const bruun_v2 ich = V2_SET1(r[2]);
        const bruun_v2 wh = V2_SET1(r[3]);
        const bruun_v2 icp = V2_SET1(r[4]);
        const bruun_v2 wp = V2_SET1(r[5]);
        const int q = s >> 3;
        const double* g0 = blk;
        const double* g1 = blk + 2 * q;
        const double* g2 = blk + 4 * q;
        const double* g3 = blk + 6 * q;
        for (int i = 0; i < q; i += 2) {
            bruun_v2 cl0, cl1, cl2, cl3, ch0, ch1, ch2, ch3;
            pair_expand_v2(V2_LD(g0 + i), V2_LD(g0 + i + q),
                           V2_LD(g1 + i), V2_LD(g1 + i + q),
                           icl, wl, hv, cl0, cl2, cl1, cl3);
            pair_expand_v2(V2_LD(g2 + i), V2_LD(g2 + i + q),
                           V2_LD(g3 + i), V2_LD(g3 + i + q),
                           ich, wh, hv, ch0, ch2, ch1, ch3);
            bruun_v2 p0, p1, p2, p3, p4, p5, p6, p7;
            pair_expand_v2(cl0, cl2, ch0, ch2, icp, wp, hv, p0, p4, p2, p6);
            pair_expand_v2(cl1, cl3, ch1, ch3, icp, wp, hv, p1, p5, p3, p7);
            V2_ST(blk + i, p0);
            V2_ST(blk + i + q, p1);
            V2_ST(blk + i + 2 * q, p2);
            V2_ST(blk + i + 3 * q, p3);
            V2_ST(blk + i + 4 * q, p4);
            V2_ST(blk + i + 5 * q, p5);
            V2_ST(blk + i + 6 * q, p6);
            V2_ST(blk + i + 7 * q, p7);
        }
    }

    void itp_rmerge_v(double* w, int S) const {
        const bruun_v2 hv = V2_SET1(0.5);
        for (int i = 0; i < S; i += 2) {
            const bruun_v2 a = V2_LD(w + i);
            const bruun_v2 b = V2_LD(w + S + i);
            V2_ST(w + i, V2_MUL(V2_ADD(a, b), hv));
            V2_ST(w + S + i, V2_MUL(V2_SUB(a, b), hv));
        }
    }

    // L2-resident sizes run two half-passes instead of one 12-stream pass:
    // u/v are re-read once, but each pass keeps at most 5 concurrent
    // power-of-2-strided streams, which measures ~20% faster there. Small
    // sizes stay fused (the re-read is pure overhead in L1).
    void itp_terminal_v(const double* w, double* out) const {
    const bruun_v2 hv = V2_SET1(0.5);
    const int q = n_ >> 2;

    const double* u  = w;
    const double* v  = w + q;
    const double* pl = w + 2 * q;
    const double* ph = w + 3 * q;

    for (int i = 0; i < q; i += 2) {
        const bruun_v2 uu  = V2_LD(u + i);
        const bruun_v2 vv  = V2_LD(v + i);
        const bruun_v2 plo = V2_LD(pl + i);
        const bruun_v2 phi = V2_LD(ph + i);

        const bruun_v2 rlo = V2_MUL(V2_ADD(uu, vv), hv);
        const bruun_v2 rhi = V2_MUL(V2_SUB(uu, vv), hv);

        V2_ST(out + i,         V2_MUL(V2_ADD(rlo, plo), hv));
        V2_ST(out + i + q,     V2_MUL(V2_ADD(rhi, phi), hv));
        V2_ST(out + i + 2 * q, V2_MUL(V2_SUB(rlo, plo), hv));
        V2_ST(out + i + 3 * q, V2_MUL(V2_SUB(rhi, phi), hv));
    }
}

    template <typename Complex>
    void inverse_interp_simd_impl(const Complex* X, double* out, double* w) const {
        BRUUN_ASSERT(n_ >= 16);
        itp_prologue(X, w);
        const int* jp = ihead_j_.data();
        const double* hr = ihead_tw_.data();
        const double* r2 = ir2_tw_.data();
        const double* sw = isw_tw_.data();
        for (int t = 0; t < itile_count_; ++t) {
            const int toff = itile_ord_[static_cast<std::size_t>(t)];
            const int S = itile_meta_[static_cast<std::size_t>(t)] >> 1;
            const bool odd = (itile_meta_[static_cast<std::size_t>(t)] & 1) != 0;
            const int s0 = odd ? 16 : 8;
            const int T = itp_area_tile(S, odd);
            if (odd) {
                for (int o = toff; o < toff + T; o += 16) {
                    itp_head16_v(X, w + o, jp[0], jp[1], hr, hr + 10, r2[0], r2[1]);
                    jp += 2;
                    hr += 20;
                    r2 += 2;
                }
            } else {
                for (int o = toff; o < toff + T; o += 8) {
                    itp_head8_v(X, w + o, *jp++, hr);
                    hr += 10;
                }
            }
            for (int s = 4 * s0; s <= T; s <<= 2) {
                for (int o = toff; o < toff + T; o += s) {
                    itp_fused4_v(w + o, s, sw);
                    sw += 6;
                }
            }
        }
        for (int S = 8; S <= half_; S <<= 1) {
            const bool odd = itp_area_odd(S);
            const int T = itp_area_tile(S, odd);
            itp_upper_walk(S, T, [&](int o, int s) {
                itp_fused4_v(w + o, s, sw);
                sw += 6;
            });
            if (S <= (n_ >> 3)) {
                itp_rmerge_v(w, S);
            }
        }
        itp_terminal_v(w, out);
    }
#endif  // BRUUN_LEVEL >= 1

    int n_;
    int half_;
    int twiddle_stage_count_;
    bool radix2_terminal_;
    heap_array<int> rev_;
    heap_array<int> twiddle_offset_;
    heap_array<double> cos_;
    heap_array<double> neg_sin_;
    heap_array<double> inv_neg_sin_;
    heap_array<double> twiddle_;
    heap_array<double> term_tw_;
    heap_array<double> inv_twiddle_;
    heap_array<double> iterm_tw_;
    heap_array<int> ihead_j_;
    heap_array<double> ihead_tw_;
    heap_array<double> ir2_tw_;
    heap_array<double> isw_tw_;
    heap_array<int> itile_ord_;
    heap_array<int> itile_meta_;
    int itile_count_ = 0;
};

} // namespace bruun
