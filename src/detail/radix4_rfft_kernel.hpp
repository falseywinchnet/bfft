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
// The scalar path below keeps the plain single-lane walk for reference and
// for BRUUN_LEVEL == 0 builds.

#include "bruun_kernel.hpp"

#include <cmath>
#include <cstddef>

namespace bruun {

class radix4_rfft_kernel {
public:
    radix4_rfft_kernel() noexcept
        : n_(0), half_(0), twiddle_stage_count_(0) {}

    bool reset(int n) {
        if (!valid_size(n)) {
            clear();
            return false;
        }

        n_ = n;
        half_ = n >> 1;

        const int logn = ilog2_pow2(n);
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
        for (int k = 0; k <= half_; ++k) {
            const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(n_);
            cos_[static_cast<std::size_t>(k)] = std::cos(theta);
            neg_sin_[static_cast<std::size_t>(k)] = -std::sin(theta);
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
            terminal_v(work, w);
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
            terminal_v(work, w);
            return;
        }
#endif
        forward_complex_scalar_impl(input, output, work);
    }

private:
    static bool valid_size(int n) {
        if (!is_power2(n)) {
            return false;
        }
        if (n < 4) {
            return false;
        }
        const int logn = ilog2_pow2(n);
        return (logn & 1) == 0;
    }

    void clear() noexcept {
        n_ = 0;
        half_ = 0;
        twiddle_stage_count_ = 0;
        rev_.clear();
        cos_.clear();
        neg_sin_.clear();
        twiddle_.clear();
        twiddle_offset_.clear();
        term_tw_.clear();
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
        // Fused seed depth: one whole L1-resident transform when n == 4096;
        // 256 for larger n, where a deeper seed block widens the gather comb
        // past what the load streams tolerate (measured regression).
        const int s_target = n_ > 4096 ? 256 : (n_ >> 2);
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
        for (; s <= (n_ >> 2); s <<= 2, stage += 2) {
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
#endif  // BRUUN_LEVEL >= 1

    int n_;
    int half_;
    int twiddle_stage_count_;
    heap_array<int> rev_;
    heap_array<int> twiddle_offset_;
    heap_array<double> cos_;
    heap_array<double> neg_sin_;
    heap_array<double> twiddle_;
    heap_array<double> term_tw_;
};

} // namespace bruun
