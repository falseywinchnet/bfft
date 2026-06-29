#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal radix-4 real FFT experiment kernel.
//
// This header keeps the standalone NEON prototype as a reusable kernel object:
// construction owns all planning tables, callers provide input/output/work
// buffers, and the SIMD path is selected through the shared BFFT two-lane
// backend abstraction instead of including one architecture directly.

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
        if (!pair_twiddle_offset_.resize(static_cast<std::size_t>(twiddle_stage_count_))) {
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

        int pair_twiddle_count = 0;
        for (int s = 4; s <= n_; s <<= 1) {
            const int q4 = s >> 2;
            const int h = q4 >> 1;
            const int pair_count = h > 1 ? ((h - 1) >> 1) : 0;
            if (pair_count > 0) {
                pair_twiddle_count += 6 * pair_count;
            }
        }
        int pair_alloc_count = pair_twiddle_count;
        if (pair_alloc_count == 0) {
            pair_alloc_count = 1;
        }
        if (!pair_twiddle_.resize(static_cast<std::size_t>(pair_alloc_count))) {
            clear();
            return false;
        }

        int pair_offset = 0;
        stage = 0;
        for (int s = 4; s <= n_; s <<= 1) {
            pair_twiddle_offset_[static_cast<std::size_t>(stage)] = pair_offset;
            const int q4 = s >> 2;
            const int h = q4 >> 1;
            const int pair_count = h > 1 ? ((h - 1) >> 1) : 0;
            if (pair_count > 0) {
                const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage - 1)];
                const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage)];
                for (int p = 0; p < pair_count; ++p) {
                    const int i = p + 1;
                    const int m = h - i;
                    double* dst = pair_twiddle_.data() + pair_offset + 6 * p;
                    dst[0] = t2[i - 1];
                    dst[1] = t2[m - 1];
                    dst[2] = t1[i - 1];
                    dst[3] = t1[m - 1];
                    dst[4] = t1[q4 - i - 1];
                    dst[5] = t1[q4 - m - 1];
                }
                pair_offset += 6 * pair_count;
            }
            ++stage;
        }
        return true;
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return half_ + 1; }
    int work_size() const noexcept { return n_; }

    void forward_scalar(const double* input, double* output_re, double* output_im, double* work) const {
        forward_impl<false>(input, output_re, output_im, work);
    }

    void forward_simd(const double* input, double* output_re, double* output_im, double* work) const {
        forward_impl<true>(input, output_re, output_im, work);
    }

    template <typename Complex>
    void forward_complex_scalar(const double* input, Complex* output, double* work) const {
        forward_complex_impl<false>(input, output, work);
    }

    template <typename Complex>
    void forward_complex_simd(const double* input, Complex* output, double* work) const {
        forward_complex_impl<true>(input, output, work);
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
        pair_twiddle_.clear();
        pair_twiddle_offset_.clear();
    }

    int stage_offset(int s) const {
        int stage = 0;
        int value = s;
        while (value > 4) {
            value >>= 1;
            ++stage;
        }
        return twiddle_offset_[static_cast<std::size_t>(stage)];
    }

    int stage_index(int s) const {
        int stage = 0;
        int value = s;
        while (value > 4) {
            value >>= 1;
            ++stage;
        }
        return stage;
    }

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

#if BRUUN_LEVEL >= 1
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

    static BRUUN_ALWAYS_INLINE bruun_v2 v2_swap(bruun_v2 v) {
#if defined(BRUUN_NEON_128)
        return vextq_f64(v, v, 1);
#elif defined(BRUUN_X86_128)
        return _mm_shuffle_pd(v, v, 1);
#else
        return V2_SETLH(v2_lane1(v), v2_lane0(v));
#endif
    }
#endif

    void merge2(double* block, int s) const {
        const int hs = s >> 1;
        const int q = s >> 2;
        const double* tw = twiddle_.data() + stage_offset(s);
        const double e_dc = block[0];
        const double e_ny = block[1];
        const double o_dc = block[hs];
        const double o_ny = block[hs + 1];
        block[0] = e_dc + o_dc;
        block[1] = e_dc - o_dc;
        block[2 * q] = e_ny;
        block[2 * q + 1] = o_ny;

        for (int k = 1; k < q - k; ++k) {
            const int m = q - k;
            double a0, b0, a1, b1, a2, b2, a3, b3;
            pair_reduce(block[2 * k], block[2 * k + 1], block[hs + 2 * k], block[hs + 2 * k + 1], tw[k - 1], a0, b0, a1, b1);
            pair_reduce(block[2 * m], block[2 * m + 1], block[hs + 2 * m], block[hs + 2 * m + 1], tw[m - 1], a2, b2, a3, b3);
            block[2 * k] = a0;
            block[2 * k + 1] = b0;
            block[2 * (hs - k)] = a1;
            block[2 * (hs - k) + 1] = b1;
            block[2 * m] = a2;
            block[2 * m + 1] = b2;
            block[2 * (hs - m)] = a3;
            block[2 * (hs - m) + 1] = b3;
        }

        if ((q & 1) == 0 && q >= 2) {
            const int k = q >> 1;
            double a0, b0, a1, b1;
            pair_reduce(block[2 * k], block[2 * k + 1], block[hs + 2 * k], block[hs + 2 * k + 1], tw[k - 1], a0, b0, a1, b1);
            block[2 * k] = a0;
            block[2 * k + 1] = b0;
            block[2 * (hs - k)] = a1;
            block[2 * (hs - k) + 1] = b1;
        }
    }

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

    template <typename Complex>
    BRUUN_ALWAYS_INLINE void write_projected_bin(Complex* output, int k, double a, double b) const {
        output[k].re = a + b * cos_[static_cast<std::size_t>(k)];
        output[k].im = b * neg_sin_[static_cast<std::size_t>(k)];
    }

#if BRUUN_LEVEL >= 1
    template <typename Complex>
    BRUUN_ALWAYS_INLINE void write_projected_pair(Complex* output, int k0, int k1, bruun_v2 a, bruun_v2 b) const {
        const bruun_v2 cv = V2_SETLH(cos_[static_cast<std::size_t>(k0)], cos_[static_cast<std::size_t>(k1)]);
        const bruun_v2 sv = V2_SETLH(neg_sin_[static_cast<std::size_t>(k0)], neg_sin_[static_cast<std::size_t>(k1)]);
        const bruun_v2 re = V2_MADD(a, b, cv);
        const bruun_v2 im = V2_MUL(b, sv);
        output[k0].re = v2_lane0(re);
        output[k0].im = v2_lane0(im);
        output[k1].re = v2_lane1(re);
        output[k1].im = v2_lane1(im);
    }

    template <typename Complex>
    BRUUN_ALWAYS_INLINE void write_projected_contiguous_pair(Complex* output, int k, bruun_v2 a, bruun_v2 b) const {
        const bruun_v2 cv = V2_LD(cos_.data() + k);
        const bruun_v2 sv = V2_LD(neg_sin_.data() + k);
        const bruun_v2 re = V2_MADD(a, b, cv);
        const bruun_v2 im = V2_MUL(b, sv);
        V2_ST(&output[k].re, V2_UNPLO(re, im));
        V2_ST(&output[k + 1].re, V2_UNPHI(re, im));
    }

    BRUUN_ALWAYS_INLINE void project_work_pair_split(const double* work, double* output_re, double* output_im,
                                                     int k) const {
        const bruun_v2 p0 = V2_LD(work + 2 * k);
        const bruun_v2 p1 = V2_LD(work + 2 * (k + 1));
        const bruun_v2 a = V2_UNPLO(p0, p1);
        const bruun_v2 b = V2_UNPHI(p0, p1);
        const bruun_v2 cv = V2_LD(cos_.data() + k);
        const bruun_v2 sv = V2_LD(neg_sin_.data() + k);
        V2_ST(output_re + k, V2_MADD(a, b, cv));
        V2_ST(output_im + k, V2_MUL(b, sv));
    }

    template <typename Complex>
    BRUUN_ALWAYS_INLINE void project_work_pair_complex(const double* work, Complex* output, int k) const {
        const bruun_v2 p0 = V2_LD(work + 2 * k);
        const bruun_v2 p1 = V2_LD(work + 2 * (k + 1));
        const bruun_v2 a = V2_UNPLO(p0, p1);
        const bruun_v2 b = V2_UNPHI(p0, p1);
        const bruun_v2 cv = V2_LD(cos_.data() + k);
        const bruun_v2 sv = V2_LD(neg_sin_.data() + k);
        const bruun_v2 re = V2_MADD(a, b, cv);
        const bruun_v2 im = V2_MUL(b, sv);
        V2_ST(&output[k].re, V2_UNPLO(re, im));
        V2_ST(&output[k + 1].re, V2_UNPHI(re, im));
    }

    void butterfly4_pair(const double* q0, const double* q1, const double* q2, const double* q3,
                         int i, int m, const double* twp, int q4, double* block, int hs) const {
        const bruun_v2 q0i = V2_LD(q0 + 2 * i);
        const bruun_v2 q0m = V2_LD(q0 + 2 * m);
        const bruun_v2 q1i = V2_LD(q1 + 2 * i);
        const bruun_v2 q1m = V2_LD(q1 + 2 * m);
        const bruun_v2 q2i = V2_LD(q2 + 2 * i);
        const bruun_v2 q2m = V2_LD(q2 + 2 * m);
        const bruun_v2 q3i = V2_LD(q3 + 2 * i);
        const bruun_v2 q3m = V2_LD(q3 + 2 * m);

        const bruun_v2 q0a = V2_UNPLO(q0i, q0m);
        const bruun_v2 q0b = V2_UNPHI(q0i, q0m);
        const bruun_v2 q1a = V2_UNPLO(q1i, q1m);
        const bruun_v2 q1b = V2_UNPHI(q1i, q1m);
        const bruun_v2 q2a = V2_UNPLO(q2i, q2m);
        const bruun_v2 q2b = V2_UNPHI(q2i, q2m);
        const bruun_v2 q3a = V2_UNPLO(q3i, q3m);
        const bruun_v2 q3b = V2_UNPHI(q3i, q3m);

        const bruun_v2 t2v = V2_LD(twp);
        const bruun_v2 t1v = V2_LD(twp + 2);
        const bruun_v2 t1q = V2_LD(twp + 4);

        bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
        pair_reduce_v2(q0a, q0b, q1a, q1b, t2v, h0la, h0lb, h0ha, h0hb);
        pair_reduce_v2(q2a, q2b, q3a, q3b, t2v, h1la, h1lb, h1ha, h1hb);

        bruun_v2 fa, fb, fpa, fpb, ga, gb, gpa, gpb;
        pair_reduce_v2(h0la, h0lb, h1la, h1lb, t1v, fa, fb, fpa, fpb);
        pair_reduce_v2(h0ha, h0hb, h1ha, h1hb, t1q, ga, gb, gpa, gpb);

        V2_ST(block + 2 * i, V2_UNPLO(fa, fb));
        V2_ST(block + 2 * m, V2_UNPHI(fa, fb));
        V2_ST(block + 2 * (hs - i), V2_UNPLO(fpa, fpb));
        V2_ST(block + 2 * (hs - m), V2_UNPHI(fpa, fpb));
        V2_ST(block + 2 * (q4 - i), V2_UNPLO(ga, gb));
        V2_ST(block + 2 * (q4 - m), V2_UNPHI(ga, gb));
        V2_ST(block + 2 * (q4 + i), V2_UNPLO(gpa, gpb));
        V2_ST(block + 2 * (q4 + m), V2_UNPHI(gpa, gpb));
    }

    template <typename Complex>
    void butterfly4_terminal_pair(const double* q0, const double* q1, const double* q2, const double* q3,
                                  int i, int m, const double* twp, int q4, Complex* output, int hs) const {
        const bruun_v2 q0i = V2_LD(q0 + 2 * i);
        const bruun_v2 q0m = V2_LD(q0 + 2 * m);
        const bruun_v2 q1i = V2_LD(q1 + 2 * i);
        const bruun_v2 q1m = V2_LD(q1 + 2 * m);
        const bruun_v2 q2i = V2_LD(q2 + 2 * i);
        const bruun_v2 q2m = V2_LD(q2 + 2 * m);
        const bruun_v2 q3i = V2_LD(q3 + 2 * i);
        const bruun_v2 q3m = V2_LD(q3 + 2 * m);

        const bruun_v2 q0a = V2_UNPLO(q0i, q0m);
        const bruun_v2 q0b = V2_UNPHI(q0i, q0m);
        const bruun_v2 q1a = V2_UNPLO(q1i, q1m);
        const bruun_v2 q1b = V2_UNPHI(q1i, q1m);
        const bruun_v2 q2a = V2_UNPLO(q2i, q2m);
        const bruun_v2 q2b = V2_UNPHI(q2i, q2m);
        const bruun_v2 q3a = V2_UNPLO(q3i, q3m);
        const bruun_v2 q3b = V2_UNPHI(q3i, q3m);

        const bruun_v2 t2v = V2_LD(twp);
        const bruun_v2 t1v = V2_LD(twp + 2);
        const bruun_v2 t1q = V2_LD(twp + 4);

        bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
        pair_reduce_v2(q0a, q0b, q1a, q1b, t2v, h0la, h0lb, h0ha, h0hb);
        pair_reduce_v2(q2a, q2b, q3a, q3b, t2v, h1la, h1lb, h1ha, h1hb);

        bruun_v2 fa, fb, fpa, fpb, ga, gb, gpa, gpb;
        pair_reduce_v2(h0la, h0lb, h1la, h1lb, t1v, fa, fb, fpa, fpb);
        pair_reduce_v2(h0ha, h0hb, h1ha, h1hb, t1q, ga, gb, gpa, gpb);

        write_projected_pair(output, i, m, fa, fb);
        write_projected_pair(output, hs - i, hs - m, fpa, fpb);
        write_projected_pair(output, q4 - i, q4 - m, ga, gb);
        write_projected_pair(output, q4 + i, q4 + m, gpa, gpb);
    }

    template <typename Complex>
    void butterfly4_terminal_adjacent_pair(const double* q0, const double* q1, const double* q2, const double* q3,
                                           int i, const double* t2, const double* t1, int q4,
                                           Complex* output, int hs) const {
        const bruun_v2 q0i = V2_LD(q0 + 2 * i);
        const bruun_v2 q0j = V2_LD(q0 + 2 * (i + 1));
        const bruun_v2 q1i = V2_LD(q1 + 2 * i);
        const bruun_v2 q1j = V2_LD(q1 + 2 * (i + 1));
        const bruun_v2 q2i = V2_LD(q2 + 2 * i);
        const bruun_v2 q2j = V2_LD(q2 + 2 * (i + 1));
        const bruun_v2 q3i = V2_LD(q3 + 2 * i);
        const bruun_v2 q3j = V2_LD(q3 + 2 * (i + 1));

        const bruun_v2 q0a = V2_UNPLO(q0i, q0j);
        const bruun_v2 q0b = V2_UNPHI(q0i, q0j);
        const bruun_v2 q1a = V2_UNPLO(q1i, q1j);
        const bruun_v2 q1b = V2_UNPHI(q1i, q1j);
        const bruun_v2 q2a = V2_UNPLO(q2i, q2j);
        const bruun_v2 q2b = V2_UNPHI(q2i, q2j);
        const bruun_v2 q3a = V2_UNPLO(q3i, q3j);
        const bruun_v2 q3b = V2_UNPHI(q3i, q3j);

        const bruun_v2 t2v = V2_LD(t2 + i - 1);
        const bruun_v2 t1v = V2_LD(t1 + i - 1);
        const bruun_v2 t1q = v2_swap(V2_LD(t1 + q4 - i - 2));

        bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
        pair_reduce_v2(q0a, q0b, q1a, q1b, t2v, h0la, h0lb, h0ha, h0hb);
        pair_reduce_v2(q2a, q2b, q3a, q3b, t2v, h1la, h1lb, h1ha, h1hb);

        bruun_v2 fa, fb, fpa, fpb, ga, gb, gpa, gpb;
        pair_reduce_v2(h0la, h0lb, h1la, h1lb, t1v, fa, fb, fpa, fpb);
        pair_reduce_v2(h0ha, h0hb, h1ha, h1hb, t1q, ga, gb, gpa, gpb);

        write_projected_contiguous_pair(output, i, fa, fb);
        write_projected_contiguous_pair(output, hs - i - 1, v2_swap(fpa), v2_swap(fpb));
        write_projected_contiguous_pair(output, q4 - i - 1, v2_swap(ga), v2_swap(gb));
        write_projected_contiguous_pair(output, q4 + i, gpa, gpb);
    }

#if defined(BRUUN_NEON_128)
    static BRUUN_ALWAYS_INLINE void store_child_pair64_lanes(double* packed, int p0, int p1,
                                                             bruun_v2 low_re, bruun_v2 low_im,
                                                             bruun_v2 high_re, bruun_v2 high_im) {
        V2_ST(packed + 4 * p0, V2_UNPLO(low_re, high_re));
        V2_ST(packed + 4 * p0 + 2, V2_UNPLO(low_im, high_im));
        V2_ST(packed + 4 * p1, V2_UNPHI(low_re, high_re));
        V2_ST(packed + 4 * p1 + 2, V2_UNPHI(low_im, high_im));
    }

    BRUUN_ALWAYS_INLINE void merge4_to_child_pairs64_neon(double* block, double* packed,
                                                          const double* t2, const double* t1, const double* twp) const {
        constexpr int q4 = 16;
        constexpr int h = 8;
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
        pair_reduce(q0ny, q1ny, q2ny, q3ny, t1[h - 1], b2a0, b2b0, b2a1, b2b1);
        block[0] = h0dc + h1dc;
        block[1] = h0dc - h1dc;
        block[2 * q4] = h0ny;
        block[2 * q4 + 1] = h1ny;
        packed[4 * (h - 1) + 0] = b2a0;
        packed[4 * (h - 1) + 1] = b2a1;
        packed[4 * (h - 1) + 2] = b2b0;
        packed[4 * (h - 1) + 3] = b2b1;

        for (int i = 1; i < h - i; ++i) {
            const int m = h - i;
            const bruun_v2 q0i = V2_LD(q0 + 2 * i);
            const bruun_v2 q0m = V2_LD(q0 + 2 * m);
            const bruun_v2 q1i = V2_LD(q1 + 2 * i);
            const bruun_v2 q1m = V2_LD(q1 + 2 * m);
            const bruun_v2 q2i = V2_LD(q2 + 2 * i);
            const bruun_v2 q2m = V2_LD(q2 + 2 * m);
            const bruun_v2 q3i = V2_LD(q3 + 2 * i);
            const bruun_v2 q3m = V2_LD(q3 + 2 * m);

            const bruun_v2 q0a = V2_UNPLO(q0i, q0m);
            const bruun_v2 q0b = V2_UNPHI(q0i, q0m);
            const bruun_v2 q1a = V2_UNPLO(q1i, q1m);
            const bruun_v2 q1b = V2_UNPHI(q1i, q1m);
            const bruun_v2 q2a = V2_UNPLO(q2i, q2m);
            const bruun_v2 q2b = V2_UNPHI(q2i, q2m);
            const bruun_v2 q3a = V2_UNPLO(q3i, q3m);
            const bruun_v2 q3b = V2_UNPHI(q3i, q3m);

            const bruun_v2 t2v = V2_LD(twp);
            const bruun_v2 t1v = V2_LD(twp + 2);
            const bruun_v2 t1q = V2_LD(twp + 4);
            twp += 6;

            bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
            pair_reduce_v2(q0a, q0b, q1a, q1b, t2v, h0la, h0lb, h0ha, h0hb);
            pair_reduce_v2(q2a, q2b, q3a, q3b, t2v, h1la, h1lb, h1ha, h1hb);

            bruun_v2 fa, fb, fpa, fpb, ga, gb, gpa, gpb;
            pair_reduce_v2(h0la, h0lb, h1la, h1lb, t1v, fa, fb, fpa, fpb);
            pair_reduce_v2(h0ha, h0hb, h1ha, h1hb, t1q, ga, gb, gpa, gpb);

            store_child_pair64_lanes(packed, i - 1, m - 1, fa, fb, fpa, fpb);
            store_child_pair64_lanes(packed, q4 - i - 1, q4 - m - 1, ga, gb, gpa, gpb);
        }

        {
            constexpr int i = h >> 1;
            double oi[8];
            butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
            packed[4 * (i - 1) + 0] = oi[0];
            packed[4 * (i - 1) + 1] = oi[2];
            packed[4 * (i - 1) + 2] = oi[1];
            packed[4 * (i - 1) + 3] = oi[3];
            packed[4 * (q4 - i - 1) + 0] = oi[4];
            packed[4 * (q4 - i - 1) + 1] = oi[6];
            packed[4 * (q4 - i - 1) + 2] = oi[5];
            packed[4 * (q4 - i - 1) + 3] = oi[7];
        }
    }

    BRUUN_ALWAYS_INLINE void merge4_child_pairs_256_neon(double* block, const double* packed,
                                                         const double* t2, const double* t1, const double* twp) const {
        constexpr int q4 = 64;
        constexpr int hs = 128;
        constexpr int h = 32;
        constexpr int pair_count = 15;
        const double* child[4] = { block, block + q4, block + 2 * q4, block + 3 * q4 };
        const double q0dc = block[0];
        const double q0ny = block[1];
        const double q1dc = block[q4];
        const double q1ny = block[q4 + 1];
        const double q2dc = block[2 * q4];
        const double q2ny = block[2 * q4 + 1];
        const double q3dc = block[3 * q4];
        const double q3ny = block[3 * q4 + 1];
        const double h0dc = q0dc + q1dc;
        const double h0ny = q0dc - q1dc;
        const double h1dc = q2dc + q3dc;
        const double h1ny = q2dc - q3dc;
        double b2a0, b2b0, b2a1, b2b1;
        pair_reduce(q0ny, q1ny, q2ny, q3ny, t1[h - 1], b2a0, b2b0, b2a1, b2b1);
        block[0] = h0dc + h1dc;
        block[1] = h0dc - h1dc;
        block[2 * q4] = h0ny;
        block[2 * q4 + 1] = h1ny;
        block[2 * h] = b2a0;
        block[2 * h + 1] = b2b0;
        block[2 * (hs - h)] = b2a1;
        block[2 * (hs - h) + 1] = b2b1;

        {
            constexpr int i = h >> 1;
            double oi[8];
            butterfly4_scalar(child[0], child[1], child[2], child[3],
                              i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
            write_butterfly4(block, hs, q4, i, oi);
        }

        for (int p = 0; p < pair_count; ++p) {
            const int i = p + 1;
            const int m = h - i;
            const double* q0p = packed + 0 * pair_count * 4 + p * 4;
            const double* q1p = packed + 1 * pair_count * 4 + p * 4;
            const double* q2p = packed + 2 * pair_count * 4 + p * 4;
            const double* q3p = packed + 3 * pair_count * 4 + p * 4;
            const bruun_v2 q0a = V2_LD(q0p);
            const bruun_v2 q0b = V2_LD(q0p + 2);
            const bruun_v2 q1a = V2_LD(q1p);
            const bruun_v2 q1b = V2_LD(q1p + 2);
            const bruun_v2 q2a = V2_LD(q2p);
            const bruun_v2 q2b = V2_LD(q2p + 2);
            const bruun_v2 q3a = V2_LD(q3p);
            const bruun_v2 q3b = V2_LD(q3p + 2);

            const bruun_v2 t2v = V2_LD(twp);
            const bruun_v2 t1v = V2_LD(twp + 2);
            const bruun_v2 t1q = V2_LD(twp + 4);
            twp += 6;

            bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
            pair_reduce_v2(q0a, q0b, q1a, q1b, t2v, h0la, h0lb, h0ha, h0hb);
            pair_reduce_v2(q2a, q2b, q3a, q3b, t2v, h1la, h1lb, h1ha, h1hb);

            bruun_v2 fa, fb, fpa, fpb, ga, gb, gpa, gpb;
            pair_reduce_v2(h0la, h0lb, h1la, h1lb, t1v, fa, fb, fpa, fpb);
            pair_reduce_v2(h0ha, h0hb, h1ha, h1hb, t1q, ga, gb, gpa, gpb);

            V2_ST(block + 2 * i, V2_UNPLO(fa, fb));
            V2_ST(block + 2 * m, V2_UNPHI(fa, fb));
            V2_ST(block + 2 * (hs - i), V2_UNPLO(fpa, fpb));
            V2_ST(block + 2 * (hs - m), V2_UNPHI(fpa, fpb));
            V2_ST(block + 2 * (q4 - i), V2_UNPLO(ga, gb));
            V2_ST(block + 2 * (q4 - m), V2_UNPHI(ga, gb));
            V2_ST(block + 2 * (q4 + i), V2_UNPLO(gpa, gpb));
            V2_ST(block + 2 * (q4 + m), V2_UNPHI(gpa, gpb));
        }
    }
#endif

#endif

    template <bool use_simd>
    void merge4_inplace(double* block, int s, const double* t2, const double* t1, const double* twp) const {
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
#if BRUUN_LEVEL >= 1
            if (use_simd) {
                butterfly4_pair(q0, q1, q2, q3, i, m, twp, q4, block, hs);
                twp += 6;
            } else
#endif
            {
                double oi[8];
                double om[8];
                butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
                butterfly4_scalar(q0, q1, q2, q3, m, t2[m - 1], t1[m - 1], t1[q4 - m - 1], om);
                write_butterfly4(block, hs, q4, i, oi);
                write_butterfly4(block, hs, q4, m, om);
            }
        }

        if (((q4 / 2) & 1) == 0 && q4 / 2 >= 2) {
            const int i = q4 >> 2;
            double oi[8];
            butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
            write_butterfly4(block, hs, q4, i, oi);
        }
    }

    template <bool use_simd, typename Complex>
    void merge4_terminal_complex(double* block, int s, const double* t2, const double* t1, const double* twp,
                                 Complex* output) const {
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
        output[0].re = h0dc + h1dc;
        output[0].im = 0.0;
        output[hs].re = h0dc - h1dc;
        output[hs].im = 0.0;
        write_projected_bin(output, q4, h0ny, h1ny);
        write_projected_bin(output, q4 / 2, b2a0, b2b0);
        write_projected_bin(output, hs - q4 / 2, b2a1, b2b1);

#if BRUUN_LEVEL >= 1
        if constexpr (use_simd) {
            int i = 1;
            for (; i + 1 < q4 / 2; i += 2) {
                butterfly4_terminal_adjacent_pair(q0, q1, q2, q3, i, t2, t1, q4, output, hs);
            }
            for (; i < q4 / 2; ++i) {
                double oi[8];
                butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
                write_projected_bin(output, i, oi[0], oi[1]);
                write_projected_bin(output, hs - i, oi[2], oi[3]);
                write_projected_bin(output, q4 - i, oi[4], oi[5]);
                write_projected_bin(output, q4 + i, oi[6], oi[7]);
            }
            (void)twp;
            return;
        }
#endif

        for (int i = 1; i < q4 / 2 - i; ++i) {
            const int m = q4 / 2 - i;
#if BRUUN_LEVEL >= 1
            if (use_simd) {
                butterfly4_terminal_pair(q0, q1, q2, q3, i, m, twp, q4, output, hs);
                twp += 6;
            } else
#endif
            {
                double oi[8];
                double om[8];
                butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
                butterfly4_scalar(q0, q1, q2, q3, m, t2[m - 1], t1[m - 1], t1[q4 - m - 1], om);
                write_projected_bin(output, i, oi[0], oi[1]);
                write_projected_bin(output, hs - i, oi[2], oi[3]);
                write_projected_bin(output, q4 - i, oi[4], oi[5]);
                write_projected_bin(output, q4 + i, oi[6], oi[7]);
                write_projected_bin(output, m, om[0], om[1]);
                write_projected_bin(output, hs - m, om[2], om[3]);
                write_projected_bin(output, q4 - m, om[4], om[5]);
                write_projected_bin(output, q4 + m, om[6], om[7]);
            }
        }

        if (((q4 / 2) & 1) == 0 && q4 / 2 >= 2) {
            const int i = q4 >> 2;
            double oi[8];
            butterfly4_scalar(q0, q1, q2, q3, i, t2[i - 1], t1[i - 1], t1[q4 - i - 1], oi);
            write_projected_bin(output, i, oi[0], oi[1]);
            write_projected_bin(output, hs - i, oi[2], oi[3]);
            write_projected_bin(output, q4 - i, oi[4], oi[5]);
            write_projected_bin(output, q4 + i, oi[6], oi[7]);
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

    BRUUN_ALWAYS_INLINE void seed_merge16_block(const double* input, double* work, int off16,
                                                const double* t16_2, const double* t16_1, const double* twp16) const {
        const int b0 = off16 >> 1;
        for (int b = b0; b < b0 + 8; b += 2) {
            seed_merge2_pair(input, work, b);
        }
        merge4_inplace<false>(work + off16, 16, t16_2, t16_1, twp16);
    }

    template <bool use_simd>
    BRUUN_ALWAYS_INLINE void seed_merge64_block(const double* input, double* work, int off64,
                                                const double* t16_2, const double* t16_1, const double* twp16,
                                                const double* t64_2, const double* t64_1, const double* twp64) const {
        for (int off16 = off64; off16 < off64 + 64; off16 += 16) {
            seed_merge16_block(input, work, off16, t16_2, t16_1, twp16);
        }
        if constexpr (use_simd) {
            merge4_inplace<true>(work + off64, 64, t64_2, t64_1, twp64);
        } else {
            merge4_inplace<false>(work + off64, 64, t64_2, t64_1, twp64);
        }
    }

    template <bool use_simd>
    BRUUN_ALWAYS_INLINE void seed_merge256_block(const double* input, double* work, int off256,
                                                 const double* t16_2, const double* t16_1, const double* twp16,
                                                 const double* t64_2, const double* t64_1, const double* twp64,
                                                 const double* t256_2, const double* t256_1, const double* twp256) const {
        if constexpr (use_simd) {
#if defined(BRUUN_NEON_128)
            constexpr int pair_count = 15;
            alignas(64) double packed[4 * pair_count * 4];
            for (int off16 = off256; off16 < off256 + 256; off16 += 16) {
                seed_merge16_block(input, work, off16, t16_2, t16_1, twp16);
            }
            for (int c = 0; c < 4; ++c) {
                const int off64 = off256 + 64 * c;
                merge4_to_child_pairs64_neon(work + off64, packed + c * pair_count * 4, t64_2, t64_1, twp64);
            }
            merge4_child_pairs_256_neon(work + off256, packed, t256_2, t256_1, twp256);
#else
            for (int off64 = off256; off64 < off256 + 256; off64 += 64) {
                seed_merge64_block<true>(input, work, off64, t16_2, t16_1, twp16, t64_2, t64_1, twp64);
            }
            merge4_inplace<true>(work + off256, 256, t256_2, t256_1, twp256);
#endif
        } else {
            for (int off64 = off256; off64 < off256 + 256; off64 += 64) {
                seed_merge64_block<false>(input, work, off64, t16_2, t16_1, twp16, t64_2, t64_1, twp64);
            }
            merge4_inplace<false>(work + off256, 256, t256_2, t256_1, twp256);
        }
    }

    template <bool use_simd>
    void compute_work(const double* input, double* work) const {
        BRUUN_ASSERT(n_ >= 4);
        const double* t16_2 = nullptr;
        const double* t16_1 = nullptr;
        const double* twp16 = nullptr;
        if (n_ >= 16) {
            t16_2 = twiddle_.data() + twiddle_offset_[1];
            t16_1 = twiddle_.data() + twiddle_offset_[2];
            twp16 = pair_twiddle_.data() + pair_twiddle_offset_[2];
        }
        if (n_ >= 256) {
            const double* t64_2 = twiddle_.data() + twiddle_offset_[3];
            const double* t64_1 = twiddle_.data() + twiddle_offset_[4];
            const double* twp64 = pair_twiddle_.data() + pair_twiddle_offset_[4];
            const double* t256_2 = twiddle_.data() + twiddle_offset_[5];
            const double* t256_1 = twiddle_.data() + twiddle_offset_[6];
            const double* twp256 = pair_twiddle_.data() + pair_twiddle_offset_[6];
            for (int off256 = 0; off256 < n_; off256 += 256) {
                seed_merge256_block<use_simd>(input, work, off256,
                                              t16_2, t16_1, twp16,
                                              t64_2, t64_1, twp64,
                                              t256_2, t256_1, twp256);
            }
        } else if (n_ >= 64) {
            const double* t64_2 = twiddle_.data() + twiddle_offset_[3];
            const double* t64_1 = twiddle_.data() + twiddle_offset_[4];
            const double* twp64 = pair_twiddle_.data() + pair_twiddle_offset_[4];
            for (int off64 = 0; off64 < n_; off64 += 64) {
                seed_merge64_block<use_simd>(input, work, off64, t16_2, t16_1, twp16, t64_2, t64_1, twp64);
            }
        } else if (n_ >= 16) {
            for (int off = 0; off < n_; off += 16) {
                seed_merge16_block(input, work, off, t16_2, t16_1, twp16);
            }
        } else {
            for (int b = 0; b < half_; b += 2) {
                seed_merge2_pair(input, work, b);
            }
        }
        int stage = 8;
        for (int s = 1024; s <= n_; s <<= 2, stage += 2) {
            const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage)];
            const double* twp = pair_twiddle_.data() + pair_twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                if constexpr (use_simd) {
                    merge4_inplace<true>(work + off, s, t2, t1, twp);
                } else {
                    merge4_inplace<false>(work + off, s, t2, t1, twp);
                }
            }
        }
    }

    template <bool use_simd>
    void compute_work_before_terminal(const double* input, double* work) const {
        BRUUN_ASSERT(n_ >= 1024);
        const double* t16_2 = twiddle_.data() + twiddle_offset_[1];
        const double* t16_1 = twiddle_.data() + twiddle_offset_[2];
        const double* twp16 = pair_twiddle_.data() + pair_twiddle_offset_[2];
        const double* t64_2 = twiddle_.data() + twiddle_offset_[3];
        const double* t64_1 = twiddle_.data() + twiddle_offset_[4];
        const double* twp64 = pair_twiddle_.data() + pair_twiddle_offset_[4];
        const double* t256_2 = twiddle_.data() + twiddle_offset_[5];
        const double* t256_1 = twiddle_.data() + twiddle_offset_[6];
        const double* twp256 = pair_twiddle_.data() + pair_twiddle_offset_[6];
        for (int off256 = 0; off256 < n_; off256 += 256) {
            seed_merge256_block<use_simd>(input, work, off256,
                                          t16_2, t16_1, twp16,
                                          t64_2, t64_1, twp64,
                                          t256_2, t256_1, twp256);
        }

        int stage = 8;
        for (int s = 1024; s < n_; s <<= 2, stage += 2) {
            const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(stage)];
            const double* twp = pair_twiddle_.data() + pair_twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                if constexpr (use_simd) {
                    merge4_inplace<true>(work + off, s, t2, t1, twp);
                } else {
                    merge4_inplace<false>(work + off, s, t2, t1, twp);
                }
            }
        }
    }

    template <bool use_simd>
    void forward_impl(const double* input, double* output_re, double* output_im, double* work) const {
        if constexpr (use_simd) {
            compute_work<true>(input, work);
            output_re[0] = work[0];
            output_im[0] = 0.0;
            output_re[half_] = work[1];
            output_im[half_] = 0.0;
#if BRUUN_LEVEL >= 1
            int k = 1;
            for (; k + 3 < half_; k += 4) {
                project_work_pair_split(work, output_re, output_im, k);
                project_work_pair_split(work, output_re, output_im, k + 2);
            }
            for (; k + 1 < half_; k += 2) {
                project_work_pair_split(work, output_re, output_im, k);
            }
            for (; k < half_; ++k) {
                const double a = work[2 * k];
                const double b = work[2 * k + 1];
                output_re[k] = a + b * cos_[static_cast<std::size_t>(k)];
                output_im[k] = b * neg_sin_[static_cast<std::size_t>(k)];
            }
#else
            for (int k = 1; k < half_; ++k) {
                const double a = work[2 * k];
                const double b = work[2 * k + 1];
                output_re[k] = a + b * cos_[static_cast<std::size_t>(k)];
                output_im[k] = b * neg_sin_[static_cast<std::size_t>(k)];
            }
#endif
            return;
        }
        compute_work<use_simd>(input, work);
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

    template <bool use_simd, typename Complex>
    void forward_complex_impl(const double* input, Complex* output, double* work) const {
        if constexpr (use_simd) {
            if (n_ >= 1024 && n_ <= 4096) {
                compute_work_before_terminal<true>(input, work);
                const int final_stage = stage_index(n_);
                const double* t2 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(final_stage - 1)];
                const double* t1 = twiddle_.data() + twiddle_offset_[static_cast<std::size_t>(final_stage)];
                const double* twp = pair_twiddle_.data() + pair_twiddle_offset_[static_cast<std::size_t>(final_stage)];
                merge4_terminal_complex<true>(work, n_, t2, t1, twp, output);
                return;
            }
            compute_work<true>(input, work);
            output[0].re = work[0];
            output[0].im = 0.0;
            output[half_].re = work[1];
            output[half_].im = 0.0;
#if BRUUN_LEVEL >= 1
            int k = 1;
            for (; k + 3 < half_; k += 4) {
                project_work_pair_complex(work, output, k);
                project_work_pair_complex(work, output, k + 2);
            }
            for (; k + 1 < half_; k += 2) {
                project_work_pair_complex(work, output, k);
            }
            for (; k < half_; ++k) {
                write_projected_bin(output, k, work[2 * k], work[2 * k + 1]);
            }
#else
            for (int k = 1; k < half_; ++k) {
                write_projected_bin(output, k, work[2 * k], work[2 * k + 1]);
            }
#endif
            return;
        }
        compute_work<use_simd>(input, work);
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

    int n_;
    int half_;
    int twiddle_stage_count_;
    heap_array<int> rev_;
    heap_array<int> twiddle_offset_;
    heap_array<int> pair_twiddle_offset_;
    heap_array<double> cos_;
    heap_array<double> neg_sin_;
    heap_array<double> twiddle_;
    heap_array<double> pair_twiddle_;
};

} // namespace bruun
