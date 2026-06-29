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
        if (!sin_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }
        for (int k = 0; k <= half_; ++k) {
            const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(n_);
            cos_[static_cast<std::size_t>(k)] = std::cos(theta);
            sin_[static_cast<std::size_t>(k)] = std::sin(theta);
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
        sin_.clear();
        twiddle_.clear();
        twiddle_offset_.clear();
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

    static BRUUN_ALWAYS_INLINE void pair_reduce(double ea, double eb, double oa, double ob, double c2,
                                                double& la, double& lb, double& ha, double& hb) {
        const double a = ea - eb;
        const double q = c2 * eb;
        const double p = c2 * ob;
        const double b = oa - ob + c2 * p;
        la = a - p;
        lb = b + q;
        ha = a + p;
        hb = b - q;
    }

#if BRUUN_LEVEL >= 1
    static BRUUN_ALWAYS_INLINE void pair_reduce_v2(bruun_v2 ea, bruun_v2 eb, bruun_v2 oa, bruun_v2 ob, bruun_v2 c2,
                                                   bruun_v2& la, bruun_v2& lb, bruun_v2& ha, bruun_v2& hb) {
        const bruun_v2 a = V2_SUB(ea, eb);
        const bruun_v2 q = V2_MUL(c2, eb);
        const bruun_v2 p = V2_MUL(c2, ob);
        const bruun_v2 b = V2_MADD(V2_SUB(oa, ob), c2, p);
        la = V2_SUB(a, p);
        lb = V2_ADD(b, q);
        ha = V2_ADD(a, p);
        hb = V2_SUB(b, q);
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

#if BRUUN_LEVEL >= 1
    void butterfly4_pair(const double* q0, const double* q1, const double* q2, const double* q3,
                         int i, int m, const double* t2, const double* t1, int q4, double* block, int hs) const {
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

        const bruun_v2 t2v = V2_SETLH(t2[i - 1], t2[m - 1]);
        const bruun_v2 t1v = V2_SETLH(t1[i - 1], t1[m - 1]);
        const bruun_v2 t1q = V2_SETLH(t1[q4 - i - 1], t1[q4 - m - 1]);

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
#endif

    template <bool use_simd>
    void merge4_inplace(double* block, int s) const {
        const int q4 = s >> 2;
        const int hs = s >> 1;
        const double* t2 = twiddle_.data() + stage_offset(hs);
        const double* t1 = twiddle_.data() + stage_offset(s);
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
                butterfly4_pair(q0, q1, q2, q3, i, m, t2, t1, q4, block, hs);
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

    template <bool use_simd>
    void forward_impl(const double* input, double* output_re, double* output_im, double* work) const {
        BRUUN_ASSERT(n_ >= 4);
        for (int b = 0; b < half_; ++b) {
            const int j = rev_[static_cast<std::size_t>(b)];
            const double a = input[j];
            const double c = input[half_ + j];
            work[2 * b] = a + c;
            work[2 * b + 1] = a - c;
        }
        for (int off = 0; off < n_; off += 4) {
            merge2(work + off, 4);
        }
        for (int s = 16; s <= n_; s <<= 2) {
            for (int off = 0; off < n_; off += s) {
                merge4_inplace<use_simd>(work + off, s);
            }
        }
        output_re[0] = work[0];
        output_im[0] = 0.0;
        output_re[half_] = work[1];
        output_im[half_] = 0.0;
        for (int k = 1; k < half_; ++k) {
            const double a = work[2 * k];
            const double b = work[2 * k + 1];
            output_re[k] = a + b * cos_[static_cast<std::size_t>(k)];
            output_im[k] = -b * sin_[static_cast<std::size_t>(k)];
        }
    }

    int n_;
    int half_;
    int twiddle_stage_count_;
    heap_array<int> rev_;
    heap_array<int> twiddle_offset_;
    heap_array<double> cos_;
    heap_array<double> sin_;
    heap_array<double> twiddle_;
};

} // namespace bruun
