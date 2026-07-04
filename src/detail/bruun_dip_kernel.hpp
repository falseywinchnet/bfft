#pragma once

// Internal DIP (diagonal-in-packets) real FFT kernel.
//
// Transport-optimal *interleaved diagonal-major* packing. The state at packet
// level e is stored row-major by diagonal coordinate, with each bin's residue
// pair held on ADJACENT rows:
//
//   row 0        : DC ridge
//   row 1        : Nyquist ridge
//   row 2d       : real residue a[d]        = Re X[d],   1 <= d < e/2
//   row 2d + 1   : phase residue b[d]       = -Im X[d],  1 <= d < e/2
//
// A stage e -> 2e splits every row into left/right column halves and applies
// the normalized Bruun cell with angle theta(d) = pi*d/e. The only angle input
// is the output diagonal coordinate d; the packet offset never enters the
// trigonometry.
//
// Why this layout (the transport point). The older R2HC packing stored a[d] at
// row d and b[d] at the *reflected* row e-d, so a single cell scattered its four
// outputs to rows {d, 2e-d, e-d, e+d} spread across the whole buffer -- four
// diverging write streams, two diverging read streams. The diagonal-major
// ordering theorem (experiments/phase_fft.py, notes/phase_packet_fft.md) says
// the address shear collapses when packets are stored by their output-frame
// diagonal. Interleaving (a,b) onto adjacent rows realizes that: each cell reads
// one contiguous 2-wide bin {2d, 2d+1} and writes two contiguous 2-wide bins
// {2d, 2d+1} and {2(e-d), 2(e-d)+1} -- two converging fat streams instead of
// four scattered thin ones. dc/ny stay on their own rows so the buffer is
// exactly n reals. This is the real folded form of the phase-packet FFT; the
// diagonal walk is transport-optimal in the middle and pays only the seed/leaf
// fold at the ends. Verified bit-exact against numpy.rfft in
// experiments/dip_transport_proto.py.

#include "bruun_simd_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace bruun {

class DIP_RFFT_kernel {
public:
    DIP_RFFT_kernel() noexcept : n_(0), half_(0) {}

    bool init(int n) {
        return reset(n);
    }

    bool reset(int n) {
        if (!is_power2(n) || n < 4) {
            clear();
            return false;
        }

        n_ = n;
        half_ = n >> 1;

        if (!cos_.resize(static_cast<std::size_t>(half_ + 1)) ||
            !sin_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }

        for (int k = 0; k <= half_; ++k) {
            const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(n_);
            cos_[static_cast<std::size_t>(k)] = std::cos(theta);
            sin_[static_cast<std::size_t>(k)] = std::sin(theta);
        }
        return true;
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return half_ + 1; }
    int work_size() const noexcept { return 2 * n_; }
    int blocked_work_size() const noexcept { return 4 * n_; }

    void forward_standard(const double* RESTRICT input,
                          complex_t* RESTRICT output,
                          double* RESTRICT work) const {
        double* buf0 = work;
        double* buf1 = work + n_;

        double* src = buf0;
        double* dst = buf1;
        int e;
        int q;
        if (n_ >= 16) {
            forward_seed16(input, buf0);
            e = 16;
            q = n_ >> 4;
        } else if (n_ >= 8) {
            forward_seed8(input, buf0);
            e = 8;
            q = n_ >> 3;
        } else {
            forward_seed4(input, buf0);
            e = 4;
            q = n_ >> 2;
        }
        while (e < n_) {
            forward_stage(src, dst, e, q);
            std::swap(src, dst);
            e <<= 1;
            q >>= 1;
        }

        // interleaved diagonal-major final level (e = n_, q = 1):
        // row 0 = DC, row 1 = Nyquist, rows (2d, 2d+1) = (Re, -Im) of bin d.
        output[0].re = src[0];
        output[0].im = 0.0;
        output[half_].re = src[1];
        output[half_].im = 0.0;
        for (int d = 1; d < half_; ++d) {
            output[d].re = src[2 * d];
            output[d].im = -src[2 * d + 1];
        }
    }

    void forward_standard_blocked(const double* RESTRICT input,
                                  complex_t* RESTRICT output,
                                  double* RESTRICT work,
                                  int max_fuse_depth = 4,
                                  int block_cols = 256) const {
        double* buf0 = work;
        double* buf1 = work + n_;
        double* tmp0 = work + 2 * static_cast<std::size_t>(n_);
        double* tmp1 = work + 3 * static_cast<std::size_t>(n_);

        int e = 1;
        int q = n_;
        int depth = choose_forward_fuse_depth(q, max_fuse_depth);
        forward_fused_stages(input, buf0, e, q, depth, block_cols, tmp0, tmp1);
        double* src = buf0;
        double* dst = buf1;
        e <<= depth;
        q >>= depth;
        while (e < n_) {
            depth = choose_forward_fuse_depth(q, max_fuse_depth);
            forward_fused_stages(src, dst, e, q, depth, block_cols, tmp0, tmp1);
            std::swap(src, dst);
            e <<= depth;
            q >>= depth;
        }

        output[0].re = src[0];
        output[0].im = 0.0;
        output[half_].re = src[1];
        output[half_].im = 0.0;
        for (int d = 1; d < half_; ++d) {
            output[d].re = src[2 * d];
            output[d].im = -src[2 * d + 1];
        }
    }

    void inverse_standard(const complex_t* RESTRICT input,
                          double* RESTRICT output,
                          double* RESTRICT work) const {
        double* buf0 = work;
        double* buf1 = work + n_;

        // interleaved diagonal-major packing (mirror of forward extraction):
        // row 0 = DC, row 1 = Nyquist, rows (2d, 2d+1) = (Re, -Im) of bin d.
        buf0[0] = input[0].re;
        buf0[1] = input[half_].re;
        for (int d = 1; d < half_; ++d) {
            buf0[2 * d] = input[d].re;
            buf0[2 * d + 1] = -input[d].im;
        }

        double* src = buf0;
        double* dst = buf1;
        int e = half_;
        int q = 2;
        while (e >= 8) {
            inverse_stage(src, dst, e, q);
            std::swap(src, dst);
            e >>= 1;
            q <<= 1;
        }
        if (n_ >= 8) {
            inverse_tail8(src, output, n_ >> 3);
        } else {
            inverse_tail4(src, output, n_ >> 2);
        }
    }

    void inverse_standard_blocked(const complex_t* RESTRICT input,
                                  double* RESTRICT output,
                                  double* RESTRICT work,
                                  int max_fuse_depth = 4,
                                  int block_cols = 256) const {
        double* buf0 = work;
        double* buf1 = work + n_;
        double* tmp0 = work + 2 * static_cast<std::size_t>(n_);
        double* tmp1 = work + 3 * static_cast<std::size_t>(n_);

        buf0[0] = input[0].re;
        buf0[1] = input[half_].re;
        for (int d = 1; d < half_; ++d) {
            buf0[2 * d] = input[d].re;
            buf0[2 * d + 1] = -input[d].im;
        }

        double* src = buf0;
        double* dst = buf1;
        int e = half_;
        int q = 2;
        while (e >= 1) {
            const int depth = choose_inverse_fuse_depth(e, max_fuse_depth);
            const bool final_block = (2 * e >> depth) == 1;
            double* stage_dst = final_block ? output : dst;
            inverse_fused_stages(src, stage_dst, e, q, depth, block_cols, tmp0, tmp1);
            if (final_block) {
                return;
            }
            std::swap(src, dst);
            e >>= depth;
            q <<= depth;
        }
    }

private:
    static inline void pair_reduce_cs(double ea, double eb,
                                      double oa, double ob,
                                      double c, double s,
                                      double& la, double& lb,
                                      double& ha, double& hb) {
        const double r = c * oa - s * ob;
        const double i = s * oa + c * ob;
        la = ea + r;
        lb = eb + i;
        ha = ea - r;
        hb = i - eb;
    }

    static inline void pair_expand_cs(double la, double lb,
                                      double ha, double hb,
                                      double c, double s,
                                      double& ea, double& eb,
                                      double& oa, double& ob) {
        const double r = 0.5 * (la - ha);
        const double i = 0.5 * (lb + hb);
        ea = 0.5 * (la + ha);
        eb = 0.5 * (lb - hb);
        oa = c * r + s * i;
        ob = c * i - s * r;
    }

    int phase_table_index(int d, int e) const noexcept {
        return static_cast<int>((static_cast<long long>(d) * n_) / (2LL * e));
    }

    static int clamp_block_cols(int block_cols) noexcept {
        return block_cols > 0 ? block_cols : 256;
    }

    static int choose_forward_fuse_depth(int q, int max_fuse_depth) noexcept {
        int depth = max_fuse_depth > 0 ? max_fuse_depth : 1;
        int limit = 0;
        for (int qq = q; qq > 1; qq >>= 1) {
            ++limit;
        }
        return depth < limit ? depth : limit;
    }

    static int choose_inverse_fuse_depth(int e, int max_fuse_depth) noexcept {
        int depth = max_fuse_depth > 0 ? max_fuse_depth : 1;
        int limit = 1;
        for (int ee = e; ee > 1; ee >>= 1) {
            ++limit;
        }
        return depth < limit ? depth : limit;
    }

    void forward_stage(const double* RESTRICT src,
                       double* RESTRICT dst,
                       int e,
                       int q) const {
        const int q2 = q >> 1;

        // DC / Nyquist ridge: interleaved rows 0 (dc) and 1 (ny) at level 2e.
        const double* RESTRICT dc = src;                       // level-e row 0
        double* RESTRICT out_dc = dst;                         // level-2e row 0
        double* RESTRICT out_ny = dst + static_cast<std::size_t>(q2);  // row 1
        for (int i = 0; i < q2; ++i) {
            const double lo = dc[i];
            const double hi = dc[q2 + i];
            out_dc[i] = lo + hi;
            out_ny[i] = lo - hi;
        }

        if (e >= 2) {
            // old Nyquist (level-e row 1) becomes new bin e/2 at rows e, e+1.
            const double* RESTRICT ny = src + static_cast<std::size_t>(q);
            double* RESTRICT out_a = dst + static_cast<std::size_t>(e) * q2;
            double* RESTRICT out_b = dst + static_cast<std::size_t>(e + 1) * q2;
            for (int i = 0; i < q2; ++i) {
                out_a[i] = ny[i];
                out_b[i] = ny[q2 + i];
            }
        }

        for (int d = 1; d < e / 2; ++d) {
            const int k = phase_table_index(d, e);
            const double c = cos_[static_cast<std::size_t>(k)];
            const double s = sin_[static_cast<std::size_t>(k)];

            // input bin d: adjacent rows 2d (a) and 2d+1 (b) of level e.
            const double* RESTRICT in_a = src + static_cast<std::size_t>(2 * d) * q;
            const double* RESTRICT in_b = src + static_cast<std::size_t>(2 * d + 1) * q;
            // output bin d -> rows 2d, 2d+1; output bin e-d -> rows 2(e-d), +1.
            double* RESTRICT out_lo = dst + static_cast<std::size_t>(2 * d) * q2;
            double* RESTRICT out_hi = dst + static_cast<std::size_t>(2 * (e - d)) * q2;

            for (int i = 0; i < q2; ++i) {
                const double ea = in_a[i];
                const double eb = in_b[i];
                const double oa = in_a[q2 + i];
                const double ob = in_b[q2 + i];
                double la, lb, ha, hb;
                pair_reduce_cs(ea, eb, oa, ob, c, s, la, lb, ha, hb);
                out_lo[i] = la;
                out_lo[q2 + i] = lb;
                out_hi[i] = ha;
                out_hi[q2 + i] = hb;
            }
        }
    }

    void forward_seed4(const double* RESTRICT input, double* RESTRICT dst) const {
        const int q = n_ >> 2;
        double* RESTRICT r0 = dst;
        double* RESTRICT r1 = dst + q;
        double* RESTRICT r2 = dst + 2 * static_cast<std::size_t>(q);
        double* RESTRICT r3 = dst + 3 * static_cast<std::size_t>(q);
        const double* RESTRICT x0 = input;
        const double* RESTRICT x1 = input + q;
        const double* RESTRICT x2 = input + 2 * static_cast<std::size_t>(q);
        const double* RESTRICT x3 = input + 3 * static_cast<std::size_t>(q);
        for (int i = 0; i < q; ++i) {
            const double a = x0[i];
            const double b = x1[i];
            const double c = x2[i];
            const double d = x3[i];
            r0[i] = a + b + c + d;   // row 0 : DC
            r1[i] = a - b + c - d;   // row 1 : Nyquist
            r2[i] = a - c;           // row 2 : a[1] = Re
            r3[i] = b - d;           // row 3 : b[1] = -Im
        }
    }

    void forward_seed8(const double* RESTRICT input, double* RESTRICT dst) const {
        const int q = n_ >> 3;
        double* RESTRICT r0 = dst;
        double* RESTRICT r1 = dst + q;
        double* RESTRICT r2 = dst + 2 * static_cast<std::size_t>(q);
        double* RESTRICT r3 = dst + 3 * static_cast<std::size_t>(q);
        double* RESTRICT r4 = dst + 4 * static_cast<std::size_t>(q);
        double* RESTRICT r5 = dst + 5 * static_cast<std::size_t>(q);
        double* RESTRICT r6 = dst + 6 * static_cast<std::size_t>(q);
        double* RESTRICT r7 = dst + 7 * static_cast<std::size_t>(q);
        const double* RESTRICT x0 = input;
        const double* RESTRICT x1 = input + q;
        const double* RESTRICT x2 = input + 2 * static_cast<std::size_t>(q);
        const double* RESTRICT x3 = input + 3 * static_cast<std::size_t>(q);
        const double* RESTRICT x4 = input + 4 * static_cast<std::size_t>(q);
        const double* RESTRICT x5 = input + 5 * static_cast<std::size_t>(q);
        const double* RESTRICT x6 = input + 6 * static_cast<std::size_t>(q);
        const double* RESTRICT x7 = input + 7 * static_cast<std::size_t>(q);
        constexpr double rt = 0.707106781186547524400844362104849039;
        for (int i = 0; i < q; ++i) {
            const double a = x0[i];
            const double b = x1[i];
            const double c = x2[i];
            const double d = x3[i];
            const double e = x4[i];
            const double f = x5[i];
            const double g = x6[i];
            const double h = x7[i];

            const double ae = a - e;
            const double bf = b - f;
            const double cg = c - g;
            const double dh = d - h;
            const double rot_r = rt * (bf - dh);
            const double rot_i = rt * (bf + dh);

            // interleaved rows: 0=dc 1=ny 2=a1 3=b1 4=a2 5=b2 6=a3 7=b3
            r0[i] = a + b + c + d + e + f + g + h;   // dc
            r1[i] = a - b + c - d + e - f + g - h;   // ny
            r2[i] = ae + rot_r;                      // a1
            r3[i] = cg + rot_i;                      // b1
            r4[i] = a - c + e - g;                   // a2
            r5[i] = b - d + f - h;                   // b2
            r6[i] = ae - rot_r;                      // a3
            r7[i] = rot_i - cg;                      // b3
        }
    }

    static inline void seed8_values(double a, double b, double c, double d,
                                    double e, double f, double g, double h,
                                    double& r0, double& r1, double& r2, double& r3,
                                    double& r4, double& r5, double& r6, double& r7) {
        constexpr double rt = 0.707106781186547524400844362104849039;
        const double ae = a - e;
        const double bf = b - f;
        const double cg = c - g;
        const double dh = d - h;
        const double rot_r = rt * (bf - dh);
        const double rot_i = rt * (bf + dh);

        r0 = a + b + c + d + e + f + g + h;
        r1 = ae + rot_r;
        r2 = a - c + e - g;
        r3 = ae - rot_r;
        r4 = a - b + c - d + e - f + g - h;
        r5 = rot_i - cg;
        r6 = b - d + f - h;
        r7 = cg + rot_i;
    }

    void forward_seed16(const double* RESTRICT input, double* RESTRICT dst) const {
        const int q = n_ >> 4;
        double* RESTRICT r0 = dst;
        double* RESTRICT r1 = dst + q;
        double* RESTRICT r2 = dst + 2 * static_cast<std::size_t>(q);
        double* RESTRICT r3 = dst + 3 * static_cast<std::size_t>(q);
        double* RESTRICT r4 = dst + 4 * static_cast<std::size_t>(q);
        double* RESTRICT r5 = dst + 5 * static_cast<std::size_t>(q);
        double* RESTRICT r6 = dst + 6 * static_cast<std::size_t>(q);
        double* RESTRICT r7 = dst + 7 * static_cast<std::size_t>(q);
        double* RESTRICT r8 = dst + 8 * static_cast<std::size_t>(q);
        double* RESTRICT r9 = dst + 9 * static_cast<std::size_t>(q);
        double* RESTRICT r10 = dst + 10 * static_cast<std::size_t>(q);
        double* RESTRICT r11 = dst + 11 * static_cast<std::size_t>(q);
        double* RESTRICT r12 = dst + 12 * static_cast<std::size_t>(q);
        double* RESTRICT r13 = dst + 13 * static_cast<std::size_t>(q);
        double* RESTRICT r14 = dst + 14 * static_cast<std::size_t>(q);
        double* RESTRICT r15 = dst + 15 * static_cast<std::size_t>(q);

        const double c1 = 0.923879532511286756128183189396788287;
        const double s1 = 0.382683432365089771728459984030398866;
        const double c2 = 0.707106781186547524400844362104849039;
        const double s2 = 0.707106781186547524400844362104849039;
        const double c3 = 0.382683432365089771728459984030398866;
        const double s3 = 0.923879532511286756128183189396788287;

        for (int i = 0; i < q; ++i) {
            double le0, le1, le2, le3, le4, le5, le6, le7;
            double ro0, ro1, ro2, ro3, ro4, ro5, ro6, ro7;
            seed8_values(input[i],
                         input[2 * static_cast<std::size_t>(q) + i],
                         input[4 * static_cast<std::size_t>(q) + i],
                         input[6 * static_cast<std::size_t>(q) + i],
                         input[8 * static_cast<std::size_t>(q) + i],
                         input[10 * static_cast<std::size_t>(q) + i],
                         input[12 * static_cast<std::size_t>(q) + i],
                         input[14 * static_cast<std::size_t>(q) + i],
                         le0, le1, le2, le3, le4, le5, le6, le7);
            seed8_values(input[static_cast<std::size_t>(q) + i],
                         input[3 * static_cast<std::size_t>(q) + i],
                         input[5 * static_cast<std::size_t>(q) + i],
                         input[7 * static_cast<std::size_t>(q) + i],
                         input[9 * static_cast<std::size_t>(q) + i],
                         input[11 * static_cast<std::size_t>(q) + i],
                         input[13 * static_cast<std::size_t>(q) + i],
                         input[15 * static_cast<std::size_t>(q) + i],
                         ro0, ro1, ro2, ro3, ro4, ro5, ro6, ro7);

            // interleaved rows: 0=dc 1=ny (2d,2d+1)=(a[d],b[d]) for d=1..7.
            r0[i] = le0 + ro0;   // dc  -> row 0
            r1[i] = le0 - ro0;   // ny  -> row 1
            r8[i] = le4;         // a4  -> row 8
            r9[i] = ro4;         // b4  -> row 9

            pair_reduce_cs(le1, le7, ro1, ro7, c1, s1, r2[i], r3[i], r14[i], r15[i]);
            pair_reduce_cs(le2, le6, ro2, ro6, c2, s2, r4[i], r5[i], r12[i], r13[i]);
            pair_reduce_cs(le3, le5, ro3, ro5, c3, s3, r6[i], r7[i], r10[i], r11[i]);
        }
    }

    void inverse_stage(const double* RESTRICT src,
                       double* RESTRICT dst,
                       int e,
                       int q) const {
        const int q2 = q >> 1;

        // DC / Nyquist ridge (level-2e interleaved rows 0, 1 -> level-e row 0).
        const double* RESTRICT in_dc = src;
        const double* RESTRICT in_ny = src + static_cast<std::size_t>(q2);
        double* RESTRICT out_dc = dst;
        for (int i = 0; i < q2; ++i) {
            const double dc = in_dc[i];
            const double ny = in_ny[i];
            out_dc[i] = 0.5 * (dc + ny);
            out_dc[q2 + i] = 0.5 * (dc - ny);
        }

        if (e >= 2) {
            // new bin e/2 (level-2e rows e, e+1) -> old Nyquist (level-e row 1).
            const double* RESTRICT in_a = src + static_cast<std::size_t>(e) * q2;
            const double* RESTRICT in_b = src + static_cast<std::size_t>(e + 1) * q2;
            double* RESTRICT out_ny = dst + static_cast<std::size_t>(q);
            for (int i = 0; i < q2; ++i) {
                out_ny[i] = in_a[i];
                out_ny[q2 + i] = in_b[i];
            }
        }

        for (int d = 1; d < e / 2; ++d) {
            const int k = phase_table_index(d, e);
            const double c = cos_[static_cast<std::size_t>(k)];
            const double s = sin_[static_cast<std::size_t>(k)];

            // level-2e bins d (rows 2d,2d+1) and e-d (rows 2(e-d),+1) ->
            // level-e bin d on adjacent rows 2d (a) and 2d+1 (b).
            const double* RESTRICT in_lo = src + static_cast<std::size_t>(2 * d) * q2;
            const double* RESTRICT in_hi = src + static_cast<std::size_t>(2 * (e - d)) * q2;
            double* RESTRICT out_a = dst + static_cast<std::size_t>(2 * d) * q;
            double* RESTRICT out_b = dst + static_cast<std::size_t>(2 * d + 1) * q;

            for (int i = 0; i < q2; ++i) {
                double ea, eb, oa, ob;
                pair_expand_cs(in_lo[i],
                               in_lo[q2 + i],
                               in_hi[i],
                               in_hi[q2 + i],
                               c,
                               s,
                               ea,
                               eb,
                               oa,
                               ob);
                out_a[i] = ea;
                out_a[q2 + i] = oa;
                out_b[i] = eb;
                out_b[q2 + i] = ob;
            }
        }
    }

    void inverse_tail4(const double* RESTRICT src, double* RESTRICT output, int q) const {
        // interleaved rows: 0=dc 1=ny 2=a1 3=b1; map to (dc, a1, ny, b1).
        const double* RESTRICT r0 = src;                                    // dc
        const double* RESTRICT r1 = src + 2 * static_cast<std::size_t>(q);  // a1
        const double* RESTRICT r2 = src + q;                               // ny
        const double* RESTRICT r3 = src + 3 * static_cast<std::size_t>(q);  // b1
        double* RESTRICT x0 = output;
        double* RESTRICT x1 = output + q;
        double* RESTRICT x2 = output + 2 * static_cast<std::size_t>(q);
        double* RESTRICT x3 = output + 3 * static_cast<std::size_t>(q);
        for (int i = 0; i < q; ++i) {
            const double ac = 0.5 * (r0[i] + r2[i]);
            const double bd = 0.5 * (r0[i] - r2[i]);
            x0[i] = 0.5 * (ac + r1[i]);
            x2[i] = 0.5 * (ac - r1[i]);
            x1[i] = 0.5 * (bd + r3[i]);
            x3[i] = 0.5 * (bd - r3[i]);
        }
    }

    void inverse_tail8(const double* RESTRICT src, double* RESTRICT output, int q) const {
        // interleaved rows 0..7 = dc,ny,a1,b1,a2,b2,a3,b3; the tail below is
        // written in the (dc,a1,a2,a3,ny,b3,b2,b1) basis, so remap the reads.
        const double* RESTRICT r0 = src;                                    // dc
        const double* RESTRICT r1 = src + 2 * static_cast<std::size_t>(q);  // a1
        const double* RESTRICT r2 = src + 4 * static_cast<std::size_t>(q);  // a2
        const double* RESTRICT r3 = src + 6 * static_cast<std::size_t>(q);  // a3
        const double* RESTRICT r4 = src + q;                               // ny
        const double* RESTRICT r5 = src + 7 * static_cast<std::size_t>(q);  // b3
        const double* RESTRICT r6 = src + 5 * static_cast<std::size_t>(q);  // b2
        const double* RESTRICT r7 = src + 3 * static_cast<std::size_t>(q);  // b1
        double* RESTRICT x0 = output;
        double* RESTRICT x1 = output + q;
        double* RESTRICT x2 = output + 2 * static_cast<std::size_t>(q);
        double* RESTRICT x3 = output + 3 * static_cast<std::size_t>(q);
        double* RESTRICT x4 = output + 4 * static_cast<std::size_t>(q);
        double* RESTRICT x5 = output + 5 * static_cast<std::size_t>(q);
        double* RESTRICT x6 = output + 6 * static_cast<std::size_t>(q);
        double* RESTRICT x7 = output + 7 * static_cast<std::size_t>(q);
        constexpr double inv_2rt = 0.707106781186547524400844362104849039;

        for (int i = 0; i < q; ++i) {
            const double even_sum = 0.5 * (r0[i] + r4[i]);
            const double odd_sum = 0.5 * (r0[i] - r4[i]);
            const double ae = 0.5 * (r1[i] + r3[i]);
            const double cg = 0.5 * (r7[i] - r5[i]);
            const double bf_minus_dh = inv_2rt * (r1[i] - r3[i]);
            const double bf_plus_dh = inv_2rt * (r5[i] + r7[i]);
            const double bf = 0.5 * (bf_minus_dh + bf_plus_dh);
            const double dh = 0.5 * (bf_plus_dh - bf_minus_dh);

            const double ae_sum = 0.5 * (even_sum + r2[i]);
            const double cg_sum = 0.5 * (even_sum - r2[i]);
            const double bf_sum = 0.5 * (odd_sum + r6[i]);
            const double dh_sum = 0.5 * (odd_sum - r6[i]);

            x0[i] = 0.5 * (ae_sum + ae);
            x4[i] = 0.5 * (ae_sum - ae);
            x2[i] = 0.5 * (cg_sum + cg);
            x6[i] = 0.5 * (cg_sum - cg);
            x1[i] = 0.5 * (bf_sum + bf);
            x5[i] = 0.5 * (bf_sum - bf);
            x3[i] = 0.5 * (dh_sum + dh);
            x7[i] = 0.5 * (dh_sum - dh);
        }
    }

    void forward_fused_stages(const double* RESTRICT src,
                              double* RESTRICT dst,
                              int e,
                              int q,
                              int depth,
                              int block_cols,
                              double* RESTRICT tmp0,
                              double* RESTRICT tmp1) const {
        const int r = 1 << depth;
        const int q_out = q >> depth;
        const int cols_per_block = clamp_block_cols(block_cols);

        for (int jb = 0; jb < q_out; jb += cols_per_block) {
            const int cols = std::min(cols_per_block, q_out - jb);
            const int q_local = r * cols;
            for (int row = 0; row < e; ++row) {
                const double* RESTRICT row_src = src + static_cast<std::size_t>(row) * q;
                double* RESTRICT row_tmp = tmp0 + static_cast<std::size_t>(row) * q_local;
                for (int k = 0; k < r; ++k) {
                    std::memcpy(row_tmp + static_cast<std::size_t>(k) * cols,
                                row_src + static_cast<std::size_t>(k) * q_out + jb,
                                sizeof(double) * static_cast<std::size_t>(cols));
                }
            }

            double* local_src = tmp0;
            double* local_dst = tmp1;
            int local_e = e;
            int local_q = q_local;
            for (int s = 0; s < depth; ++s) {
                forward_stage(local_src, local_dst, local_e, local_q);
                std::swap(local_src, local_dst);
                local_e <<= 1;
                local_q >>= 1;
            }

            for (int row = 0; row < e * r; ++row) {
                std::memcpy(dst + static_cast<std::size_t>(row) * q_out + jb,
                            local_src + static_cast<std::size_t>(row) * cols,
                            sizeof(double) * static_cast<std::size_t>(cols));
            }
        }
    }

    void inverse_fused_stages(const double* RESTRICT src,
                              double* RESTRICT dst,
                              int e,
                              int q,
                              int depth,
                              int block_cols,
                              double* RESTRICT tmp0,
                              double* RESTRICT tmp1) const {
        const int r = 1 << depth;
        const int q_in = q >> 1;
        const int q_out = q_in * r;
        const int rows_in = 2 * e;
        const int rows_out = rows_in >> depth;
        const int cols_per_block = clamp_block_cols(block_cols);

        for (int jb = 0; jb < q_in; jb += cols_per_block) {
            const int cols = std::min(cols_per_block, q_in - jb);
            for (int row = 0; row < rows_in; ++row) {
                std::memcpy(tmp0 + static_cast<std::size_t>(row) * cols,
                            src + static_cast<std::size_t>(row) * q_in + jb,
                            sizeof(double) * static_cast<std::size_t>(cols));
            }

            double* local_src = tmp0;
            double* local_dst = tmp1;
            int local_e = e;
            int local_q = 2 * cols;
            for (int s = 0; s < depth; ++s) {
                inverse_stage(local_src, local_dst, local_e, local_q);
                std::swap(local_src, local_dst);
                local_e >>= 1;
                local_q <<= 1;
            }

            for (int row = 0; row < rows_out; ++row) {
                const double* RESTRICT row_src = local_src + static_cast<std::size_t>(row) * (r * cols);
                double* RESTRICT row_dst = dst + static_cast<std::size_t>(row) * q_out;
                for (int k = 0; k < r; ++k) {
                    std::memcpy(row_dst + static_cast<std::size_t>(k) * q_in + jb,
                                row_src + static_cast<std::size_t>(k) * cols,
                                sizeof(double) * static_cast<std::size_t>(cols));
                }
            }
        }
    }

    void clear() noexcept {
        n_ = 0;
        half_ = 0;
        (void)cos_.resize(0);
        (void)sin_.resize(0);
    }

    int n_;
    int half_;
    heap_array<double> cos_;
    heap_array<double> sin_;
};

} // namespace bruun
