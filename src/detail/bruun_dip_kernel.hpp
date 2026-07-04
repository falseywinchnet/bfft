#pragma once

// Internal DIP (diagonal-in-packets) real FFT kernel.
//
// THE WALK (2026-07-04 rework). One in-place, depth-first descent of the Bruun
// diagonal tree; no ping-pong, no scratch, no interior permutation.
//
// State = contiguous SPANS. A bin span at packet level e is the phase packet
// (d, e): the residue pair (a, b) = (Re, -Im) of diagonal d, stored
// [ a : w | b : w ] with w = n/e. The ridge span is [ dc : w | ny : w ].
// The stage cell splits each row into (even | odd) column halves and applies
// the normalized Bruun rotation theta(d) = pi*d/e -- writing the two children
// (d, 2e) and (e-d, 2e) INTO THE PARENT'S OWN ADDRESSES:
//
//   [ ea : w2 | oa : w2 | eb : w2 | ob : w2 ]   parent (d, e)
//        |         |         |         |        cell_fwd_ip: four streams,
//        v         v         v         v        the SAME addresses read+written
//   [ la : w2 | lb : w2 | ha : w2 | hb : w2 ] = [ child (d, 2e) | child (e-d, 2e) ]
//
// so every span is self-contained forever after: the walk recurses into the
// children independently (depth-first), and once a span fits in cache its
// whole subtree completes cache-resident. The cache transition is emergent --
// no block sizes, no fuse depths, no tmp buffers. The Nyquist promotion is
// free: the ridge's ny row IS the promoted bin (e/2, 2e) in place (a = left
// half, b = right half), so the old copy_span pass no longer exists.
//
// THE DISORDER LEDGER. The bit-reversal cannot be destroyed, only placed.
// This walk splits it across the two boundaries, each paid in its cheap
// currency:
//   - input side: the seed reads 16 comb streams (long-range strides but
//     sequential within each stream, prefetch-friendly) and writes 16
//     contiguous rows -- one streaming pass.
//   - output side: leaves stay IN PLACE (bin d's (a, b) pair ends at
//     work[leaf_[d]]), and one boundary pass walks the output sequentially,
//     gather-reading the leaf pairs through the precomputed leaf_ table --
//     the same cost class as the DIF's residues_to_complex (IDX) or the
//     DIT's input comb gather. Measured: fusing the scatter into the walk
//     instead costs ~2x the whole transform at n >= 16K (far-range 16-byte
//     writes from every terminal); the separated pass streams.
// The interior pays ZERO rearrangement: no transposes, no swaps, no sorting.
// Span addresses run in tree order; the ANGLE law stays purely diagonal:
// theta = pi*d/e, served at every level by the one scale-indexed length-n
// table via phase_table_index(d, e) = d*n/(2e).
//
// SIMD. Cells are isoclinic column rotations: (c, s) broadcast, four
// contiguous vector streams, loads and stores to the same addresses, no
// permutes at any width (AVX2 / V2 / scalar cascade; w2 is 1 or an even power
// of two, so there is never a ragged tail). In-place operation halves the
// memory traffic of the old out-of-place stage and the work buffer drops from
// 2n to n doubles.
//
// CONCURRENCY. The kernel is const after init; forward/inverse touch only the
// caller's work buffer (exactly n doubles) and the output. No internal
// scratch, no allocation, no shared mutable state.
//
// Validation lineage: experiments/phase_fft.py (the diagonal walk + the
// ordering theorem), experiments/dip_transport_symmetry.py (boundary-transport
// conservation; why interior transposes/ping-pongs were rejected),
// notes/dip_phase_packet_design.md (design history).

#include "bruun_simd_backend.hpp"

#include <cmath>

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
            !sin_.resize(static_cast<std::size_t>(half_ + 1)) ||
            !leaf_.resize(static_cast<std::size_t>(half_ > 1 ? half_ : 1))) {
            clear();
            return false;
        }

        for (int k = 0; k <= half_; ++k) {
            const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(n_);
            cos_[static_cast<std::size_t>(k)] = std::cos(theta);
            sin_[static_cast<std::size_t>(k)] = std::sin(theta);
        }

        // leaf_[d] = work offset where bin d's (a, b) pair lands after the
        // descent -- the boundary permutation, resolved once here.
        if (n_ >= 16) {
            const int q = n_ >> 4;
            map_ridge(0, q, 16);
            map_span(2 * q, q, 4, 16);
            map_span(4 * q, q, 2, 16);
            map_span(6 * q, q, 6, 16);
            map_span(8 * q, q, 1, 16);
            map_span(10 * q, q, 7, 16);
            map_span(12 * q, q, 3, 16);
            map_span(14 * q, q, 5, 16);
        } else if (n_ == 8) {
            map_ridge(0, 1, 8);
            map_span(2, 1, 2, 8);
            map_span(4, 1, 1, 8);
            map_span(6, 1, 3, 8);
        } else {
            map_ridge(0, 1, 4);
            map_span(2, 1, 1, 4);
        }
        return true;
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return half_ + 1; }
    int work_size() const noexcept { return n_; }

    void forward_standard(const double* RESTRICT input,
                          complex_t* RESTRICT output,
                          double* RESTRICT work) const {
        if (n_ >= 16) {
            forward_seed16(input, work);
            // level-16 slot order (tree order): [R][B4][B2][B6][B1][B7][B3][B5]
            const int q = n_ >> 4;
            fwd_ridge(work, q, 16);
            fwd_span(work + 2 * static_cast<std::size_t>(q), q, 4, 16);
            fwd_span(work + 4 * static_cast<std::size_t>(q), q, 2, 16);
            fwd_span(work + 6 * static_cast<std::size_t>(q), q, 6, 16);
            fwd_span(work + 8 * static_cast<std::size_t>(q), q, 1, 16);
            fwd_span(work + 10 * static_cast<std::size_t>(q), q, 7, 16);
            fwd_span(work + 12 * static_cast<std::size_t>(q), q, 3, 16);
            fwd_span(work + 14 * static_cast<std::size_t>(q), q, 5, 16);
        } else if (n_ == 8) {
            forward_seed8(input, work);
            // level-8 slot order: [R][B2][B1][B3], q = 1
            fwd_span(work + 2, 1, 2, 8);
            fwd_span(work + 4, 1, 1, 8);
            fwd_span(work + 6, 1, 3, 8);
        } else {
            forward_seed4(input, work);
        }

        // Boundary pass: sequential output walk, gather-read of leaf pairs.
        output[0].re = work[0];
        output[0].im = 0.0;
        output[half_].re = work[1];
        output[half_].im = 0.0;
        const int* RESTRICT lf = leaf_.data();
        for (int d = 1; d < half_; ++d) {
            const int p = lf[d];
            output[d].re = work[p];
            output[d].im = -work[p + 1];
        }
    }

    void inverse_standard(const complex_t* RESTRICT input,
                          double* RESTRICT output,
                          double* RESTRICT work) const {
        // The inverse boundary is fused into the descent instead of paid as a
        // separate injection pass: leaf gathers are scattered READS of the
        // input (cheap; loads overlap) with writes landing in the cache-hot
        // span, where an injection pass would pay scattered WRITES (measured
        // ~1.5x slower at n >= 64K).
        if (n_ >= 8) {
            // gather + ascend the level-8 subtrees, then unseed the slot
            // state [R][B2][B1][B3] to time order.
            const int q = n_ >> 3;
            inv_ridge(work, q, 8, input);
            inv_span(work + 2 * static_cast<std::size_t>(q), q, 2, 8, input);
            inv_span(work + 4 * static_cast<std::size_t>(q), q, 1, 8, input);
            inv_span(work + 6 * static_cast<std::size_t>(q), q, 3, 8, input);
            inverse_tail8(work, output, q);
        } else {
            work[0] = input[0].re;      // dc
            work[1] = input[2].re;      // ny
            work[2] = input[1].re;      // a1
            work[3] = -input[1].im;     // b1
            inverse_tail4(work, output, 1);
        }
    }

private:
    // Scalar reference cell (the algebra realized by cell_fwd_ip / cell_inv_ip;
    // also used directly by the seeds).
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

    // -----------------------------------------------------------------------
    // In-place span cells. The twiddle (c, s) is constant along the column
    // axis (isoclinic rotation): broadcast it and run pure vertical SIMD on
    // four contiguous streams, storing to the same addresses just loaded.
    // w2 is 1 or an even power of two, so the cascade has no ragged tail.
    // -----------------------------------------------------------------------

    // Forward Bruun cell on the span [ ea | oa | eb | ob ] (each w2 wide):
    // parent (d, e) -> children [ la | lb ] = (d, 2e), [ ha | hb ] = (e-d, 2e).
    static BRUUN_ALWAYS_INLINE void cell_fwd_ip(double* RESTRICT v, int w2,
                                                double c, double s) {
        double* RESTRICT ea = v;
        double* RESTRICT oa = v + w2;
        double* RESTRICT eb = v + 2 * static_cast<std::size_t>(w2);
        double* RESTRICT ob = v + 3 * static_cast<std::size_t>(w2);
        int i = 0;
#if BRUUN_LEVEL >= 2
        {
            const __m256d vc = _mm256_set1_pd(c);
            const __m256d vs = _mm256_set1_pd(s);
            for (; i + 3 < w2; i += 4) {
                const __m256d a = _mm256_loadu_pd(ea + i);
                const __m256d o = _mm256_loadu_pd(oa + i);
                const __m256d b = _mm256_loadu_pd(eb + i);
                const __m256d p = _mm256_loadu_pd(ob + i);
                const __m256d R = _mm256_fmsub_pd(vc, o, _mm256_mul_pd(vs, p));
                const __m256d I = _mm256_fmadd_pd(vs, o, _mm256_mul_pd(vc, p));
                _mm256_storeu_pd(ea + i, _mm256_add_pd(a, R));
                _mm256_storeu_pd(oa + i, _mm256_add_pd(b, I));
                _mm256_storeu_pd(eb + i, _mm256_sub_pd(a, R));
                _mm256_storeu_pd(ob + i, _mm256_sub_pd(I, b));
            }
        }
#endif
#if BRUUN_LEVEL >= 1
        {
            const bruun_v2 vc = V2_SET1(c);
            const bruun_v2 vs = V2_SET1(s);
            for (; i + 1 < w2; i += 2) {
                const bruun_v2 a = V2_LD(ea + i);
                const bruun_v2 o = V2_LD(oa + i);
                const bruun_v2 b = V2_LD(eb + i);
                const bruun_v2 p = V2_LD(ob + i);
                const bruun_v2 R = V2_MSUB(V2_MUL(vc, o), vs, p);
                const bruun_v2 I = V2_MADD(V2_MUL(vs, o), vc, p);
                V2_ST(ea + i, V2_ADD(a, R));
                V2_ST(oa + i, V2_ADD(b, I));
                V2_ST(eb + i, V2_SUB(a, R));
                V2_ST(ob + i, V2_SUB(I, b));
            }
        }
#endif
        for (; i < w2; ++i) {
            const double a = ea[i];
            const double o = oa[i];
            const double b = eb[i];
            const double p = ob[i];
            const double R = c * o - s * p;
            const double I = s * o + c * p;
            ea[i] = a + R;    // la
            oa[i] = b + I;    // lb
            eb[i] = a - R;    // ha
            ob[i] = I - b;    // hb
        }
    }

    // Inverse Bruun cell on the span [ la | lb | ha | hb ] -> [ ea | oa | eb | ob ].
    // hc = 0.5*c, hs = 0.5*s fold the expand 0.5 into the twiddle for the
    // rotated outputs; the even outputs keep an explicit 0.5.
    static BRUUN_ALWAYS_INLINE void cell_inv_ip(double* RESTRICT v, int w2,
                                                double hc, double hs) {
        double* RESTRICT la = v;
        double* RESTRICT lb = v + w2;
        double* RESTRICT ha = v + 2 * static_cast<std::size_t>(w2);
        double* RESTRICT hb = v + 3 * static_cast<std::size_t>(w2);
        int i = 0;
#if BRUUN_LEVEL >= 2
        {
            const __m256d vhc = _mm256_set1_pd(hc);
            const __m256d vhs = _mm256_set1_pd(hs);
            const __m256d half = _mm256_set1_pd(0.5);
            for (; i + 3 < w2; i += 4) {
                const __m256d A = _mm256_loadu_pd(la + i);
                const __m256d B = _mm256_loadu_pd(lb + i);
                const __m256d C = _mm256_loadu_pd(ha + i);
                const __m256d D = _mm256_loadu_pd(hb + i);
                const __m256d u = _mm256_sub_pd(A, C);
                const __m256d t = _mm256_add_pd(B, D);
                _mm256_storeu_pd(la + i, _mm256_mul_pd(half, _mm256_add_pd(A, C)));
                _mm256_storeu_pd(lb + i, _mm256_fmadd_pd(vhc, u, _mm256_mul_pd(vhs, t)));
                _mm256_storeu_pd(ha + i, _mm256_mul_pd(half, _mm256_sub_pd(B, D)));
                _mm256_storeu_pd(hb + i, _mm256_fmsub_pd(vhc, t, _mm256_mul_pd(vhs, u)));
            }
        }
#endif
#if BRUUN_LEVEL >= 1
        {
            const bruun_v2 vhc = V2_SET1(hc);
            const bruun_v2 vhs = V2_SET1(hs);
            const bruun_v2 half = V2_SET1(0.5);
            for (; i + 1 < w2; i += 2) {
                const bruun_v2 A = V2_LD(la + i);
                const bruun_v2 B = V2_LD(lb + i);
                const bruun_v2 C = V2_LD(ha + i);
                const bruun_v2 D = V2_LD(hb + i);
                const bruun_v2 u = V2_SUB(A, C);
                const bruun_v2 t = V2_ADD(B, D);
                V2_ST(la + i, V2_MUL(half, V2_ADD(A, C)));
                V2_ST(lb + i, V2_MADD(V2_MUL(vhc, u), vhs, t));
                V2_ST(ha + i, V2_MUL(half, V2_SUB(B, D)));
                V2_ST(hb + i, V2_MSUB(V2_MUL(vhc, t), vhs, u));
            }
        }
#endif
        for (; i < w2; ++i) {
            const double A = la[i];
            const double B = lb[i];
            const double C = ha[i];
            const double D = hb[i];
            const double u = A - C;
            const double t = B + D;
            la[i] = 0.5 * (A + C);      // ea
            lb[i] = hc * u + hs * t;    // oa
            ha[i] = 0.5 * (B - D);      // eb
            hb[i] = hc * t - hs * u;    // ob
        }
    }

    // Fused two-level forward cell (the DIP mirror of the DIF's norm2_fused):
    // parent (d, e) plus both child cells in ONE pass over the span's eight
    // w4-wide streams -- half the span traffic and loop overhead of two
    // radix-2 passes. Stream layout in: [ a0 a1 a2 a3 | b0 b1 b2 b3 ]
    // (a-row and b-row quarters); out: the four grandchild pairs
    // (d,4e) (2e-d,4e) (e-d,4e) (e+d,4e) in tree order.
    static BRUUN_ALWAYS_INLINE void cell4_fwd_ip(double* RESTRICT v, int w4,
                                                 double c0, double s0,
                                                 double cl, double sl,
                                                 double ch, double sh) {
        double* RESTRICT a0 = v;
        double* RESTRICT a1 = v + w4;
        double* RESTRICT a2 = v + 2 * static_cast<std::size_t>(w4);
        double* RESTRICT a3 = v + 3 * static_cast<std::size_t>(w4);
        double* RESTRICT b0 = v + 4 * static_cast<std::size_t>(w4);
        double* RESTRICT b1 = v + 5 * static_cast<std::size_t>(w4);
        double* RESTRICT b2 = v + 6 * static_cast<std::size_t>(w4);
        double* RESTRICT b3 = v + 7 * static_cast<std::size_t>(w4);
        int i = 0;
#if BRUUN_LEVEL >= 2
        {
            const __m256d vc0 = _mm256_set1_pd(c0);
            const __m256d vs0 = _mm256_set1_pd(s0);
            const __m256d vcl = _mm256_set1_pd(cl);
            const __m256d vsl = _mm256_set1_pd(sl);
            const __m256d vch = _mm256_set1_pd(ch);
            const __m256d vsh = _mm256_set1_pd(sh);
            for (; i + 3 < w4; i += 4) {
                const __m256d A0 = _mm256_loadu_pd(a0 + i);
                const __m256d A1 = _mm256_loadu_pd(a1 + i);
                const __m256d A2 = _mm256_loadu_pd(a2 + i);
                const __m256d A3 = _mm256_loadu_pd(a3 + i);
                const __m256d B0 = _mm256_loadu_pd(b0 + i);
                const __m256d B1 = _mm256_loadu_pd(b1 + i);
                const __m256d B2 = _mm256_loadu_pd(b2 + i);
                const __m256d B3 = _mm256_loadu_pd(b3 + i);
                const __m256d R0 = _mm256_fmsub_pd(vc0, A2, _mm256_mul_pd(vs0, B2));
                const __m256d I0 = _mm256_fmadd_pd(vs0, A2, _mm256_mul_pd(vc0, B2));
                const __m256d R1 = _mm256_fmsub_pd(vc0, A3, _mm256_mul_pd(vs0, B3));
                const __m256d I1 = _mm256_fmadd_pd(vs0, A3, _mm256_mul_pd(vc0, B3));
                const __m256d la0 = _mm256_add_pd(A0, R0);
                const __m256d la1 = _mm256_add_pd(A1, R1);
                const __m256d lb0 = _mm256_add_pd(B0, I0);
                const __m256d lb1 = _mm256_add_pd(B1, I1);
                const __m256d ha0 = _mm256_sub_pd(A0, R0);
                const __m256d ha1 = _mm256_sub_pd(A1, R1);
                const __m256d hb0 = _mm256_sub_pd(I0, B0);
                const __m256d hb1 = _mm256_sub_pd(I1, B1);
                const __m256d Rl = _mm256_fmsub_pd(vcl, la1, _mm256_mul_pd(vsl, lb1));
                const __m256d Il = _mm256_fmadd_pd(vsl, la1, _mm256_mul_pd(vcl, lb1));
                const __m256d Rh = _mm256_fmsub_pd(vch, ha1, _mm256_mul_pd(vsh, hb1));
                const __m256d Ih = _mm256_fmadd_pd(vsh, ha1, _mm256_mul_pd(vch, hb1));
                _mm256_storeu_pd(a0 + i, _mm256_add_pd(la0, Rl));
                _mm256_storeu_pd(a1 + i, _mm256_add_pd(lb0, Il));
                _mm256_storeu_pd(a2 + i, _mm256_sub_pd(la0, Rl));
                _mm256_storeu_pd(a3 + i, _mm256_sub_pd(Il, lb0));
                _mm256_storeu_pd(b0 + i, _mm256_add_pd(ha0, Rh));
                _mm256_storeu_pd(b1 + i, _mm256_add_pd(hb0, Ih));
                _mm256_storeu_pd(b2 + i, _mm256_sub_pd(ha0, Rh));
                _mm256_storeu_pd(b3 + i, _mm256_sub_pd(Ih, hb0));
            }
        }
#endif
#if BRUUN_LEVEL >= 1
        {
            const bruun_v2 vc0 = V2_SET1(c0);
            const bruun_v2 vs0 = V2_SET1(s0);
            const bruun_v2 vcl = V2_SET1(cl);
            const bruun_v2 vsl = V2_SET1(sl);
            const bruun_v2 vch = V2_SET1(ch);
            const bruun_v2 vsh = V2_SET1(sh);
            for (; i + 1 < w4; i += 2) {
                const bruun_v2 A0 = V2_LD(a0 + i);
                const bruun_v2 A1 = V2_LD(a1 + i);
                const bruun_v2 A2 = V2_LD(a2 + i);
                const bruun_v2 A3 = V2_LD(a3 + i);
                const bruun_v2 B0 = V2_LD(b0 + i);
                const bruun_v2 B1 = V2_LD(b1 + i);
                const bruun_v2 B2 = V2_LD(b2 + i);
                const bruun_v2 B3 = V2_LD(b3 + i);
                const bruun_v2 R0 = V2_MSUB(V2_MUL(vc0, A2), vs0, B2);
                const bruun_v2 I0 = V2_MADD(V2_MUL(vs0, A2), vc0, B2);
                const bruun_v2 R1 = V2_MSUB(V2_MUL(vc0, A3), vs0, B3);
                const bruun_v2 I1 = V2_MADD(V2_MUL(vs0, A3), vc0, B3);
                const bruun_v2 la0 = V2_ADD(A0, R0);
                const bruun_v2 la1 = V2_ADD(A1, R1);
                const bruun_v2 lb0 = V2_ADD(B0, I0);
                const bruun_v2 lb1 = V2_ADD(B1, I1);
                const bruun_v2 ha0 = V2_SUB(A0, R0);
                const bruun_v2 ha1 = V2_SUB(A1, R1);
                const bruun_v2 hb0 = V2_SUB(I0, B0);
                const bruun_v2 hb1 = V2_SUB(I1, B1);
                const bruun_v2 Rl = V2_MSUB(V2_MUL(vcl, la1), vsl, lb1);
                const bruun_v2 Il = V2_MADD(V2_MUL(vsl, la1), vcl, lb1);
                const bruun_v2 Rh = V2_MSUB(V2_MUL(vch, ha1), vsh, hb1);
                const bruun_v2 Ih = V2_MADD(V2_MUL(vsh, ha1), vch, hb1);
                V2_ST(a0 + i, V2_ADD(la0, Rl));
                V2_ST(a1 + i, V2_ADD(lb0, Il));
                V2_ST(a2 + i, V2_SUB(la0, Rl));
                V2_ST(a3 + i, V2_SUB(Il, lb0));
                V2_ST(b0 + i, V2_ADD(ha0, Rh));
                V2_ST(b1 + i, V2_ADD(hb0, Ih));
                V2_ST(b2 + i, V2_SUB(ha0, Rh));
                V2_ST(b3 + i, V2_SUB(Ih, hb0));
            }
        }
#endif
        for (; i < w4; ++i) {
            const double A0 = a0[i];
            const double A1 = a1[i];
            const double A2 = a2[i];
            const double A3 = a3[i];
            const double B0 = b0[i];
            const double B1 = b1[i];
            const double B2 = b2[i];
            const double B3 = b3[i];
            const double R0 = c0 * A2 - s0 * B2;
            const double I0 = s0 * A2 + c0 * B2;
            const double R1 = c0 * A3 - s0 * B3;
            const double I1 = s0 * A3 + c0 * B3;
            const double la0 = A0 + R0;
            const double la1 = A1 + R1;
            const double lb0 = B0 + I0;
            const double lb1 = B1 + I1;
            const double ha0 = A0 - R0;
            const double ha1 = A1 - R1;
            const double hb0 = I0 - B0;
            const double hb1 = I1 - B1;
            const double Rl = cl * la1 - sl * lb1;
            const double Il = sl * la1 + cl * lb1;
            const double Rh = ch * ha1 - sh * hb1;
            const double Ih = sh * ha1 + ch * hb1;
            a0[i] = la0 + Rl;
            a1[i] = lb0 + Il;
            a2[i] = la0 - Rl;
            a3[i] = Il - lb0;
            b0[i] = ha0 + Rh;
            b1[i] = hb0 + Ih;
            b2[i] = ha0 - Rh;
            b3[i] = Ih - hb0;
        }
    }

    // Fused two-level inverse cell: both child inverse cells plus the parent
    // inverse cell in one pass. Twiddles arrive pre-halved (hc = 0.5*c).
    static BRUUN_ALWAYS_INLINE void cell4_inv_ip(double* RESTRICT v, int w4,
                                                 double hc0, double hs0,
                                                 double hcl, double hsl,
                                                 double hch, double hsh) {
        double* RESTRICT a0 = v;
        double* RESTRICT a1 = v + w4;
        double* RESTRICT a2 = v + 2 * static_cast<std::size_t>(w4);
        double* RESTRICT a3 = v + 3 * static_cast<std::size_t>(w4);
        double* RESTRICT b0 = v + 4 * static_cast<std::size_t>(w4);
        double* RESTRICT b1 = v + 5 * static_cast<std::size_t>(w4);
        double* RESTRICT b2 = v + 6 * static_cast<std::size_t>(w4);
        double* RESTRICT b3 = v + 7 * static_cast<std::size_t>(w4);
        int i = 0;
#if BRUUN_LEVEL >= 2
        {
            const __m256d vhc0 = _mm256_set1_pd(hc0);
            const __m256d vhs0 = _mm256_set1_pd(hs0);
            const __m256d vhcl = _mm256_set1_pd(hcl);
            const __m256d vhsl = _mm256_set1_pd(hsl);
            const __m256d vhch = _mm256_set1_pd(hch);
            const __m256d vhsh = _mm256_set1_pd(hsh);
            const __m256d half = _mm256_set1_pd(0.5);
            for (; i + 3 < w4; i += 4) {
                const __m256d X0 = _mm256_loadu_pd(a0 + i);
                const __m256d X1 = _mm256_loadu_pd(a1 + i);
                const __m256d X2 = _mm256_loadu_pd(a2 + i);
                const __m256d X3 = _mm256_loadu_pd(a3 + i);
                const __m256d X4 = _mm256_loadu_pd(b0 + i);
                const __m256d X5 = _mm256_loadu_pd(b1 + i);
                const __m256d X6 = _mm256_loadu_pd(b2 + i);
                const __m256d X7 = _mm256_loadu_pd(b3 + i);
                const __m256d u = _mm256_sub_pd(X0, X2);
                const __m256d t = _mm256_add_pd(X1, X3);
                const __m256d la0 = _mm256_mul_pd(half, _mm256_add_pd(X0, X2));
                const __m256d la1 = _mm256_fmadd_pd(vhcl, u, _mm256_mul_pd(vhsl, t));
                const __m256d lb0 = _mm256_mul_pd(half, _mm256_sub_pd(X1, X3));
                const __m256d lb1 = _mm256_fmsub_pd(vhcl, t, _mm256_mul_pd(vhsl, u));
                const __m256d u2 = _mm256_sub_pd(X4, X6);
                const __m256d t2 = _mm256_add_pd(X5, X7);
                const __m256d ha0 = _mm256_mul_pd(half, _mm256_add_pd(X4, X6));
                const __m256d ha1 = _mm256_fmadd_pd(vhch, u2, _mm256_mul_pd(vhsh, t2));
                const __m256d hb0 = _mm256_mul_pd(half, _mm256_sub_pd(X5, X7));
                const __m256d hb1 = _mm256_fmsub_pd(vhch, t2, _mm256_mul_pd(vhsh, u2));
                const __m256d u3 = _mm256_sub_pd(la0, ha0);
                const __m256d t3 = _mm256_add_pd(lb0, hb0);
                const __m256d u4 = _mm256_sub_pd(la1, ha1);
                const __m256d t4 = _mm256_add_pd(lb1, hb1);
                _mm256_storeu_pd(a0 + i, _mm256_mul_pd(half, _mm256_add_pd(la0, ha0)));
                _mm256_storeu_pd(a1 + i, _mm256_mul_pd(half, _mm256_add_pd(la1, ha1)));
                _mm256_storeu_pd(a2 + i, _mm256_fmadd_pd(vhc0, u3, _mm256_mul_pd(vhs0, t3)));
                _mm256_storeu_pd(a3 + i, _mm256_fmadd_pd(vhc0, u4, _mm256_mul_pd(vhs0, t4)));
                _mm256_storeu_pd(b0 + i, _mm256_mul_pd(half, _mm256_sub_pd(lb0, hb0)));
                _mm256_storeu_pd(b1 + i, _mm256_mul_pd(half, _mm256_sub_pd(lb1, hb1)));
                _mm256_storeu_pd(b2 + i, _mm256_fmsub_pd(vhc0, t3, _mm256_mul_pd(vhs0, u3)));
                _mm256_storeu_pd(b3 + i, _mm256_fmsub_pd(vhc0, t4, _mm256_mul_pd(vhs0, u4)));
            }
        }
#endif
#if BRUUN_LEVEL >= 1
        {
            const bruun_v2 vhc0 = V2_SET1(hc0);
            const bruun_v2 vhs0 = V2_SET1(hs0);
            const bruun_v2 vhcl = V2_SET1(hcl);
            const bruun_v2 vhsl = V2_SET1(hsl);
            const bruun_v2 vhch = V2_SET1(hch);
            const bruun_v2 vhsh = V2_SET1(hsh);
            const bruun_v2 half = V2_SET1(0.5);
            for (; i + 1 < w4; i += 2) {
                const bruun_v2 X0 = V2_LD(a0 + i);
                const bruun_v2 X1 = V2_LD(a1 + i);
                const bruun_v2 X2 = V2_LD(a2 + i);
                const bruun_v2 X3 = V2_LD(a3 + i);
                const bruun_v2 X4 = V2_LD(b0 + i);
                const bruun_v2 X5 = V2_LD(b1 + i);
                const bruun_v2 X6 = V2_LD(b2 + i);
                const bruun_v2 X7 = V2_LD(b3 + i);
                const bruun_v2 u = V2_SUB(X0, X2);
                const bruun_v2 t = V2_ADD(X1, X3);
                const bruun_v2 la0 = V2_MUL(half, V2_ADD(X0, X2));
                const bruun_v2 la1 = V2_MADD(V2_MUL(vhcl, u), vhsl, t);
                const bruun_v2 lb0 = V2_MUL(half, V2_SUB(X1, X3));
                const bruun_v2 lb1 = V2_MSUB(V2_MUL(vhcl, t), vhsl, u);
                const bruun_v2 u2 = V2_SUB(X4, X6);
                const bruun_v2 t2 = V2_ADD(X5, X7);
                const bruun_v2 ha0 = V2_MUL(half, V2_ADD(X4, X6));
                const bruun_v2 ha1 = V2_MADD(V2_MUL(vhch, u2), vhsh, t2);
                const bruun_v2 hb0 = V2_MUL(half, V2_SUB(X5, X7));
                const bruun_v2 hb1 = V2_MSUB(V2_MUL(vhch, t2), vhsh, u2);
                const bruun_v2 u3 = V2_SUB(la0, ha0);
                const bruun_v2 t3 = V2_ADD(lb0, hb0);
                const bruun_v2 u4 = V2_SUB(la1, ha1);
                const bruun_v2 t4 = V2_ADD(lb1, hb1);
                V2_ST(a0 + i, V2_MUL(half, V2_ADD(la0, ha0)));
                V2_ST(a1 + i, V2_MUL(half, V2_ADD(la1, ha1)));
                V2_ST(a2 + i, V2_MADD(V2_MUL(vhc0, u3), vhs0, t3));
                V2_ST(a3 + i, V2_MADD(V2_MUL(vhc0, u4), vhs0, t4));
                V2_ST(b0 + i, V2_MUL(half, V2_SUB(lb0, hb0)));
                V2_ST(b1 + i, V2_MUL(half, V2_SUB(lb1, hb1)));
                V2_ST(b2 + i, V2_MSUB(V2_MUL(vhc0, t3), vhs0, u3));
                V2_ST(b3 + i, V2_MSUB(V2_MUL(vhc0, t4), vhs0, u4));
            }
        }
#endif
        for (; i < w4; ++i) {
            const double X0 = a0[i];
            const double X1 = a1[i];
            const double X2 = a2[i];
            const double X3 = a3[i];
            const double X4 = b0[i];
            const double X5 = b1[i];
            const double X6 = b2[i];
            const double X7 = b3[i];
            const double u = X0 - X2;
            const double t = X1 + X3;
            const double la0 = 0.5 * (X0 + X2);
            const double la1 = hcl * u + hsl * t;
            const double lb0 = 0.5 * (X1 - X3);
            const double lb1 = hcl * t - hsl * u;
            const double u2 = X4 - X6;
            const double t2 = X5 + X7;
            const double ha0 = 0.5 * (X4 + X6);
            const double ha1 = hch * u2 + hsh * t2;
            const double hb0 = 0.5 * (X5 - X7);
            const double hb1 = hch * t2 - hsh * u2;
            const double u3 = la0 - ha0;
            const double t3 = lb0 + hb0;
            const double u4 = la1 - ha1;
            const double t4 = lb1 + hb1;
            a0[i] = 0.5 * (la0 + ha0);
            a1[i] = 0.5 * (la1 + ha1);
            a2[i] = hc0 * u3 + hs0 * t3;
            a3[i] = hc0 * u4 + hs0 * t4;
            b0[i] = 0.5 * (lb0 - hb0);
            b1[i] = 0.5 * (lb1 - hb1);
            b2[i] = hc0 * t3 - hs0 * u3;
            b3[i] = hc0 * t4 - hs0 * u4;
        }
    }

    // In-place ridge fold on the dc row [ lo : q2 | hi : q2 ] -> [ dc' | ny' ].
    static BRUUN_ALWAYS_INLINE void binom_fwd_ip(double* RESTRICT v, int q2) {
        double* RESTRICT lo = v;
        double* RESTRICT hi = v + q2;
        int i = 0;
#if BRUUN_LEVEL >= 2
        for (; i + 3 < q2; i += 4) {
            const __m256d a = _mm256_loadu_pd(lo + i);
            const __m256d b = _mm256_loadu_pd(hi + i);
            _mm256_storeu_pd(lo + i, _mm256_add_pd(a, b));
            _mm256_storeu_pd(hi + i, _mm256_sub_pd(a, b));
        }
#endif
#if BRUUN_LEVEL >= 1
        for (; i + 1 < q2; i += 2) {
            const bruun_v2 a = V2_LD(lo + i);
            const bruun_v2 b = V2_LD(hi + i);
            V2_ST(lo + i, V2_ADD(a, b));
            V2_ST(hi + i, V2_SUB(a, b));
        }
#endif
        for (; i < q2; ++i) {
            const double a = lo[i];
            const double b = hi[i];
            lo[i] = a + b;
            hi[i] = a - b;
        }
    }

    // In-place ridge unfold: [ dc' | ny' ] -> [ lo | hi ] with the 0.5.
    static BRUUN_ALWAYS_INLINE void binom_inv_ip(double* RESTRICT v, int q2) {
        double* RESTRICT lo = v;
        double* RESTRICT hi = v + q2;
        int i = 0;
#if BRUUN_LEVEL >= 2
        {
            const __m256d half = _mm256_set1_pd(0.5);
            for (; i + 3 < q2; i += 4) {
                const __m256d a = _mm256_loadu_pd(lo + i);
                const __m256d b = _mm256_loadu_pd(hi + i);
                _mm256_storeu_pd(lo + i, _mm256_mul_pd(half, _mm256_add_pd(a, b)));
                _mm256_storeu_pd(hi + i, _mm256_mul_pd(half, _mm256_sub_pd(a, b)));
            }
        }
#endif
#if BRUUN_LEVEL >= 1
        {
            const bruun_v2 half = V2_SET1(0.5);
            for (; i + 1 < q2; i += 2) {
                const bruun_v2 a = V2_LD(lo + i);
                const bruun_v2 b = V2_LD(hi + i);
                V2_ST(lo + i, V2_MUL(half, V2_ADD(a, b)));
                V2_ST(hi + i, V2_MUL(half, V2_SUB(a, b)));
            }
        }
#endif
        for (; i < q2; ++i) {
            const double a = lo[i];
            const double b = hi[i];
            lo[i] = 0.5 * (a + b);
            hi[i] = 0.5 * (a - b);
        }
    }

    int phase_table_index(int d, int e) const noexcept {
        return static_cast<int>((static_cast<long long>(d) * n_) / (2LL * e));
    }

    // -----------------------------------------------------------------------
    // The depth-first walk. fwd_span descends the packet (d, e) held in
    // [ a : w | b : w ] at v; leaves (w == 1) stay in place -- the boundary
    // pass in forward_standard/inverse_standard resolves the permutation.
    // The w == 8 terminal straight-lines the last three levels (7 cells) so
    // the deep recursion collapses to one call per 16 doubles.
    // -----------------------------------------------------------------------

    void span8_fwd(double* RESTRICT v, int d, int e) const {
        const int e2 = e << 1;
        const int e4 = e2 << 1;
        const int k0 = phase_table_index(d, e);
        const int kl = phase_table_index(d, e2);
        const int kh = phase_table_index(e - d, e2);
        const int k00 = phase_table_index(d, e4);
        const int k01 = phase_table_index(e2 - d, e4);
        const int k10 = phase_table_index(e - d, e4);
        const int k11 = phase_table_index(e + d, e4);
        cell_fwd_ip(v, 4, cos_[static_cast<std::size_t>(k0)], sin_[static_cast<std::size_t>(k0)]);
        cell_fwd_ip(v, 2, cos_[static_cast<std::size_t>(kl)], sin_[static_cast<std::size_t>(kl)]);
        cell_fwd_ip(v + 8, 2, cos_[static_cast<std::size_t>(kh)], sin_[static_cast<std::size_t>(kh)]);
        cell_fwd_ip(v, 1, cos_[static_cast<std::size_t>(k00)], sin_[static_cast<std::size_t>(k00)]);
        cell_fwd_ip(v + 4, 1, cos_[static_cast<std::size_t>(k01)], sin_[static_cast<std::size_t>(k01)]);
        cell_fwd_ip(v + 8, 1, cos_[static_cast<std::size_t>(k10)], sin_[static_cast<std::size_t>(k10)]);
        cell_fwd_ip(v + 12, 1, cos_[static_cast<std::size_t>(k11)], sin_[static_cast<std::size_t>(k11)]);
    }

    void span8_inv(double* RESTRICT v, int d, int e,
                   const complex_t* RESTRICT in) const {
        const int e2 = e << 1;
        const int e4 = e2 << 1;
        const int e3 = e2 + e;
        // gather the eight leaves of this subtree (scattered reads, hot writes)
        v[0] = in[d].re;             v[1] = -in[d].im;
        v[2] = in[e4 - d].re;        v[3] = -in[e4 - d].im;
        v[4] = in[e2 - d].re;        v[5] = -in[e2 - d].im;
        v[6] = in[e2 + d].re;        v[7] = -in[e2 + d].im;
        v[8] = in[e - d].re;         v[9] = -in[e - d].im;
        v[10] = in[e3 + d].re;       v[11] = -in[e3 + d].im;
        v[12] = in[e + d].re;        v[13] = -in[e + d].im;
        v[14] = in[e3 - d].re;       v[15] = -in[e3 - d].im;
        const int k0 = phase_table_index(d, e);
        const int kl = phase_table_index(d, e2);
        const int kh = phase_table_index(e - d, e2);
        const int k00 = phase_table_index(d, e4);
        const int k01 = phase_table_index(e2 - d, e4);
        const int k10 = phase_table_index(e - d, e4);
        const int k11 = phase_table_index(e + d, e4);
        cell_inv_ip(v, 1, 0.5 * cos_[static_cast<std::size_t>(k00)], 0.5 * sin_[static_cast<std::size_t>(k00)]);
        cell_inv_ip(v + 4, 1, 0.5 * cos_[static_cast<std::size_t>(k01)], 0.5 * sin_[static_cast<std::size_t>(k01)]);
        cell_inv_ip(v + 8, 1, 0.5 * cos_[static_cast<std::size_t>(k10)], 0.5 * sin_[static_cast<std::size_t>(k10)]);
        cell_inv_ip(v + 12, 1, 0.5 * cos_[static_cast<std::size_t>(k11)], 0.5 * sin_[static_cast<std::size_t>(k11)]);
        cell_inv_ip(v, 2, 0.5 * cos_[static_cast<std::size_t>(kl)], 0.5 * sin_[static_cast<std::size_t>(kl)]);
        cell_inv_ip(v + 8, 2, 0.5 * cos_[static_cast<std::size_t>(kh)], 0.5 * sin_[static_cast<std::size_t>(kh)]);
        cell_inv_ip(v, 4, 0.5 * cos_[static_cast<std::size_t>(k0)], 0.5 * sin_[static_cast<std::size_t>(k0)]);
    }

    void fwd_span(double* RESTRICT v, int w, int d, int e) const {
        if (w >= 32) {
            // radix-4 step: fused two-level cell, recurse into grandchildren.
            const int w4 = w >> 2;
            const int e2 = e << 1;
            const int e4 = e2 << 1;
            const int k0 = phase_table_index(d, e);
            const int kl = phase_table_index(d, e2);
            const int kh = phase_table_index(e - d, e2);
            cell4_fwd_ip(v, w4,
                         cos_[static_cast<std::size_t>(k0)], sin_[static_cast<std::size_t>(k0)],
                         cos_[static_cast<std::size_t>(kl)], sin_[static_cast<std::size_t>(kl)],
                         cos_[static_cast<std::size_t>(kh)], sin_[static_cast<std::size_t>(kh)]);
            fwd_span(v, w4, d, e4);
            fwd_span(v + (w >> 1), w4, e2 - d, e4);
            fwd_span(v + w, w4, e - d, e4);
            fwd_span(v + w + (w >> 1), w4, e + d, e4);
            return;
        }
        if (w == 16) {
            const int k = phase_table_index(d, e);
            cell_fwd_ip(v, 8, cos_[static_cast<std::size_t>(k)],
                        sin_[static_cast<std::size_t>(k)]);
            const int e2 = e << 1;
            span8_fwd(v, d, e2);
            span8_fwd(v + 16, e - d, e2);
            return;
        }
        if (w == 8) {
            span8_fwd(v, d, e);
            return;
        }
        if (w == 1) {
            return;
        }
        const int w2 = w >> 1;
        const int k = phase_table_index(d, e);
        cell_fwd_ip(v, w2, cos_[static_cast<std::size_t>(k)],
                    sin_[static_cast<std::size_t>(k)]);
        if (w == 2) {
            return;
        }
        const int e2 = e << 1;
        fwd_span(v, w2, d, e2);
        fwd_span(v + w, w2, e - d, e2);
    }

    void fwd_ridge(double* RESTRICT v, int q, int e) const {
        if (q == 1) {
            return;    // dc/ny final at v[0], v[1]
        }
        const int q2 = q >> 1;
        binom_fwd_ip(v, q2);
        const int e2 = e << 1;
        fwd_ridge(v, q2, e2);
        fwd_span(v + q, q2, e >> 1, e2);   // promoted bin (e/2, 2e), free
    }

    void inv_span(double* RESTRICT v, int w, int d, int e,
                  const complex_t* RESTRICT in) const {
        if (w >= 32) {
            // radix-4 step: ascend grandchildren, then fused two-level cell.
            const int w4 = w >> 2;
            const int e2 = e << 1;
            const int e4 = e2 << 1;
            inv_span(v, w4, d, e4, in);
            inv_span(v + (w >> 1), w4, e2 - d, e4, in);
            inv_span(v + w, w4, e - d, e4, in);
            inv_span(v + w + (w >> 1), w4, e + d, e4, in);
            const int k0 = phase_table_index(d, e);
            const int kl = phase_table_index(d, e2);
            const int kh = phase_table_index(e - d, e2);
            cell4_inv_ip(v, w4,
                         0.5 * cos_[static_cast<std::size_t>(k0)], 0.5 * sin_[static_cast<std::size_t>(k0)],
                         0.5 * cos_[static_cast<std::size_t>(kl)], 0.5 * sin_[static_cast<std::size_t>(kl)],
                         0.5 * cos_[static_cast<std::size_t>(kh)], 0.5 * sin_[static_cast<std::size_t>(kh)]);
            return;
        }
        if (w == 16) {
            const int e2 = e << 1;
            span8_inv(v, d, e2, in);
            span8_inv(v + 16, e - d, e2, in);
            const int k = phase_table_index(d, e);
            cell_inv_ip(v, 8, 0.5 * cos_[static_cast<std::size_t>(k)],
                        0.5 * sin_[static_cast<std::size_t>(k)]);
            return;
        }
        if (w == 8) {
            span8_inv(v, d, e, in);
            return;
        }
        if (w == 1) {
            v[0] = in[d].re;
            v[1] = -in[d].im;
            return;
        }
        const int w2 = w >> 1;
        if (w == 2) {
            v[0] = in[d].re;
            v[1] = -in[d].im;
            v[2] = in[e - d].re;
            v[3] = -in[e - d].im;
        } else {
            const int e2 = e << 1;
            inv_span(v, w2, d, e2, in);
            inv_span(v + w, w2, e - d, e2, in);
        }
        const int k = phase_table_index(d, e);
        cell_inv_ip(v, w2, 0.5 * cos_[static_cast<std::size_t>(k)],
                    0.5 * sin_[static_cast<std::size_t>(k)]);
    }

    void inv_ridge(double* RESTRICT v, int q, int e,
                   const complex_t* RESTRICT in) const {
        if (q == 1) {
            v[0] = in[0].re;
            v[1] = in[half_].re;
            return;
        }
        const int q2 = q >> 1;
        const int e2 = e << 1;
        inv_ridge(v, q2, e2, in);
        inv_span(v + q, q2, e >> 1, e2, in);
        binom_inv_ip(v, q2);
    }

    // -----------------------------------------------------------------------
    // Boundary permutation build (init only): where each bin's leaf pair
    // lands. Mirrors the descent's address math exactly.
    // -----------------------------------------------------------------------

    void map_span(int off, int w, int d, int e) {
        if (w == 1) {
            leaf_[static_cast<std::size_t>(d)] = off;
            return;
        }
        const int w2 = w >> 1;
        const int e2 = e << 1;
        map_span(off, w2, d, e2);
        map_span(off + w, w2, e - d, e2);
    }

    void map_ridge(int off, int q, int e) {
        if (q == 1) {
            return;    // dc/ny fixed at work[0], work[1]
        }
        const int q2 = q >> 1;
        const int e2 = e << 1;
        map_ridge(off, q2, e2);
        map_span(off + q, q2, e >> 1, e2);
    }

    // -----------------------------------------------------------------------
    // Seeds: one streaming pass, comb reads -> contiguous slot-row writes.
    // Slot order is the tree (descent) order of the walk above.
    // -----------------------------------------------------------------------

    void forward_seed4(const double* RESTRICT input, double* RESTRICT dst) const {
        const int q = n_ >> 2;
        // level-4 slots: [ dc | ny | a1 | b1 ]
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
            r0[i] = a + b + c + d;   // dc
            r1[i] = a - b + c - d;   // ny
            r2[i] = a - c;           // a1 = Re
            r3[i] = b - d;           // b1 = -Im
        }
    }

    void forward_seed8(const double* RESTRICT input, double* RESTRICT dst) const {
        const int q = n_ >> 3;
        // level-8 slots: [ dc | ny | a2 | b2 | a1 | b1 | a3 | b3 ]
        double* RESTRICT r0 = dst;                                    // dc
        double* RESTRICT r1 = dst + q;                                // ny
        double* RESTRICT r2 = dst + 4 * static_cast<std::size_t>(q);  // a1
        double* RESTRICT r3 = dst + 5 * static_cast<std::size_t>(q);  // b1
        double* RESTRICT r4 = dst + 2 * static_cast<std::size_t>(q);  // a2
        double* RESTRICT r5 = dst + 3 * static_cast<std::size_t>(q);  // b2
        double* RESTRICT r6 = dst + 6 * static_cast<std::size_t>(q);  // a3
        double* RESTRICT r7 = dst + 7 * static_cast<std::size_t>(q);  // b3
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
        // level-16 slots: [ dc | ny | a4 b4 | a2 b2 | a6 b6 | a1 b1 | a7 b7 | a3 b3 | a5 b5 ]
        double* RESTRICT r0 = dst;                                     // dc
        double* RESTRICT r1 = dst + q;                                 // ny
        double* RESTRICT r2 = dst + 8 * static_cast<std::size_t>(q);   // a1
        double* RESTRICT r3 = dst + 9 * static_cast<std::size_t>(q);   // b1
        double* RESTRICT r4 = dst + 4 * static_cast<std::size_t>(q);   // a2
        double* RESTRICT r5 = dst + 5 * static_cast<std::size_t>(q);   // b2
        double* RESTRICT r6 = dst + 12 * static_cast<std::size_t>(q);  // a3
        double* RESTRICT r7 = dst + 13 * static_cast<std::size_t>(q);  // b3
        double* RESTRICT r8 = dst + 2 * static_cast<std::size_t>(q);   // a4
        double* RESTRICT r9 = dst + 3 * static_cast<std::size_t>(q);   // b4
        double* RESTRICT r10 = dst + 14 * static_cast<std::size_t>(q); // a5
        double* RESTRICT r11 = dst + 15 * static_cast<std::size_t>(q); // b5
        double* RESTRICT r12 = dst + 6 * static_cast<std::size_t>(q);  // a6
        double* RESTRICT r13 = dst + 7 * static_cast<std::size_t>(q);  // b6
        double* RESTRICT r14 = dst + 10 * static_cast<std::size_t>(q); // a7
        double* RESTRICT r15 = dst + 11 * static_cast<std::size_t>(q); // b7

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

            r0[i] = le0 + ro0;   // dc
            r1[i] = le0 - ro0;   // ny
            r8[i] = le4;         // a4
            r9[i] = ro4;         // b4

            pair_reduce_cs(le1, le7, ro1, ro7, c1, s1, r2[i], r3[i], r14[i], r15[i]);
            pair_reduce_cs(le2, le6, ro2, ro6, c2, s2, r4[i], r5[i], r12[i], r13[i]);
            pair_reduce_cs(le3, le5, ro3, ro5, c3, s3, r6[i], r7[i], r10[i], r11[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Inverse tails: level-8 / level-4 slot state -> time, one streaming pass.
    // -----------------------------------------------------------------------

    void inverse_tail4(const double* RESTRICT src, double* RESTRICT output, int q) const {
        // level-4 slots: [ dc | ny | a1 | b1 ]; the tail below is written in
        // the (dc, a1, ny, b1) basis, so remap the reads.
        const double* RESTRICT r0 = src;                                    // dc
        const double* RESTRICT r1 = src + 2 * static_cast<std::size_t>(q);  // a1
        const double* RESTRICT r2 = src + q;                                // ny
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
        // level-8 slots: [ dc | ny | a2 b2 | a1 b1 | a3 b3 ]; the tail below is
        // written in the (dc, a1, a2, a3, ny, b3, b2, b1) basis, so remap.
        const double* RESTRICT r0 = src;                                    // dc
        const double* RESTRICT r1 = src + 4 * static_cast<std::size_t>(q);  // a1
        const double* RESTRICT r2 = src + 2 * static_cast<std::size_t>(q);  // a2
        const double* RESTRICT r3 = src + 6 * static_cast<std::size_t>(q);  // a3
        const double* RESTRICT r4 = src + q;                                // ny
        const double* RESTRICT r5 = src + 7 * static_cast<std::size_t>(q);  // b3
        const double* RESTRICT r6 = src + 3 * static_cast<std::size_t>(q);  // b2
        const double* RESTRICT r7 = src + 5 * static_cast<std::size_t>(q);  // b1
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

    void clear() noexcept {
        n_ = 0;
        half_ = 0;
        (void)cos_.resize(0);
        (void)sin_.resize(0);
        (void)leaf_.resize(0);
    }

    int n_;
    int half_;
    heap_array<double> cos_;
    heap_array<double> sin_;
    heap_array<int> leaf_;
};

} // namespace bruun
