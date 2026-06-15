#ifndef BFFT_EXPERIMENTS_RADIX4_BRUUN_KERNEL_HPP
#define BFFT_EXPERIMENTS_RADIX4_BRUUN_KERNEL_HPP

// Experimental radix-4-shaped Bruun split notes and scalar reference kernel.
//
// Goal
// ----
// Find a legal four-child real CRT basis for the normalized Bruun tree whose
// local map is cheaper than the existing two-level composition. The current
// fused implementation in src/detail/bruun_kernel.hpp performs one parent
// normalized split and the two child normalized splits while values are still in
// registers. That is already radix-4-shaped for locality, but not for arithmetic.
//
// Current cost for q parent lanes:
//     parent split over q lanes:          4q mul + 6q add
//     two child splits over q/2 lanes:    4q mul + 6q add
//     total:                             8q mul + 12q add = 20q flops
//
// Target for a mathematically useful radix-4 Bruun node:
//     at most 16q flops to tie the split-radix asymptotic constant,
//     less than 16q flops to be worth promoting beyond a locality kernel.
//
// Basis attempts recorded so far
// ------------------------------
// 1. Chebyshev basis, y = T_m(x).
//    T_4(y) - a = 8 y^4 - 8 y^2 + 1 - a is an even quartic, so its four roots
//    are paired as +/-sqrt((1 +/- sqrt((1 + a) / 2)) / 2). This preserves real
//    coefficients and gives a clean four-child CRT description. In the local
//    normalized coordinates, however, evaluating the four children still needs
//    the same two independent plane mixes unless a later level can absorb a
//    child-dependent scale. No sub-16q local map has fallen out of this chart.
//
// 2. Scaled local-complex basis, J = (z - c) / s.
//    This is the shipped normalized basis at quadratic leaves. It makes each
//    local quotient complex-like with J^2 = -1, but the two grandchildren carry
//    different half-angle constants. A direct four-child map becomes two child
//    rotations after the parent add/sub layer. Without allowing nonlocal scale
//    debt, it is algebraically the same cost as the composed binary split.
//
// 3. True four-child residue map.
//    Mapping directly from the parent quadratic-residue block to four quadratic
//    grandchildren is legal and local when expressed as the CRT product of the
//    parent split and both child splits. The scalar kernel below is that legal
//    reference. It should be used as the comparison oracle for any proposed
//    alternative basis. At present it is intentionally not wired into the build:
//    it records the search state and gives future experiments a tiny, auditable
//    kernel with the exact current arithmetic count.
//
// Discovery test
// --------------
// A candidate only changes the asymptotic story if it produces the same four
// residues as bruun_radix4_reference_split while using <= 16q flops locally, or
// if it carries explicit scale debt and proves that the debt cancels across
// neighboring levels without increasing the total above that bound.

#include <cstddef>

namespace bfft_experiments {

struct BruunRadix4Cost {
    std::size_t multiplications;
    std::size_t additions;
    std::size_t flops;
};

inline BruunRadix4Cost bruun_radix4_reference_cost(std::size_t q) {
    BruunRadix4Cost cost{};
    cost.multiplications = 8 * q;
    cost.additions = 12 * q;
    cost.flops = cost.multiplications + cost.additions;
    return cost;
}

// Scalar reference for the legal four-child CRT map. Layout matches
// norm2_fused: p contains A0, B0, A1, B1 blocks of q doubles. Each block is
// divided into low/high halves for the two child splits.
inline void bruun_radix4_reference_split(double* p,
                                         std::size_t q,
                                         double c,
                                         double s,
                                         double c0,
                                         double s0,
                                         double c1,
                                         double s1) {
    const std::size_t qh = q >> 1;
    double* A0 = p;
    double* B0 = p + q;
    double* A1 = p + 2 * q;
    double* B1 = p + 3 * q;

    for (std::size_t n = 0; n < qh; ++n) {
        const double a0n = A0[n];
        const double a0h = A0[qh + n];
        const double b0n = B0[n];
        const double b0h = B0[qh + n];
        const double a1n = A1[n];
        const double a1h = A1[qh + n];
        const double b1n = B1[n];
        const double b1h = B1[qh + n];

        const double parent_real_low = c * b0n - s * b1n;
        const double parent_imag_low = s * b0n + c * b1n;
        const double parent_real_high = c * b0h - s * b1h;
        const double parent_imag_high = s * b0h + c * b1h;

        const double child0_a_low = a0n + parent_real_low;
        const double child0_a_high = a0h + parent_real_high;
        const double child0_b_low = a1n + parent_imag_low;
        const double child0_b_high = a1h + parent_imag_high;

        const double child1_a_low = a0n - parent_real_low;
        const double child1_a_high = a0h - parent_real_high;
        const double child1_b_low = parent_imag_low - a1n;
        const double child1_b_high = parent_imag_high - a1h;

        const double child0_real = c0 * child0_a_high - s0 * child0_b_high;
        const double child0_imag = s0 * child0_a_high + c0 * child0_b_high;
        const double child1_real = c1 * child1_a_high - s1 * child1_b_high;
        const double child1_imag = s1 * child1_a_high + c1 * child1_b_high;

        A0[n] = child0_a_low + child0_real;
        A0[qh + n] = child0_b_low + child0_imag;
        B0[n] = child0_a_low - child0_real;
        B0[qh + n] = child0_imag - child0_b_low;

        A1[n] = child1_a_low + child1_real;
        A1[qh + n] = child1_b_low + child1_imag;
        B1[n] = child1_a_low - child1_real;
        B1[qh + n] = child1_imag - child1_b_low;
    }
}

}  // namespace bfft_experiments

#endif  // BFFT_EXPERIMENTS_RADIX4_BRUUN_KERNEL_HPP
