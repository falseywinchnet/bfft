#ifndef BFFT_CHEB_DCT_ASSEMBLE_HPP
#define BFFT_CHEB_DCT_ASSEMBLE_HPP

// Real FFT assembled from DCT-I + DST-I (ROADMAP Phase 5, the "easy" route in the
// difficulty matrix). For real x[0..N-1] the rFFT bins X[k] = C[k] - i S[k] split
// into a cosine projection C and a sine projection S on the Lobatto grid
// cos(2 pi k / N). Folding x symmetrically/antisymmetrically turns those into a
// DCT-I and a DST-I of half the length:
//
//   M = N/2
//   E[0]=x[0], E[M]=x[M], E[m]=(x[m]+x[N-m])/2  (m=1..M-1)   -> Re X = DCT-I_{M+1}(E)
//   O[j]=x[j+1]-x[N-(j+1)]  (j=0..M-2)                       -> Im X[b]=-DST-I_{M-1}(O)[b-1]/2
//   Im X[0] = Im X[M] = 0   (DC and Nyquist are real)
//
// The fold is N/2 adds + N/2 subs; the combine is zero-multiply. This file proves
// the assembly is correct (validated against the shipped Bruun rFFT and a direct
// DFT) and gives a benchmarkable real FFT built entirely from DCT/DST endpoints.
//
// Caveat made explicit by this construction: the rFFT lives on the Lobatto grid,
// so the DCT-I/DST-I here go through DCT1/DST1 (themselves real FFTs of length
// ~N). The assembled path therefore does ~2 FFTs' work and is NOT competitive
// with the single fused Bruun kernel -- consistent with the capstone's finding
// that the fused path is the locality optimum. A tree-native rFFT would need a
// first-kind-grid factorization of the FFT, which the radix-4 Chebyshev tree does
// not provide (see Phase 4). N must be a power of two with N >= 8.

#include <cstddef>
#include <vector>

#include <bfft/bfft.hpp>

#include "dct1_dst1_via_fft.hpp"

namespace cheb_dct_assemble {

class AssembledRFFT {
public:
    static bool supported(int n) {
        return n >= 8 && (n & (n - 1)) == 0;  // power of two; M-1 >= 3
    }

    explicit AssembledRFFT(int n)
        : n_(n), m_(n / 2), dct1_(n / 2 + 1), dst1_(n / 2 - 1) {
        e_.assign(m_ + 1, 0.0);
        o_.assign(m_ - 1, 0.0);
        re_.assign(m_ + 1, 0.0);
        ds_.assign(m_ - 1, 0.0);
    }

    int size() const { return n_; }
    int bins() const { return m_ + 1; }

    // X must hold m_+1 = N/2+1 complex bins.
    void forward(const double* input, bfft::complex* X) {
        // Symmetric fold -> DCT-I gives Re X.
        e_[0] = input[0];
        e_[m_] = input[m_];
        for (int m = 1; m < m_; ++m) e_[m] = 0.5 * (input[m] + input[n_ - m]);
        dct1_.forward(e_.data(), re_.data());

        // Antisymmetric fold -> DST-I gives Im X (interior bins only).
        for (int j = 0; j < m_ - 1; ++j) o_[j] = input[j + 1] - input[n_ - (j + 1)];
        dst1_.forward(o_.data(), ds_.data());

        X[0] = bfft::complex{re_[0], 0.0};
        X[m_] = bfft::complex{re_[m_], 0.0};
        for (int b = 1; b < m_; ++b) X[b] = bfft::complex{re_[b], -0.5 * ds_[b - 1]};
    }

private:
    int n_, m_;
    cheb_dct_endpoint::DCT1 dct1_;  // size M+1
    cheb_dct_endpoint::DST1 dst1_;  // size M-1
    std::vector<double> e_, o_, re_, ds_;
};

}  // namespace cheb_dct_assemble

#endif  // BFFT_CHEB_DCT_ASSEMBLE_HPP
