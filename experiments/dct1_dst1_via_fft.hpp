#ifndef BFFT_DCT1_DST1_VIA_FFT_HPP
#define BFFT_DCT1_DST1_VIA_FFT_HPP

// DCT-I / DST-I endpoints (ROADMAP Phase 4) via the shipped BFFT real FFT.
//
// Why these are different from Phases 1-3
// --------------------------------------
// DCT-II/III/IV and DST-II/III/IV all live on the first-kind Chebyshev grid
// x_k = cos(pi(2k+1)/2n) -- the roots of T_n -- which is exactly what the radix-4
// Chebyshev tree (ChebDCT) evaluates at. DCT-I/DST-I instead live on the
// Chebyshev-Lobatto grid x_k = cos(pi k/(n-1)), the extrema of T_{n-1}, which
// INCLUDES the endpoints +-1. The first-kind tree does not produce that grid, so
// there is no in-tree fast path; the endpoint cases need a different engine.
//
// The natural one is the real FFT of the (anti)symmetric extension -- which is
// also the "easy" FFT<->DCT/DST route in the ROADMAP difficulty matrix, so this
// header doubles as the first concrete step of Phase 5. It reuses BFFT's own
// power-of-two real FFT (src/detail/bruun_kernel.hpp via bfft::plan), with all
// buffers preallocated in init() and zero per-call heap allocation.
//
// Conventions (FFTW, unnormalized), verified against the direct oracle:
//   DCT-I (REDFT00), logical 2(n-1):  Y_k = Re V[k],      V = rFFT of the
//     symmetric extension v (length M=2(n-1)): v[0..n-1]=X, v[M-j]=X[j], 1<=j<=n-2.
//   DST-I (RODFT00), logical 2(n+1):  Y_k = -Im V[k+1],   V = rFFT of the
//     antisymmetric extension v (length M=2(n+1)): v[j+1]=X[j], v[M-(j+1)]=-X[j].
// Both are self-inverse up to 2(n-1) / 2(n+1) respectively.
//
// Size constraint: M must be a power of two for BFFT. So DCT-I needs n-1 a power
// of two (n = 2^m + 1) and DST-I needs n+1 a power of two (n = 2^m - 1). Other n
// fall back to the direct oracle (chebyshev_dct_kernel.hpp); a general-n fast
// path would use a mixed-radix real FFT, out of scope here.

#include <cstddef>
#include <vector>

#include <bfft/bfft.hpp>

namespace cheb_dct_endpoint {

inline bool is_pow2(int m) { return m >= 4 && (m & (m - 1)) == 0; }

// ---------------------------------------------------------------------------
// DCT-I (REDFT00) via the real FFT of the symmetric extension.
// ---------------------------------------------------------------------------
class DCT1 {
public:
    static bool supported(int n) { return n >= 3 && is_pow2(2 * (n - 1)); }

    explicit DCT1(int n)
        : n_(n), m_(2 * (n - 1)), plan_(static_cast<std::size_t>(2 * (n - 1))) {
        ext_.assign(m_, 0.0);
        work_.assign(plan_.work_size(), 0.0);
        scratch_.assign(plan_.native_scratch_size(), bfft::complex{0.0, 0.0});
        spec_.assign(plan_.bins(), bfft::complex{0.0, 0.0});
    }

    int size() const { return n_; }

    void forward(const double* input, double* output) {
        // Symmetric extension: v[j]=X[j] (j=0..n-1), v[M-j]=X[j] (j=1..n-2).
        for (int j = 0; j < n_; ++j) ext_[j] = input[j];
        for (int j = 1; j < n_ - 1; ++j) ext_[m_ - j] = input[j];
        plan_.forward(ext_.data(), spec_.data(), work_.data(), scratch_.data());
        for (int k = 0; k < n_; ++k) output[k] = spec_[k].re;  // bins 0..M/2 = 0..n-1
    }

    // Self-inverse up to 2(n-1).
    void inverse(const double* input, double* output) {
        forward(input, output);
        const double s = 1.0 / (2.0 * (n_ - 1));
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }

private:
    int n_, m_;
    bfft::plan plan_;
    std::vector<double> ext_, work_;
    std::vector<bfft::complex> scratch_, spec_;
};

// ---------------------------------------------------------------------------
// DST-I (RODFT00) via the real FFT of the antisymmetric extension.
// ---------------------------------------------------------------------------
class DST1 {
public:
    static bool supported(int n) { return n >= 3 && is_pow2(2 * (n + 1)); }

    explicit DST1(int n)
        : n_(n), m_(2 * (n + 1)), plan_(static_cast<std::size_t>(2 * (n + 1))) {
        ext_.assign(m_, 0.0);
        work_.assign(plan_.work_size(), 0.0);
        scratch_.assign(plan_.native_scratch_size(), bfft::complex{0.0, 0.0});
        spec_.assign(plan_.bins(), bfft::complex{0.0, 0.0});
    }

    int size() const { return n_; }

    void forward(const double* input, double* output) {
        // Antisymmetric extension: v[j+1]=X[j], v[M-(j+1)]=-X[j], j=0..n-1.
        for (int j = 0; j < m_; ++j) ext_[j] = 0.0;
        for (int j = 0; j < n_; ++j) {
            ext_[j + 1] = input[j];
            ext_[m_ - (j + 1)] = -input[j];
        }
        plan_.forward(ext_.data(), spec_.data(), work_.data(), scratch_.data());
        for (int k = 0; k < n_; ++k) output[k] = -spec_[k + 1].im;  // bins 1..n
    }

    // Self-inverse up to 2(n+1).
    void inverse(const double* input, double* output) {
        forward(input, output);
        const double s = 1.0 / (2.0 * (n_ + 1));
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }

private:
    int n_, m_;
    bfft::plan plan_;
    std::vector<double> ext_, work_;
    std::vector<bfft::complex> scratch_, spec_;
};

}  // namespace cheb_dct_endpoint

#endif  // BFFT_DCT1_DST1_VIA_FFT_HPP
