#ifndef BFFT_BRUUN_DCT_HPP
#define BFFT_BRUUN_DCT_HPP

// Bruun-native, repack-free DCT-II / DCT-III (ROADMAP Phase: factorization).
//
// Lineage. This is the Bruun real-residue tree with the imaginary (sine /
// Chebyshev-U) half pruned -- a *real* decimation-in-frequency over the
// Chebyshev factorization, fed by the kernel's binomial fold. It is NOT a tower
// transform: it consumes flat Chebyshev coefficients (DCT-III) / samples
// (DCT-II) directly, so there is no flat->tower repack (the repack was an
// artifact of the experimental eval_monomial, not of Bruun), and it produces
// natural order (no bin permutation). Power of two, any N >= 2.
//
// DCT-III (synthesis / eval), the transpose's partner -- evaluate the flat
// Chebyshev series sum_j c_j T_j at the roots of T_n, recursively:
//   even coeffs c[0::2]  ->  half-size DCT-III      (T_{2i} = T_i o T_2)
//   odd  coeffs c[1::2]  ->  bidiagonal g, then half-size DCT-III, x 2 cos
//   butterfly: Y[k] = E[k] + 2 t_k O[k],  Y[n-1-k] = E[k] - 2 t_k O[k]
// with t_k = cos(pi(2k+1)/2n). The "2 cos" / bidiagonal is the U-companion the
// full complex Bruun node would carry; here it is handled with O(N) adds.
//
// DCT-II (analysis) is the exact transpose (DCT-II = 2 * DCT-III^T). Transposing
// turns the trailing butterfly into a *leading fold* -- the Bruun binomial fold
// of mirror-paired samples -- and the recursion runs the other way:
//   fold:  E'[k] = X[k] + X[n-1-k],  O'[k] = 2 t_k (X[k] - X[n-1-k])
//   recurse on E', O'  (each half-size DCT-II)
//   bidiagonal^T sweep on the O' result, then interleave back to coefficients.
// Both verified against the direct matrices and each other (transpose identity).
//
// This first cut is recursive with one plan-owned scratch buffer (no per-call
// heap allocation); radix-4 fusion and an iterative schedule are the follow-up
// performance pass, mirroring the FFT kernel.

#include <cmath>
#include <cstddef>
#include <cstdlib>

#if defined(__GNUC__) || defined(__clang__)
#define BFFT_BDCT_RESTRICT __restrict__
#else
#define BFFT_BDCT_RESTRICT
#endif

namespace bruun_dct {

constexpr double kPi = 3.141592653589793238462643383279502884;

inline double* aligned_alloc_doubles(std::size_t count) {
    if (count == 0) count = 1;
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(count * sizeof(double), 64);
#else
    if (posix_memalign(&p, 64, count * sizeof(double)) != 0) p = nullptr;
#endif
    return static_cast<double*>(p);
}
inline void aligned_free_doubles(double* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

inline bool is_pow2(int n) { return n >= 2 && (n & (n - 1)) == 0; }

class BruunDCT {
public:
    BruunDCT() = default;
    ~BruunDCT() {
        aligned_free_doubles(tw_);
        aligned_free_doubles(scratch_);
    }
    BruunDCT(const BruunDCT&) = delete;
    BruunDCT& operator=(const BruunDCT&) = delete;

    // n must be a power of two >= 2.
    bool init(int n) {
        if (!is_pow2(n)) return false;
        n_ = n;
        tw_ = aligned_alloc_doubles(n);       // 2*cos table, by recursion level
        scratch_ = aligned_alloc_doubles(n);
        if (!tw_ || !scratch_) return false;
        // For each recursion size s = n, n/2, ..., 2: s/2 twiddles 2 cos(pi(2k+1)/2s)
        // packed at base offset (n - s).
        for (int s = n; s >= 2; s >>= 1) {
            const int base = n - s;
            for (int k = 0; k < s / 2; ++k)
                tw_[base + k] = 2.0 * std::cos(kPi * (2.0 * k + 1.0) / (2.0 * s));
        }
        return true;
    }

    int size() const { return n_; }

    // Forward DCT-II (FFTW REDFT10): Y_k = 2 sum_j X_j cos(pi(2j+1)k/2n).
    // output must differ from input; no per-call heap allocation.
    void dct2_forward(const double* BFFT_BDCT_RESTRICT input,
                      double* BFFT_BDCT_RESTRICT output) const {
        for (int i = 0; i < n_; ++i) output[i] = input[i];
        analysis(output, 0, n_);
        for (int i = 0; i < n_; ++i) output[i] *= 2.0;  // DCT-II = 2 * recT
    }

    // Forward DCT-III (FFTW REDFT01): Y_k = X_0 + 2 sum_{j>=1} X_j cos(pi j(2k+1)/2n).
    void dct3_forward(const double* BFFT_BDCT_RESTRICT input,
                      double* BFFT_BDCT_RESTRICT output) const {
        output[0] = input[0];
        for (int j = 1; j < n_; ++j) output[j] = 2.0 * input[j];  // flat coeffs c
        synthesis(output, 0, n_);
    }

    // Inverses via the FFTW pair relations (REDFT01 . REDFT10 = 2n).
    void dct2_inverse(const double* BFFT_BDCT_RESTRICT input,
                      double* BFFT_BDCT_RESTRICT output) const {
        dct3_forward(input, output);
        const double s = 1.0 / (2.0 * n_);
        for (int i = 0; i < n_; ++i) output[i] *= s;
    }
    void dct3_inverse(const double* BFFT_BDCT_RESTRICT input,
                      double* BFFT_BDCT_RESTRICT output) const {
        dct2_forward(input, output);
        const double s = 1.0 / (2.0 * n_);
        for (int i = 0; i < n_; ++i) output[i] *= s;
    }

private:
    int n_ = 0;
    double* tw_ = nullptr;        // 2*cos twiddles, packed by recursion size
    double* scratch_ = nullptr;   // n_ doubles, reused depth-first

    const double* tw_for(int s) const { return tw_ + (n_ - s); }

    // DCT-III synthesis: in place on p[off, off+s) (flat coeffs -> values).
    void synthesis(double* BFFT_BDCT_RESTRICT p, int off, int s) const {
        if (s == 1) return;
        const int m = s / 2;
        double* BFFT_BDCT_RESTRICT c = p + off;
        double* BFFT_BDCT_RESTRICT sc = scratch_;

        // De-interleave: even = c[0::2] -> [0,m), odd = c[1::2] -> [m,s).
        for (int i = 0; i < m; ++i) { sc[i] = c[2 * i]; sc[m + i] = c[2 * i + 1]; }
        for (int i = 0; i < s; ++i) c[i] = sc[i];

        // Bidiagonal solve on the odd half -> g (backward sweep, in place).
        double* BFFT_BDCT_RESTRICT od = c + m;     // g[m-1]=od[m-1] kept
        for (int j = m - 2; j >= 1; --j) od[j] = od[j] - od[j + 1];
        if (m > 1) od[0] = 0.5 * (od[0] - od[1]);
        else od[0] = 0.5 * od[0];                  // m==1: g0 = odd0/2

        synthesis(p, off, m);        // E = DCT-III(even)
        synthesis(p, off + m, m);    // O = DCT-III(g)

        // Butterfly with 2 t_k: Y[k]=E+2t O, Y[s-1-k]=E-2t O.
        const double* BFFT_BDCT_RESTRICT tw = tw_for(s);
        for (int k = 0; k < m; ++k) {
            const double pterm = tw[k] * c[m + k];   // 2 t_k * O[k]
            sc[k] = c[k] + pterm;
            sc[s - 1 - k] = c[k] - pterm;
        }
        for (int i = 0; i < s; ++i) c[i] = sc[i];
    }

    // DCT-II analysis: in place on p[off, off+s). Transpose of synthesis.
    void analysis(double* BFFT_BDCT_RESTRICT p, int off, int s) const {
        if (s == 1) return;
        const int m = s / 2;
        double* BFFT_BDCT_RESTRICT c = p + off;
        double* BFFT_BDCT_RESTRICT sc = scratch_;

        // Fold (butterfly^T): E'[k]=X[k]+X[s-1-k], O'[k]=2 t_k (X[k]-X[s-1-k]).
        const double* BFFT_BDCT_RESTRICT tw = tw_for(s);
        for (int k = 0; k < m; ++k) {
            const double a = c[k], b = c[s - 1 - k];
            sc[k] = a + b;
            sc[m + k] = tw[k] * (a - b);
        }
        for (int i = 0; i < s; ++i) c[i] = sc[i];

        analysis(p, off, m);         // recurse on E'
        analysis(p, off + m, m);     // recurse on O'

        // Bidiagonal^T sweep on the O' result (forward, in place):
        //   odd'[0] = 0.5 g'[0];  odd'[i] = g'[i] - odd'[i-1].
        double* BFFT_BDCT_RESTRICT oh = c + m;
        oh[0] = 0.5 * oh[0];
        for (int i = 1; i < m; ++i) oh[i] = oh[i] - oh[i - 1];

        // Interleave (Split^T): coeff[2i]=Ehat[i], coeff[2i+1]=odd'[i].
        for (int i = 0; i < m; ++i) { sc[2 * i] = c[i]; sc[2 * i + 1] = c[m + i]; }
        for (int i = 0; i < s; ++i) c[i] = sc[i];
    }
};

}  // namespace bruun_dct

#endif  // BFFT_BRUUN_DCT_HPP
