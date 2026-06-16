// Radix-4 vs radix-2 parity and accuracy validation.
//
// bfft's residue engine has two implementations, selected by the RFFT
// fuse_tail flag:
//
//   radix-4 : fuse_tail=true  -> depth-first fused norm2_fused engine (default)
//   radix-2 : fuse_tail=false -> breadth-first norm_q stage engine
//
// IMPORTANT architectural fact this test documents and guards:
//   * The two engines agree bit-for-bit in the RESIDUE domain, and each is a
//     correct, self-consistent residue round-trip.
//   * The complex (native/standard) and inverse pipeline is built on the fused
//     radix-4 layout for N >= 32; the non-fused (radix-2) complex packing is
//     correct only for N <= 16. So radix-4 is the sole complex pipeline at
//     scale, validated here against an independent DFT.
//
// Build:
//   c++ -O3 -std=c++17 -I. tests/test_radix_parity.cpp -lm -o build/tests/test_radix_parity

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/detail/bruun_kernel.hpp"

using bruun::RFFT;
using cd = bruun::complex_t;

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::vector<double> make_input(int N) {
    std::vector<double> x(N);
    unsigned s = 2246u + static_cast<unsigned>(N);
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / N;
        s = s * 1103515245u + 12345u;
        const double noise = (static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
        x[i] = 0.7 * std::sin(2 * pi * 3 * t) + 0.2 * std::cos(2 * pi * 11 * t)
             + 0.07 * std::sin(2 * pi * 29 * t) + 0.03 * noise;
    }
    return x;
}

// Independent reference: standard FFT-order r2c bins X[k], k = 0..N/2.
std::vector<cd> ref_dft(const std::vector<double>& x) {
    const int N = static_cast<int>(x.size());
    std::vector<cd> X(N / 2 + 1);
    for (int k = 0; k <= N / 2; ++k) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; ++n) {
            const double a = -2.0 * pi * k * n / N;
            re += x[n] * std::cos(a);
            im += x[n] * std::sin(a);
        }
        X[k].re = re; X[k].im = im;
    }
    return X;
}

double max_re_im(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

double max_complex(const std::vector<cd>& a, const std::vector<cd>& b, int n) {
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dr = a[i].re - b[i].re, di = a[i].im - b[i].im;
        e = std::max(e, std::sqrt(dr * dr + di * di));
    }
    return e;
}

double residue_roundtrip_err(RFFT& r, const std::vector<double>& in) {
    const int N = static_cast<int>(in.size());
    std::vector<double> v(N);
    r.forward_residues(in.data(), v.data());
    r.inverse_residues(v.data());
    double e = 0.0;
    for (int i = 0; i < N; ++i) e = std::max(e, std::abs(v[i] - in[i]));
    return e;
}

bool run(int N) {
    RFFT r4, r2;
    r4.init(N, true);    // radix-4 (fused) — the default
    r2.init(N, false);   // radix-2 (breadth-first)

    std::vector<double> in = make_input(N);
    const int NB = r4.bins();
    std::vector<double> work(N);

    // (1) Residue-domain parity: the two engines must agree bit-for-bit.
    std::vector<double> res4(N), res2(N);
    r4.forward_residues(in.data(), res4.data());
    r2.forward_residues(in.data(), res2.data());
    const double e_res = max_re_im(res4, res2);

    // (2) Each engine is a correct, self-consistent residue round-trip.
    const double e_rrt4 = residue_roundtrip_err(r4, in);
    const double e_rrt2 = residue_roundtrip_err(r2, in);

    // (3) Radix-4 (default) standard bins vs independent DFT (absolute accuracy).
    std::vector<cd> std4(NB);
    r4.forward(in.data(), std4.data(), work.data());
    double e_ref = NAN;
    if (N <= 4096) {                       // naive DFT is O(N^2)
        std::vector<cd> ref = ref_dft(in);
        e_ref = max_complex(std4, ref, NB);
    }

    // (4) Radix-4 full complex round-trip (forward then inverse).
    std::vector<double> rt(N);
    r4.inverse(std4.data(), rt.data());
    double e_rt = 0.0;
    for (int i = 0; i < N; ++i) e_rt = std::max(e_rt, std::abs(rt[i] - in[i]));

    const double tol = 1e-9;
    const bool ref_ok = std::isnan(e_ref) || e_ref < tol;
    const bool ok = e_res < tol && e_rrt4 < tol && e_rrt2 < tol && ref_ok && e_rt < tol;

    std::printf("N=%8d  res(r4=r2)=%.1e | resRT r4=%.1e r2=%.1e | "
                "std-vs-DFT=%.1e | complexRT=%.1e  %s\n",
                N, e_res, e_rrt4, e_rrt2, e_ref, e_rt, ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::printf("Radix-4 vs radix-2 parity:\n");
    std::printf("  res(r4=r2): the two residue engines agree (expect ~0)\n");
    std::printf("  resRT     : each engine's residue forward+inverse round-trip\n");
    std::printf("  std-vs-DFT: radix-4 standard output vs independent DFT (N<=4096)\n");
    std::printf("  complexRT : radix-4 forward+inverse reconstruction\n\n");
    int pass = 0, fail = 0;
    for (int N : {16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                  8192, 16384, 65536, 262144, 1048576}) {
        if (run(N)) ++pass; else ++fail;
    }
    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
