// test_dit.cpp — verification harness for bruun_DIT_kernel.hpp
#include "../src/detail/bruun_DIT_kernel.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using bruun_dit::RFFT_DIT;
using bruun_dit::RFFT_DIT_F32;
using bruun_dit::complex_t;
using bruun_dit::complex_f32_t;

static void naive_rfft(const std::vector<double>& x, std::vector<complex_t>& X) {
    const int n = (int)x.size();
    for (int k = 0; k <= n / 2; ++k) {
        double re = 0.0, im = 0.0;
        for (int t = 0; t < n; ++t) {
            const double a = -2.0 * M_PI * (double)k * (double)t / (double)n;
            re += x[t] * std::cos(a);
            im += x[t] * std::sin(a);
        }
        X[k].re = re; X[k].im = im;
    }
}

int main() {
    std::printf("backend: %s\n", bruun_dit::simd_backend_name());
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    int fails = 0;

    for (int N = 4; N <= 65536; N <<= 1) {
        RFFT_DIT plan(N);
        std::vector<double> x(N), work(plan.work_size()), back(N);
        std::vector<complex_t> X(plan.bins());
        for (auto& v : x) v = nd(rng);

        plan.forward(x.data(), X.data(), work.data());
        std::vector<double> xbr(N);
        const int half = N / 2;
        const int bits = bruun_dit::ilog2_pow2(half);
        for (int b = 0; b < half; ++b) {
            const int j = bruun_dit::bitrev_int(b, bits);
            xbr[b] = x[j];
            xbr[half + b] = x[half + j];
        }
        std::vector<complex_t> Xbr(plan.bins());
        plan.forward_bitreversed(xbr.data(), Xbr.data(), work.data());
        std::vector<double> xprepared(N);
        for (int b = 0; b < half; ++b) {
            const int j = bruun_dit::bitrev_int(b, bits);
            const double a = x[j];
            const double c = x[half + j];
            xprepared[2 * b] = a + c;
            xprepared[2 * b + 1] = a - c;
        }
        std::vector<complex_t> Xprepared(plan.bins());
        plan.forward_pregathered(xprepared.data(), Xprepared.data(), work.data());
        std::vector<complex_t> Xtiled(plan.bins());
        plan.forward_tiled(x.data(), Xtiled.data(), work.data());

        double fwd_err = -1.0;
        double br_err = 0.0;
        double prepared_err = 0.0;
        double tiled_err = 0.0;
        for (int k = 0; k <= N / 2; ++k) {
            br_err = std::max(br_err, std::abs(Xbr[k].re - X[k].re));
            br_err = std::max(br_err, std::abs(Xbr[k].im - X[k].im));
            prepared_err = std::max(prepared_err, std::abs(Xprepared[k].re - X[k].re));
            prepared_err = std::max(prepared_err, std::abs(Xprepared[k].im - X[k].im));
            tiled_err = std::max(tiled_err, std::abs(Xtiled[k].re - X[k].re));
            tiled_err = std::max(tiled_err, std::abs(Xtiled[k].im - X[k].im));
        }
        if (N <= 4096) {
            std::vector<complex_t> ref(plan.bins());
            naive_rfft(x, ref);
            fwd_err = 0.0;
            for (int k = 0; k <= N / 2; ++k) {
                fwd_err = std::max(fwd_err, std::abs(X[k].re - ref[k].re));
                fwd_err = std::max(fwd_err, std::abs(X[k].im - ref[k].im));
            }
        }

        plan.inverse(X.data(), back.data(), work.data());
        double rt = 0.0;
        for (int i = 0; i < N; ++i) rt = std::max(rt, std::abs(back[i] - x[i]));

        // halfcomplex round trip independently
        std::vector<double> hc(plan.halfcomplex_size()), back2(N);
        plan.forward_halfcomplex(x.data(), hc.data());
        plan.inverse_halfcomplex(hc.data(), back2.data());
        double rt2 = 0.0;
        for (int i = 0; i < N; ++i) rt2 = std::max(rt2, std::abs(back2[i] - x[i]));

        const double ftol = 1e-9 * std::sqrt((double)N);
        const double rtol = 1e-13 * std::log2((double)N);
        const bool ok = (fwd_err < 0 || fwd_err < ftol) && br_err < ftol && prepared_err < ftol && tiled_err < ftol && rt < rtol && rt2 < rtol;
        if (!ok) ++fails;
        std::printf("N=%7d  fwd_vs_naive %s  br %.3e  prep %.3e  tiled %.3e  roundtrip %.3e  hc_roundtrip %.3e  %s\n",
                    N, fwd_err < 0 ? "   skipped " :
                    ([](double e){ static char b[32]; std::snprintf(b, 32, "%.3e", e); return (const char*)b; })(fwd_err),
                    br_err, prepared_err, tiled_err, rt, rt2, ok ? "ok" : "FAIL");

        RFFT_DIT_F32 planf(N);
        std::vector<float> xf(N), xbrf(N), xprepf(N), workf(planf.work_size());
        for (int i = 0; i < N; ++i) xf[i] = static_cast<float>(x[i]);
        for (int b = 0; b < half; ++b) {
            const int j = bruun_dit::bitrev_int(b, bits);
            xbrf[b] = xf[j];
            xbrf[half + b] = xf[half + j];
            const float a = xf[j];
            const float c = xf[half + j];
            xprepf[2 * b] = a + c;
            xprepf[2 * b + 1] = a - c;
        }

        std::vector<complex_f32_t> Xf(planf.bins());
        std::vector<complex_f32_t> Xftiled(planf.bins());
        std::vector<complex_f32_t> Xfbr(planf.bins());
        std::vector<complex_f32_t> Xfprep(planf.bins());
        planf.forward(xf.data(), Xf.data(), workf.data());
        planf.forward_tiled(xf.data(), Xftiled.data(), workf.data());
        planf.forward_bitreversed(xbrf.data(), Xfbr.data(), workf.data());
        planf.forward_pregathered(xprepf.data(), Xfprep.data(), workf.data());

        double f32_tiled_err = 0.0;
        double f32_br_err = 0.0;
        double f32_prep_err = 0.0;
        double f32_vs_double = 0.0;
        for (int k = 0; k <= N / 2; ++k) {
            f32_tiled_err = std::max(f32_tiled_err, std::abs(static_cast<double>(Xftiled[k].re - Xf[k].re)));
            f32_tiled_err = std::max(f32_tiled_err, std::abs(static_cast<double>(Xftiled[k].im - Xf[k].im)));
            f32_br_err = std::max(f32_br_err, std::abs(static_cast<double>(Xfbr[k].re - Xf[k].re)));
            f32_br_err = std::max(f32_br_err, std::abs(static_cast<double>(Xfbr[k].im - Xf[k].im)));
            f32_prep_err = std::max(f32_prep_err, std::abs(static_cast<double>(Xfprep[k].re - Xf[k].re)));
            f32_prep_err = std::max(f32_prep_err, std::abs(static_cast<double>(Xfprep[k].im - Xf[k].im)));
            f32_vs_double = std::max(f32_vs_double, std::abs(static_cast<double>(Xf[k].re) - X[k].re));
            f32_vs_double = std::max(f32_vs_double, std::abs(static_cast<double>(Xf[k].im) - X[k].im));
        }

        const double f32_path_tol = 1e-5 * std::sqrt((double)N);
        const double f32_ref_tol = 2e-4 * std::sqrt((double)N);
        const bool f32_ok = f32_tiled_err < f32_path_tol
                         && f32_br_err < f32_path_tol
                         && f32_prep_err < f32_path_tol
                         && f32_vs_double < f32_ref_tol;
        if (!f32_ok) ++fails;
        std::printf("          f32_vs_f64 %.3e  f32_br %.3e  f32_prep %.3e  f32_tiled %.3e  %s\n",
                    f32_vs_double, f32_br_err, f32_prep_err, f32_tiled_err, f32_ok ? "ok" : "FAIL");
    }

    // circular convolution via pointwise halfcomplex multiply vs O(N^2) reference
    {
        const int N = 1024;
        RFFT_DIT plan(N);
        std::vector<double> x(N), h(N), ref(N, 0.0);
        for (auto& v : x) v = nd(rng);
        for (auto& v : h) v = nd(rng);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                ref[(i + j) % N] += x[i] * h[j];
        std::vector<double> hx(N), hh(N), out(N);
        plan.forward_halfcomplex(x.data(), hx.data());
        plan.forward_halfcomplex(h.data(), hh.data());
        RFFT_DIT::pointwise_multiply(hx.data(), hh.data(), N);
        plan.inverse_halfcomplex(hx.data(), out.data());
        double e = 0.0;
        for (int i = 0; i < N; ++i) e = std::max(e, std::abs(out[i] - ref[i]));
        const bool ok = e < 1e-9;
        if (!ok) ++fails;
        std::printf("convolution N=%d  max err %.3e  %s\n", N, e, ok ? "ok" : "FAIL");
    }

    // large-N round trip
    {
        const int N = 1 << 20;
        RFFT_DIT plan(N);
        std::vector<double> x(N), work(N), back(N);
        std::vector<complex_t> X(plan.bins());
        for (auto& v : x) v = nd(rng);
        plan.forward(x.data(), X.data(), work.data());
        plan.inverse(X.data(), back.data(), work.data());
        double rt = 0.0;
        for (int i = 0; i < N; ++i) rt = std::max(rt, std::abs(back[i] - x[i]));
        const bool ok = rt < 1e-12;
        if (!ok) ++fails;
        std::printf("roundtrip N=%d  max err %.3e  %s\n", N, rt, ok ? "ok" : "FAIL");
    }

    std::printf(fails ? "FAILURES: %d\n" : "all tests passed\n", fails);
    return fails ? 1 : 0;
}
