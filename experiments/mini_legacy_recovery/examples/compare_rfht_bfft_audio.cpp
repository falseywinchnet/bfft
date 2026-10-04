// compare_rfht_bfft_audio.cpp
//
// Side-by-side benchmark for Youssef's rfht_polar.h and bfft.
//
// Measures:
//   1. forward polar speed: real -> mag,phase
//   2. polar round-trip speed/error: real -> mag,phase -> real
//   3. rfht phase/magnitude difference against bfft forward_mag_phase
//
// Expected layout from repo root:
//   ./rfht_polar.h
//   ./include/bfft/bfft.hpp
//   ./build/libbfft.a
//
// Build on x86_64 AVX2/FMA:
//   c++ -O3 -DNDEBUG -std=c++17 -I. -Iinclude -mavx2 -mfma \
//     compare_rfht_bfft_audio.cpp build/libbfft.a -lm \
//     -o build/examples/compare_rfht_bfft_audio
//
// Run:
//   ./build/examples/compare_rfht_bfft_audio [N=2048] [frames=1024]
//
// Note:
//   The pasted rfht_polar.h uses <immintrin.h> and AVX2 intrinsics.
//   It will not compile as-is for native Apple Silicon/NEON.
//   Use an x86_64 AVX2 machine or an x86_64/Rosetta toolchain.

#include <bfft/bfft.hpp>

extern "C" {
#include "rfht_polar.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double tau = 2.0 * pi;

struct err_stats {
    double mse = 0.0;
    double signal_power = 0.0;
    double snr_db = 0.0;
    double max_abs = 0.0;
};

struct phase_stats {
    double max_phase_abs = 0.0;
    double rms_phase = 0.0;
    double max_mag_abs = 0.0;
    double max_mag_rel = 0.0;
};

struct timed_err {
    double seconds = 0.0;
    double msamples_per_s = 0.0;
    err_stats err;
    double sink = 0.0;
};

struct timed_forward {
    double seconds = 0.0;
    double msamples_per_s = 0.0;
    double sink = 0.0;
};

std::size_t parse_size(const char* s, std::size_t fallback) {
    if (!s) return fallback;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || v == 0) {
        throw std::runtime_error("bad unsigned integer argument");
    }
    return static_cast<std::size_t>(v);
}

bool is_power2(std::size_t n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

double wrap_phase_diff(double a, double b) {
    double d = a - b;
    d = std::fmod(d + pi, tau);
    if (d < 0.0) d += tau;
    return d - pi;
}

std::vector<double> make_audio_frames(std::size_t n, std::size_t frames) {
    std::vector<double> x(n * frames);
    std::mt19937_64 rng(20260626);

    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    std::uniform_real_distribution<double> phase_dist(0.0, tau);
    std::uniform_real_distribution<double> amp_jitter(0.85, 1.15);

    const double fs = 48000.0;
    const double freqs[] = {
        55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0
    };
    const double amps[] = {
        0.45, 0.30, 0.20, 0.12, 0.08, 0.05, 0.025, 0.012
    };

    std::vector<double> phases(sizeof(freqs) / sizeof(freqs[0]));
    for (double& p : phases) p = phase_dist(rng);

    for (std::size_t f = 0; f < frames; ++f) {
        const double frame_gain = amp_jitter(rng);
        for (std::size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(f * n + i) / fs;
            double v = 0.0;

            for (std::size_t k = 0; k < sizeof(freqs) / sizeof(freqs[0]); ++k) {
                v += amps[k] * std::sin(tau * freqs[k] * t + phases[k]);
            }

            v += 1e-4 * noise(rng);
            x[f * n + i] = 0.75 * frame_gain * v;
        }
    }

    return x;
}

err_stats compute_err(const std::vector<double>& ref, const std::vector<double>& got) {
    if (ref.size() != got.size()) throw std::runtime_error("size mismatch");

    long double sum_e2 = 0.0L;
    long double sum_x2 = 0.0L;
    double max_abs = 0.0;

    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double e = got[i] - ref[i];
        sum_e2 += static_cast<long double>(e) * e;
        sum_x2 += static_cast<long double>(ref[i]) * ref[i];
        max_abs = std::max(max_abs, std::abs(e));
    }

    err_stats out;
    out.mse = static_cast<double>(sum_e2 / static_cast<long double>(ref.size()));
    out.signal_power = static_cast<double>(sum_x2 / static_cast<long double>(ref.size()));
    out.max_abs = max_abs;
    out.snr_db = out.mse > 0.0 ? 10.0 * std::log10(out.signal_power / out.mse) : INFINITY;
    return out;
}

phase_stats compare_forward_to_bfft(const std::vector<double>& bmag,
                                    const std::vector<double>& bph,
                                    const std::vector<double>& rmag,
                                    const std::vector<double>& rph) {
    if (bmag.size() != rmag.size() || bph.size() != rph.size() || bmag.size() != bph.size()) {
        throw std::runtime_error("spectrum size mismatch");
    }

    long double sum_phase2 = 0.0L;
    double max_phase_abs = 0.0;
    double max_mag_abs = 0.0;
    double max_mag_rel = 0.0;

    for (std::size_t k = 0; k < bmag.size(); ++k) {
        const double dp = wrap_phase_diff(rph[k], bph[k]);
        const double abs_dp = std::abs(dp);
        max_phase_abs = std::max(max_phase_abs, abs_dp);
        sum_phase2 += static_cast<long double>(dp) * dp;

        const double dm = rmag[k] - bmag[k];
        max_mag_abs = std::max(max_mag_abs, std::abs(dm));
        if (bmag[k] > 0.0) {
            max_mag_rel = std::max(max_mag_rel, std::abs(dm) / bmag[k]);
        }
    }

    phase_stats out;
    out.max_phase_abs = max_phase_abs;
    out.rms_phase = std::sqrt(static_cast<double>(sum_phase2 / static_cast<long double>(bmag.size())));
    out.max_mag_abs = max_mag_abs;
    out.max_mag_rel = max_mag_rel;
    return out;
}

timed_forward bench_bfft_forward(const bfft::plan& plan,
                                 const std::vector<double>& input,
                                 std::size_t n,
                                 std::size_t frames,
                                 std::vector<bfft::complex>* last = nullptr) {
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> mp(plan.bins());
    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t f = 0; f < frames; ++f) {
        const double* in = input.data() + f * n;
        plan.forward_mag_phase(in, mp.data(), work.data());
        sink += mp[(f * 17) % mp.size()].re + 0.125 * mp[(f * 19) % mp.size()].im;
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (last) *last = mp;

    timed_forward out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.msamples_per_s = static_cast<double>(n * frames) / out.seconds / 1.0e6;
    out.sink = sink;
    return out;
}

timed_forward bench_rfht_forward(rfht_plan* rp,
                                 const std::vector<double>& input,
                                 std::size_t n,
                                 std::size_t frames,
                                 std::vector<double>* last_mag = nullptr,
                                 std::vector<double>* last_phase = nullptr) {
    std::vector<double> H(n);
    std::vector<double> mag(n / 2 + 1);
    std::vector<double> phase(n / 2 + 1);
    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t f = 0; f < frames; ++f) {
        const double* in = input.data() + f * n;
        rfht_polar(rp, in, H.data(), mag.data(), phase.data());
        sink += mag[(f * 17) % mag.size()] + 0.125 * phase[(f * 19) % phase.size()];
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (last_mag) *last_mag = mag;
    if (last_phase) *last_phase = phase;

    timed_forward out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.msamples_per_s = static_cast<double>(n * frames) / out.seconds / 1.0e6;
    out.sink = sink;
    return out;
}

timed_err bench_bfft_cycle(const bfft::plan& plan,
                           const std::vector<double>& input,
                           std::size_t n,
                           std::size_t frames) {
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> mp(plan.bins());
    std::vector<double> frame_out(n);
    std::vector<double> recon(input.size());
    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t f = 0; f < frames; ++f) {
        const double* in = input.data() + f * n;
        plan.forward_mag_phase(in, mp.data(), work.data());
        plan.inverse_mag_phase(mp.data(), frame_out.data());
        std::copy(frame_out.begin(), frame_out.end(), recon.begin() + f * n);
        sink += frame_out[(f * 17) & (n - 1)];
    }
    const auto t1 = std::chrono::steady_clock::now();

    timed_err out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.msamples_per_s = static_cast<double>(n * frames) / out.seconds / 1.0e6;
    out.err = compute_err(input, recon);
    out.sink = sink;
    return out;
}

timed_err bench_rfht_cycle(rfht_plan* rp,
                           const std::vector<double>& input,
                           std::size_t n,
                           std::size_t frames) {
    std::vector<double> H(n);
    std::vector<double> mag(n / 2 + 1);
    std::vector<double> phase(n / 2 + 1);
    std::vector<double> frame_out(n);
    std::vector<double> recon(input.size());
    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t f = 0; f < frames; ++f) {
        const double* in = input.data() + f * n;
        rfht_polar(rp, in, H.data(), mag.data(), phase.data());
        rfht_ipolar(rp, mag.data(), phase.data(), H.data(), frame_out.data());
        std::copy(frame_out.begin(), frame_out.end(), recon.begin() + f * n);
        sink += frame_out[(f * 17) & (n - 1)];
    }
    const auto t1 = std::chrono::steady_clock::now();

    timed_err out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.msamples_per_s = static_cast<double>(n * frames) / out.seconds / 1.0e6;
    out.err = compute_err(input, recon);
    out.sink = sink;
    return out;
}

void print_forward(const char* name, const timed_forward& r) {
    std::printf("%-24s time %.6f s  %.3f Msamples/s  sink %.17g\n",
                name,
                r.seconds,
                r.msamples_per_s,
                r.sink);
}

void print_cycle(const char* name, const timed_err& r) {
    std::printf("%-24s time %.6f s  %.3f Msamples/s  MSE %.17g  SNR %.3f dB  max_abs %.17g  sink %.17g\n",
                name,
                r.seconds,
                r.msamples_per_s,
                r.err.mse,
                r.err.snr_db,
                r.err.max_abs,
                r.sink);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t n = 2048;
        std::size_t frames = 1024;

        if (argc > 1) n = parse_size(argv[1], n);
        if (argc > 2) frames = parse_size(argv[2], frames);

        if (!is_power2(n) || n < 8) {
            throw std::runtime_error("N must be a power of two >= 8");
        }
        if (n > static_cast<std::size_t>(INT32_MAX)) {
            throw std::runtime_error("rfht_plan uses int N; N too large");
        }

        const std::vector<double> input = make_audio_frames(n, frames);

        bfft::plan bp(n);
        rfht_plan* rp = rfht_plan_create(static_cast<int>(n));
        if (!rp) throw std::runtime_error("rfht_plan_create failed");

        std::printf("RFHT vs BFFT audio polar benchmark\n");
        std::printf("N=%zu frames=%zu total_samples=%zu\n", n, frames, n * frames);
        std::printf("bfft backend=%s version=%s\n", bfft::backend_name().c_str(), bfft::version_string().c_str());
        std::printf("\nforward only: real -> mag,phase\n");

        std::vector<bfft::complex> b_last;
        std::vector<double> r_last_mag;
        std::vector<double> r_last_phase;

        const timed_forward bfw = bench_bfft_forward(bp, input, n, frames, &b_last);
        const timed_forward rfw = bench_rfht_forward(rp, input, n, frames, &r_last_mag, &r_last_phase);

        print_forward("bfft forward polar", bfw);
        print_forward("rfht forward polar", rfw);

        std::vector<double> b_last_mag(bp.bins());
        std::vector<double> b_last_phase(bp.bins());
        for (std::size_t k = 0; k < bp.bins(); ++k) {
            b_last_mag[k] = b_last[k].re;
            b_last_phase[k] = b_last[k].im;
        }

        const phase_stats ps = compare_forward_to_bfft(b_last_mag, b_last_phase, r_last_mag, r_last_phase);
        std::printf("\nrfht forward compared to bfft on last frame:\n");
        std::printf("max_phase_abs %.17g rad  rms_phase %.17g rad  max_mag_abs %.17g  max_mag_rel %.17g\n",
                    ps.max_phase_abs,
                    ps.rms_phase,
                    ps.max_mag_abs,
                    ps.max_mag_rel);

        std::printf("\nround trip: real -> mag,phase -> real\n");
        const timed_err bcy = bench_bfft_cycle(bp, input, n, frames);
        const timed_err rcy = bench_rfht_cycle(rp, input, n, frames);

        print_cycle("bfft polar cycle", bcy);
        print_cycle("rfht polar cycle", rcy);

        std::printf("\nratios:\n");
        std::printf("forward speed rfht/bfft: %.17g\n", rfw.msamples_per_s / bfw.msamples_per_s);
        std::printf("cycle speed rfht/bfft:   %.17g\n", rcy.msamples_per_s / bcy.msamples_per_s);
        std::printf("cycle MSE rfht/bfft:     %.17g\n", rcy.err.mse / bcy.err.mse);

        rfht_plan_destroy(rp);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "usage: compare_rfht_bfft_audio [N=2048] [frames=1024]\n");
        return 2;
    }
}
