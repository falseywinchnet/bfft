// compare_rfht_neon_bfft_audio.cpp
//
// Compare Youssef's Apple Silicon RFHT polar path against BFFT on the same
// NEON machine.
//
// From bfft repo root:
//   cp "/mnt/data/rfht_neon (2).h" rfht_neon.h
//   cp /mnt/data/compare_rfht_neon_bfft_audio.cpp examples/compare_rfht_neon_bfft_audio.cpp
//
// Fast RFHT path, default rfht_neon.h behavior:
//   clang++ -O3 -DNDEBUG -std=c++17 -mcpu=native -I. -Iinclude \
//     examples/compare_rfht_neon_bfft_audio.cpp build/libbfft.a -lm \
//     -o build/examples/compare_rfht_neon_bfft_audio_fast
//
// Accurate RFHT path, using libm atan2/sin/cos inside RFHT:
//   clang++ -O3 -DNDEBUG -std=c++17 -mcpu=native -DRFHT_ACCURATE -I. -Iinclude \
//     examples/compare_rfht_neon_bfft_audio.cpp build/libbfft.a -lm \
//     -o build/examples/compare_rfht_neon_bfft_audio_accurate
//
// Run:
//   ./build/examples/compare_rfht_neon_bfft_audio_fast 2048 4096
//   ./build/examples/compare_rfht_neon_bfft_audio_accurate 2048 4096

#include <bfft/bfft.hpp>

#include "rfht_neon.h"

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

double dbc_from_abs(double x) {
    if (x == 0.0) return -INFINITY;
    return 20.0 * std::log10(x);
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

timed_forward bench_bfft_forward(const bfft::plan& plan,
                                 const std::vector<double>& input,
                                 std::size_t n,
                                 std::size_t frames) {
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> mp(plan.bins());

    double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (std::size_t f = 0; f < frames; ++f) {
        plan.forward_mag_phase(input.data() + f * n, mp.data(), work.data());
        sink += mp[(f * 17) % mp.size()].re + 0.125 * mp[(f * 19) % mp.size()].im;
    }

    const auto t1 = std::chrono::steady_clock::now();

    timed_forward out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.msamples_per_s = static_cast<double>(n * frames) / out.seconds / 1.0e6;
    out.sink = sink;
    return out;
}

timed_forward bench_rfht_forward(rfht_plan* plan,
                                 const std::vector<double>& input,
                                 std::size_t n,
                                 std::size_t frames) {
    std::vector<double> H(n);
    std::vector<double> mag(n / 2 + 1);
    std::vector<double> phase(n / 2 + 1);

    double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (std::size_t f = 0; f < frames; ++f) {
        rfht_polar(plan, input.data() + f * n, H.data(), mag.data(), phase.data());
        sink += mag[(f * 17) % mag.size()] + 0.125 * phase[(f * 19) % phase.size()];
    }

    const auto t1 = std::chrono::steady_clock::now();

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
        plan.forward_mag_phase(input.data() + f * n, mp.data(), work.data());
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

timed_err bench_rfht_cycle(rfht_plan* plan,
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
        rfht_polar(plan, input.data() + f * n, H.data(), mag.data(), phase.data());
        rfht_ipolar(plan, mag.data(), phase.data(), H.data(), frame_out.data());
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

phase_stats compare_rfht_to_bfft_all(const bfft::plan& bplan,
                                      rfht_plan* rplan,
                                      const std::vector<double>& input,
                                      std::size_t n,
                                      std::size_t frames) {
    std::vector<double> bwork(bplan.work_size());
    std::vector<bfft::complex> bmp(bplan.bins());

    std::vector<double> H(n);
    std::vector<double> rmag(n / 2 + 1);
    std::vector<double> rphase(n / 2 + 1);

    long double sum_phase2 = 0.0L;
    std::size_t count = 0;
    double max_phase_abs = 0.0;
    double max_mag_abs = 0.0;
    double max_mag_rel = 0.0;

    for (std::size_t f = 0; f < frames; ++f) {
        const double* x = input.data() + f * n;

        bplan.forward_mag_phase(x, bmp.data(), bwork.data());
        rfht_polar(rplan, x, H.data(), rmag.data(), rphase.data());

        for (std::size_t k = 0; k < bmp.size(); ++k) {
            const double dp = wrap_phase_diff(rphase[k], bmp[k].im);
            const double abs_dp = std::abs(dp);
            max_phase_abs = std::max(max_phase_abs, abs_dp);
            sum_phase2 += static_cast<long double>(dp) * dp;

            const double dm = rmag[k] - bmp[k].re;
            max_mag_abs = std::max(max_mag_abs, std::abs(dm));
            if (bmp[k].re > 0.0) {
                max_mag_rel = std::max(max_mag_rel, std::abs(dm) / bmp[k].re);
            }
            ++count;
        }
    }

    phase_stats out;
    out.max_phase_abs = max_phase_abs;
    out.rms_phase = std::sqrt(static_cast<double>(sum_phase2 / static_cast<long double>(count)));
    out.max_mag_abs = max_mag_abs;
    out.max_mag_rel = max_mag_rel;
    return out;
}

void print_forward(const char* name, const timed_forward& r) {
    std::printf("%-24s time %.6f s  %.3f Msamples/s  sink %.17g\n",
                name, r.seconds, r.msamples_per_s, r.sink);
}

void print_cycle(const char* name, const timed_err& r) {
    std::printf("%-24s time %.6f s  %.3f Msamples/s  MSE %.17g  SNR %.3f dB  max_abs %.17g  sink %.17g\n",
                name, r.seconds, r.msamples_per_s, r.err.mse, r.err.snr_db, r.err.max_abs, r.sink);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t n = 2048;
        std::size_t frames = 4096;

        if (argc > 1) n = parse_size(argv[1], n);
        if (argc > 2) frames = parse_size(argv[2], frames);

        if (!is_power2(n) || n < 8) {
            throw std::runtime_error("N must be a power of two >= 8");
        }
        if (n > static_cast<std::size_t>(INT32_MAX)) {
            throw std::runtime_error("rfht_plan uses int N; N too large");
        }

        const std::vector<double> input = make_audio_frames(n, frames);

        bfft::plan bplan(n);
        rfht_plan* rplan = rfht_plan_create(static_cast<int>(n));
        if (!rplan) throw std::runtime_error("rfht_plan_create failed");

        std::printf("RFHT-NEON vs BFFT audio polar benchmark\n");
#ifdef RFHT_ACCURATE
        std::printf("RFHT mode=ACCURATE libm atan2/sin/cos\n");
#else
        std::printf("RFHT mode=FAST polynomial atan/sincos\n");
#endif
        std::printf("N=%zu frames=%zu total_samples=%zu\n", n, frames, n * frames);
        std::printf("bfft backend=%s version=%s\n", bfft::backend_name().c_str(), bfft::version_string().c_str());

        std::printf("\nforward only: real -> mag,phase\n");
        const timed_forward bfw = bench_bfft_forward(bplan, input, n, frames);
        const timed_forward rfw = bench_rfht_forward(rplan, input, n, frames);
        print_forward("bfft forward polar", bfw);
        print_forward("rfht forward polar", rfw);
        std::printf("forward speed rfht/bfft: %.17g\n", rfw.msamples_per_s / bfw.msamples_per_s);

        std::printf("\nrfht forward compared to bfft over all frames/bins:\n");
        const phase_stats ps = compare_rfht_to_bfft_all(bplan, rplan, input, n, frames);
        std::printf("max_phase_abs %.17g rad  dBc %.3f  rms_phase %.17g rad  max_mag_abs %.17g  max_mag_rel %.17g\n",
                    ps.max_phase_abs,
                    dbc_from_abs(ps.max_phase_abs),
                    ps.rms_phase,
                    ps.max_mag_abs,
                    ps.max_mag_rel);

        std::printf("\nround trip: real -> mag,phase -> real\n");
        const timed_err bcy = bench_bfft_cycle(bplan, input, n, frames);
        const timed_err rcy = bench_rfht_cycle(rplan, input, n, frames);
        print_cycle("bfft polar cycle", bcy);
        print_cycle("rfht polar cycle", rcy);
        std::printf("cycle speed rfht/bfft:   %.17g\n", rcy.msamples_per_s / bcy.msamples_per_s);
        std::printf("cycle MSE rfht/bfft:     %.17g\n", rcy.err.mse / bcy.err.mse);

        rfht_plan_destroy(rplan);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "usage: compare_rfht_neon_bfft_audio [N=2048] [frames=4096]\n");
        return 2;
    }
}
