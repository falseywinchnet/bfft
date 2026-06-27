#include <bfft/bfft.hpp>

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

struct stats {
    double mse = 0.0;
    double signal_power = 0.0;
    double max_abs = 0.0;
    double snr_db = 0.0;
};

struct result {
    double seconds = 0.0;
    stats err;
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

double parse_double(const char* s, double fallback) {
    if (!s) return fallback;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(v)) {
        throw std::runtime_error("bad floating argument");
    }
    return v;
}

bool is_power2(std::size_t n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

std::vector<double> make_audio_frames(std::size_t n, std::size_t frames) {
    std::vector<double> x(n * frames);
    std::mt19937_64 rng(20260626);

    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    std::uniform_real_distribution<double> phase_dist(0.0, tau);
    std::uniform_real_distribution<double> amp_jitter(0.85, 1.15);

    // Audio-ish deterministic mixture. Frequencies are bin-centered for the
    // default fs/n layout, plus a low noise floor. The point is round-trip
    // numerical behavior, not psychoacoustic modeling.
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

            // Small broadband component; this prevents only testing sparse spectra.
            v += 1e-4 * noise(rng);

            // Keep signal comfortably inside [-1, 1].
            x[f * n + i] = 0.75 * frame_gain * v;
        }
    }

    return x;
}

stats compute_stats(const std::vector<double>& ref, const std::vector<double>& got) {
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

    stats out;
    out.mse = static_cast<double>(sum_e2 / static_cast<long double>(ref.size()));
    out.signal_power = static_cast<double>(sum_x2 / static_cast<long double>(ref.size()));
    out.max_abs = max_abs;
    out.snr_db = (out.mse > 0.0)
        ? 10.0 * std::log10(out.signal_power / out.mse)
        : INFINITY;
    return out;
}

void quantize_phase(std::vector<bfft::complex>& mp, double step) {
    if (step <= 0.0) return;

    for (bfft::complex& z : mp) {
        double p = z.im;
        p = std::round(p / step) * step;
        if (p >= tau) p -= tau;
        if (p < 0.0) p += tau;
        z.im = p;
    }
}

result bench_mag_phase_cycle(const bfft::plan& plan,
                             const std::vector<double>& input,
                             std::size_t n,
                             std::size_t frames,
                             double phase_quant_step) {
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> mp(plan.bins());
    std::vector<double> recon_frame(n);
    std::vector<double> recon(input.size());

    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t f = 0; f < frames; ++f) {
        const double* in = input.data() + f * n;
        plan.forward_mag_phase(in, mp.data(), work.data());

        quantize_phase(mp, phase_quant_step);

        plan.inverse_mag_phase(mp.data(), recon_frame.data());
        std::copy(recon_frame.begin(), recon_frame.end(), recon.begin() + f * n);
        sink += recon_frame[(f * 17) & (n - 1)];
    }
    const auto t1 = std::chrono::steady_clock::now();

    result r;
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.err = compute_stats(input, recon);
    r.sink = sink;
    return r;
}

result bench_rect_cycle(const bfft::plan& plan,
                        const std::vector<double>& input,
                        std::size_t n,
                        std::size_t frames) {
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> spectrum(plan.bins());
    std::vector<bfft::complex> scratch(plan.native_scratch_size());
    std::vector<double> recon_frame(n);
    std::vector<double> recon(input.size());

    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t f = 0; f < frames; ++f) {
        const double* in = input.data() + f * n;
        plan.forward(in, spectrum.data(), work.data(), scratch.data());
        plan.inverse(spectrum.data(), recon_frame.data());
        std::copy(recon_frame.begin(), recon_frame.end(), recon.begin() + f * n);
        sink += recon_frame[(f * 17) & (n - 1)];
    }
    const auto t1 = std::chrono::steady_clock::now();

    result r;
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.err = compute_stats(input, recon);
    r.sink = sink;
    return r;
}

void print_result(const char* name, const result& r, std::size_t samples) {
    const double ms = 1000.0 * r.seconds;
    const double msamples_per_s = static_cast<double>(samples) / r.seconds / 1.0e6;

    std::printf("%-28s time %.6f s  %.3f Msamples/s  MSE %.17g  SNR %.3f dB  max_abs %.17g  sink %.17g\n",
                name,
                r.seconds,
                msamples_per_s,
                r.err.mse,
                r.err.snr_db,
                r.err.max_abs,
                r.sink);
    (void)ms;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t n = 2048;
        std::size_t frames = 4096;
        double phase_quant_step = 0.0;

        if (argc > 1) n = parse_size(argv[1], n);
        if (argc > 2) frames = parse_size(argv[2], frames);
        if (argc > 3) phase_quant_step = parse_double(argv[3], phase_quant_step);

        if (!is_power2(n) || n < 4) {
            throw std::runtime_error("N must be a power of two >= 4");
        }
        if (phase_quant_step < 0.0) {
            throw std::runtime_error("phase quantization step must be >= 0");
        }

        bfft::plan plan(n);
        const std::vector<double> input = make_audio_frames(n, frames);
        const std::size_t total_samples = n * frames;

        std::printf("BFFT audio round-trip benchmark: real -> mag,phase -> real\n");
        std::printf("backend=%s version=%s\n", bfft::backend_name().c_str(), bfft::version_string().c_str());
        std::printf("N=%zu frames=%zu total_samples=%zu phase_quant_step=%.17g rad\n",
                    n, frames, total_samples, phase_quant_step);

        if (phase_quant_step > 0.0) {
            const double worst_half_step = 0.5 * phase_quant_step;
            const double dbc = -20.0 * std::log10(worst_half_step);
            std::printf("phase quantization worst half-step %.17g rad ~= -%.3f dBc relative phase floor\n",
                        worst_half_step, dbc);
        }

        // Run rectangular first as the baseline numerical round-trip.
        const result rect = bench_rect_cycle(plan, input, n, frames);
        const result polar = bench_mag_phase_cycle(plan, input, n, frames, phase_quant_step);

        print_result("rectangular fwd+inv", rect, total_samples);
        print_result("magphase fwd+inv", polar, total_samples);

        const double mse_ratio = polar.err.mse / rect.err.mse;
        std::printf("magphase/rect MSE ratio: %.17g\n", mse_ratio);

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "usage: audio_magphase_roundtrip_benchmark [N=2048] [frames=4096] [phase_quant_step_rad=0]\n");
        return 2;
    }
}
