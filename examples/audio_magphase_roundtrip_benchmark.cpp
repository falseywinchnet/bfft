#include <bfft/bfft.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr double pi = 3.141592653589793238462643383279502884;

double seconds_since(clock_type::time_point start, clock_type::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

} // namespace

int main(int argc, char** argv) {
    std::size_t n = 2048;
    std::size_t frames = 4096;
    if (argc > 1) {
        n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        frames = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    }

    bfft::plan plan(n);
    std::vector<double> input(n);
    std::vector<double> rect_roundtrip(n);
    std::vector<double> magphase_roundtrip(n);
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> rect(plan.bins());
    std::vector<bfft::complex> polar(plan.bins());
    std::vector<bfft::complex> native_tmp(plan.bins());

    std::mt19937_64 rng(20260627);
    std::uniform_real_distribution<double> noise(-0.01, 0.01);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        input[i] = 0.7 * std::sin(2.0 * pi * 17.0 * t) + 0.2 * std::sin(2.0 * pi * 113.0 * t) + noise(rng);
    }

    double rect_forward_seconds = 0.0;
    double rect_inverse_seconds = 0.0;
    double magphase_forward_seconds = 0.0;
    double magphase_inverse_seconds = 0.0;
    double sink = 0.0;

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto a = clock_type::now();
        plan.forward(input.data(), rect.data(), work.data(), native_tmp.data());
        const auto b = clock_type::now();
        plan.inverse(rect.data(), rect_roundtrip.data());
        const auto c = clock_type::now();
        plan.forward_mag_phase(input.data(), polar.data(), work.data());
        const auto d = clock_type::now();
        plan.inverse_mag_phase(polar.data(), magphase_roundtrip.data());
        const auto e = clock_type::now();

        rect_forward_seconds += seconds_since(a, b);
        rect_inverse_seconds += seconds_since(b, c);
        magphase_forward_seconds += seconds_since(c, d);
        magphase_inverse_seconds += seconds_since(d, e);
        sink += rect_roundtrip[frame % n] + magphase_roundtrip[(frame * 17) % n];
    }

    double mse = 0.0;
    double signal = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double error = magphase_roundtrip[i] - input[i];
        mse += error * error;
        signal += input[i] * input[i];
        max_abs = std::max(max_abs, std::fabs(error));
    }
    mse /= static_cast<double>(n);
    signal /= static_cast<double>(n);
    const double snr = 10.0 * std::log10(signal / mse);
    const double frame_scale = static_cast<double>(frames) / 1000.0;
    const double rect_total = rect_forward_seconds + rect_inverse_seconds;
    const double magphase_total = magphase_forward_seconds + magphase_inverse_seconds;

    std::printf("N=%zu frames=%zu sink=%.17g\n", n, frames, sink);
    std::printf("rect forward                 %.6f s, %.3f kframes/s\n", rect_forward_seconds, frame_scale / rect_forward_seconds);
    std::printf("rect inverse                 %.6f s, %.3f kframes/s\n", rect_inverse_seconds, frame_scale / rect_inverse_seconds);
    std::printf("rect fwd+inv                 %.6f s, %.3f kframes/s\n", rect_total, frame_scale / rect_total);
    std::printf("magphase forward_mag_phase   %.6f s, %.3f kframes/s\n", magphase_forward_seconds, frame_scale / magphase_forward_seconds);
    std::printf("magphase inverse_mag_phase   %.6f s, %.3f kframes/s\n", magphase_inverse_seconds, frame_scale / magphase_inverse_seconds);
    std::printf("magphase fwd+inv             %.6f s, %.3f kframes/s\n", magphase_total, frame_scale / magphase_total);
    std::printf("MSE %.17g\n", mse);
    std::printf("SNR %.6f dB\n", snr);
    std::printf("max_abs %.17g\n", max_abs);
    return 0;
}
