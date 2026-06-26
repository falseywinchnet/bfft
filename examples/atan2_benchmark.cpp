#include "../src/detail/bruun_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double tau = 2.0 * pi;

struct sample {
    double x;
    double y;
    double mag;
};

double normalize_phase(double phase) {
    if (phase < 0.0) {
        phase += tau;
    }
    return phase;
}

float normalize_phase_f32(float phase) {
    if (phase < 0.0f) {
        phase += static_cast<float>(tau);
    }
    return phase;
}

std::vector<sample> make_samples(std::size_t count) {
    std::vector<sample> samples(count);
    std::mt19937_64 rng(20260626);
    std::uniform_real_distribution<double> radius_dist(-12.0, 12.0);
    std::uniform_real_distribution<double> angle_dist(-pi, pi);

    for (sample& value : samples) {
        const double radius = std::exp(radius_dist(rng));
        const double angle = angle_dist(rng);
        value.x = radius * std::cos(angle);
        value.y = radius * std::sin(angle);
        value.mag = radius;
    }
    return samples;
}

template <typename Func>
double time_loop(const std::vector<sample>& samples, Func func, double& sink) {
    const auto start = std::chrono::steady_clock::now();
    double total = 0.0;
    for (const sample& value : samples) {
        total += static_cast<double>(func(value));
    }
    const auto stop = std::chrono::steady_clock::now();
    sink += total;
    return std::chrono::duration<double>(stop - start).count();
}

} // namespace

int main(int argc, char** argv) {
    std::size_t count = 1u << 22;
    if (argc > 1) {
        count = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }

    const std::vector<sample> samples = make_samples(count);
    double sink = 0.0;

    const double std_double_seconds = time_loop(samples, [](const sample& value) {
        return normalize_phase(std::atan2(value.y, value.x));
    }, sink);

    const double bfft_double_seconds = time_loop(samples, [](const sample& value) {
        return bruun::bruun_phase_atan2_mag(value.y, value.x, value.mag);
    }, sink);

    const double std_float_seconds = time_loop(samples, [](const sample& value) {
        const float y = static_cast<float>(value.y);
        const float x = static_cast<float>(value.x);
        return normalize_phase_f32(std::atan2(y, x));
    }, sink);

    const double bfft_float_seconds = time_loop(samples, [](const sample& value) {
        const float y = static_cast<float>(value.y);
        const float x = static_cast<float>(value.x);
        const float mag = static_cast<float>(value.mag);
        return bruun::bruun_phase_atan2_mag_f32(y, x, mag);
    }, sink);

    const double scale = static_cast<double>(count) / 1000000.0;
    std::printf("samples=%zu sink=%.17g\n", count, sink);
    std::printf("std::atan2 double: %.6f s, %.3f Mphase/s\n", std_double_seconds, scale / std_double_seconds);
    std::printf("bfft atan2 double: %.6f s, %.3f Mphase/s\n", bfft_double_seconds, scale / bfft_double_seconds);
    std::printf("std::atan2 float : %.6f s, %.3f Mphase/s\n", std_float_seconds, scale / std_float_seconds);
    std::printf("bfft atan2 float : %.6f s, %.3f Mphase/s\n", bfft_float_seconds, scale / bfft_float_seconds);
    return 0;
}
