#include <bfft/bfft.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::vector<bfft::complex> naive_rfft(const std::vector<double>& input) {
    const std::size_t n = input.size();
    std::vector<bfft::complex> output(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (std::size_t t = 0; t < n; ++t) {
            const double angle = -2.0 * pi * static_cast<double>(k * t) / static_cast<double>(n);
            re += input[t] * std::cos(angle);
            im += input[t] * std::sin(angle);
        }
        output[k].re = re;
        output[k].im = im;
    }
    return output;
}

double max_spectrum_error(const std::vector<bfft::complex>& a, const std::vector<bfft::complex>& b) {
    double error = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, std::fabs(a[i].re - b[i].re));
        error = std::max(error, std::fabs(a[i].im - b[i].im));
    }
    return error;
}

bool check_size(std::size_t n) {
    bfft::plan plan(n);
    std::vector<double> input(n);
    std::mt19937_64 rng(n);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : input) {
        value = dist(rng);
    }

    std::vector<bfft::complex> output(plan.bins());
    std::vector<bfft::complex> scratch(plan.native_scratch_size());
    std::vector<double> work(plan.work_size());
    plan.forward(input.data(), output.data(), work.data(), scratch.data());

    const std::vector<bfft::complex> expected = naive_rfft(input);
    const double error = max_spectrum_error(output, expected);
    const double tolerance = 1e-8 * static_cast<double>(n);
    if (error > tolerance) {
        std::fprintf(stderr, "n=%zu spectrum error %.17g exceeds %.17g\n", n, error, tolerance);
        return false;
    }

    std::vector<double> roundtrip(n);
    plan.inverse(output.data(), roundtrip.data());
    double inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        inverse_error = std::max(inverse_error, std::fabs(roundtrip[i] - input[i]));
    }
    if (inverse_error > tolerance) {
        std::fprintf(stderr, "n=%zu inverse error %.17g exceeds %.17g\n", n, inverse_error, tolerance);
        return false;
    }

    std::vector<bfft::complex> native(plan.bins());
    std::vector<bfft::complex> standard(plan.bins());
    plan.forward_native(input.data(), native.data(), work.data());
    plan.native_to_standard(native.data(), standard.data());
    const double native_error = max_spectrum_error(standard, output);
    if (native_error > tolerance) {
        std::fprintf(stderr, "n=%zu native conversion error %.17g exceeds %.17g\n", n, native_error, tolerance);
        return false;
    }

    return true;
}

} // namespace

int main(void) {
    const std::size_t sizes[] = {4, 8, 16, 32, 64, 256, 1024};
    for (std::size_t n : sizes) {
        if (!check_size(n)) {
            return 1;
        }
    }
    std::printf("correctness ok backend=%s\n", bfft::backend_name().c_str());
    return 0;
}
