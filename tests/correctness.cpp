#include <bfft/bfft.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
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

std::vector<bfft::complex_f32> naive_rfft_f32(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<bfft::complex_f32> output(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (std::size_t t = 0; t < n; ++t) {
            const double angle = -2.0 * pi * static_cast<double>(k * t) / static_cast<double>(n);
            re += static_cast<double>(input[t]) * std::cos(angle);
            im += static_cast<double>(input[t]) * std::sin(angle);
        }
        output[k].re = static_cast<float>(re);
        output[k].im = static_cast<float>(im);
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

double max_spectrum_error_f32(const std::vector<bfft::complex_f32>& a,
                              const std::vector<bfft::complex_f32>& b) {
    double error = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, static_cast<double>(std::fabs(a[i].re - b[i].re)));
        error = std::max(error, static_cast<double>(std::fabs(a[i].im - b[i].im)));
    }
    return error;
}

double max_magnitude_error(const std::vector<double>& magnitudes,
                           const std::vector<bfft::complex>& spectrum) {
    double error = 0.0;
    for (std::size_t i = 0; i < magnitudes.size(); ++i) {
        const double expected = std::hypot(spectrum[i].re, spectrum[i].im);
        error = std::max(error, std::fabs(magnitudes[i] - expected));
    }
    return error;
}

double max_magnitude_error_f32(const std::vector<float>& magnitudes,
                               const std::vector<bfft::complex_f32>& spectrum) {
    double error = 0.0;
    for (std::size_t i = 0; i < magnitudes.size(); ++i) {
        const double expected = std::hypot(static_cast<double>(spectrum[i].re),
                                          static_cast<double>(spectrum[i].im));
        error = std::max(error, std::fabs(static_cast<double>(magnitudes[i]) - expected));
    }
    return error;
}

double bh7_window(std::size_t i, std::size_t n) {
    const double theta = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n);
    constexpr double a0 = 0.27105140069342;
    constexpr double a1 = 0.43329793923448;
    constexpr double a2 = 0.21812299954311;
    constexpr double a3 = 0.06592544638803;
    constexpr double a4 = 0.01081174209837;
    constexpr double a5 = 0.00077658482522;
    constexpr double a6 = 0.00001388721735;

    return a0
         - a1 * std::cos(theta)
         + a2 * std::cos(2.0 * theta)
         - a3 * std::cos(3.0 * theta)
         + a4 * std::cos(4.0 * theta)
         - a5 * std::cos(5.0 * theta)
         + a6 * std::cos(6.0 * theta);
}

bool is_bh7_main_lobe(std::size_t bin, std::size_t k) {
    constexpr std::size_t radius = 6;
    const std::size_t lo = k > radius ? k - radius : 0;
    const std::size_t hi = k + radius;
    return bin >= lo && bin <= hi;
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

    std::vector<double> magnitudes(plan.bins());
    plan.forward_magnitude(input.data(), magnitudes.data(), work.data());
    const double magnitude_error = max_magnitude_error(magnitudes, output);
    if (magnitude_error > tolerance) {
        std::fprintf(stderr, "n=%zu magnitude error %.17g exceeds %.17g\n", n, magnitude_error, tolerance);
        return false;
    }

    const std::vector<double> vector_magnitudes = plan.forward_magnitude(input);
    const double vector_magnitude_error = max_magnitude_error(vector_magnitudes, output);
    if (vector_magnitude_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu vector magnitude error %.17g exceeds %.17g\n",
                     n,
                     vector_magnitude_error,
                     tolerance);
        return false;
    }

    return true;
}

bool check_size_f32(std::size_t n) {
    bfft::plan plan(n);
    std::vector<float> input(n);
    std::mt19937 rng(static_cast<unsigned int>(n));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : input) {
        value = dist(rng);
    }

    std::vector<bfft::complex_f32> output(plan.bins());
    std::vector<bfft::complex_f32> scratch(plan.native_scratch_size());
    std::vector<float> work(plan.work_size_f32());
    plan.forward_f32(input.data(), output.data(), work.data(), scratch.data());

    const std::vector<bfft::complex_f32> expected = naive_rfft_f32(input);
    const double error = max_spectrum_error_f32(output, expected);
    const double tolerance = 8e-5 * static_cast<double>(n);
    if (error > tolerance) {
        std::fprintf(stderr, "n=%zu f32 spectrum error %.17g exceeds %.17g\n", n, error, tolerance);
        return false;
    }

    std::vector<float> roundtrip(n);
    plan.inverse_f32(output.data(), roundtrip.data());
    double inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        inverse_error = std::max(inverse_error, static_cast<double>(std::fabs(roundtrip[i] - input[i])));
    }
    if (inverse_error > tolerance) {
        std::fprintf(stderr, "n=%zu f32 inverse error %.17g exceeds %.17g\n", n, inverse_error, tolerance);
        return false;
    }

    std::vector<bfft::complex_f32> native(plan.bins());
    std::vector<bfft::complex_f32> standard(plan.bins());
    plan.forward_native_f32(input.data(), native.data(), work.data());
    plan.native_to_standard_f32(native.data(), standard.data());
    const double native_error = max_spectrum_error_f32(standard, output);
    if (native_error > tolerance) {
        std::fprintf(stderr, "n=%zu f32 native conversion error %.17g exceeds %.17g\n", n, native_error, tolerance);
        return false;
    }

    std::vector<float> native_roundtrip(n);
    plan.inverse_native_f32(native.data(), native_roundtrip.data());
    double native_inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        native_inverse_error = std::max(native_inverse_error,
                                        static_cast<double>(std::fabs(native_roundtrip[i] - input[i])));
    }
    if (native_inverse_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu f32 native inverse error %.17g exceeds %.17g\n",
                     n,
                     native_inverse_error,
                     tolerance);
        return false;
    }

    const std::vector<bfft::complex_f32> vector_output = plan.forward_f32(input);
    const double vector_error = max_spectrum_error_f32(vector_output, output);
    if (vector_error > tolerance) {
        std::fprintf(stderr, "n=%zu f32 vector API error %.17g exceeds %.17g\n", n, vector_error, tolerance);
        return false;
    }

    std::vector<float> magnitudes(plan.bins());
    plan.forward_magnitude_f32(input.data(), magnitudes.data(), work.data());
    const double magnitude_error = max_magnitude_error_f32(magnitudes, output);
    if (magnitude_error > tolerance) {
        std::fprintf(stderr, "n=%zu f32 magnitude error %.17g exceeds %.17g\n", n, magnitude_error, tolerance);
        return false;
    }

    const std::vector<float> vector_magnitudes = plan.forward_magnitude_f32(input);
    const double vector_magnitude_error = max_magnitude_error_f32(vector_magnitudes, output);
    if (vector_magnitude_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu f32 vector magnitude error %.17g exceeds %.17g\n",
                     n,
                     vector_magnitude_error,
                     tolerance);
        return false;
    }

    return true;
}


bool check_bh7_f32_native_sfdr(void) {
    constexpr std::size_t n = 32768;
    constexpr std::size_t k = 7;
    constexpr double target_db = 144.0;

    bfft::plan plan(n);
    std::vector<float> input(n);
    const double omega = 2.0 * pi * static_cast<double>(k) / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(std::sin(omega * static_cast<double>(i)) * bh7_window(i, n));
    }

    std::vector<float> work(plan.work_size_f32());
    std::vector<bfft::complex_f32> native(plan.bins());
    std::vector<bfft::complex_f32> standard(plan.bins());
    plan.forward_native_f32(input.data(), native.data(), work.data());
    plan.native_to_standard_f32(native.data(), standard.data());

    const double carrier = std::hypot(static_cast<double>(standard[k].re), static_cast<double>(standard[k].im));
    double max_spur = 0.0;
    std::size_t spur_bin = 0;
    for (std::size_t i = 0; i < standard.size(); ++i) {
        if (is_bh7_main_lobe(i, k)) {
            continue;
        }
        const double spur = std::hypot(static_cast<double>(standard[i].re), static_cast<double>(standard[i].im));
        if (spur > max_spur) {
            max_spur = spur;
            spur_bin = i;
        }
    }

    const double sfdr_db = max_spur > 0.0 ? 20.0 * std::log10(carrier / max_spur)
                                          : std::numeric_limits<double>::infinity();
    if (sfdr_db < target_db) {
        std::fprintf(stderr,
                     "f32 native BH7 SFDR %.8f dB below %.8f dB, spur bin %zu\n",
                     sfdr_db,
                     target_db,
                     spur_bin);
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
        if (!check_size_f32(n)) {
            return 1;
        }
    }
    if (!check_bh7_f32_native_sfdr()) {
        return 1;
    }
    std::printf("correctness ok backend=%s\n", bfft::backend_name().c_str());
    return 0;
}
