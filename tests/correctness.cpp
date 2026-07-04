#include <bfft/bfft.hpp>
#include "../src/detail/bruun_dip_kernel.hpp"
#include "../src/detail/bruun_dif_kernel.hpp"
#include "../src/detail/bruun_dit_kernel.hpp"

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

double max_mag_phase_error(const std::vector<bfft::complex>& polar,
                           const std::vector<bfft::complex>& spectrum) {
    double error = 0.0;
    for (std::size_t i = 0; i < polar.size(); ++i) {
        const double re = polar[i].re * std::cos(polar[i].im);
        const double im = polar[i].re * std::sin(polar[i].im);
        error = std::max(error, std::fabs(re - spectrum[i].re));
        error = std::max(error, std::fabs(im - spectrum[i].im));
    }
    return error;
}

double max_mag_phase_error_f32(const std::vector<bfft::complex_f32>& polar,
                               const std::vector<bfft::complex_f32>& spectrum) {
    double error = 0.0;
    for (std::size_t i = 0; i < polar.size(); ++i) {
        const double mag = static_cast<double>(polar[i].re);
        const double phase = static_cast<double>(polar[i].im);
        const double re = mag * std::cos(phase);
        const double im = mag * std::sin(phase);
        error = std::max(error, std::fabs(re - static_cast<double>(spectrum[i].re)));
        error = std::max(error, std::fabs(im - static_cast<double>(spectrum[i].im)));
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

    std::vector<bfft::complex> polar(plan.bins());
    plan.forward_mag_phase(input.data(), polar.data(), work.data());
    const double mag_phase_error = max_mag_phase_error(polar, output);
    if (mag_phase_error > tolerance) {
        std::fprintf(stderr, "n=%zu mag-phase error %.17g exceeds %.17g\n", n, mag_phase_error, tolerance);
        return false;
    }

    std::vector<double> mag_phase_roundtrip(n);
    plan.inverse_mag_phase(polar.data(), mag_phase_roundtrip.data());
    double mag_phase_inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mag_phase_inverse_error = std::max(mag_phase_inverse_error, std::fabs(mag_phase_roundtrip[i] - input[i]));
    }
    if (mag_phase_inverse_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu mag-phase inverse error %.17g exceeds %.17g\n",
                     n,
                     mag_phase_inverse_error,
                     tolerance);
        return false;
    }

    const std::vector<bfft::complex> vector_polar = plan.forward_mag_phase(input);
    if (vector_polar.size() != plan.bins()) {
        std::fprintf(stderr, "n=%zu vector mag-phase bin count mismatch\n", n);
        return false;
    }
    const double vector_mag_phase_error = max_mag_phase_error(vector_polar, output);
    if (vector_mag_phase_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu vector mag-phase error %.17g exceeds %.17g\n",
                     n,
                     vector_mag_phase_error,
                     tolerance);
        return false;
    }

    bool threw_wrong_size = false;
    try {
        std::vector<double> wrong_input(n + 1);
        (void)plan.forward_mag_phase(wrong_input);
    } catch (const bfft::error&) {
        threw_wrong_size = true;
    }
    if (!threw_wrong_size) {
        std::fprintf(stderr, "n=%zu vector mag-phase wrong size did not throw\n", n);
        return false;
    }

    const std::vector<double> vector_mag_phase_roundtrip = plan.inverse_mag_phase(vector_polar);
    double vector_mag_phase_inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        vector_mag_phase_inverse_error = std::max(vector_mag_phase_inverse_error,
                                                 std::fabs(vector_mag_phase_roundtrip[i] - input[i]));
    }
    if (vector_mag_phase_inverse_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu vector mag-phase inverse error %.17g exceeds %.17g\n",
                     n,
                     vector_mag_phase_inverse_error,
                     tolerance);
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

    std::vector<bfft::complex_f32> polar(plan.bins());
    plan.forward_mag_phase_f32(input.data(), polar.data(), work.data());
    const double mag_phase_error = max_mag_phase_error_f32(polar, output);
    if (mag_phase_error > tolerance) {
        std::fprintf(stderr, "n=%zu f32 mag-phase error %.17g exceeds %.17g\n", n, mag_phase_error, tolerance);
        return false;
    }

    std::vector<float> mag_phase_roundtrip(n);
    plan.inverse_mag_phase_f32(polar.data(), mag_phase_roundtrip.data());
    double mag_phase_inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mag_phase_inverse_error = std::max(mag_phase_inverse_error,
                                           static_cast<double>(std::fabs(mag_phase_roundtrip[i] - input[i])));
    }
    if (mag_phase_inverse_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu f32 mag-phase inverse error %.17g exceeds %.17g\n",
                     n,
                     mag_phase_inverse_error,
                     tolerance);
        return false;
    }

    const std::vector<bfft::complex_f32> vector_polar = plan.forward_mag_phase_f32(input);
    if (vector_polar.size() != plan.bins()) {
        std::fprintf(stderr, "n=%zu f32 vector mag-phase bin count mismatch\n", n);
        return false;
    }
    const double vector_mag_phase_error = max_mag_phase_error_f32(vector_polar, output);
    if (vector_mag_phase_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu f32 vector mag-phase error %.17g exceeds %.17g\n",
                     n,
                     vector_mag_phase_error,
                     tolerance);
        return false;
    }

    bool threw_wrong_size = false;
    try {
        std::vector<float> wrong_input(n + 1);
        (void)plan.forward_mag_phase_f32(wrong_input);
    } catch (const bfft::error&) {
        threw_wrong_size = true;
    }
    if (!threw_wrong_size) {
        std::fprintf(stderr, "n=%zu f32 vector mag-phase wrong size did not throw\n", n);
        return false;
    }

    const std::vector<float> vector_mag_phase_roundtrip = plan.inverse_mag_phase_f32(vector_polar);
    double vector_mag_phase_inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        vector_mag_phase_inverse_error = std::max(vector_mag_phase_inverse_error,
                                                 static_cast<double>(std::fabs(vector_mag_phase_roundtrip[i] - input[i])));
    }
    if (vector_mag_phase_inverse_error > tolerance) {
        std::fprintf(stderr,
                     "n=%zu f32 vector mag-phase inverse error %.17g exceeds %.17g\n",
                     n,
                     vector_mag_phase_inverse_error,
                     tolerance);
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

bool check_dit_f32_direct(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    bruun::DIT_RFFT_kernel plan;
    if (!plan.init(static_cast<int>(n))) {
        std::fprintf(stderr, "n=%zu DIT f32 direct plan init failed\n", n);
        return false;
    }

    std::vector<float> input(n);
    std::mt19937 rng(static_cast<unsigned int>(0xD17u + n));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : input) {
        value = dist(rng);
    }

    std::vector<bruun::complex_f32_t> output(plan.bins());
    std::vector<float> work(static_cast<std::size_t>(plan.work_size_f32()));
    plan.forward_simd_f32(input.data(), output.data(), work.data());

    const std::vector<bfft::complex_f32> expected = naive_rfft_f32(input);
    double spectrum_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        spectrum_error = std::max(spectrum_error,
                                  static_cast<double>(std::fabs(output[i].re - expected[i].re)));
        spectrum_error = std::max(spectrum_error,
                                  static_cast<double>(std::fabs(output[i].im - expected[i].im)));
    }
    const double tolerance = 8e-5 * static_cast<double>(n);
    if (spectrum_error > tolerance) {
        std::fprintf(stderr, "n=%zu DIT f32 direct spectrum error %.17g exceeds %.17g\n",
                     n, spectrum_error, tolerance);
        return false;
    }

    std::vector<float> roundtrip(n);
    plan.inverse_simd_f32(output.data(), roundtrip.data(), work.data());
    double inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        inverse_error = std::max(inverse_error,
                                 static_cast<double>(std::fabs(roundtrip[i] - input[i])));
    }
    if (inverse_error > tolerance) {
        std::fprintf(stderr, "n=%zu DIT f32 direct inverse error %.17g exceeds %.17g\n",
                     n, inverse_error, tolerance);
        return false;
    }

    return true;
}

bool check_dip_direct(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    bruun::DIP_RFFT_kernel plan;
    if (!plan.init(static_cast<int>(n))) {
        std::fprintf(stderr, "n=%zu DIP direct plan init failed\n", n);
        return false;
    }

    std::vector<double> input(n);
    std::mt19937_64 rng(0xD1A60A1ULL + n);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : input) {
        value = dist(rng);
    }

    std::vector<bruun::complex_t> output(plan.bins());
    std::vector<bruun::complex_t> blocked_output(plan.bins());
    std::vector<double> work(static_cast<std::size_t>(plan.work_size()));
    std::vector<double> blocked_work(static_cast<std::size_t>(plan.blocked_work_size()));
    plan.forward_standard(input.data(), output.data(), work.data());
    plan.forward_standard_blocked(input.data(), blocked_output.data(), blocked_work.data());

    const std::vector<bfft::complex> expected = naive_rfft(input);
    double spectrum_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        spectrum_error = std::max(spectrum_error, std::fabs(output[i].re - expected[i].re));
        spectrum_error = std::max(spectrum_error, std::fabs(output[i].im - expected[i].im));
    }

    const double tolerance = 1e-8 * static_cast<double>(n);
    if (spectrum_error > tolerance) {
        std::fprintf(stderr, "n=%zu DIP direct spectrum error %.17g exceeds %.17g\n",
                     n, spectrum_error, tolerance);
        return false;
    }

    double blocked_spectrum_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        blocked_spectrum_error = std::max(blocked_spectrum_error,
                                          std::fabs(blocked_output[i].re - expected[i].re));
        blocked_spectrum_error = std::max(blocked_spectrum_error,
                                          std::fabs(blocked_output[i].im - expected[i].im));
    }
    if (blocked_spectrum_error > tolerance) {
        std::fprintf(stderr, "n=%zu blocked DIP direct spectrum error %.17g exceeds %.17g\n",
                     n, blocked_spectrum_error, tolerance);
        return false;
    }

    std::vector<double> roundtrip(n);
    plan.inverse_standard(output.data(), roundtrip.data(), work.data());
    double inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        inverse_error = std::max(inverse_error, std::fabs(roundtrip[i] - input[i]));
    }
    if (inverse_error > tolerance) {
        std::fprintf(stderr, "n=%zu DIP direct inverse error %.17g exceeds %.17g\n",
                     n, inverse_error, tolerance);
        return false;
    }

    std::vector<double> blocked_roundtrip(n);
    plan.inverse_standard_blocked(blocked_output.data(), blocked_roundtrip.data(), blocked_work.data());
    double blocked_inverse_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        blocked_inverse_error = std::max(blocked_inverse_error, std::fabs(blocked_roundtrip[i] - input[i]));
    }
    if (blocked_inverse_error > tolerance) {
        std::fprintf(stderr, "n=%zu blocked DIP direct inverse error %.17g exceeds %.17g\n",
                     n, blocked_inverse_error, tolerance);
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

bool check_dc_nyquist_phase(void) {
    bfft::plan plan(4);
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> polar(plan.bins());

    std::vector<double> input = {1.0, 1.0, 1.0, 1.0};
    plan.forward_mag_phase(input.data(), polar.data(), work.data());
    if (polar[0].im != 0.0 || polar[2].im != 0.0) {
        std::fprintf(stderr, "positive DC/Nyquist phase was not zero\n");
        return false;
    }

    input = {-1.0, -1.0, -1.0, -1.0};
    plan.forward_mag_phase(input.data(), polar.data(), work.data());
    if (std::fabs(polar[0].im - pi) > 1e-12 || polar[2].im != 0.0) {
        std::fprintf(stderr, "negative DC phase was not pi or zero Nyquist was not zero\n");
        return false;
    }

    input = {-1.0, 1.0, -1.0, 1.0};
    plan.forward_mag_phase(input.data(), polar.data(), work.data());
    if (polar[0].im != 0.0 || std::fabs(polar[2].im - pi) > 1e-12) {
        std::fprintf(stderr, "zero DC or negative Nyquist phase was wrong\n");
        return false;
    }

    input = {0.0, 0.0, 0.0, 0.0};
    plan.forward_mag_phase(input.data(), polar.data(), work.data());
    if (polar[0].im != 0.0 || polar[2].im != 0.0) {
        std::fprintf(stderr, "zero DC/Nyquist phase was not zero\n");
        return false;
    }

    return true;
}


bool check_slope_phase_accuracy(void) {
    constexpr double target_rad = 6.309573444801929e-8;
    double max_double_error = 0.0;
    double max_float_error = 0.0;
    double float_error_sum_sq = 0.0;
    std::size_t float_count = 0;

    for (int i = 0; i <= 100000; ++i) {
        const double theta = (0.5 * pi) * static_cast<double>(i) / 100000.0;
        const double x = std::cos(theta);
        const double y = std::sin(theta);
        const double phase = bruun::bruun_phase_atan2_mag(y, x, 1.0);
        max_double_error = std::max(max_double_error, std::fabs(phase - theta));

        const float xf = static_cast<float>(x);
        const float yf = static_cast<float>(y);
        const float magf = std::sqrt(xf * xf + yf * yf);
        const float phasef = bruun::bruun_phase_atan2_mag_f32(yf, xf, magf);
        const float reff = std::atan2(yf, xf);
        const double ferr = std::fabs(static_cast<double>(phasef - reff));
        max_float_error = std::max(max_float_error, ferr);
        float_error_sum_sq += ferr * ferr;
        ++float_count;
    }

    std::mt19937_64 rng(20260627);
    std::uniform_real_distribution<double> angle_dist(-pi, pi);
    std::uniform_real_distribution<double> radius_dist(-12.0, 12.0);
    for (int i = 0; i < 100000; ++i) {
        const double theta = angle_dist(rng);
        const double radius = std::exp(radius_dist(rng));
        const double x = radius * std::cos(theta);
        const double y = radius * std::sin(theta);
        double ref = std::atan2(y, x);
        if (ref < 0.0) {
            ref += 2.0 * pi;
        }
        const double phase = bruun::bruun_phase_atan2_mag(y, x, radius);
        double err = std::fabs(phase - ref);
        err = std::min(err, std::fabs(err - 2.0 * pi));
        max_double_error = std::max(max_double_error, err);

        const float xf = static_cast<float>(x);
        const float yf = static_cast<float>(y);
        const float magf = std::sqrt(xf * xf + yf * yf);
        float reff = std::atan2(yf, xf);
        if (reff < 0.0f) {
            reff += static_cast<float>(2.0 * pi);
        }
        const float phasef = bruun::bruun_phase_atan2_mag_f32(yf, xf, magf);
        double ferr = std::fabs(static_cast<double>(phasef - reff));
        ferr = std::min(ferr, std::fabs(ferr - 2.0 * pi));
        max_float_error = std::max(max_float_error, ferr);
        float_error_sum_sq += ferr * ferr;
        ++float_count;
    }

    const double float_rms = std::sqrt(float_error_sum_sq / static_cast<double>(float_count));
    std::printf("phase slope accuracy: double max %.17g, f32 max %.17g, f32 rms %.17g\n",
                max_double_error,
                max_float_error,
                float_rms);

    if (max_double_error > target_rad) {
        std::fprintf(stderr,
                     "double slope256 linear Q512 phase error %.17g exceeds %.17g\n",
                     max_double_error,
                     target_rad);
        return false;
    }
    return true;
}

bool check_rejects_non_power_of_two(void) {
    bfft_plan* raw = nullptr;
    const bfft_status status = bfft_plan_create(15, &raw);
    if (status != BFFT_ERROR_INVALID_ARGUMENT || raw != nullptr) {
        std::fprintf(stderr, "non-power-of-two plan creation was not rejected\n");
        bfft_plan_destroy(raw);
        return false;
    }
    return true;
}

} // namespace

int main(void) {
    if (!check_rejects_non_power_of_two()) {
        return 1;
    }
    if (!check_dc_nyquist_phase()) {
        return 1;
    }
    if (!check_slope_phase_accuracy()) {
        return 1;
    }
    const std::size_t sizes[] = {4, 8, 16, 32, 64, 128, 256, 1024};
    for (std::size_t n : sizes) {
        if (!check_size(n)) {
            return 1;
        }
        if (!check_size_f32(n)) {
            return 1;
        }
        if (!check_dit_f32_direct(n)) {
            return 1;
        }
        if (!check_dip_direct(n)) {
            return 1;
        }
    }
    if (!check_bh7_f32_native_sfdr()) {
        return 1;
    }
    std::printf("correctness ok backend=%s\n", bfft::backend_name().c_str());
    return 0;
}
