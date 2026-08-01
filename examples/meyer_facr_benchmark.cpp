#include <bfft/meyer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

double run(std::size_t h, std::size_t w, int solver, int repeats) {
    const std::size_t n = h * w;
    std::vector<double> image(n), cartoon(n), texture(n);
    for (std::size_t i = 0; i < n; ++i)
        image[i] = 127.0 + 60.0 * std::sin(0.013 * double(i));
    bfft_meyer_plan* plan = nullptr;
    if (bfft_meyer_plan_create(h, w, 0.05, 40.0, 24, 1, 0.0, 4,
                               &plan) != BFFT_OK)
        return -1.0;
    if (solver != 0 && bfft_meyer_plan_set_solver(plan, solver) != BFFT_OK)
        return -1.0;
    bfft_meyer_split(plan, image.data(), cartoon.data(), texture.data());
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r)
        bfft_meyer_split(plan, image.data(), cartoon.data(), texture.data());
    const auto end = std::chrono::steady_clock::now();
    bfft_meyer_plan_destroy(plan);
    return std::chrono::duration<double, std::milli>(end - start).count() /
        repeats;
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 8;
    while (result < value) result *= 2;
    return result;
}

std::size_t reflected_index(std::ptrdiff_t coordinate, std::size_t length) {
    if (coordinate < 0) return std::size_t(-coordinate - 1);
    if (coordinate >= std::ptrdiff_t(length))
        return std::size_t(2 * std::ptrdiff_t(length) - coordinate - 1);
    return std::size_t(coordinate);
}

std::vector<double> make_source(std::size_t h, std::size_t w) {
    std::vector<double> source(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            const double edge = x > w / 3 && y > h / 4 ? 38.0 : -24.0;
            source[y * w + x] = 126.0 + edge
                + 21.0 * std::sin(0.071 * double(x))
                + 13.0 * std::cos(0.053 * double(y))
                + 7.0 * std::sin(0.037 * double(x + 2 * y));
        }
    }
    return source;
}

std::vector<double> symmetric_extend(
    const std::vector<double>& source, std::size_t h, std::size_t w,
    std::size_t ph, std::size_t pw) {
    std::vector<double> padded(ph * pw);
    const std::ptrdiff_t top = std::ptrdiff_t(ph - h) / 2;
    const std::ptrdiff_t left = std::ptrdiff_t(pw - w) / 2;
    for (std::size_t y = 0; y < ph; ++y) {
        const std::size_t sy = reflected_index(std::ptrdiff_t(y) - top, h);
        for (std::size_t x = 0; x < pw; ++x) {
            const std::size_t sx = reflected_index(std::ptrdiff_t(x) - left, w);
            padded[y * pw + x] = source[sy * w + sx];
        }
    }
    return padded;
}

struct result {
    double milliseconds = 0.0;
    std::vector<double> cartoon;
};

result run_extended(const std::vector<double>& source,
                    std::size_t h, std::size_t w,
                    std::size_t ph, std::size_t pw,
                    int solver, int threads, int repeats) {
    std::vector<double> input = symmetric_extend(source, h, w, ph, pw);
    std::vector<double> cartoon(ph * pw), texture(ph * pw);
    bfft_meyer_plan* plan = nullptr;
    if (bfft_meyer_plan_create(ph, pw, 0.05, 40.0, 1, 1, 0.0, threads,
                               &plan) != BFFT_OK)
        throw std::runtime_error("plan creation failed");
    if (solver != 0 && bfft_meyer_plan_set_solver(plan, solver) != BFFT_OK) {
        bfft_meyer_plan_destroy(plan);
        throw std::runtime_error("solver selection failed");
    }
    if (bfft_meyer_split(plan, input.data(), cartoon.data(), texture.data())
        != BFFT_OK) {
        bfft_meyer_plan_destroy(plan);
        throw std::runtime_error("warm split failed");
    }
    std::vector<double> samples;
    samples.reserve(std::size_t(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        if (bfft_meyer_split(plan, input.data(), cartoon.data(), texture.data())
            != BFFT_OK) {
            bfft_meyer_plan_destroy(plan);
            throw std::runtime_error("timed split failed");
        }
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(
            end - start).count());
    }
    bfft_meyer_plan_destroy(plan);
    std::sort(samples.begin(), samples.end());

    result answer;
    answer.milliseconds = samples[samples.size() / 2];
    answer.cartoon.resize(h * w);
    const std::size_t top = (ph - h) / 2;
    const std::size_t left = (pw - w) / 2;
    for (std::size_t y = 0; y < h; ++y)
        std::copy_n(cartoon.data() + (y + top) * pw + left, w,
                    answer.cartoon.data() + y * w);
    return answer;
}

double rms_difference(const std::vector<double>& a,
                      const std::vector<double>& b,
                      std::size_t border = 0,
                      std::size_t h = 0, std::size_t w = 0) {
    double energy = 0.0;
    std::size_t count = 0;
    if (h == 0 || w == 0) {
        h = 1;
        w = a.size();
    }
    for (std::size_t y = border; y + border < h; ++y) {
        for (std::size_t x = border; x + border < w; ++x) {
            const double difference = a[y * w + x] - b[y * w + x];
            energy += difference * difference;
            ++count;
        }
    }
    return std::sqrt(energy / std::max<std::size_t>(count, 1));
}

void compare_shape(std::size_t h, std::size_t w, int threads, int repeats) {
    const std::size_t ph = next_power_of_two(h);
    const std::size_t pw = next_power_of_two(w);
    const std::vector<double> source = make_source(h, w);
    const result spectral = run_extended(
        source, h, w, ph, pw, 0, threads, repeats);
    const result row = run_extended(
        source, h, w, h, pw, 1, threads, repeats);
    const result column = run_extended(
        source, h, w, ph, w, 1, threads, repeats);
    const std::size_t row_area = h * pw;
    const std::size_t column_area = ph * w;
    const bool video_sized = h * w >= 720 * 1280;
    const bool current_is_row = row_area <= column_area ||
        (video_sized && 3 * row_area <= 4 * column_area);
    const result& current = current_is_row ? row : column;
    const result& other = current_is_row ? column : row;
    const std::size_t border = std::min({h, w, std::size_t(32)}) / 4;
    std::printf(
        "\nsource %zux%zu, T%d; full spectral %zux%zu %.3f ms\n"
        "  FACR row FFT    %zux%zu %.3f ms  %.2fx spectral, RMS %.4g"
        " (inner %.4g)\n"
        "  FACR column FFT %zux%zu %.3f ms  %.2fx spectral, RMS %.4g"
        " (inner %.4g)\n"
        "  cost-aware choice: %s, %.3f ms; alternate/current %.2fx,"
        " orientation RMS %.4g\n",
        h, w, threads, ph, pw, spectral.milliseconds,
        h, pw, row.milliseconds, spectral.milliseconds / row.milliseconds,
        rms_difference(row.cartoon, spectral.cartoon),
        rms_difference(row.cartoon, spectral.cartoon, border, h, w),
        ph, w, column.milliseconds,
        spectral.milliseconds / column.milliseconds,
        rms_difference(column.cartoon, spectral.cartoon),
        rms_difference(column.cartoon, spectral.cartoon, border, h, w),
        current_is_row ? "row" : "column", current.milliseconds,
        other.milliseconds / current.milliseconds,
        rms_difference(row.cartoon, column.cartoon));
}

}  // namespace

int main() {
    for (std::size_t n : {std::size_t(256), std::size_t(512),
                          std::size_t(1024)}) {
        const int repeats = n < 1024 ? 5 : 2;
        const double spectral = run(n, n, 0, repeats);
        const double facr = run(n, n, 1, repeats);
        std::printf("%zux%zu spectral %.2f ms, FACR %.2f ms, %.2fx\n",
                    n, n, spectral, facr, spectral / facr);
    }
    for (const auto& shape :
         {std::pair<std::size_t, std::size_t>{300, 512}, {512, 300}}) {
        const double facr = run(shape.first, shape.second, 1, 5);
        const double padded = run(512, 512, 0, 5);
        std::printf("%zux%zu FACR %.2f ms vs 512x512 spectral %.2f ms, %.2fx\n",
                    shape.first, shape.second, facr, padded, padded / facr);
    }

    std::puts("\nBoth-non-power-of-two reflected-input comparison");
    compare_shape(360, 640, 4, 5);
    compare_shape(480, 854, 4, 5);
    compare_shape(720, 1280, 4, 5);
    compare_shape(1280, 720, 4, 5);
    compare_shape(900, 1600, 4, 3);
    compare_shape(1000, 1536, 4, 3);
    compare_shape(1536, 1000, 4, 3);
    compare_shape(1080, 1920, 4, 3);
    compare_shape(1920, 1080, 4, 3);
    compare_shape(1440, 2560, 4, 3);
    compare_shape(2560, 1440, 4, 3);
    std::puts("\nSingle-thread orientation controls");
    compare_shape(720, 1280, 1, 3);
    compare_shape(1280, 720, 1, 3);
    compare_shape(1000, 1536, 1, 3);
    compare_shape(1536, 1000, 1, 3);
}
