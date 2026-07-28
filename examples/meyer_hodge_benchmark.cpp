#include <bfft/meyer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

void scene(std::vector<double>& image, std::size_t h, std::size_t w) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    unsigned state = 0x6d2b79f5u;
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            state = 1664525u * state + 1013904223u;
            const double noise =
                (double(state >> 8) / 16777216.0 - 0.5) * 4.0;
            double value = 45.0 + 125.0 * double(x) / double(w);
            value += y > h / 2 ? 38.0 : 0.0;
            value += 21.0 *
                std::cos(2.0 * pi * double(x + y) / 7.0);
            if (x < w / 2)
                value += 13.0 *
                    std::cos(2.0 * pi * double(y) / 5.0);
            image[y * w + x] = value + noise;
        }
    }
}

double objective(const std::vector<double>& u,
                 const std::vector<double>& g,
                 std::size_t h, std::size_t w, double c) {
    double value = 0.0;
    for (std::size_t y = 0; y < h; ++y) {
        const std::size_t yn = y + 1 == h ? 0 : y + 1;
        for (std::size_t x = 0; x < w; ++x) {
            const std::size_t xn = x + 1 == w ? 0 : x + 1;
            const std::size_t i = y * w + x;
            const double gx = u[y * w + xn] - u[i];
            const double gy = u[yn * w + x] - u[i];
            const double residual = u[i] - g[i];
            value += std::sqrt(gx * gx + gy * gy) +
                0.5 * c * residual * residual;
        }
    }
    return value;
}

struct measurement {
    double milliseconds = 0.0;
    double objective = 0.0;
    int sweeps = 0;
    bool hodge_applied = false;
};

measurement measure(bfft_meyer_plan* plan,
                    const std::vector<double>& image,
                    std::vector<double>& output,
                    std::size_t h, std::size_t w,
                    double c, double eta, int max_sweeps, double tol,
                    int hodge_after, int repeats) {
    std::vector<double> samples;
    samples.reserve(std::size_t(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        const bfft_status status = hodge_after > 0
            ? bfft_meyer_rof_accelerated(
                  plan, image.data(), output.data(), c, eta, max_sweeps,
                  tol, hodge_after)
            : bfft_meyer_rof(
                  plan, image.data(), output.data(), c, eta, max_sweeps,
                  tol);
        const auto end = std::chrono::steady_clock::now();
        if (status != BFFT_OK) return {};
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return {
        samples[samples.size() / 2],
        objective(output, image, h, w, c),
        bfft_meyer_plan_last_rof_sweeps(plan),
        bfft_meyer_plan_last_rof_hodge_applied(plan) != 0,
    };
}

}  // namespace

int main() {
    constexpr double c = 0.05, eta = 0.10;
    std::puts(
        "size  mode       sweeps  hodge  median_ms  objective_excess");
    for (const std::size_t size :
         {std::size_t(128), std::size_t(256), std::size_t(512)}) {
        const std::size_t count = size * size;
        std::vector<double> image(count), output(count), reference(count);
        scene(image, size, size);
        bfft_meyer_plan* plan = nullptr;
        if (bfft_meyer_plan_create(
                size, size, c, 40.0, 1, 1, 0.0, 1, &plan) != BFFT_OK)
            return 1;

        if (bfft_meyer_rof(
                plan, image.data(), reference.data(), c, eta, 1024, 0.0) !=
            BFFT_OK)
            return 2;
        const double reference_objective =
            objective(reference, image, size, size, c);
        const int repeats = size < 512 ? 7 : 3;
        const measurement immediate_plain = measure(
            plan, image, output, size, size, c, eta, 4, 0.0, 0, repeats);
        const measurement immediate_hodge = measure(
            plan, image, output, size, size, c, eta, 4, 0.0, 4, repeats);
        const measurement fixed_plain = measure(
            plan, image, output, size, size, c, eta, 8, 0.0, 0, repeats);
        const measurement fixed_hodge = measure(
            plan, image, output, size, size, c, eta, 8, 0.0, 4, repeats);
        const measurement tol_plain = measure(
            plan, image, output, size, size, c, eta, 1024, 1e-3, 0,
            repeats);
        const measurement tol_hodge = measure(
            plan, image, output, size, size, c, eta, 1024, 1e-3, 4,
            repeats);
        const measurement high_plain = measure(
            plan, image, output, size, size, c, eta, 1024, 1e-5, 0,
            repeats);
        const measurement high_hodge = measure(
            plan, image, output, size, size, c, eta, 1024, 1e-5, 4,
            repeats);

        const auto report = [&](const char* mode, const measurement& m) {
            std::printf(
                "%4zu  %-10s %6d  %5s  %9.3f  %16.6g\n",
                size, mode, m.sweeps, m.hodge_applied ? "yes" : "no",
                m.milliseconds, m.objective - reference_objective);
        };
        report("plain-4", immediate_plain);
        report("hodge-4", immediate_hodge);
        report("plain-8", fixed_plain);
        report("hodge-8", fixed_hodge);
        report("plain-1e-3", tol_plain);
        report("hodge-1e-3", tol_hodge);
        report("plain-1e-5", high_plain);
        report("hodge-1e-5", high_hodge);
        bfft_meyer_plan_destroy(plan);
    }
}
