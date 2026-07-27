#include <bfft/meyer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
    for (const auto shape :
         {std::pair<std::size_t, std::size_t>{300, 512}, {512, 300}}) {
        const double facr = run(shape.first, shape.second, 1, 5);
        const double padded = run(512, 512, 0, 5);
        std::printf("%zux%zu FACR %.2f ms vs 512x512 spectral %.2f ms, %.2fx\n",
                    shape.first, shape.second, facr, padded, padded / facr);
    }
}
