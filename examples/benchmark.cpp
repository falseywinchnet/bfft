#include <bfft/bfft.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

std::size_t parse_size(int argc, char** argv) {
    if (argc < 2) {
        return 4096;
    }
    return static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
}

int parse_repetitions(int argc, char** argv) {
    if (argc < 3) {
        return 200;
    }
    return std::atoi(argv[2]);
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t n = parse_size(argc, argv);
    const int repetitions = parse_repetitions(argc, argv);

    try {
        bfft::plan plan(n);
        std::vector<double> input(plan.size());
        std::vector<double> work(plan.work_size());
        std::vector<bfft::complex> spectrum(plan.bins());
        std::vector<bfft::complex> scratch(plan.native_scratch_size());

        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (double& value : input) {
            value = dist(rng);
        }

        plan.forward(input.data(), spectrum.data(), work.data(), scratch.data());

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            plan.forward(input.data(), spectrum.data(), work.data(), scratch.data());
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        const double us = seconds * 1000000.0 / static_cast<double>(repetitions);

        std::printf("BFFT benchmark\n");
        std::printf("  n: %zu\n", plan.size());
        std::printf("  bins: %zu\n", plan.bins());
        std::printf("  backend: %s\n", bfft::backend_name().c_str());
        std::printf("  standard policy: %s\n", plan.standard_policy().c_str());
        std::printf("  repetitions: %d\n", repetitions);
        std::printf("  forward time: %.3f us\n", us);
        std::printf("  first bins: (%g,%g) (%g,%g)\n",
                    spectrum[0].re,
                    spectrum[0].im,
                    spectrum[1].re,
                    spectrum[1].im);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "bfft benchmark failed: %s\n", exc.what());
        return 1;
    }

    return 0;
}
