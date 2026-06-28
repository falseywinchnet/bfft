#include <bfft/bfft.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

int main() {
    const std::size_t n = 1024;

    try {
        bfft::plan plan(n);
        std::vector<double> input(plan.size());
        std::vector<double> work(plan.work_size());
        std::vector<bfft::complex> output(plan.bins());
        std::vector<bfft::complex> scratch(plan.native_scratch_size());

        for (std::size_t i = 0; i < plan.size(); ++i) {
            input[i] = std::sin(2.0 * 3.14159265358979323846 * static_cast<double>(i) /
                                static_cast<double>(plan.size()));
        }

        plan.forward(input.data(), output.data(), work.data(), scratch.data());

        std::printf("backend=%s policy=%s bin1=(%.6f, %.6f)\n",
                    bfft::backend_name().c_str(),
                    plan.standard_policy().c_str(),
                    output[1].re,
                    output[1].im);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "bfft demo failed: %s\n", exc.what());
        return 1;
    }

    return 0;
}
