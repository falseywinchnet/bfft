#include "retained_transport.h"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    constexpr size_t nodes = 4;
    constexpr size_t modes = 2;
    const std::array<pft_retained_block_f64, 1> blocks{{
        {0, 2, 0, 2, 0, 2,
         PFT_BLOCK_SOURCE_CONTIGUOUS | PFT_BLOCK_RECEIVER_CONTIGUOUS,
         0, 0.4, 0.3, 0.7, 0.1}}};
    const std::array<uint32_t, 2> source_nodes{{0, 1}};
    const std::array<double, 2> source_factors{{0.8, 0.5}};
    const std::array<uint32_t, 2> receiver_nodes{{2, 3}};
    const std::array<double, 2> receiver_factors{{0.3, 0.4}};
    const std::array<uint8_t, nodes> active{{1, 1, 0, 0}};
    const std::array<double, modes> linear{{1.0, 0.0}};
    const std::array<double, modes> circular{{0.0, -1.0}};
    const std::array<double, modes * nodes> power{{2.0, 3.0, 0.0, 0.0,
                                                  -1.0, 4.0, 0.0, 0.0}};
    std::array<double, modes * nodes> output{};
    std::array<double, modes> scratch{};
    pft_retained_stats stats{};
    const int status = pft_retained_apply_f64(
        nodes, modes, blocks.size(), blocks.data(), source_nodes.data(),
        source_factors.data(), receiver_nodes.data(), receiver_factors.data(),
        active.data(), linear.data(), circular.data(), power.data(), output.data(),
        scratch.data(), &stats);
    if (status != 0) {
        std::cerr << "kernel status " << status << '\n';
        return 1;
    }
    const double first = (0.8 * 2.0 + 0.5 * 3.0) * std::exp(-0.4 * (0.3 + 0.7));
    const double second = (0.8 * -1.0 + 0.5 * 4.0) * std::exp(-0.4 * (0.3 + 0.1));
    const double tolerance = 2.0e-15;
    if (std::abs(output[2] - 0.3 * first) > tolerance ||
        std::abs(output[3] - 0.4 * first) > tolerance ||
        std::abs(output[nodes + 2] - 0.3 * second) > tolerance ||
        std::abs(output[nodes + 3] - 0.4 * second) > tolerance ||
        stats.block_applications != 1 ||
        stats.gathered_source_coefficients != 2 ||
        stats.scattered_receiver_coefficients != 2) {
        std::cerr << "retained-rank result mismatch\n";
        return 2;
    }
    pft_retained_plan* plan = nullptr;
    if (pft_retained_plan_create_f64(
            nodes, blocks.size(), blocks.data(), source_nodes.size(),
            source_nodes.data(), source_factors.data(), receiver_nodes.size(),
            receiver_nodes.data(), receiver_factors.data(), &plan) != 0 ||
        plan == nullptr) {
        std::cerr << "plan creation failed\n";
        return 3;
    }
    output.fill(0.0);
    const int plan_status = pft_retained_plan_apply_f64(
        plan, modes, active.data(), linear.data(), circular.data(), power.data(),
        output.data(), scratch.data(), &stats);
    pft_retained_plan_destroy(plan);
    if (plan_status != 0 ||
        std::abs(output[2] - 0.3 * first) > tolerance ||
        std::abs(output[nodes + 3] - 0.4 * second) > tolerance) {
        std::cerr << "compiled-plan result mismatch\n";
        return 4;
    }
    // A newly inserted or fully occluded node can have no transport blocks.
    // Its bound mode plan must still apply the zero operator successfully.
    pft_retained_plan* empty_plan = nullptr;
    pft_retained_mode_plan* empty_modes = nullptr;
    if (pft_retained_plan_create_f64(nodes, 0, nullptr, 0, nullptr, nullptr,
            0, nullptr, nullptr, &empty_plan) != 0 ||
        pft_retained_mode_plan_create_f64(empty_plan, modes, linear.data(),
            circular.data(), &empty_modes) != 0) return 5;
    output.fill(17.0);
    const int empty_status = pft_retained_mode_plan_apply_f64(empty_modes,
        active.data(), power.data(), output.data(), scratch.data(), &stats);
    pft_retained_mode_plan_destroy(empty_modes);
    pft_retained_plan_destroy(empty_plan);
    if (empty_status != 0 || stats.block_applications != 0) return 6;
    for (double value : output) if (value != 0.0) return 7;
    std::cout << "retained transport self-test: ok ("
              << pft_retained_backend() << ")\n";
    return 0;
}
