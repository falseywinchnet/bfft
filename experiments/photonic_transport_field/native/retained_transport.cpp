#include "retained_transport.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

inline double dot_contiguous(
    const double* PFT_RESTRICT factor,
    const double* PFT_RESTRICT value,
    size_t count) noexcept {
    size_t index = 0;
    double sum = 0.0;
#if defined(__aarch64__)
    float64x2_t accumulator = vdupq_n_f64(0.0);
    for (; index + 2 <= count; index += 2) {
        accumulator = vfmaq_f64(
            accumulator,
            vld1q_f64(factor + index),
            vld1q_f64(value + index));
    }
    sum = vaddvq_f64(accumulator);
#endif
    for (; index < count; ++index) {
        sum += factor[index] * value[index];
    }
    return sum;
}

inline double dot_indexed(
    const uint32_t* PFT_RESTRICT node,
    const double* PFT_RESTRICT factor,
    const double* PFT_RESTRICT value,
    size_t count) noexcept {
    size_t index = 0;
    double sum = 0.0;
#if defined(__aarch64__)
    float64x2_t accumulator = vdupq_n_f64(0.0);
    for (; index + 2 <= count; index += 2) {
        const float64x2_t gathered = {
            value[node[index]], value[node[index + 1]]};
        accumulator = vfmaq_f64(
            accumulator, vld1q_f64(factor + index), gathered);
    }
    sum = vaddvq_f64(accumulator);
#endif
    for (; index < count; ++index) {
        sum += factor[index] * value[node[index]];
    }
    return sum;
}

inline void scatter_contiguous(
    double* PFT_RESTRICT destination,
    const double* PFT_RESTRICT factor,
    size_t count,
    double gathered) noexcept {
    size_t index = 0;
#if defined(__aarch64__)
    const float64x2_t value = vdupq_n_f64(gathered);
    for (; index + 2 <= count; index += 2) {
        vst1q_f64(
            destination + index,
            vfmaq_f64(
                vld1q_f64(destination + index),
                vld1q_f64(factor + index),
                value));
    }
#endif
    for (; index < count; ++index) {
        destination[index] += factor[index] * gathered;
    }
}

inline void scatter_indexed(
    double* PFT_RESTRICT destination,
    const uint32_t* PFT_RESTRICT node,
    const double* PFT_RESTRICT factor,
    size_t count,
    double gathered) noexcept {
    for (size_t index = 0; index < count; ++index) {
        destination[node[index]] += factor[index] * gathered;
    }
}

}  // namespace

struct pft_retained_plan {
    size_t node_count = 0;
    std::vector<pft_retained_block_f64> blocks;
    std::vector<uint32_t> source_nodes;
    std::vector<double> source_factors;
    std::vector<uint32_t> receiver_nodes;
    std::vector<double> receiver_factors;
};

struct pft_retained_mode_plan {
    const pft_retained_plan* plan = nullptr;
    size_t mode_count = 0;
    std::vector<double> extinction;
};

namespace {

int apply_impl(
    size_t node_count,
    size_t mode_count,
    size_t block_count,
    const pft_retained_block_f64* PFT_RESTRICT blocks,
    const uint32_t* PFT_RESTRICT source_nodes,
    const double* PFT_RESTRICT source_factors,
    const uint32_t* PFT_RESTRICT receiver_nodes,
    const double* PFT_RESTRICT receiver_factors,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats,
    bool validate_static,
    const double* PFT_RESTRICT retained_extinction) {
    if ((node_count && (!active_nodes || !powers || !output)) ||
        (mode_count &&
         (!scratch ||
          (block_count && !retained_extinction && (!linear_peakedness || !chirality)))) ||
        (block_count &&
         (!blocks || !source_nodes || !source_factors ||
          !receiver_nodes || !receiver_factors))) {
        return 1;
    }
    if (node_count > std::numeric_limits<uint32_t>::max()) {
        return 2;
    }
    std::memset(output, 0, node_count * mode_count * sizeof(double));
    pft_retained_stats local{};
    for (size_t block_index = 0; block_index < block_count; ++block_index) {
        const pft_retained_block_f64& block = blocks[block_index];
        const uint32_t* block_sources = source_nodes + block.source_offset;
        const double* block_source_factors = source_factors + block.source_offset;
        const uint32_t* block_receivers = receiver_nodes + block.receiver_offset;
        const double* block_receiver_factors = receiver_factors + block.receiver_offset;
        bool active = false;
        for (size_t index = 0; index < block.source_count; ++index) {
            if (validate_static &&
                (block_sources[index] >= node_count ||
                 !std::isfinite(block_source_factors[index]))) {
                return 3;
            }
            if (active_nodes[block_sources[index]]) {
                active = true;
                ++local.gathered_source_coefficients;
            }
        }
        if (validate_static) {
            for (size_t index = 0; index < block.receiver_count; ++index) {
                if (block_receivers[index] >= node_count ||
                    !std::isfinite(block_receiver_factors[index])) {
                    return 4;
                }
            }
        }
        if (!active) {
            continue;
        }
        bool nonzero = false;
        double transported_l1 = 0.0;
        for (size_t mode = 0; mode < mode_count; ++mode) {
            const double* mode_power = powers + mode * node_count;
            double gathered;
            if (block.flags & PFT_BLOCK_SOURCE_CONTIGUOUS) {
                if (validate_static &&
                    static_cast<size_t>(block.source_base) + block.source_count >
                        node_count) {
                    return 5;
                }
                gathered = dot_contiguous(
                    block_source_factors,
                    mode_power + block.source_base,
                    block.source_count);
            } else {
                gathered = dot_indexed(
                    block_sources,
                    block_source_factors,
                    mode_power,
                    block.source_count);
            }
            double retention;
            if (retained_extinction) {
                retention = retained_extinction[block_index * mode_count + mode];
            } else {
                const double mode_extinction =
                    block.extinction +
                    block.linear_diffusive_extinction * linear_peakedness[mode] +
                    block.circular_diffusive_extinction * std::abs(chirality[mode]);
                retention = std::exp(-block.optical_depth * mode_extinction);
            }
            const double value = gathered * retention;
            if (!std::isfinite(value)) {
                return 6;
            }
            scratch[mode] = value;
            transported_l1 += std::abs(value);
            nonzero = nonzero || value != 0.0;
        }
        if (!nonzero) {
            continue;
        }
        ++local.block_applications;
        local.scattered_receiver_coefficients += block.receiver_count;
        if (block.flags & PFT_BLOCK_SOURCE_CONTIGUOUS) {
            ++local.contiguous_source_blocks;
        }
        if (block.flags & PFT_BLOCK_RECEIVER_CONTIGUOUS) {
            if (validate_static &&
                static_cast<size_t>(block.receiver_base) + block.receiver_count >
                    node_count) {
                return 7;
            }
            ++local.contiguous_receiver_blocks;
        }
        if (transported_l1 == 0.0) {
            continue;
        }
        for (size_t mode = 0; mode < mode_count; ++mode) {
            const double gathered = scratch[mode];
            if (gathered == 0.0) {
                continue;
            }
            double* mode_output = output + mode * node_count;
            if (block.flags & PFT_BLOCK_RECEIVER_CONTIGUOUS) {
                scatter_contiguous(
                    mode_output + block.receiver_base,
                    block_receiver_factors,
                    block.receiver_count,
                    gathered);
            } else {
                scatter_indexed(
                    mode_output,
                    block_receivers,
                    block_receiver_factors,
                    block.receiver_count,
                    gathered);
            }
        }
    }
    if (stats) {
        *stats = local;
    }
    return 0;
}

}  // namespace

extern "C" int pft_retained_apply_f64(
    size_t node_count,
    size_t mode_count,
    size_t block_count,
    const pft_retained_block_f64* PFT_RESTRICT blocks,
    const uint32_t* PFT_RESTRICT source_nodes,
    const double* PFT_RESTRICT source_factors,
    const uint32_t* PFT_RESTRICT receiver_nodes,
    const double* PFT_RESTRICT receiver_factors,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats) {
    return apply_impl(
        node_count, mode_count, block_count, blocks, source_nodes,
        source_factors, receiver_nodes, receiver_factors, active_nodes,
        linear_peakedness, chirality, powers, output, scratch, stats, true,
        nullptr);
}

extern "C" int pft_retained_plan_create_f64(
    size_t node_count,
    size_t block_count,
    const pft_retained_block_f64* PFT_RESTRICT blocks,
    size_t source_coefficient_count,
    const uint32_t* PFT_RESTRICT source_nodes,
    const double* PFT_RESTRICT source_factors,
    size_t receiver_coefficient_count,
    const uint32_t* PFT_RESTRICT receiver_nodes,
    const double* PFT_RESTRICT receiver_factors,
    pft_retained_plan** PFT_RESTRICT output_plan) {
    if (!output_plan || !node_count || node_count > std::numeric_limits<uint32_t>::max() ||
        (block_count && !blocks) ||
        (source_coefficient_count && (!source_nodes || !source_factors)) ||
        (receiver_coefficient_count && (!receiver_nodes || !receiver_factors))) {
        return 1;
    }
    *output_plan = nullptr;
    for (size_t block_index = 0; block_index < block_count; ++block_index) {
        const auto& block = blocks[block_index];
        if (static_cast<size_t>(block.source_offset) + block.source_count >
                source_coefficient_count ||
            static_cast<size_t>(block.receiver_offset) + block.receiver_count >
                receiver_coefficient_count ||
            !std::isfinite(block.optical_depth) || block.optical_depth < 0.0 ||
            !std::isfinite(block.extinction) || block.extinction < 0.0 ||
            !std::isfinite(block.linear_diffusive_extinction) ||
            block.linear_diffusive_extinction < 0.0 ||
            !std::isfinite(block.circular_diffusive_extinction) ||
            block.circular_diffusive_extinction < 0.0) {
            return 2;
        }
        for (size_t index = 0; index < block.source_count; ++index) {
            const size_t position = block.source_offset + index;
            if (source_nodes[position] >= node_count ||
                !std::isfinite(source_factors[position])) {
                return 3;
            }
        }
        for (size_t index = 0; index < block.receiver_count; ++index) {
            const size_t position = block.receiver_offset + index;
            if (receiver_nodes[position] >= node_count ||
                !std::isfinite(receiver_factors[position])) {
                return 4;
            }
        }
    }
    try {
        auto plan = std::make_unique<pft_retained_plan>();
        plan->node_count = node_count;
        if (block_count) {
            plan->blocks.assign(blocks, blocks + block_count);
        }
        if (source_coefficient_count) {
            plan->source_nodes.assign(
                source_nodes, source_nodes + source_coefficient_count);
            plan->source_factors.assign(
                source_factors, source_factors + source_coefficient_count);
        }
        if (receiver_coefficient_count) {
            plan->receiver_nodes.assign(
                receiver_nodes, receiver_nodes + receiver_coefficient_count);
            plan->receiver_factors.assign(
                receiver_factors, receiver_factors + receiver_coefficient_count);
        }
        *output_plan = plan.release();
    } catch (const std::bad_alloc&) {
        return 5;
    }
    return 0;
}

extern "C" void pft_retained_plan_destroy(pft_retained_plan* plan) {
    delete plan;
}

extern "C" int pft_retained_plan_apply_f64(
    const pft_retained_plan* plan,
    size_t mode_count,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats) {
    if (!plan) {
        return 1;
    }
    return apply_impl(
        plan->node_count, mode_count, plan->blocks.size(), plan->blocks.data(),
        plan->source_nodes.data(), plan->source_factors.data(),
        plan->receiver_nodes.data(), plan->receiver_factors.data(), active_nodes,
        linear_peakedness, chirality, powers, output, scratch, stats, false,
        nullptr);
}

extern "C" int pft_retained_mode_plan_create_f64(
    const pft_retained_plan* plan,
    size_t mode_count,
    const double* PFT_RESTRICT linear_peakedness,
    const double* PFT_RESTRICT chirality,
    pft_retained_mode_plan** PFT_RESTRICT output_plan) {
    if (!plan || !mode_count || !linear_peakedness || !chirality || !output_plan) {
        return 1;
    }
    *output_plan = nullptr;
    try {
        auto mode_plan = std::make_unique<pft_retained_mode_plan>();
        mode_plan->plan = plan;
        mode_plan->mode_count = mode_count;
        mode_plan->extinction.resize(plan->blocks.size() * mode_count);
        for (size_t mode = 0; mode < mode_count; ++mode) {
            if (!std::isfinite(linear_peakedness[mode]) ||
                !std::isfinite(chirality[mode])) {
                return 2;
            }
        }
        for (size_t block_index = 0; block_index < plan->blocks.size(); ++block_index) {
            const auto& block = plan->blocks[block_index];
            for (size_t mode = 0; mode < mode_count; ++mode) {
                const double mode_extinction =
                    block.extinction +
                    block.linear_diffusive_extinction * linear_peakedness[mode] +
                    block.circular_diffusive_extinction * std::abs(chirality[mode]);
                mode_plan->extinction[block_index * mode_count + mode] =
                    std::exp(-block.optical_depth * mode_extinction);
            }
        }
        *output_plan = mode_plan.release();
    } catch (const std::bad_alloc&) {
        return 3;
    }
    return 0;
}

extern "C" void pft_retained_mode_plan_destroy(pft_retained_mode_plan* plan) {
    delete plan;
}

extern "C" int pft_retained_mode_plan_apply_f64(
    const pft_retained_mode_plan* mode_plan,
    const uint8_t* PFT_RESTRICT active_nodes,
    const double* PFT_RESTRICT powers,
    double* PFT_RESTRICT output,
    double* PFT_RESTRICT scratch,
    pft_retained_stats* PFT_RESTRICT stats) {
    if (!mode_plan || !mode_plan->plan) {
        return 1;
    }
    const auto* plan = mode_plan->plan;
    return apply_impl(
        plan->node_count, mode_plan->mode_count, plan->blocks.size(),
        plan->blocks.data(), plan->source_nodes.data(),
        plan->source_factors.data(), plan->receiver_nodes.data(),
        plan->receiver_factors.data(), active_nodes, nullptr, nullptr, powers,
        output, scratch, stats, false, mode_plan->extinction.data());
}

extern "C" const char* pft_retained_backend(void) {
#if defined(__aarch64__)
    return "arm64-neon-f64";
#elif defined(__AVX2__)
    return "x86-avx2-f64";
#else
    return "portable-f64";
#endif
}
