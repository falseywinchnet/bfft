#include <bfft/vision.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "vision_kernels: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void require_close(const std::vector<double>& actual,
                   const std::vector<double>& expected,
                   double tolerance,
                   const std::string& label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    double largest = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        largest = std::max(largest, std::abs(actual[i] - expected[i]));
    }
    if (largest > tolerance) {
        fail(label + " max absolute error " + std::to_string(largest));
    }
}

void test_normal_assembly_and_render() {
    constexpr std::size_t pixels = 37;
    constexpr std::size_t cells = 5;
    constexpr std::size_t width = 4;
    constexpr std::size_t channels = 3;
    constexpr std::size_t blocks = cells * cells;
    constexpr std::size_t block_area = width * width;
    constexpr std::size_t rhs_stride = width * channels;

    std::mt19937_64 rng(0xBFF7u);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> fraction(0.05, 0.95);

    std::vector<std::int32_t> owner(pixels);
    std::vector<std::int32_t> runner(pixels);
    std::vector<std::uint8_t> has_runner(pixels);
    std::vector<double> w1(pixels);
    std::vector<double> w2(pixels);
    std::vector<double> first(pixels * width);
    std::vector<double> second(pixels * width);
    std::vector<double> target(pixels * channels);
    std::vector<std::int64_t> diagonal(cells);
    std::vector<std::int64_t> forward(pixels, -1);
    std::vector<std::int64_t> reverse(pixels, -1);

    for (std::size_t cell = 0; cell < cells; ++cell) {
        diagonal[cell] = static_cast<std::int64_t>(cell * cells + cell);
    }
    for (std::size_t p = 0; p < pixels; ++p) {
        owner[p] = static_cast<std::int32_t>((3 * p + 1) % cells);
        has_runner[p] = static_cast<std::uint8_t>((p % 6) != 0);
        if (has_runner[p] != 0) {
            runner[p] =
                static_cast<std::int32_t>((owner[p] + 1 + p % 3) % cells);
            if (runner[p] == owner[p]) {
                runner[p] = static_cast<std::int32_t>(
                    (static_cast<std::size_t>(runner[p]) + 1) % cells);
            }
            const std::size_t i = static_cast<std::size_t>(owner[p]);
            const std::size_t j = static_cast<std::size_t>(runner[p]);
            forward[p] = static_cast<std::int64_t>(i * cells + j);
            reverse[p] = static_cast<std::int64_t>(j * cells + i);
            w1[p] = fraction(rng);
            w2[p] = 1.0 - w1[p];
        } else {
            runner[p] = owner[p];
            w1[p] = 1.0;
            w2[p] = 0.0;
        }
        for (std::size_t a = 0; a < width; ++a) {
            first[p * width + a] = value(rng);
            second[p * width + a] =
                has_runner[p] != 0 ? value(rng) : first[p * width + a];
        }
        for (std::size_t channel = 0; channel < channels; ++channel) {
            target[p * channels + channel] = value(rng);
        }
    }

    std::vector<double> normal(blocks * block_area,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> rhs(cells * rhs_stride,
                            std::numeric_limits<double>::quiet_NaN());
    require(bfft_vision_assemble_normal(
                pixels, cells, width, blocks, owner.data(), runner.data(),
                has_runner.data(), w1.data(), w2.data(), first.data(),
                second.data(), target.data(), diagonal.data(), forward.data(),
                reverse.data(), normal.data(), rhs.data()) == BFFT_OK,
            "normal assembly returned an error");

    std::vector<double> normal_reference(blocks * block_area, 0.0);
    std::vector<double> rhs_reference(cells * rhs_stride, 0.0);
    for (std::size_t p = 0; p < pixels; ++p) {
        const std::size_t i = static_cast<std::size_t>(owner[p]);
        const std::size_t j = static_cast<std::size_t>(runner[p]);
        for (std::size_t a = 0; a < width; ++a) {
            const double u = w1[p] * first[p * width + a];
            for (std::size_t b = 0; b < width; ++b) {
                normal_reference[
                    (i * cells + i) * block_area + a * width + b] +=
                    u * (w1[p] * first[p * width + b]);
            }
            for (std::size_t channel = 0; channel < channels; ++channel) {
                rhs_reference[i * rhs_stride + a * channels + channel] +=
                    u * target[p * channels + channel];
            }
        }
        if (has_runner[p] == 0) {
            continue;
        }
        for (std::size_t a = 0; a < width; ++a) {
            const double u = w1[p] * first[p * width + a];
            const double v = w2[p] * second[p * width + a];
            for (std::size_t b = 0; b < width; ++b) {
                const double ub = w1[p] * first[p * width + b];
                const double vb = w2[p] * second[p * width + b];
                normal_reference[
                    (j * cells + j) * block_area + a * width + b] += v * vb;
                normal_reference[
                    (i * cells + j) * block_area + a * width + b] += u * vb;
                normal_reference[
                    (j * cells + i) * block_area + a * width + b] += v * ub;
            }
            for (std::size_t channel = 0; channel < channels; ++channel) {
                rhs_reference[j * rhs_stride + a * channels + channel] +=
                    v * target[p * channels + channel];
            }
        }
    }
    require_close(normal, normal_reference, 2e-15, "normal blocks");
    require_close(rhs, rhs_reference, 2e-15, "right-hand side");

    std::vector<double> coeff(cells * width * channels);
    for (double& entry : coeff) {
        entry = value(rng);
    }
    std::vector<double> pred_owner(pixels * channels);
    std::vector<double> pred_runner(pixels * channels);
    std::vector<double> field(pixels * channels);
    require(bfft_vision_render_affine(
                pixels, cells, width, owner.data(), runner.data(), w1.data(),
                w2.data(), first.data(), second.data(), coeff.data(),
                pred_owner.data(), pred_runner.data(), field.data()) == BFFT_OK,
            "affine render returned an error");

    std::vector<double> pred_owner_reference(pixels * channels, 0.0);
    std::vector<double> pred_runner_reference(pixels * channels, 0.0);
    std::vector<double> field_reference(pixels * channels, 0.0);
    for (std::size_t p = 0; p < pixels; ++p) {
        const std::size_t i = static_cast<std::size_t>(owner[p]);
        const std::size_t j = static_cast<std::size_t>(runner[p]);
        for (std::size_t channel = 0; channel < channels; ++channel) {
            double left = 0.0;
            double right = 0.0;
            for (std::size_t a = 0; a < width; ++a) {
                left += coeff[(i * width + a) * channels + channel] *
                        first[p * width + a];
                right += coeff[(j * width + a) * channels + channel] *
                         second[p * width + a];
            }
            pred_owner_reference[p * channels + channel] = left;
            pred_runner_reference[p * channels + channel] = right;
            field_reference[p * channels + channel] =
                w1[p] * left + w2[p] * right;
        }
    }
    require_close(pred_owner, pred_owner_reference, 0.0, "owner prediction");
    require_close(pred_runner, pred_runner_reference, 0.0,
                  "runner prediction");
    require_close(field, field_reference, 0.0, "rendered field");

    const std::int32_t saved = owner[3];
    owner[3] = static_cast<std::int32_t>(cells);
    require(bfft_vision_assemble_normal(
                pixels, cells, width, blocks, owner.data(), runner.data(),
                has_runner.data(), w1.data(), w2.data(), first.data(),
                second.data(), target.data(), diagonal.data(), forward.data(),
                reverse.data(), normal.data(), rhs.data()) ==
                BFFT_ERROR_INVALID_ARGUMENT,
            "invalid owner was accepted");
    owner[3] = saved;
}

void test_residual_ridge_scan() {
    constexpr std::size_t pixels = 43;
    constexpr std::size_t cells = 4;
    constexpr std::size_t angles = 7;
    constexpr std::size_t bins = 13;
    constexpr double spacing = 2.75;
    constexpr double span = 2.5;

    std::mt19937_64 rng(0x51D6Eu);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> positive(0.1, 1.0);
    std::vector<std::int32_t> owner(pixels);
    std::vector<double> weight(pixels);
    std::vector<double> residual(pixels * 3);
    std::vector<double> dx(pixels);
    std::vector<double> dy(pixels);
    std::vector<double> cosine(angles);
    std::vector<double> sine(angles);
    const std::vector<double> channel_weight{1.0, 0.6, 0.35};

    for (std::size_t p = 0; p < pixels; ++p) {
        owner[p] = static_cast<std::int32_t>((p * 7 + 2) % cells);
        weight[p] = positive(rng);
        dx[p] = 5.0 * value(rng);
        dy[p] = 5.0 * value(rng);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            residual[p * 3 + channel] = value(rng);
        }
    }
    for (std::size_t angle = 0; angle < angles; ++angle) {
        const double theta =
            std::acos(-1.0) * static_cast<double>(angle) /
            static_cast<double>(angles);
        cosine[angle] = std::cos(theta);
        sine[angle] = std::sin(theta);
    }

    std::vector<double> score(cells);
    std::vector<std::int32_t> best_angle(cells);
    std::vector<std::int32_t> best_bin(cells);
    require(bfft_vision_scan_residual_ridges(
                pixels, cells, angles, bins, spacing, span, owner.data(),
                weight.data(), residual.data(), dx.data(), dy.data(),
                cosine.data(), sine.data(), channel_weight.data(), score.data(),
                best_angle.data(), best_bin.data()) == BFFT_OK,
            "ridge scan returned an error");

    std::vector<double> score_reference(cells, 0.0);
    std::vector<std::int32_t> angle_reference(cells, 0);
    std::vector<std::int32_t> bin_reference(cells, 0);
    const double scale = static_cast<double>(bins) / (2.0 * span);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        double mass = 0.0;
        double total[3] = {0.0, 0.0, 0.0};
        for (std::size_t p = 0; p < pixels; ++p) {
            if (static_cast<std::size_t>(owner[p]) != cell) {
                continue;
            }
            mass += weight[p];
            for (std::size_t channel = 0; channel < 3; ++channel) {
                total[channel] +=
                    weight[p] * residual[p * 3 + channel];
            }
        }
        mass = std::max(mass, 1e-9);
        bool seen = false;
        for (std::size_t angle = 0; angle < angles; ++angle) {
            std::vector<double> histogram(bins * 3, 0.0);
            for (std::size_t p = 0; p < pixels; ++p) {
                if (static_cast<std::size_t>(owner[p]) != cell) {
                    continue;
                }
                const double projection =
                    (dx[p] * cosine[angle] + dy[p] * sine[angle]) / spacing;
                std::int64_t bin = static_cast<std::int64_t>(
                    (projection + span) * scale);
                bin = std::max<std::int64_t>(0, bin);
                bin = std::min<std::int64_t>(
                    static_cast<std::int64_t>(bins - 1), bin);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    histogram[static_cast<std::size_t>(bin) * 3 + channel] +=
                        weight[p] * residual[p * 3 + channel];
                }
            }
            double run[3] = {0.0, 0.0, 0.0};
            double local_top = 0.0;
            std::size_t local_bin = 0;
            for (std::size_t bin = 0; bin < bins; ++bin) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    run[channel] += histogram[bin * 3 + channel];
                }
                const double c0 = total[0] - 2.0 * run[0];
                const double c1 = total[1] - 2.0 * run[1];
                const double c2 = total[2] - 2.0 * run[2];
                const double candidate =
                    (channel_weight[0] * c0 * c0 +
                     channel_weight[1] * c1 * c1 +
                     channel_weight[2] * c2 * c2) /
                    mass;
                if (bin == 0 || candidate > local_top) {
                    local_top = candidate;
                    local_bin = bin;
                }
            }
            if (!seen || local_top > score_reference[cell]) {
                seen = true;
                score_reference[cell] = local_top;
                angle_reference[cell] = static_cast<std::int32_t>(angle);
                bin_reference[cell] = static_cast<std::int32_t>(local_bin);
            }
        }
    }

    require_close(score, score_reference, 2e-15, "ridge score");
    require(best_angle == angle_reference, "ridge angle differs");
    require(best_bin == bin_reference, "ridge threshold differs");
}

}  // namespace

int main() {
    test_normal_assembly_and_render();
    test_residual_ridge_scan();
    std::cout << "vision kernels: scalar-reference agreement passed\n";
    return 0;
}
