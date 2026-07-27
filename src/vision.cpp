#include <bfft/vision.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

bool checked_product(std::size_t a, std::size_t b, std::size_t* result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

bool valid_cell(std::int32_t index, std::size_t cell_count) {
    return index >= 0 && static_cast<std::size_t>(index) < cell_count;
}

bool valid_slot(std::int64_t index, std::size_t block_count) {
    return index >= 0 && static_cast<std::size_t>(index) < block_count;
}

}  // namespace

extern "C" {

bfft_status bfft_vision_assemble_normal(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t basis_width,
    std::size_t normal_block_count,
    const std::int32_t* owner,
    const std::int32_t* runner,
    const std::uint8_t* has_runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* target,
    const std::int64_t* diagonal_block,
    const std::int64_t* owner_runner_block,
    const std::int64_t* runner_owner_block,
    double* normal_blocks,
    double* rhs) {
    if (pixel_count == 0 || cell_count == 0 || basis_width == 0 ||
        normal_block_count == 0 || owner == nullptr || runner == nullptr ||
        has_runner == nullptr || owner_weight == nullptr ||
        runner_weight == nullptr || owner_basis == nullptr ||
        runner_basis == nullptr || target == nullptr ||
        diagonal_block == nullptr || owner_runner_block == nullptr ||
        runner_owner_block == nullptr || normal_blocks == nullptr ||
        rhs == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    std::size_t block_area = 0;
    std::size_t normal_size = 0;
    std::size_t rhs_width = 0;
    std::size_t rhs_size = 0;
    if (!checked_product(basis_width, basis_width, &block_area) ||
        !checked_product(normal_block_count, block_area, &normal_size) ||
        !checked_product(basis_width, std::size_t{3}, &rhs_width) ||
        !checked_product(cell_count, rhs_width, &rhs_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (!valid_slot(diagonal_block[cell], normal_block_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }
    for (std::size_t p = 0; p < pixel_count; ++p) {
        if (!valid_cell(owner[p], cell_count) ||
            !valid_cell(runner[p], cell_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        if (has_runner[p] != 0 &&
            (!valid_slot(owner_runner_block[p], normal_block_count) ||
             !valid_slot(runner_owner_block[p], normal_block_count))) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    std::fill(normal_blocks, normal_blocks + normal_size, 0.0);
    std::fill(rhs, rhs + rhs_size, 0.0);

    for (std::size_t p = 0; p < pixel_count; ++p) {
        const std::size_t i = static_cast<std::size_t>(owner[p]);
        const double* first = owner_basis + p * basis_width;
        const double* target_p = target + p * 3;
        const double wi = owner_weight[p];
        double* diagonal_i =
            normal_blocks +
            static_cast<std::size_t>(diagonal_block[i]) * block_area;
        double* rhs_i = rhs + i * rhs_width;

        for (std::size_t a = 0; a < basis_width; ++a) {
            const double ua = wi * first[a];
            double* diagonal_row = diagonal_i + a * basis_width;
            for (std::size_t b = 0; b < basis_width; ++b) {
                diagonal_row[b] += ua * (wi * first[b]);
            }
            double* rhs_row = rhs_i + a * 3;
            rhs_row[0] += ua * target_p[0];
            rhs_row[1] += ua * target_p[1];
            rhs_row[2] += ua * target_p[2];
        }

        if (has_runner[p] == 0) {
            continue;
        }

        const std::size_t j = static_cast<std::size_t>(runner[p]);
        const double* second = runner_basis + p * basis_width;
        const double wj = runner_weight[p];
        double* diagonal_j =
            normal_blocks +
            static_cast<std::size_t>(diagonal_block[j]) * block_area;
        double* cross_ij =
            normal_blocks +
            static_cast<std::size_t>(owner_runner_block[p]) * block_area;
        double* cross_ji =
            normal_blocks +
            static_cast<std::size_t>(runner_owner_block[p]) * block_area;
        double* rhs_j = rhs + j * rhs_width;

        for (std::size_t a = 0; a < basis_width; ++a) {
            const double ua = wi * first[a];
            const double va = wj * second[a];
            double* diagonal_row = diagonal_j + a * basis_width;
            double* cross_ij_row = cross_ij + a * basis_width;
            double* cross_ji_row = cross_ji + a * basis_width;
            for (std::size_t b = 0; b < basis_width; ++b) {
                const double ub = wi * first[b];
                const double vb = wj * second[b];
                diagonal_row[b] += va * vb;
                cross_ij_row[b] += ua * vb;
                cross_ji_row[b] += va * ub;
            }
            double* rhs_row = rhs_j + a * 3;
            rhs_row[0] += va * target_p[0];
            rhs_row[1] += va * target_p[1];
            rhs_row[2] += va * target_p[2];
        }
    }
    return BFFT_OK;
}

bfft_status bfft_vision_render_affine(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t basis_width,
    const std::int32_t* owner,
    const std::int32_t* runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* coeff,
    double* owner_prediction,
    double* runner_prediction,
    double* field) {
    if (pixel_count == 0 || cell_count == 0 || basis_width == 0 ||
        owner == nullptr || runner == nullptr || owner_weight == nullptr ||
        runner_weight == nullptr || owner_basis == nullptr ||
        runner_basis == nullptr || coeff == nullptr ||
        owner_prediction == nullptr || runner_prediction == nullptr ||
        field == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t p = 0; p < pixel_count; ++p) {
        if (!valid_cell(owner[p], cell_count) ||
            !valid_cell(runner[p], cell_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    std::size_t coefficient_stride = 0;
    if (!checked_product(basis_width, std::size_t{3},
                         &coefficient_stride)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    for (std::size_t p = 0; p < pixel_count; ++p) {
        const std::size_t i = static_cast<std::size_t>(owner[p]);
        const std::size_t j = static_cast<std::size_t>(runner[p]);
        const double* first = owner_basis + p * basis_width;
        const double* second = runner_basis + p * basis_width;
        const double* coeff_i = coeff + i * coefficient_stride;
        const double* coeff_j = coeff + j * coefficient_stride;
        double* pred_i = owner_prediction + p * 3;
        double* pred_j = runner_prediction + p * 3;
        double* result = field + p * 3;

        for (std::size_t channel = 0; channel < 3; ++channel) {
            double left = 0.0;
            double right = 0.0;
            for (std::size_t a = 0; a < basis_width; ++a) {
                left += coeff_i[a * 3 + channel] * first[a];
                right += coeff_j[a * 3 + channel] * second[a];
            }
            pred_i[channel] = left;
            pred_j[channel] = right;
            result[channel] =
                owner_weight[p] * left + runner_weight[p] * right;
        }
    }
    return BFFT_OK;
}

bfft_status bfft_vision_scan_residual_ridges(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t angle_count,
    std::size_t bin_count,
    double spacing,
    double span,
    const std::int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* dx,
    const double* dy,
    const double* angle_cos,
    const double* angle_sin,
    const double* channel_weight,
    double* score,
    std::int32_t* best_angle,
    std::int32_t* best_bin) {
    if (pixel_count == 0 || cell_count == 0 || angle_count == 0 ||
        bin_count == 0 || !std::isfinite(spacing) || spacing <= 0.0 ||
        !std::isfinite(span) || span <= 0.0 ||
        angle_count >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        bin_count >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        owner == nullptr || pixel_weight == nullptr || residual == nullptr ||
        dx == nullptr || dy == nullptr || angle_cos == nullptr ||
        angle_sin == nullptr || channel_weight == nullptr || score == nullptr ||
        best_angle == nullptr || best_bin == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t p = 0; p < pixel_count; ++p) {
        if (!valid_cell(owner[p], cell_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    std::size_t angle_bins = 0;
    std::size_t cell_angle_bins = 0;
    std::size_t accumulator_size = 0;
    if (!checked_product(angle_count, bin_count, &angle_bins) ||
        !checked_product(cell_count, angle_bins, &cell_angle_bins) ||
        !checked_product(cell_angle_bins, std::size_t{3},
                         &accumulator_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<double> accumulator(accumulator_size, 0.0);
        std::vector<double> mass(cell_count, 0.0);
        std::vector<double> total(cell_count * 3, 0.0);
        const double scale = static_cast<double>(bin_count) / (2.0 * span);

        for (std::size_t p = 0; p < pixel_count; ++p) {
            const std::size_t cell = static_cast<std::size_t>(owner[p]);
            const double phi = pixel_weight[p];
            const double r0 = phi * residual[p * 3];
            const double r1 = phi * residual[p * 3 + 1];
            const double r2 = phi * residual[p * 3 + 2];
            mass[cell] += phi;
            total[cell * 3] += r0;
            total[cell * 3 + 1] += r1;
            total[cell * 3 + 2] += r2;

            const double px = dx[p] / spacing;
            const double py = dy[p] / spacing;
            const std::size_t cell_base = cell * angle_bins * 3;
            for (std::size_t angle = 0; angle < angle_count; ++angle) {
                const double projection =
                    px * angle_cos[angle] + py * angle_sin[angle];
                std::int64_t bin = static_cast<std::int64_t>(
                    (projection + span) * scale);
                bin = std::max<std::int64_t>(0, bin);
                bin = std::min<std::int64_t>(
                    static_cast<std::int64_t>(bin_count - 1), bin);
                const std::size_t slot =
                    cell_base + (angle * bin_count +
                                 static_cast<std::size_t>(bin)) * 3;
                accumulator[slot] += r0;
                accumulator[slot + 1] += r1;
                accumulator[slot + 2] += r2;
            }
        }

        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const double cell_mass = std::max(mass[cell], 1e-9);
            const double total0 = total[cell * 3];
            const double total1 = total[cell * 3 + 1];
            const double total2 = total[cell * 3 + 2];
            const std::size_t cell_base = cell * angle_bins * 3;
            bool seen = false;
            double top = 0.0;
            std::size_t top_angle = 0;
            std::size_t top_bin = 0;

            for (std::size_t angle = 0; angle < angle_count; ++angle) {
                double run0 = 0.0;
                double run1 = 0.0;
                double run2 = 0.0;
                double local_top = 0.0;
                std::size_t local_bin = 0;
                for (std::size_t bin = 0; bin < bin_count; ++bin) {
                    const std::size_t slot =
                        cell_base + (angle * bin_count + bin) * 3;
                    run0 += accumulator[slot];
                    run1 += accumulator[slot + 1];
                    run2 += accumulator[slot + 2];
                    const double c0 = total0 - 2.0 * run0;
                    const double c1 = total1 - 2.0 * run1;
                    const double c2 = total2 - 2.0 * run2;
                    const double value =
                        (channel_weight[0] * c0 * c0 +
                         channel_weight[1] * c1 * c1 +
                         channel_weight[2] * c2 * c2) /
                        cell_mass;
                    if (bin == 0 || value > local_top) {
                        local_top = value;
                        local_bin = bin;
                    }
                }
                if (!seen || local_top > top) {
                    seen = true;
                    top = local_top;
                    top_angle = angle;
                    top_bin = local_bin;
                }
            }
            score[cell] = top;
            best_angle[cell] = static_cast<std::int32_t>(top_angle);
            best_bin[cell] = static_cast<std::int32_t>(top_bin);
        }
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_support_forward(
    std::size_t sample_count, std::size_t pixel_count,
    std::size_t cell_count, const std::int32_t* rows,
    const std::int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y,
    const double* coefficient, double* pixel) {
    if (sample_count == 0 || pixel_count == 0 || cell_count == 0 ||
        rows == nullptr || sites == nullptr || weight == nullptr ||
        basis_x == nullptr || basis_y == nullptr || coefficient == nullptr ||
        pixel == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    std::fill(pixel, pixel + pixel_count, 0.0);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::int32_t row = rows[sample];
        const std::int32_t site = sites[sample];
        if (row < 0 || static_cast<std::size_t>(row) >= pixel_count ||
            !valid_cell(site, cell_count))
            return BFFT_ERROR_INVALID_ARGUMENT;
        const double* c =
            coefficient + 3 * static_cast<std::size_t>(site);
        pixel[static_cast<std::size_t>(row)] += weight[sample] * (
            c[0] + basis_x[sample] * c[1] + basis_y[sample] * c[2]);
    }
    return BFFT_OK;
}

bfft_status bfft_vision_support_transpose(
    std::size_t sample_count, std::size_t pixel_count,
    std::size_t cell_count, const std::int32_t* rows,
    const std::int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* pixel,
    double* coefficient) {
    if (sample_count == 0 || pixel_count == 0 || cell_count == 0 ||
        rows == nullptr || sites == nullptr || weight == nullptr ||
        basis_x == nullptr || basis_y == nullptr || pixel == nullptr ||
        coefficient == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    std::fill(coefficient, coefficient + 3 * cell_count, 0.0);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::int32_t row = rows[sample];
        const std::int32_t site = sites[sample];
        if (row < 0 || static_cast<std::size_t>(row) >= pixel_count ||
            !valid_cell(site, cell_count))
            return BFFT_ERROR_INVALID_ARGUMENT;
        const double value =
            weight[sample] * pixel[static_cast<std::size_t>(row)];
        double* c = coefficient + 3 * static_cast<std::size_t>(site);
        c[0] += value;
        c[1] += value * basis_x[sample];
        c[2] += value * basis_y[sample];
    }
    return BFFT_OK;
}

bfft_status bfft_vision_support_normal_apply(
    std::size_t sample_count, std::size_t pixel_count,
    std::size_t cell_count, const std::int32_t* rows,
    const std::int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y,
    const double* coefficient, double* pixel_scratch, double* output) {
    const bfft_status forward = bfft_vision_support_forward(
        sample_count, pixel_count, cell_count, rows, sites, weight,
        basis_x, basis_y, coefficient, pixel_scratch);
    if (forward != BFFT_OK) return forward;
    return bfft_vision_support_transpose(
        sample_count, pixel_count, cell_count, rows, sites, weight,
        basis_x, basis_y, pixel_scratch, output);
}

}  // extern "C"
