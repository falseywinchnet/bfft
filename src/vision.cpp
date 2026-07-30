#include <bfft/vision.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <thread>
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

bfft_status bfft_vision_scan_paired_offsets(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t bin_count,
    double span,
    const std::int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* projection,
    const double* channel_weight,
    double* score,
    std::int32_t* best_bin) {
    if (pixel_count == 0 || cell_count == 0 || bin_count == 0 ||
        !std::isfinite(span) || span <= 0.0 ||
        bin_count >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        owner == nullptr || pixel_weight == nullptr || residual == nullptr ||
        projection == nullptr || channel_weight == nullptr || score == nullptr ||
        best_bin == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<std::size_t> offset(cell_count + 1, 0);
        for (std::size_t p = 0; p < pixel_count; ++p) {
            if (!valid_cell(owner[p], cell_count) ||
                !std::isfinite(pixel_weight[p]) ||
                !std::isfinite(projection[p])) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            ++offset[static_cast<std::size_t>(owner[p]) + 1];
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            offset[cell + 1] += offset[cell];
        }
        std::vector<std::size_t> cursor(offset.begin(), offset.end() - 1);
        std::vector<std::size_t> order(pixel_count);
        for (std::size_t p = 0; p < pixel_count; ++p) {
            const std::size_t cell = static_cast<std::size_t>(owner[p]);
            order[cursor[cell]++] = p;
        }

        std::size_t histogram_size = 0;
        if (!checked_product(bin_count, std::size_t{3}, &histogram_size)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        std::vector<double> histogram(histogram_size, 0.0);
        const double scale = static_cast<double>(bin_count) / (2.0 * span);

        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            std::fill(histogram.begin(), histogram.end(), 0.0);
            double mass = 0.0;
            double total0 = 0.0;
            double total1 = 0.0;
            double total2 = 0.0;
            for (std::size_t at = offset[cell]; at < offset[cell + 1]; ++at) {
                const std::size_t p = order[at];
                const double phi = pixel_weight[p];
                const double r0 = phi * residual[p * 3];
                const double r1 = phi * residual[p * 3 + 1];
                const double r2 = phi * residual[p * 3 + 2];
                mass += phi;
                total0 += r0;
                total1 += r1;
                total2 += r2;
                std::int64_t bin = static_cast<std::int64_t>(
                    (projection[p] + span) * scale);
                bin = std::max<std::int64_t>(0, bin);
                bin = std::min<std::int64_t>(
                    static_cast<std::int64_t>(bin_count - 1), bin);
                const std::size_t slot = static_cast<std::size_t>(bin) * 3;
                histogram[slot] += r0;
                histogram[slot + 1] += r1;
                histogram[slot + 2] += r2;
            }

            const double denominator = std::max(mass, 1e-9);
            double run0 = 0.0;
            double run1 = 0.0;
            double run2 = 0.0;
            double top = 0.0;
            std::size_t top_bin = 0;
            bool seen = false;
            for (std::size_t bin = 0; bin < bin_count; ++bin) {
                const std::size_t slot = bin * 3;
                run0 += histogram[slot];
                run1 += histogram[slot + 1];
                run2 += histogram[slot + 2];
                const double contrast0 = total0 - 2.0 * run0;
                const double contrast1 = total1 - 2.0 * run1;
                const double contrast2 = total2 - 2.0 * run2;
                const double value =
                    (channel_weight[0] * contrast0 * contrast0 +
                     channel_weight[1] * contrast1 * contrast1 +
                     channel_weight[2] * contrast2 * contrast2) /
                    denominator;
                if (!seen || value > top) {
                    top = value;
                    top_bin = bin;
                    seen = true;
                }
            }
            score[cell] = top;
            best_bin[cell] = static_cast<std::int32_t>(top_bin);
        }
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
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

bfft_status bfft_vision_curvature_population_f32(
    std::size_t height, std::size_t width,
    const float* precision_xx, const float* precision_xy,
    const float* precision_yy, const float* base_measure,
    double base_implied_cells,
    float* corrected_measure, float* director_curvature,
    float* sagitta_ratio, float* population_factor,
    double* corrected_implied_cells) {
    if (height < 2 || width < 2 ||
        precision_xx == nullptr || precision_xy == nullptr ||
        precision_yy == nullptr || base_measure == nullptr ||
        corrected_measure == nullptr || director_curvature == nullptr ||
        sagitta_ratio == nullptr || population_factor == nullptr ||
        corrected_implied_cells == nullptr ||
        !std::isfinite(base_implied_cells) || base_implied_cells <= 0.0) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t pixels = 0;
    if (!checked_product(height, width, &pixels)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<double> doubled_cosine(pixels);
        std::vector<double> doubled_sine(pixels);
        std::vector<double> corrected_raw(pixels);
        for (std::size_t p = 0; p < pixels; ++p) {
            const double qxx = static_cast<double>(precision_xx[p]);
            const double qxy = static_cast<double>(precision_xy[p]);
            const double qyy = static_cast<double>(precision_yy[p]);
            const double measure = static_cast<double>(base_measure[p]);
            if (!std::isfinite(qxx) || !std::isfinite(qxy) ||
                !std::isfinite(qyy) || !std::isfinite(measure) ||
                measure < 0.0) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            const double discriminant =
                std::hypot(qxx - qyy, 2.0 * qxy);
            const double safe = std::max(discriminant, 1e-30);
            doubled_cosine[p] = (qxx - qyy) / safe;
            doubled_sine[p] = 2.0 * qxy / safe;
        }

        const auto difference_x = [width](
            const std::vector<double>& field,
            std::size_t y, std::size_t x) {
            const std::size_t p = y * width + x;
            if (x == 0) return field[p + 1] - field[p];
            if (x + 1 == width) return field[p] - field[p - 1];
            return 0.5 * (field[p + 1] - field[p - 1]);
        };
        const auto difference_y = [height, width](
            const std::vector<double>& field,
            std::size_t y, std::size_t x) {
            const std::size_t p = y * width + x;
            if (y == 0) return field[p + width] - field[p];
            if (y + 1 == height) return field[p] - field[p - width];
            return 0.5 * (field[p + width] - field[p - width]);
        };

        double implied = 0.0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t p = y * width + x;
                const double qxx = static_cast<double>(precision_xx[p]);
                const double qxy = static_cast<double>(precision_xy[p]);
                const double qyy = static_cast<double>(precision_yy[p]);
                const double trace = qxx + qyy;
                const double discriminant =
                    std::hypot(qxx - qyy, 2.0 * qxy);
                const double coherence =
                    discriminant / std::max(trace, 1e-30);
                const double u = doubled_cosine[p];
                const double v = doubled_sine[p];
                const double theta_x = 0.5 * (
                    u * difference_x(doubled_sine, y, x) -
                    v * difference_x(doubled_cosine, y, x));
                const double theta_y = 0.5 * (
                    u * difference_y(doubled_sine, y, x) -
                    v * difference_y(doubled_cosine, y, x));
                const double normal_x =
                    std::sqrt(std::max(0.5 * (1.0 + u), 0.0));
                const double sign_source =
                    std::abs(v) > 1e-30 ? v : 1.0;
                const double normal_y = std::copysign(
                    std::sqrt(std::max(0.5 * (1.0 - u), 0.0)),
                    sign_source);
                const double tangent_x = -normal_y;
                const double tangent_y = normal_x;
                const double curvature = coherence * std::abs(
                    tangent_x * theta_x + tangent_y * theta_y);
                const double high = std::max(
                    0.5 * (trace + discriminant), 1e-30);
                const double low = std::max(
                    0.5 * (trace - discriminant), 1e-30);
                const double tangent_span = 1.0 / std::sqrt(low);
                const double normal_span = 1.0 / std::sqrt(high);
                const double ratio =
                    curvature * tangent_span * tangent_span /
                    std::max(2.0 * normal_span, 1e-30);
                const double factor = std::sqrt(std::max(1.0, ratio));
                const double raw =
                    static_cast<double>(base_measure[p]) *
                    base_implied_cells * factor;
                corrected_raw[p] = raw;
                implied += raw;
                director_curvature[p] = static_cast<float>(curvature);
                sagitta_ratio[p] = static_cast<float>(ratio);
                population_factor[p] = static_cast<float>(factor);
            }
        }
        if (!std::isfinite(implied) || implied <= 0.0) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        const double inverse_implied = 1.0 / implied;
        for (std::size_t p = 0; p < pixels; ++p) {
            corrected_measure[p] = static_cast<float>(
                corrected_raw[p] * inverse_implied);
        }
        *corrected_implied_cells = implied;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_soft_support_diffuse(
    std::size_t height, std::size_t width, std::size_t channels,
    std::size_t passes, std::size_t thread_count, double coupling,
    const double* field,
    const double* horizontal, const double* vertical,
    const double* diagonal_down_right, const double* diagonal_down_left,
    double* output, double* scratch) {
    if (height < 2 || width < 2 || channels == 0 ||
        !std::isfinite(coupling) || coupling < 0.0 ||
        field == nullptr || horizontal == nullptr || vertical == nullptr ||
        diagonal_down_right == nullptr || diagonal_down_left == nullptr ||
        output == nullptr || scratch == nullptr || output == scratch) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t pixels = 0;
    std::size_t values = 0;
    if (!checked_product(height, width, &pixels) ||
        !checked_product(pixels, channels, &values)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::copy(field, field + values, output);
    if (passes == 0 || coupling == 0.0) {
        return BFFT_OK;
    }

    try {
    const std::size_t horizontal_stride = width - 1;
    std::vector<double> inverse_denominator(pixels);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            double denominator = 1.0;
            if (x > 0)
                denominator += coupling *
                    horizontal[y * horizontal_stride + x - 1];
            if (x + 1 < width)
                denominator += coupling *
                    horizontal[y * horizontal_stride + x];
            if (y > 0)
                denominator += coupling * vertical[(y - 1) * width + x];
            if (y + 1 < height)
                denominator += coupling * vertical[y * width + x];
            if (x > 0 && y > 0)
                denominator += coupling * diagonal_down_right[
                    (y - 1) * horizontal_stride + x - 1];
            if (x + 1 < width && y + 1 < height)
                denominator += coupling * diagonal_down_right[
                    y * horizontal_stride + x];
            if (x + 1 < width && y > 0)
                denominator += coupling * diagonal_down_left[
                    (y - 1) * horizontal_stride + x];
            if (x > 0 && y + 1 < height)
                denominator += coupling * diagonal_down_left[
                    y * horizontal_stride + x - 1];
            inverse_denominator[y * width + x] = 1.0 / denominator;
        }
    }

    double* source = output;
    double* target = scratch;
    std::size_t workers = thread_count == 0 ? 1 : thread_count;
    workers = std::min(workers, height);
    workers = std::min<std::size_t>(workers, 64);
    const auto process_rows = [&](std::size_t y_begin, std::size_t y_end) {
        for (std::size_t y = y_begin; y < y_end; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t p = y * width + x;
                const std::size_t base = p * channels;
                for (std::size_t channel = 0;
                     channel < channels; ++channel) {
                    double numerator = source[base + channel];
                    if (x > 0)
                        numerator += coupling *
                            horizontal[y * horizontal_stride + x - 1] *
                            source[(p - 1) * channels + channel];
                    if (x + 1 < width)
                        numerator += coupling *
                            horizontal[y * horizontal_stride + x] *
                            source[(p + 1) * channels + channel];
                    if (y > 0)
                        numerator += coupling *
                            vertical[(y - 1) * width + x] *
                            source[(p - width) * channels + channel];
                    if (y + 1 < height)
                        numerator += coupling *
                            vertical[y * width + x] *
                            source[(p + width) * channels + channel];
                    if (x > 0 && y > 0)
                        numerator += coupling * diagonal_down_right[
                            (y - 1) * horizontal_stride + x - 1] *
                            source[(p - width - 1) * channels + channel];
                    if (x + 1 < width && y + 1 < height)
                        numerator += coupling * diagonal_down_right[
                            y * horizontal_stride + x] *
                            source[(p + width + 1) * channels + channel];
                    if (x + 1 < width && y > 0)
                        numerator += coupling * diagonal_down_left[
                            (y - 1) * horizontal_stride + x] *
                            source[(p - width + 1) * channels + channel];
                    if (x > 0 && y + 1 < height)
                        numerator += coupling * diagonal_down_left[
                            y * horizontal_stride + x - 1] *
                            source[(p + width - 1) * channels + channel];
                    target[base + channel] =
                        numerator * inverse_denominator[p];
                }
            }
        }
    };
    for (std::size_t pass = 0; pass < passes; ++pass) {
        if (workers == 1) {
            process_rows(0, height);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t y_begin = height * worker / workers;
                const std::size_t y_end = height * (worker + 1) / workers;
                pool.emplace_back(process_rows, y_begin, y_end);
            }
            for (auto& worker : pool) {
                worker.join();
            }
        }
        std::swap(source, target);
    }
    if (source != output) {
        std::copy(source, source + values, output);
    }
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_fast_march_first_label(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int32_t* seed_pixel,
    const double* seed_value,
    const std::int32_t* seed_label,
    const double* seed_gradient_x,
    const double* seed_gradient_y,
    const std::int32_t* directions,
    const double* direction_costs,
    const std::uint8_t* direction_valid,
    const double* cardinal_costs,
    const std::int64_t* inverse_offset,
    std::size_t inverse_count,
    const std::int32_t* inverse_receiver,
    const double* mxx,
    const double* mxy,
    const double* myy,
    std::int32_t* owner,
    double* distance,
    double* gradient_x,
    double* gradient_y,
    double* source_gradient_x,
    double* source_gradient_y,
    std::int32_t* parent_first,
    std::int32_t* parent_second,
    double* parent_fraction,
    std::int32_t* acceptance_order,
    std::size_t* accepted_count,
    std::size_t* push_count,
    std::size_t* maximum_heap_size) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        seed_pixel == nullptr || seed_value == nullptr ||
        seed_label == nullptr || seed_gradient_x == nullptr ||
        seed_gradient_y == nullptr || directions == nullptr ||
        direction_costs == nullptr || direction_valid == nullptr ||
        cardinal_costs == nullptr || inverse_offset == nullptr ||
        (inverse_count != 0 && inverse_receiver == nullptr) ||
        mxx == nullptr || mxy == nullptr || myy == nullptr ||
        owner == nullptr || distance == nullptr ||
        push_count == nullptr || maximum_heap_size == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    const bool full_output = gradient_x != nullptr;
    if (
        (gradient_y != nullptr) != full_output ||
        (source_gradient_x != nullptr) != full_output ||
        (source_gradient_y != nullptr) != full_output ||
        (parent_first != nullptr) != full_output ||
        (parent_second != nullptr) != full_output ||
        (parent_fraction != nullptr) != full_output ||
        (acceptance_order != nullptr) != full_output ||
        (accepted_count != nullptr) != full_output
    ) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (inverse_offset[0] != 0 ||
        inverse_offset[pixels] < 0 ||
        static_cast<std::size_t>(inverse_offset[pixels]) != inverse_count) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    /*
       The Python boundary constructs this CSR once and owns its shape checks.
       Do not rescan every pixel and incidence on every global walk: that
       validation alone is an image-sized pass through the hottest API.
    */
    for (std::size_t seed = 0; seed < seed_count; ++seed) {
        if (seed_pixel[seed] < 0 ||
            static_cast<std::size_t>(seed_pixel[seed]) >= pixels ||
            seed_label[seed] < 0) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    const auto metric_norm = [](
        double dx, double dy, double a, double b, double c) {
        return std::sqrt(std::max(
            a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
            1e-30));
    };
    const auto simplex_candidate = [&metric_norm](
        double first_value,
        double second_value,
        double first_x,
        double first_y,
        double second_x,
        double second_y,
        double a,
        double b,
        double c,
        double* fraction) {
        const double delta_value = second_value - first_value;
        const double delta_x = second_x - first_x;
        const double delta_y = second_y - first_y;
        const double quadratic_a =
            a * delta_x * delta_x +
            2.0 * b * delta_x * delta_y +
            c * delta_y * delta_y;
        const double quadratic_b =
            delta_x * (a * first_x + b * first_y) +
            delta_y * (b * first_x + c * first_y);
        const double quadratic_c =
            a * first_x * first_x +
            2.0 * b * first_x * first_y +
            c * first_y * first_y;
        const double first_length =
            std::sqrt(std::max(quadratic_c, 1e-30));
        const double derivative_first =
            delta_value + quadratic_b / first_length;
        if (derivative_first >= 0.0) {
            *fraction = 0.0;
            return first_value + first_length;
        }
        const double second_quadratic =
            quadratic_a + 2.0 * quadratic_b + quadratic_c;
        const double second_length =
            std::sqrt(std::max(second_quadratic, 1e-30));
        const double derivative_second =
            delta_value +
            (quadratic_a + quadratic_b) / second_length;
        if (derivative_second <= 0.0) {
            *fraction = 1.0;
            return second_value + second_length;
        }
        const double area = std::max(
            quadratic_a * quadratic_c - quadratic_b * quadratic_b,
            0.0);
        const double causal = std::max(
            quadratic_a - delta_value * delta_value,
            1e-30);
        double t = (
            -quadratic_b -
            delta_value * std::sqrt(area / causal)
        ) / std::max(quadratic_a, 1e-30);
        t = std::min(std::max(t, 0.0), 1.0);
        const double rx = first_x + t * delta_x;
        const double ry = first_y + t * delta_y;
        *fraction = t;
        return (
            first_value +
            t * delta_value +
            metric_norm(rx, ry, a, b, c));
    };

    try {
        std::vector<double> tentative(pixels, infinity);
        std::vector<std::int32_t> tentative_label(pixels, -1);
        std::vector<double> tentative_gradient_x(pixels, 0.0);
        std::vector<double> tentative_gradient_y(pixels, 0.0);
        std::vector<double> tentative_source_gradient_x(pixels, 0.0);
        std::vector<double> tentative_source_gradient_y(pixels, 0.0);
        std::vector<std::int32_t> tentative_parent_first(pixels, -1);
        std::vector<std::int32_t> tentative_parent_second(pixels, -1);
        std::vector<double> tentative_parent_fraction(pixels, 0.0);
        std::vector<std::uint8_t> accepted(pixels, 0);
        std::vector<double> heap_value(pixels);
        std::vector<std::int32_t> heap_pixel(pixels);
        std::vector<std::int32_t> heap_position(pixels, -1);

        std::fill(distance, distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});
        if (full_output) {
            std::fill(gradient_x, gradient_x + pixels, 0.0);
            std::fill(gradient_y, gradient_y + pixels, 0.0);
            std::fill(source_gradient_x, source_gradient_x + pixels, 0.0);
            std::fill(source_gradient_y, source_gradient_y + pixels, 0.0);
            std::fill(parent_first, parent_first + pixels, std::int32_t{-1});
            std::fill(parent_second, parent_second + pixels, std::int32_t{-1});
            std::fill(parent_fraction, parent_fraction + pixels, 0.0);
        }

        std::size_t heap_size = 0;
        std::size_t pushes = 0;
        std::size_t maximum_heap = 0;
        std::size_t acceptance_size = 0;

        const auto bubble_up = [&] (std::size_t child) {
            while (child > 0) {
                const std::size_t parent = (child - 1) / 2;
                if (heap_value[parent] <= heap_value[child]) {
                    break;
                }
                std::swap(heap_value[parent], heap_value[child]);
                std::swap(heap_pixel[parent], heap_pixel[child]);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[parent])] = static_cast<std::int32_t>(parent);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[child])] = static_cast<std::int32_t>(child);
                child = parent;
            }
        };

        for (std::size_t seed = 0; seed < seed_count; ++seed) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[seed]);
            const double value = seed_value[seed];
            if (value >= tentative[pixel]) {
                continue;
            }
            tentative[pixel] = value;
            tentative_label[pixel] = seed_label[seed];
            tentative_gradient_x[pixel] = seed_gradient_x[seed];
            tentative_gradient_y[pixel] = seed_gradient_y[seed];
            tentative_source_gradient_x[pixel] = -seed_gradient_x[seed];
            tentative_source_gradient_y[pixel] = -seed_gradient_y[seed];
            std::int32_t child_position = heap_position[pixel];
            if (child_position < 0) {
                child_position = static_cast<std::int32_t>(heap_size);
                heap_pixel[heap_size] = static_cast<std::int32_t>(pixel);
                heap_position[pixel] = child_position;
                ++heap_size;
            }
            const std::size_t child =
                static_cast<std::size_t>(child_position);
            heap_value[child] = value;
            ++pushes;
            maximum_heap = std::max(maximum_heap, heap_size);
            bubble_up(child);
        }

        static constexpr int cardinal_x[4] = {1, -1, 0, 0};
        static constexpr int cardinal_y[4] = {0, 0, 1, -1};
        while (heap_size > 0) {
            const double value = heap_value[0];
            const std::size_t pixel =
                static_cast<std::size_t>(heap_pixel[0]);
            --heap_size;
            heap_position[pixel] = -2;
            if (heap_size > 0) {
                heap_value[0] = heap_value[heap_size];
                heap_pixel[0] = heap_pixel[heap_size];
                heap_position[static_cast<std::size_t>(heap_pixel[0])] = 0;
                std::size_t node = 0;
                for (;;) {
                    const std::size_t left = 2 * node + 1;
                    const std::size_t right = left + 1;
                    std::size_t smallest = node;
                    if (left < heap_size &&
                        heap_value[left] < heap_value[smallest]) {
                        smallest = left;
                    }
                    if (right < heap_size &&
                        heap_value[right] < heap_value[smallest]) {
                        smallest = right;
                    }
                    if (smallest == node) {
                        break;
                    }
                    std::swap(heap_value[node], heap_value[smallest]);
                    std::swap(heap_pixel[node], heap_pixel[smallest]);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[node])] = static_cast<std::int32_t>(node);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[smallest])] =
                            static_cast<std::int32_t>(smallest);
                    node = smallest;
                }
            }

            const std::int32_t label = tentative_label[pixel];
            accepted[pixel] = 1;
            distance[pixel] = value;
            owner[pixel] = label;
            if (full_output) {
                gradient_x[pixel] = tentative_gradient_x[pixel];
                gradient_y[pixel] = tentative_gradient_y[pixel];
                source_gradient_x[pixel] =
                    tentative_source_gradient_x[pixel];
                source_gradient_y[pixel] =
                    tentative_source_gradient_y[pixel];
                parent_first[pixel] = tentative_parent_first[pixel];
                parent_second[pixel] = tentative_parent_second[pixel];
                parent_fraction[pixel] = tentative_parent_fraction[pixel];
                acceptance_order[acceptance_size] =
                    static_cast<std::int32_t>(pixel);
            }
            ++acceptance_size;

            const std::size_t begin =
                static_cast<std::size_t>(inverse_offset[pixel]);
            const std::size_t end =
                static_cast<std::size_t>(inverse_offset[pixel + 1]);
            for (std::size_t incidence = begin;
                 incidence < end; ++incidence) {
                const std::size_t receiver =
                    static_cast<std::size_t>(inverse_receiver[incidence]);
                if (accepted[receiver] != 0) {
                    continue;
                }
                const std::size_t ry = receiver / width;
                const std::size_t rx = receiver - ry * width;
                const double a = mxx[receiver];
                const double b = mxy[receiver];
                const double c = myy[receiver];
                double best_value = tentative[receiver];
                std::int32_t best_label = tentative_label[receiver];
                double best_gradient_x = tentative_gradient_x[receiver];
                double best_gradient_y = tentative_gradient_y[receiver];
                double best_source_gradient_x =
                    tentative_source_gradient_x[receiver];
                double best_source_gradient_y =
                    tentative_source_gradient_y[receiver];
                std::int32_t best_parent_first =
                    tentative_parent_first[receiver];
                std::int32_t best_parent_second =
                    tentative_parent_second[receiver];
                double best_parent_fraction =
                    tentative_parent_fraction[receiver];

                const std::size_t cardinal_base = receiver * 4;
                for (std::size_t cardinal = 0;
                     cardinal < 4; ++cardinal) {
                    const int ux = cardinal_x[cardinal];
                    const int uy = cardinal_y[cardinal];
                    const std::int64_t nx =
                        static_cast<std::int64_t>(rx) + ux;
                    const std::int64_t ny =
                        static_cast<std::int64_t>(ry) + uy;
                    if (nx < 0 || nx >= static_cast<std::int64_t>(width) ||
                        ny < 0 || ny >= static_cast<std::int64_t>(height)) {
                        continue;
                    }
                    const std::size_t neighbour =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    if (accepted[neighbour] == 0) {
                        continue;
                    }
                    const double candidate =
                        distance[neighbour] +
                        cardinal_costs[cardinal_base + cardinal];
                    if (candidate < best_value) {
                        best_value = candidate;
                        best_label = owner[neighbour];
                        const double local_length =
                            metric_norm(ux, uy, a, b, c);
                        best_gradient_x =
                            -(a * ux + b * uy) / local_length;
                        best_gradient_y =
                            -(b * ux + c * uy) / local_length;
                        if (full_output) {
                            best_source_gradient_x =
                                source_gradient_x[neighbour];
                            best_source_gradient_y =
                                source_gradient_y[neighbour];
                        }
                        best_parent_first =
                            static_cast<std::int32_t>(neighbour);
                        best_parent_second = -1;
                        best_parent_fraction = 0.0;
                    }
                }

                const std::size_t direction_base = receiver * 6;
                for (std::size_t direction = 0;
                     direction < 6; ++direction) {
                    const std::size_t next = (direction + 1) % 6;
                    const std::size_t first_vector =
                        (direction_base + direction) * 2;
                    const std::size_t second_vector =
                        (direction_base + next) * 2;
                    const int ux = directions[first_vector];
                    const int uy = directions[first_vector + 1];
                    const int vx = directions[second_vector];
                    const int vy = directions[second_vector + 1];
                    const std::int64_t first_x =
                        static_cast<std::int64_t>(rx) + ux;
                    const std::int64_t first_y =
                        static_cast<std::int64_t>(ry) + uy;
                    const std::int64_t second_x =
                        static_cast<std::int64_t>(rx) + vx;
                    const std::int64_t second_y =
                        static_cast<std::int64_t>(ry) + vy;
                    const bool first_inside =
                        first_x >= 0 &&
                        first_x < static_cast<std::int64_t>(width) &&
                        first_y >= 0 &&
                        first_y < static_cast<std::int64_t>(height);
                    const bool second_inside =
                        second_x >= 0 &&
                        second_x < static_cast<std::int64_t>(width) &&
                        second_y >= 0 &&
                        second_y < static_cast<std::int64_t>(height);
                    const std::size_t first_pixel = first_inside
                        ? static_cast<std::size_t>(first_y) * width +
                            static_cast<std::size_t>(first_x)
                        : pixels;
                    const std::size_t second_pixel = second_inside
                        ? static_cast<std::size_t>(second_y) * width +
                            static_cast<std::size_t>(second_x)
                        : pixels;

                    if (first_inside && accepted[first_pixel] != 0) {
                        const double candidate =
                            distance[first_pixel] +
                            direction_costs[direction_base + direction];
                        if (direction_valid[
                                direction_base + direction] != 0 &&
                            candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                            const double local_length =
                                metric_norm(ux, uy, a, b, c);
                            best_gradient_x =
                                -(a * ux + b * uy) / local_length;
                            best_gradient_y =
                                -(b * ux + c * uy) / local_length;
                            if (full_output) {
                                best_source_gradient_x =
                                    source_gradient_x[first_pixel];
                                best_source_gradient_y =
                                    source_gradient_y[first_pixel];
                            }
                            best_parent_first =
                                static_cast<std::int32_t>(first_pixel);
                            best_parent_second = -1;
                            best_parent_fraction = 0.0;
                        }
                    }
                    if (second_inside && accepted[second_pixel] != 0) {
                        const double candidate =
                            distance[second_pixel] +
                            direction_costs[direction_base + next];
                        if (direction_valid[
                                direction_base + next] != 0 &&
                            candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[second_pixel];
                            const double local_length =
                                metric_norm(vx, vy, a, b, c);
                            best_gradient_x =
                                -(a * vx + b * vy) / local_length;
                            best_gradient_y =
                                -(b * vx + c * vy) / local_length;
                            if (full_output) {
                                best_source_gradient_x =
                                    source_gradient_x[second_pixel];
                                best_source_gradient_y =
                                    source_gradient_y[second_pixel];
                            }
                            best_parent_first =
                                static_cast<std::int32_t>(second_pixel);
                            best_parent_second = -1;
                            best_parent_fraction = 0.0;
                        }
                    }
                    if (first_inside && second_inside &&
                        accepted[first_pixel] != 0 &&
                        accepted[second_pixel] != 0 &&
                        owner[first_pixel] == owner[second_pixel] &&
                        direction_valid[
                            direction_base + direction] != 0 &&
                        direction_valid[direction_base + next] != 0) {
                        double fraction = 0.0;
                        const double candidate = simplex_candidate(
                            distance[first_pixel],
                            distance[second_pixel],
                            ux,
                            uy,
                            vx,
                            vy,
                            a,
                            b,
                            c,
                            &fraction);
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                            const double foot_x =
                                ux + fraction * (vx - ux);
                            const double foot_y =
                                uy + fraction * (vy - uy);
                            const double local_length =
                                metric_norm(foot_x, foot_y, a, b, c);
                            best_gradient_x = -(
                                a * foot_x + b * foot_y) / local_length;
                            best_gradient_y = -(
                                b * foot_x + c * foot_y) / local_length;
                            if (full_output) {
                                best_source_gradient_x =
                                    (1.0 - fraction) *
                                        source_gradient_x[first_pixel] +
                                    fraction *
                                        source_gradient_x[second_pixel];
                                best_source_gradient_y =
                                    (1.0 - fraction) *
                                        source_gradient_y[first_pixel] +
                                    fraction *
                                        source_gradient_y[second_pixel];
                            }
                            best_parent_first =
                                static_cast<std::int32_t>(first_pixel);
                            best_parent_second =
                                static_cast<std::int32_t>(second_pixel);
                            best_parent_fraction = fraction;
                        }
                    }
                }

                if (best_value + 1e-12 >= tentative[receiver]) {
                    continue;
                }
                tentative[receiver] = best_value;
                tentative_label[receiver] = best_label;
                tentative_gradient_x[receiver] = best_gradient_x;
                tentative_gradient_y[receiver] = best_gradient_y;
                tentative_source_gradient_x[receiver] =
                    best_source_gradient_x;
                tentative_source_gradient_y[receiver] =
                    best_source_gradient_y;
                tentative_parent_first[receiver] = best_parent_first;
                tentative_parent_second[receiver] = best_parent_second;
                tentative_parent_fraction[receiver] = best_parent_fraction;
                std::int32_t child_position = heap_position[receiver];
                if (child_position < 0) {
                    child_position = static_cast<std::int32_t>(heap_size);
                    heap_pixel[heap_size] =
                        static_cast<std::int32_t>(receiver);
                    heap_position[receiver] = child_position;
                    ++heap_size;
                }
                const std::size_t child =
                    static_cast<std::size_t>(child_position);
                heap_value[child] = best_value;
                ++pushes;
                maximum_heap = std::max(maximum_heap, heap_size);
                bubble_up(child);
            }
        }

        if (full_output) {
            *accepted_count = acceptance_size;
        }
        *push_count = pushes;
        *maximum_heap_size = maximum_heap;
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_fast_march_labels(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int32_t* seed_pixel,
    const double* seed_value,
    const std::int32_t* seed_label,
    const double* seed_gradient_x,
    const double* seed_gradient_y,
    const std::int32_t* directions,
    const double* direction_costs,
    const std::uint8_t* direction_valid,
    const double* cardinal_costs,
    const std::int64_t* inverse_offset,
    std::size_t inverse_count,
    const std::int32_t* inverse_receiver,
    const double* mxx,
    const double* mxy,
    const double* myy,
    std::int32_t* owner,
    double* distance,
    std::size_t* push_count,
    std::size_t* maximum_heap_size) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        seed_pixel == nullptr || seed_value == nullptr ||
        seed_label == nullptr || seed_gradient_x == nullptr ||
        seed_gradient_y == nullptr || directions == nullptr ||
        direction_costs == nullptr || direction_valid == nullptr ||
        cardinal_costs == nullptr || inverse_offset == nullptr ||
        (inverse_count != 0 && inverse_receiver == nullptr) ||
        mxx == nullptr || mxy == nullptr || myy == nullptr ||
        owner == nullptr || distance == nullptr ||
        push_count == nullptr || maximum_heap_size == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (inverse_offset[0] != 0 ||
        inverse_offset[pixels] < 0 ||
        static_cast<std::size_t>(inverse_offset[pixels]) != inverse_count) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t seed = 0; seed < seed_count; ++seed) {
        if (seed_pixel[seed] < 0 ||
            static_cast<std::size_t>(seed_pixel[seed]) >= pixels ||
            seed_label[seed] < 0) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    const auto metric_norm = [](
        double dx, double dy, double a, double b, double c) {
        return std::sqrt(std::max(
            a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
            1e-30));
    };
    const auto simplex_candidate = [&metric_norm](
        double first_value,
        double second_value,
        double first_x,
        double first_y,
        double second_x,
        double second_y,
        double a,
        double b,
        double c) {
        const double delta_value = second_value - first_value;
        const double delta_x = second_x - first_x;
        const double delta_y = second_y - first_y;
        const double quadratic_a =
            a * delta_x * delta_x +
            2.0 * b * delta_x * delta_y +
            c * delta_y * delta_y;
        const double quadratic_b =
            delta_x * (a * first_x + b * first_y) +
            delta_y * (b * first_x + c * first_y);
        const double quadratic_c =
            a * first_x * first_x +
            2.0 * b * first_x * first_y +
            c * first_y * first_y;
        const double first_length =
            std::sqrt(std::max(quadratic_c, 1e-30));
        const double derivative_first =
            delta_value + quadratic_b / first_length;
        if (derivative_first >= 0.0) {
            return first_value + first_length;
        }
        const double second_quadratic =
            quadratic_a + 2.0 * quadratic_b + quadratic_c;
        const double second_length =
            std::sqrt(std::max(second_quadratic, 1e-30));
        const double derivative_second =
            delta_value +
            (quadratic_a + quadratic_b) / second_length;
        if (derivative_second <= 0.0) {
            return second_value + second_length;
        }
        const double area = std::max(
            quadratic_a * quadratic_c - quadratic_b * quadratic_b,
            0.0);
        const double causal = std::max(
            quadratic_a - delta_value * delta_value,
            1e-30);
        double fraction = (
            -quadratic_b -
            delta_value * std::sqrt(area / causal)
        ) / std::max(quadratic_a, 1e-30);
        fraction = std::min(std::max(fraction, 0.0), 1.0);
        const double rx = first_x + fraction * delta_x;
        const double ry = first_y + fraction * delta_y;
        return (
            first_value +
            fraction * delta_value +
            metric_norm(rx, ry, a, b, c));
    };

    try {
        std::vector<double> tentative(pixels, infinity);
        std::vector<std::int32_t> tentative_label(pixels, -1);
        std::vector<std::uint8_t> accepted(pixels, 0);
        std::vector<double> heap_value(pixels);
        std::vector<std::int32_t> heap_pixel(pixels);
        std::vector<std::int32_t> heap_position(pixels, -1);
        std::fill(distance, distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});

        std::size_t heap_size = 0;
        std::size_t pushes = 0;
        std::size_t maximum_heap = 0;
        const auto bubble_up = [&] (std::size_t child) {
            while (child > 0) {
                const std::size_t parent = (child - 1) / 2;
                if (heap_value[parent] <= heap_value[child]) {
                    break;
                }
                std::swap(heap_value[parent], heap_value[child]);
                std::swap(heap_pixel[parent], heap_pixel[child]);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[parent])] = static_cast<std::int32_t>(parent);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[child])] = static_cast<std::int32_t>(child);
                child = parent;
            }
        };

        for (std::size_t seed = 0; seed < seed_count; ++seed) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[seed]);
            const double value = seed_value[seed];
            if (value >= tentative[pixel]) {
                continue;
            }
            tentative[pixel] = value;
            tentative_label[pixel] = seed_label[seed];
            std::int32_t child_position = heap_position[pixel];
            if (child_position < 0) {
                child_position = static_cast<std::int32_t>(heap_size);
                heap_pixel[heap_size] = static_cast<std::int32_t>(pixel);
                heap_position[pixel] = child_position;
                ++heap_size;
            }
            const std::size_t child =
                static_cast<std::size_t>(child_position);
            heap_value[child] = value;
            ++pushes;
            maximum_heap = std::max(maximum_heap, heap_size);
            bubble_up(child);
        }

        static constexpr int cardinal_x[4] = {1, -1, 0, 0};
        static constexpr int cardinal_y[4] = {0, 0, 1, -1};
        while (heap_size > 0) {
            const double value = heap_value[0];
            const std::size_t pixel =
                static_cast<std::size_t>(heap_pixel[0]);
            --heap_size;
            heap_position[pixel] = -2;
            if (heap_size > 0) {
                heap_value[0] = heap_value[heap_size];
                heap_pixel[0] = heap_pixel[heap_size];
                heap_position[static_cast<std::size_t>(heap_pixel[0])] = 0;
                std::size_t node = 0;
                for (;;) {
                    const std::size_t left = 2 * node + 1;
                    const std::size_t right = left + 1;
                    std::size_t smallest = node;
                    if (left < heap_size &&
                        heap_value[left] < heap_value[smallest]) {
                        smallest = left;
                    }
                    if (right < heap_size &&
                        heap_value[right] < heap_value[smallest]) {
                        smallest = right;
                    }
                    if (smallest == node) {
                        break;
                    }
                    std::swap(heap_value[node], heap_value[smallest]);
                    std::swap(heap_pixel[node], heap_pixel[smallest]);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[node])] = static_cast<std::int32_t>(node);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[smallest])] =
                            static_cast<std::int32_t>(smallest);
                    node = smallest;
                }
            }

            accepted[pixel] = 1;
            distance[pixel] = value;
            owner[pixel] = tentative_label[pixel];
            const std::size_t begin =
                static_cast<std::size_t>(inverse_offset[pixel]);
            const std::size_t end =
                static_cast<std::size_t>(inverse_offset[pixel + 1]);
            for (std::size_t incidence = begin;
                 incidence < end; ++incidence) {
                const std::size_t receiver =
                    static_cast<std::size_t>(inverse_receiver[incidence]);
                if (accepted[receiver] != 0) {
                    continue;
                }
                const std::size_t ry = receiver / width;
                const std::size_t rx = receiver - ry * width;
                const double a = mxx[receiver];
                const double b = mxy[receiver];
                const double c = myy[receiver];
                double best_value = tentative[receiver];
                std::int32_t best_label = tentative_label[receiver];

                const std::size_t cardinal_base = receiver * 4;
                for (std::size_t cardinal = 0;
                     cardinal < 4; ++cardinal) {
                    const int ux = cardinal_x[cardinal];
                    const int uy = cardinal_y[cardinal];
                    const std::int64_t nx =
                        static_cast<std::int64_t>(rx) + ux;
                    const std::int64_t ny =
                        static_cast<std::int64_t>(ry) + uy;
                    if (nx < 0 || nx >= static_cast<std::int64_t>(width) ||
                        ny < 0 || ny >= static_cast<std::int64_t>(height)) {
                        continue;
                    }
                    const std::size_t neighbour =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    if (accepted[neighbour] == 0) {
                        continue;
                    }
                    const double candidate =
                        distance[neighbour] +
                        cardinal_costs[cardinal_base + cardinal];
                    if (candidate < best_value) {
                        best_value = candidate;
                        best_label = owner[neighbour];
                    }
                }

                const std::size_t direction_base = receiver * 6;
                for (std::size_t direction = 0;
                     direction < 6; ++direction) {
                    const std::size_t next = (direction + 1) % 6;
                    const std::size_t first_vector =
                        (direction_base + direction) * 2;
                    const std::size_t second_vector =
                        (direction_base + next) * 2;
                    const int ux = directions[first_vector];
                    const int uy = directions[first_vector + 1];
                    const int vx = directions[second_vector];
                    const int vy = directions[second_vector + 1];
                    const std::int64_t first_x =
                        static_cast<std::int64_t>(rx) + ux;
                    const std::int64_t first_y =
                        static_cast<std::int64_t>(ry) + uy;
                    const std::int64_t second_x =
                        static_cast<std::int64_t>(rx) + vx;
                    const std::int64_t second_y =
                        static_cast<std::int64_t>(ry) + vy;
                    const bool first_inside =
                        first_x >= 0 &&
                        first_x < static_cast<std::int64_t>(width) &&
                        first_y >= 0 &&
                        first_y < static_cast<std::int64_t>(height);
                    const bool second_inside =
                        second_x >= 0 &&
                        second_x < static_cast<std::int64_t>(width) &&
                        second_y >= 0 &&
                        second_y < static_cast<std::int64_t>(height);
                    const std::size_t first_pixel = first_inside
                        ? static_cast<std::size_t>(first_y) * width +
                            static_cast<std::size_t>(first_x)
                        : pixels;
                    const std::size_t second_pixel = second_inside
                        ? static_cast<std::size_t>(second_y) * width +
                            static_cast<std::size_t>(second_x)
                        : pixels;

                    if (first_inside && accepted[first_pixel] != 0) {
                        const double candidate =
                            distance[first_pixel] +
                            direction_costs[direction_base + direction];
                        if (direction_valid[
                                direction_base + direction] != 0 &&
                            candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                        }
                    }
                    if (second_inside && accepted[second_pixel] != 0) {
                        const double candidate =
                            distance[second_pixel] +
                            direction_costs[direction_base + next];
                        if (direction_valid[
                                direction_base + next] != 0 &&
                            candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[second_pixel];
                        }
                    }
                    if (first_inside && second_inside &&
                        accepted[first_pixel] != 0 &&
                        accepted[second_pixel] != 0 &&
                        owner[first_pixel] == owner[second_pixel] &&
                        direction_valid[
                            direction_base + direction] != 0 &&
                        direction_valid[direction_base + next] != 0) {
                        const double candidate = simplex_candidate(
                            distance[first_pixel],
                            distance[second_pixel],
                            ux,
                            uy,
                            vx,
                            vy,
                            a,
                            b,
                            c);
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                        }
                    }
                }

                if (best_value + 1e-12 >= tentative[receiver]) {
                    continue;
                }
                tentative[receiver] = best_value;
                tentative_label[receiver] = best_label;
                std::int32_t child_position = heap_position[receiver];
                if (child_position < 0) {
                    child_position = static_cast<std::int32_t>(heap_size);
                    heap_pixel[heap_size] =
                        static_cast<std::int32_t>(receiver);
                    heap_position[receiver] = child_position;
                    ++heap_size;
                }
                const std::size_t child =
                    static_cast<std::size_t>(child_position);
                heap_value[child] = best_value;
                ++pushes;
                maximum_heap = std::max(maximum_heap, heap_size);
                bubble_up(child);
            }
        }
        *push_count = pushes;
        *maximum_heap_size = maximum_heap;
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_metric_edge_costs_f32(
    std::size_t height,
    std::size_t width,
    const float* precision_xx,
    const float* precision_xy,
    const float* precision_yy,
    const float* boundary_xx,
    const float* boundary_xy,
    const float* boundary_yy,
    double precision_gain,
    double boundary_gain,
    float* direction_costs) {
    std::size_t pixels = 0;
    const bool has_boundary = boundary_gain > 0.0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        precision_xx == nullptr || precision_xy == nullptr ||
        precision_yy == nullptr || direction_costs == nullptr ||
        !std::isfinite(precision_gain) || precision_gain < 0.0 ||
        !std::isfinite(boundary_gain) || boundary_gain < 0.0 ||
        (has_boundary && (
            boundary_xx == nullptr || boundary_xy == nullptr ||
            boundary_yy == nullptr))) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    static constexpr int dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static constexpr int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const float infinity = std::numeric_limits<float>::infinity();
    std::fill(
        direction_costs, direction_costs + 8 * pixels, infinity);
    const auto metric_component = [
        precision_gain,
        boundary_gain,
        has_boundary
    ](
        const float* precision,
        const float* boundary,
        std::size_t pixel,
        double identity) {
        double value =
            identity +
            precision_gain * static_cast<double>(precision[pixel]);
        if (has_boundary) {
            value += boundary_gain *
                static_cast<double>(boundary[pixel]);
        }
        return value;
    };
    for (std::size_t direction = 0; direction < 8; ++direction) {
        const int step_x = dx[direction];
        const int step_y = dy[direction];
        const std::size_t y_begin =
            step_y < 0 ? static_cast<std::size_t>(-step_y) : 0;
        const std::size_t y_end =
            step_y > 0 ? height - static_cast<std::size_t>(step_y) : height;
        const std::size_t x_begin =
            step_x < 0 ? static_cast<std::size_t>(-step_x) : 0;
        const std::size_t x_end =
            step_x > 0 ? width - static_cast<std::size_t>(step_x) : width;
        float* output = direction_costs + direction * pixels;
        for (std::size_t y = y_begin; y < y_end; ++y) {
            for (std::size_t x = x_begin; x < x_end; ++x) {
                const std::size_t source = y * width + x;
                const std::size_t receiver =
                    static_cast<std::size_t>(
                        static_cast<std::int64_t>(y) + step_y) * width +
                    static_cast<std::size_t>(
                        static_cast<std::int64_t>(x) + step_x);
                const double source_xx = metric_component(
                    precision_xx, boundary_xx, source, 1.0);
                const double source_xy = metric_component(
                    precision_xy, boundary_xy, source, 0.0);
                const double source_yy = metric_component(
                    precision_yy, boundary_yy, source, 1.0);
                const double receiver_xx = metric_component(
                    precision_xx, boundary_xx, receiver, 1.0);
                const double receiver_xy = metric_component(
                    precision_xy, boundary_xy, receiver, 0.0);
                const double receiver_yy = metric_component(
                    precision_yy, boundary_yy, receiver, 1.0);
                const double a = 0.5 * (source_xx + receiver_xx);
                const double b = 0.5 * (source_xy + receiver_xy);
                const double c = 0.5 * (source_yy + receiver_yy);
                const double quadratic =
                    step_x * step_x * a +
                    2.0 * step_x * step_y * b +
                    step_y * step_y * c;
                output[source] = static_cast<float>(
                    std::sqrt(std::max(quadratic, 1e-8)));
            }
        }
    }
    return BFFT_OK;
}

bfft_status bfft_vision_bucket_first_label(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int64_t* seed_pixel,
    const double* reach,
    const float* direction_costs,
    double delta,
    std::size_t span,
    double shift,
    std::int32_t* owner,
    double* distance,
    std::int32_t* parent,
    std::size_t* push_count) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        seed_pixel == nullptr || reach == nullptr ||
        direction_costs == nullptr || !(delta > 0.0) ||
        span == 0 || owner == nullptr || distance == nullptr ||
        parent == nullptr || push_count == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t site = 0; site < seed_count; ++site) {
        if (seed_pixel[site] < 0 ||
            static_cast<std::size_t>(seed_pixel[site]) >= pixels) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    constexpr double tolerance = 1e-12;
    static constexpr int dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static constexpr int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    try {
        const std::size_t bucket_count = span + 2;
        std::vector<std::int32_t> head(bucket_count, -1);
        std::vector<double> key;
        std::vector<std::int32_t> entry_pixel;
        std::vector<std::int32_t> next;
        const std::size_t reserve =
            std::min(
                pixels * 2 + 256,
                std::numeric_limits<std::size_t>::max() / 4);
        key.reserve(reserve);
        entry_pixel.reserve(reserve);
        next.reserve(reserve);
        std::fill(distance, distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});
        std::fill(parent, parent + pixels, std::int32_t{-1});

        std::size_t alive = 0;
        const auto insert = [&](
            double value, std::size_t pixel) {
            const std::int64_t slot = static_cast<std::int64_t>(
                (value + shift) / delta);
            const std::size_t bucket = static_cast<std::size_t>(
                slot % static_cast<std::int64_t>(bucket_count));
            const std::int32_t index =
                static_cast<std::int32_t>(key.size());
            key.push_back(value);
            entry_pixel.push_back(static_cast<std::int32_t>(pixel));
            next.push_back(head[bucket]);
            head[bucket] = index;
            ++alive;
        };

        for (std::size_t site = 0; site < seed_count; ++site) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[site]);
            const double value = -reach[site];
            if (value < distance[pixel]) {
                distance[pixel] = value;
                owner[pixel] = static_cast<std::int32_t>(site);
                parent[pixel] = -1;
                insert(value, pixel);
            }
        }

        std::size_t current = 0;
        std::size_t guard = 0;
        const std::size_t limit =
            bucket_count * (pixels + 16);
        while (alive > 0 && guard < limit) {
            const std::size_t bucket = current % bucket_count;
            std::int32_t entry = head[bucket];
            if (entry < 0) {
                ++current;
                ++guard;
                continue;
            }
            head[bucket] = -1;
            while (entry >= 0) {
                const std::size_t index =
                    static_cast<std::size_t>(entry);
                const double value = key[index];
                const std::size_t pixel = static_cast<std::size_t>(
                    entry_pixel[index]);
                entry = next[index];
                --alive;
                if (value > distance[pixel] + tolerance) {
                    continue;
                }
                const std::int32_t site = owner[pixel];

                const std::size_t y = pixel / width;
                const std::size_t x = pixel - y * width;
                for (std::size_t direction = 0;
                     direction < 8; ++direction) {
                    const std::int64_t nx =
                        static_cast<std::int64_t>(x) + dx[direction];
                    const std::int64_t ny =
                        static_cast<std::int64_t>(y) + dy[direction];
                    if (
                        nx < 0 || nx >= static_cast<std::int64_t>(width) ||
                        ny < 0 || ny >= static_cast<std::int64_t>(height)
                    ) {
                        continue;
                    }
                    const std::size_t receiver =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    const double candidate =
                        value + static_cast<double>(
                            direction_costs[
                                direction * pixels + pixel]);
                    if (candidate + tolerance >= distance[receiver]) {
                        continue;
                    }
                    distance[receiver] = candidate;
                    owner[receiver] = site;
                    parent[receiver] =
                        static_cast<std::int32_t>(pixel);
                    insert(candidate, receiver);
                }
            }
            ++current;
            ++guard;
        }
        *push_count = key.size();
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_hard_affine_fit(
    std::size_t height,
    std::size_t width,
    std::size_t cell_count,
    const std::int32_t* labels,
    const double* target,
    double* basis,
    double* count,
    double* radius,
    double* centroid,
    double* reconstruction) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 || cell_count == 0 ||
        !checked_product(height, width, &pixels) ||
        labels == nullptr || target == nullptr || basis == nullptr ||
        count == nullptr || radius == nullptr || centroid == nullptr ||
        reconstruction == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    try {
        std::vector<double> centroid_sum(cell_count * 2, 0.0);
        std::vector<double> second_moment(cell_count * 3, 0.0);
        std::vector<double> rhs(cell_count * 9, 0.0);
        std::vector<double> coefficient(cell_count * 9, 0.0);
        std::fill(count, count + cell_count, 0.0);

        /* First reduction: mass, centroid, and constant-channel RHS. */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::int32_t label_value = labels[pixel];
            if (!valid_cell(label_value, cell_count)) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            const std::size_t cell =
                static_cast<std::size_t>(label_value);
            const std::size_t y = pixel / width;
            const std::size_t x = pixel - y * width;
            const double px =
                (static_cast<double>(x) + 0.5) /
                    static_cast<double>(width) - 0.5;
            const double py =
                (static_cast<double>(y) + 0.5) /
                    static_cast<double>(height) - 0.5;
            count[cell] += 1.0;
            centroid_sum[cell * 2] += px;
            centroid_sum[cell * 2 + 1] += py;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                rhs[cell * 9 + channel] += target[pixel * 3 + channel];
            }
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            count[cell] = std::max(count[cell], 1.0);
            centroid[cell * 2] =
                centroid_sum[cell * 2] / count[cell];
            centroid[cell * 2 + 1] =
                centroid_sum[cell * 2 + 1] / count[cell];
        }

        /* Second reduction: the physical covariance needed to set r. */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const std::size_t y = pixel / width;
            const std::size_t x = pixel - y * width;
            const double px =
                (static_cast<double>(x) + 0.5) /
                    static_cast<double>(width) - 0.5;
            const double py =
                (static_cast<double>(y) + 0.5) /
                    static_cast<double>(height) - 0.5;
            const double dx = px - centroid[cell * 2];
            const double dy = py - centroid[cell * 2 + 1];
            second_moment[cell * 3] += dx * dx;
            second_moment[cell * 3 + 2] += dy * dy;
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            radius[cell] = std::sqrt(std::max(
                (second_moment[cell * 3] +
                 second_moment[cell * 3 + 2]) / count[cell],
                1e-30));
        }
        std::fill(second_moment.begin(), second_moment.end(), 0.0);

        /*
           Third reduction: normalized slope block and slope-channel RHS.
           Writing the basis here lets the ridge rung reuse this exact frame.
        */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const std::size_t y = pixel / width;
            const std::size_t x = pixel - y * width;
            const double px =
                (static_cast<double>(x) + 0.5) /
                    static_cast<double>(width) - 0.5;
            const double py =
                (static_cast<double>(y) + 0.5) /
                    static_cast<double>(height) - 0.5;
            const double ux =
                (px - centroid[cell * 2]) / radius[cell];
            const double uy =
                (py - centroid[cell * 2 + 1]) / radius[cell];
            basis[pixel * 3] = 1.0;
            basis[pixel * 3 + 1] = ux;
            basis[pixel * 3 + 2] = uy;
            second_moment[cell * 3] += ux * ux;
            second_moment[cell * 3 + 1] += ux * uy;
            second_moment[cell * 3 + 2] += uy * uy;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const double sample = target[pixel * 3 + channel];
                rhs[cell * 9 + 3 + channel] += ux * sample;
                rhs[cell * 9 + 6 + channel] += uy * sample;
            }
        }

        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const double gradient_regularization =
                1e-5 * count[cell] /
                std::max(radius[cell] * radius[cell], 1e-30);
            const double a =
                second_moment[cell * 3] + gradient_regularization;
            const double b = second_moment[cell * 3 + 1];
            const double c =
                second_moment[cell * 3 + 2] + gradient_regularization;
            const double determinant =
                std::max(a * c - b * b, 1e-30);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const double rhs0 = rhs[cell * 9 + channel];
                const double rhsx = rhs[cell * 9 + 3 + channel];
                const double rhsy = rhs[cell * 9 + 6 + channel];
                coefficient[cell * 9 + channel] =
                    rhs0 / ((1.0 + 1e-7) * count[cell]);
                coefficient[cell * 9 + 3 + channel] =
                    (c * rhsx - b * rhsy) / determinant;
                coefficient[cell * 9 + 6 + channel] =
                    (a * rhsy - b * rhsx) / determinant;
            }
        }

        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const double ux = basis[pixel * 3 + 1];
            const double uy = basis[pixel * 3 + 2];
            for (std::size_t channel = 0; channel < 3; ++channel) {
                reconstruction[pixel * 3 + channel] =
                    coefficient[cell * 9 + channel] +
                    ux * coefficient[cell * 9 + 3 + channel] +
                    uy * coefficient[cell * 9 + 6 + channel];
            }
        }
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_hard_basis_refit(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t basis_width,
    const std::int32_t* labels,
    const double* design,
    const double* target,
    const double* count,
    const double* radius,
    double* reconstruction) {
    if (pixel_count == 0 || cell_count == 0 || basis_width < 3 ||
        labels == nullptr || design == nullptr || target == nullptr ||
        count == nullptr || radius == nullptr || reconstruction == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t block_area = 0;
    std::size_t normal_size = 0;
    std::size_t rhs_width = 0;
    std::size_t rhs_size = 0;
    if (!checked_product(basis_width, basis_width, &block_area) ||
        !checked_product(cell_count, block_area, &normal_size) ||
        !checked_product(basis_width, std::size_t{3}, &rhs_width) ||
        !checked_product(cell_count, rhs_width, &rhs_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    try {
        std::vector<double> normal(normal_size, 0.0);
        std::vector<double> rhs(rhs_size, 0.0);
        std::vector<double> coefficient(rhs_size, 0.0);
        std::vector<double> factor(block_area, 0.0);
        std::vector<double> solve_rhs(rhs_width, 0.0);

        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const std::int32_t label_value = labels[pixel];
            if (!valid_cell(label_value, cell_count)) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            const std::size_t cell =
                static_cast<std::size_t>(label_value);
            const double* row = design + pixel * basis_width;
            double* block = normal.data() + cell * block_area;
            double* cell_rhs = rhs.data() + cell * rhs_width;
            for (std::size_t first = 0; first < basis_width; ++first) {
                const double u = row[first];
                for (std::size_t second = first;
                     second < basis_width; ++second) {
                    block[first * basis_width + second] +=
                        u * row[second];
                }
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    cell_rhs[first * 3 + channel] +=
                        u * target[pixel * 3 + channel];
                }
            }
        }

        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            double* block = normal.data() + cell * block_area;
            for (std::size_t row = 0; row < basis_width; ++row) {
                for (std::size_t column = 0; column < row; ++column) {
                    block[row * basis_width + column] =
                        block[column * basis_width + row];
                }
            }
            block[0] += 1e-7 * count[cell];
            const double gradient_regularization =
                1e-5 * count[cell] /
                std::max(radius[cell] * radius[cell], 1e-30);
            block[basis_width + 1] += gradient_regularization;
            block[2 * basis_width + 2] += gradient_regularization;
            for (std::size_t component = 3;
                 component < basis_width; ++component) {
                block[component * basis_width + component] +=
                    2e-5 * count[cell];
            }

            const double* cell_rhs = rhs.data() + cell * rhs_width;
            double* cell_coefficient =
                coefficient.data() + cell * rhs_width;
            std::copy(block, block + block_area, factor.begin());
            std::copy(cell_rhs, cell_rhs + rhs_width, solve_rhs.begin());
            /*
               Directly eliminate the fixed, tiny per-cell system. Partial
               pivoting retains relative accuracy in very thin, nearly
               collinear sliver cells without calling a generic factorization
               library or constructing a sparse solver.
            */
            for (std::size_t column = 0;
                 column < basis_width; ++column) {
                std::size_t pivot = column;
                double pivot_size = std::abs(
                    factor[column * basis_width + column]);
                for (std::size_t row = column + 1;
                     row < basis_width; ++row) {
                    const double candidate = std::abs(
                        factor[row * basis_width + column]);
                    if (candidate > pivot_size) {
                        pivot = row;
                        pivot_size = candidate;
                    }
                }
                if (!(pivot_size > 0.0) || !std::isfinite(pivot_size)) {
                    return BFFT_ERROR_INTERNAL;
                }
                if (pivot != column) {
                    for (std::size_t inner = 0;
                         inner < basis_width; ++inner) {
                        std::swap(
                            factor[column * basis_width + inner],
                            factor[pivot * basis_width + inner]);
                    }
                    for (std::size_t channel = 0;
                         channel < 3; ++channel) {
                        std::swap(
                            solve_rhs[column * 3 + channel],
                            solve_rhs[pivot * 3 + channel]);
                    }
                }
                const double diagonal =
                    factor[column * basis_width + column];
                for (std::size_t row = column + 1;
                     row < basis_width; ++row) {
                    const double multiplier =
                        factor[row * basis_width + column] / diagonal;
                    factor[row * basis_width + column] = multiplier;
                    for (std::size_t inner = column + 1;
                         inner < basis_width; ++inner) {
                        factor[row * basis_width + inner] -=
                            multiplier *
                            factor[column * basis_width + inner];
                    }
                    for (std::size_t channel = 0;
                         channel < 3; ++channel) {
                        solve_rhs[row * 3 + channel] -=
                            multiplier *
                            solve_rhs[column * 3 + channel];
                    }
                }
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                for (std::size_t reverse = basis_width;
                     reverse-- > 0;) {
                    double value = solve_rhs[reverse * 3 + channel];
                    for (std::size_t inner = reverse + 1;
                         inner < basis_width; ++inner) {
                        value -=
                            factor[reverse * basis_width + inner] *
                            cell_coefficient[inner * 3 + channel];
                    }
                    cell_coefficient[reverse * 3 + channel] =
                        value /
                        factor[reverse * basis_width + reverse];
                }
            }
        }

        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const double* row = design + pixel * basis_width;
            const double* cell_coefficient =
                coefficient.data() + cell * rhs_width;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                double value = 0.0;
                for (std::size_t component = 0;
                     component < basis_width; ++component) {
                    value += row[component] *
                        cell_coefficient[component * 3 + channel];
                }
                reconstruction[pixel * 3 + channel] = value;
            }
        }
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

}  // extern "C"
