#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define PDEB_EXPORT __declspec(dllexport)
#else
#define PDEB_EXPORT __attribute__((visibility("default")))
#endif

namespace {

bool valid_dimensions(std::int64_t pixels, int channels, int atoms) {
    return pixels > 0 && channels > 0 && atoms > 0
        && pixels <= std::numeric_limits<std::int64_t>::max() / channels;
}

bool valid_batch_dimensions(
    std::int64_t pixels, int channels, int contributions, int plans
) {
    return valid_dimensions(pixels, channels, contributions) && plans > 0
        && static_cast<std::uint64_t>(plans)
            <= std::numeric_limits<std::uint64_t>::max()
                / static_cast<std::uint64_t>(pixels);
}

std::int64_t reflect_index(std::int64_t index, std::int64_t size) {
    if (size <= 1) {
        return 0;
    }
    const std::int64_t period = 2 * size;
    std::int64_t folded = index % period;
    if (folded < 0) {
        folded += period;
    }
    return folded < size ? folded : period - 1 - folded;
}

int covariance_apply(
    const double* input,
    double* output,
    std::int64_t height,
    std::int64_t width,
    int channels,
    const double* axes,
    const double* side_weights,
    int spatial_side_weights,
    bool adjoint
) {
    if (input == nullptr || output == nullptr || axes == nullptr
        || side_weights == nullptr
        || height <= 0 || width <= 0 || channels <= 0
        || height > std::numeric_limits<std::int64_t>::max() / width) {
        return 1;
    }
    const std::int64_t pixels = height * width;
    if (!valid_dimensions(pixels, channels, 1)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    std::fill(output, output + values, 0.0);
    for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
        const double low_x = axes[4 * pixel];
        const double low_y = axes[4 * pixel + 1];
        const double high_x = axes[4 * pixel + 2];
        const double high_y = axes[4 * pixel + 3];
        const double sigma_coordinates[3] = {-1.0, 0.0, 1.0};
        const double* pixel_side_weights = side_weights
            + (spatial_side_weights != 0 ? 2 * pixel : 0);
        const double low_weight = pixel_side_weights[0];
        const double high_weight = pixel_side_weights[1];
        if (low_weight <= 0.0 || low_weight >= 0.5
            || high_weight <= 0.0 || high_weight >= 0.5) {
            return 3;
        }
        const double low_sigma_weights[3] = {
            low_weight, 1.0 - 2.0 * low_weight, low_weight,
        };
        const double high_sigma_weights[3] = {
            high_weight, 1.0 - 2.0 * high_weight, high_weight,
        };
        const std::int64_t y = pixel / width;
        const std::int64_t x = pixel - y * width;
        for (int low_index = 0; low_index < 3; ++low_index) {
          for (int high_index = 0; high_index < 3; ++high_index) {
            const double displacement_x =
                sigma_coordinates[low_index] * low_x
                + sigma_coordinates[high_index] * high_x;
            const double displacement_y =
                sigma_coordinates[low_index] * low_y
                + sigma_coordinates[high_index] * high_y;
            const double source_x = static_cast<double>(x) - displacement_x;
            const double source_y = static_cast<double>(y) - displacement_y;
            const std::int64_t x0 = static_cast<std::int64_t>(std::floor(source_x));
            const std::int64_t y0 = static_cast<std::int64_t>(std::floor(source_y));
            const double fraction_x = source_x - static_cast<double>(x0);
            const double fraction_y = source_y - static_cast<double>(y0);
            const double interpolation[4] = {
                (1.0 - fraction_x) * (1.0 - fraction_y),
                fraction_x * (1.0 - fraction_y),
                (1.0 - fraction_x) * fraction_y,
                fraction_x * fraction_y,
            };
            const std::int64_t offsets[4][2] = {
                {0, 0}, {1, 0}, {0, 1}, {1, 1},
            };
            for (int corner = 0; corner < 4; ++corner) {
                const std::int64_t reflected_x = reflect_index(
                    x0 + offsets[corner][0], width);
                const std::int64_t reflected_y = reflect_index(
                    y0 + offsets[corner][1], height);
                const std::int64_t sampled = reflected_y * width + reflected_x;
                const double coefficient = low_sigma_weights[low_index]
                    * high_sigma_weights[high_index] * interpolation[corner];
                const std::int64_t input_pixel = adjoint ? pixel : sampled;
                const std::int64_t output_pixel = adjoint ? sampled : pixel;
                for (int channel = 0; channel < channels; ++channel) {
                    output[output_pixel * channels + channel] += coefficient
                        * input[input_pixel * channels + channel];
                }
            }
          }
        }
    }
    return 0;
}

int covariance_batch_apply(
    const double* input,
    double* output,
    std::int64_t height,
    std::int64_t width,
    int channels,
    const double* axes,
    const double* side_weights,
    int spatial_side_weights,
    int plans,
    bool adjoint
) {
    if (input == nullptr || output == nullptr || axes == nullptr
        || side_weights == nullptr || height <= 0 || width <= 0
        || channels <= 0 || plans <= 0
        || height > std::numeric_limits<std::int64_t>::max() / width) {
        return 1;
    }
    const std::int64_t pixels = height * width;
    if (!valid_batch_dimensions(pixels, channels, 1, plans)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    const std::int64_t axis_values = 4 * pixels;
    const std::int64_t side_values = spatial_side_weights != 0
        ? 2 * pixels : 2;
    std::vector<int> statuses(static_cast<std::size_t>(plans), 0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(plans));
    for (int plan = 0; plan < plans; ++plan) {
        workers.emplace_back([=, &statuses]() {
            statuses[static_cast<std::size_t>(plan)] = covariance_apply(
                input + static_cast<std::int64_t>(plan) * values,
                output + static_cast<std::int64_t>(plan) * values,
                height,
                width,
                channels,
                axes + static_cast<std::int64_t>(plan) * axis_values,
                side_weights + static_cast<std::int64_t>(plan) * side_values,
                spatial_side_weights,
                adjoint);
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (int status : statuses) {
        if (status != 0) {
            return status;
        }
    }
    return 0;
}

}  // namespace

extern "C" {

PDEB_EXPORT int pdeb_path_operator_abi_version() {
    return 6;
}

PDEB_EXPORT const char* pdeb_path_operator_backend() {
    return "native_cxx_exposure_transport_v6";
}

PDEB_EXPORT int pdeb_covariance_forward(
    const double* input,
    double* output,
    std::int64_t height,
    std::int64_t width,
    int channels,
    const double* covariance,
    const double* side_weights,
    int spatial_side_weights
) {
    return covariance_apply(
        input, output, height, width, channels, covariance, side_weights,
        spatial_side_weights, false);
}

PDEB_EXPORT int pdeb_covariance_adjoint(
    const double* input,
    double* output,
    std::int64_t height,
    std::int64_t width,
    int channels,
    const double* covariance,
    const double* side_weights,
    int spatial_side_weights
) {
    return covariance_apply(
        input, output, height, width, channels, covariance, side_weights,
        spatial_side_weights, true);
}

PDEB_EXPORT int pdeb_covariance_batch_forward(
    const double* input,
    double* output,
    std::int64_t height,
    std::int64_t width,
    int channels,
    const double* covariance,
    const double* side_weights,
    int spatial_side_weights,
    int plans
) {
    return covariance_batch_apply(
        input, output, height, width, channels, covariance, side_weights,
        spatial_side_weights, plans, false);
}

PDEB_EXPORT int pdeb_covariance_batch_adjoint(
    const double* input,
    double* output,
    std::int64_t height,
    std::int64_t width,
    int channels,
    const double* covariance,
    const double* side_weights,
    int spatial_side_weights,
    int plans
) {
    return covariance_batch_apply(
        input, output, height, width, channels, covariance, side_weights,
        spatial_side_weights, plans, true);
}

PDEB_EXPORT int pdeb_path_forward(
    const double* input,
    double* output,
    std::int64_t pixels,
    int channels,
    const std::int64_t* source_indices,
    const double* weights,
    int atoms
) {
    if (input == nullptr || output == nullptr || source_indices == nullptr
        || weights == nullptr || !valid_dimensions(pixels, channels, atoms)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    std::fill(output, output + values, 0.0);
    for (int atom = 0; atom < atoms; ++atom) {
        const double weight = weights[atom];
        const std::int64_t* atom_indices = source_indices + atom * pixels;
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
            const std::int64_t source = atom_indices[pixel];
            if (source < 0 || source >= pixels) {
                return 2;
            }
            const std::int64_t destination_offset = pixel * channels;
            const std::int64_t source_offset = source * channels;
            for (int channel = 0; channel < channels; ++channel) {
                output[destination_offset + channel] +=
                    weight * input[source_offset + channel];
            }
        }
    }
    return 0;
}

PDEB_EXPORT int pdeb_path_adjoint(
    const double* input,
    double* output,
    std::int64_t pixels,
    int channels,
    const std::int64_t* source_indices,
    const double* weights,
    int atoms
) {
    if (input == nullptr || output == nullptr || source_indices == nullptr
        || weights == nullptr || !valid_dimensions(pixels, channels, atoms)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    std::fill(output, output + values, 0.0);
    for (int atom = 0; atom < atoms; ++atom) {
        const double weight = weights[atom];
        const std::int64_t* atom_indices = source_indices + atom * pixels;
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
            const std::int64_t destination = atom_indices[pixel];
            if (destination < 0 || destination >= pixels) {
                return 2;
            }
            const std::int64_t source_offset = pixel * channels;
            const std::int64_t destination_offset = destination * channels;
            for (int channel = 0; channel < channels; ++channel) {
                output[destination_offset + channel] +=
                    weight * input[source_offset + channel];
            }
        }
    }
    return 0;
}

PDEB_EXPORT int pdeb_spatial_forward(
    const double* input,
    double* output,
    std::int64_t pixels,
    int channels,
    const std::int64_t* source_indices,
    const double* coefficients,
    int contributions
) {
    if (input == nullptr || output == nullptr || source_indices == nullptr
        || coefficients == nullptr
        || !valid_dimensions(pixels, channels, contributions)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    std::fill(output, output + values, 0.0);
    for (int contribution = 0; contribution < contributions; ++contribution) {
        const std::int64_t* contribution_indices =
            source_indices + contribution * pixels;
        const double* contribution_coefficients =
            coefficients + contribution * pixels;
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
            const std::int64_t source = contribution_indices[pixel];
            if (source < 0 || source >= pixels) {
                return 2;
            }
            const double coefficient = contribution_coefficients[pixel];
            const std::int64_t destination_offset = pixel * channels;
            const std::int64_t source_offset = source * channels;
            for (int channel = 0; channel < channels; ++channel) {
                output[destination_offset + channel] +=
                    coefficient * input[source_offset + channel];
            }
        }
    }
    return 0;
}

PDEB_EXPORT int pdeb_spatial_adjoint(
    const double* input,
    double* output,
    std::int64_t pixels,
    int channels,
    const std::int64_t* source_indices,
    const double* coefficients,
    int contributions
) {
    if (input == nullptr || output == nullptr || source_indices == nullptr
        || coefficients == nullptr
        || !valid_dimensions(pixels, channels, contributions)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    std::fill(output, output + values, 0.0);
    for (int contribution = 0; contribution < contributions; ++contribution) {
        const std::int64_t* contribution_indices =
            source_indices + contribution * pixels;
        const double* contribution_coefficients =
            coefficients + contribution * pixels;
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
            const std::int64_t destination = contribution_indices[pixel];
            if (destination < 0 || destination >= pixels) {
                return 2;
            }
            const double coefficient = contribution_coefficients[pixel];
            const std::int64_t source_offset = pixel * channels;
            const std::int64_t destination_offset = destination * channels;
            for (int channel = 0; channel < channels; ++channel) {
                output[destination_offset + channel] +=
                    coefficient * input[source_offset + channel];
            }
        }
    }
    return 0;
}

PDEB_EXPORT int pdeb_spatial_batch_forward(
    const double* input,
    double* output,
    std::int64_t pixels,
    int channels,
    const std::int64_t* source_indices,
    const double* coefficients,
    int maximum_contributions,
    const int* contribution_counts,
    int plans
) {
    if (input == nullptr || output == nullptr || source_indices == nullptr
        || coefficients == nullptr || contribution_counts == nullptr
        || !valid_batch_dimensions(
            pixels, channels, maximum_contributions, plans)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    const std::int64_t plan_entries =
        pixels * static_cast<std::int64_t>(maximum_contributions);
    for (int plan = 0; plan < plans; ++plan) {
        const int contributions = contribution_counts[plan];
        if (contributions <= 0 || contributions > maximum_contributions) {
            return 3;
        }
        const double* plan_input = input + plan * values;
        double* plan_output = output + plan * values;
        const std::int64_t* plan_indices =
            source_indices + plan * plan_entries;
        const double* plan_coefficients = coefficients + plan * plan_entries;
        std::fill(plan_output, plan_output + values, 0.0);
        for (int contribution = 0; contribution < contributions; ++contribution) {
            const std::int64_t* contribution_indices =
                plan_indices + contribution * pixels;
            const double* contribution_coefficients =
                plan_coefficients + contribution * pixels;
            for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
                const std::int64_t source = contribution_indices[pixel];
                if (source < 0 || source >= pixels) {
                    return 2;
                }
                const double coefficient = contribution_coefficients[pixel];
                const std::int64_t destination_offset = pixel * channels;
                const std::int64_t source_offset = source * channels;
                for (int channel = 0; channel < channels; ++channel) {
                    plan_output[destination_offset + channel] +=
                        coefficient * plan_input[source_offset + channel];
                }
            }
        }
    }
    return 0;
}

PDEB_EXPORT int pdeb_spatial_batch_adjoint(
    const double* input,
    double* output,
    std::int64_t pixels,
    int channels,
    const std::int64_t* source_indices,
    const double* coefficients,
    int maximum_contributions,
    const int* contribution_counts,
    int plans
) {
    if (input == nullptr || output == nullptr || source_indices == nullptr
        || coefficients == nullptr || contribution_counts == nullptr
        || !valid_batch_dimensions(
            pixels, channels, maximum_contributions, plans)) {
        return 1;
    }
    const std::int64_t values = pixels * static_cast<std::int64_t>(channels);
    const std::int64_t plan_entries =
        pixels * static_cast<std::int64_t>(maximum_contributions);
    for (int plan = 0; plan < plans; ++plan) {
        const int contributions = contribution_counts[plan];
        if (contributions <= 0 || contributions > maximum_contributions) {
            return 3;
        }
        const double* plan_input = input + plan * values;
        double* plan_output = output + plan * values;
        const std::int64_t* plan_indices =
            source_indices + plan * plan_entries;
        const double* plan_coefficients = coefficients + plan * plan_entries;
        std::fill(plan_output, plan_output + values, 0.0);
        for (int contribution = 0; contribution < contributions; ++contribution) {
            const std::int64_t* contribution_indices =
                plan_indices + contribution * pixels;
            const double* contribution_coefficients =
                plan_coefficients + contribution * pixels;
            for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
                const std::int64_t destination = contribution_indices[pixel];
                if (destination < 0 || destination >= pixels) {
                    return 2;
                }
                const double coefficient = contribution_coefficients[pixel];
                const std::int64_t source_offset = pixel * channels;
                const std::int64_t destination_offset = destination * channels;
                for (int channel = 0; channel < channels; ++channel) {
                    plan_output[destination_offset + channel] +=
                        coefficient * plan_input[source_offset + channel];
                }
            }
        }
    }
    return 0;
}

}  // extern "C"
