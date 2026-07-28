#include <high_vision/high_vision.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

double rmse(const std::vector<float> &actual,
	    const std::vector<float> &expected)
{
	double squared = 0.0;
	for (std::size_t i = 0; i < actual.size(); ++i) {
		const double error =
			static_cast<double>(actual[i]) - expected[i];
		squared += error * error;
	}
	return std::sqrt(squared / static_cast<double>(actual.size()));
}

double rectangle_mean(const std::vector<float> &image, std::size_t width,
		      std::size_t x0, std::size_t y0, std::size_t x1,
		      std::size_t y1)
{
	double sum = 0.0;
	std::size_t count = 0;
	for (std::size_t y = y0; y < y1; ++y)
		for (std::size_t x = x0; x < x1; ++x) {
			sum += image[y * width + x];
			++count;
		}
	return sum / static_cast<double>(count);
}

double interior_rmse(const std::vector<float> &actual,
		     const std::vector<float> &expected,
		     std::size_t width, std::size_t height,
		     std::size_t margin)
{
	double squared = 0.0;
	std::size_t count = 0;
	for (std::size_t y = margin; y + margin < height; ++y) {
		for (std::size_t x = margin; x + margin < width; ++x) {
			const double error =
				static_cast<double>(actual[y * width + x]) -
				expected[y * width + x];
			squared += error * error;
			++count;
		}
	}
	return std::sqrt(squared / static_cast<double>(count));
}

} // namespace

int main()
{
	constexpr std::size_t width = 128;
	constexpr std::size_t height = 96;
	constexpr std::size_t pixels = width * height;
	std::vector<float> truth(pixels);
	for (std::size_t y = 0; y < height; ++y) {
		for (std::size_t x = 0; x < width; ++x) {
			const float gradient =
				0.025f + 0.045f * static_cast<float>(x) /
						  static_cast<float>(width - 1);
			const float structure =
				0.010f * std::sin(0.17f * static_cast<float>(x)) *
				std::cos(0.13f * static_cast<float>(y));
			truth[y * width + x] =
				std::clamp(gradient + structure, 0.008f, 0.12f);
		}
	}

	high_vision::Config config;
	config.mode = high_vision::Mode::night_integrator;
	config.registration_radius = 0;
	config.local_search_radius = 6;
	config.tile_size = 8;
	config.scene_cut_threshold = 1.0f;
	config.support_limit = 120.0f;
	config.support_decay = 0.997f;
	config.change_threshold = 0.15f;
	config.tone_strength = 0.0f;
	config.local_contrast = 0.0f;
	high_vision::Processor processor(config);
	high_vision::Config likelihood_config = config;
	likelihood_config.mode = high_vision::Mode::night_likelihood;
	high_vision::Processor likelihood_processor(likelihood_config);

	std::mt19937 random(0x65f00dU);
	std::normal_distribution<float> gaussian(0.0f, 1.0f);
	std::vector<float> frame(pixels);
	std::vector<float> output(pixels);
	auto sample = [&](const std::vector<float> &scene) {
		for (std::size_t i = 0; i < pixels; ++i) {
			const float sigma = std::sqrt(
				config.read_noise * config.read_noise +
				config.shot_noise * config.shot_noise *
					std::max(scene[i], 0.0f));
			frame[i] =
				std::clamp(scene[i] + sigma * gaussian(random),
					   0.0f, 1.0f);
		}
	};

	double raw_squared = 0.0;
	std::size_t raw_samples = 0;
	const auto started = std::chrono::steady_clock::now();
	for (int index = 0; index < 120; ++index) {
		sample(truth);
		for (std::size_t i = 0; i < pixels; ++i) {
			const double error =
				static_cast<double>(frame[i]) - truth[i];
			raw_squared += error * error;
			++raw_samples;
		}
		if (!processor.process(frame.data(), width, output.data(), width,
				      width, height) ||
		    !likelihood_processor.process(
			    frame.data(), width, output.data(), width, width,
			    height))
			return 1;
	}
	const double elapsed_ms =
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - started)
			.count();
	const double raw_rmse =
		std::sqrt(raw_squared / static_cast<double>(raw_samples));
	const double belief_rmse = rmse(processor.belief(), truth);
	const double likelihood_belief_rmse =
		rmse(likelihood_processor.belief(), truth);
	const double old_object_level =
		rectangle_mean(processor.belief(), width, 44, 30, 84, 66);
	const double likelihood_old_object_level = rectangle_mean(
		likelihood_processor.belief(), width, 44, 30, 84, 66);

	std::vector<float> occluded = truth;
	for (std::size_t y = 30; y < 66; ++y)
		for (std::size_t x = 44; x < 84; ++x)
			occluded[y * width + x] = 0.006f;
	const double target_level =
		rectangle_mean(occluded, width, 44, 30, 84, 66);

	std::vector<double> recovery;
	std::vector<double> likelihood_recovery;
	for (int index = 1; index <= 8; ++index) {
		sample(occluded);
		if (!processor.process(frame.data(), width, output.data(), width,
				      width, height) ||
		    !likelihood_processor.process(
			    frame.data(), width, output.data(), width, width,
			    height))
			return 1;
		if (index == 1 || index == 2 || index == 4 || index == 8) {
			const double level = rectangle_mean(
				processor.belief(), width, 44, 30, 84, 66);
			const double likelihood_level = rectangle_mean(
				likelihood_processor.belief(), width, 44, 30,
				84, 66);
			recovery.push_back(std::clamp(
				(old_object_level - level) /
					std::max(old_object_level - target_level,
						 1e-9),
				0.0, 1.0));
			likelihood_recovery.push_back(std::clamp(
				(likelihood_old_object_level - likelihood_level) /
					std::max(likelihood_old_object_level -
							 target_level,
						 1e-9),
				0.0, 1.0));
		}
	}

	// A realistic temporal mean cannot remove a detector-fixed field. Exercise
	// the two gauges independently: scene content translates, while the
	// nuisance pattern remains nailed to detector coordinates.
	high_vision::Config pattern_config = config;
	pattern_config.registration_radius = 3;
	pattern_config.local_search_radius = 0;
	pattern_config.support_limit = 48.0f;
	pattern_config.support_decay = 0.98f;
	pattern_config.change_threshold = 0.20f;
	pattern_config.sensor_pattern_learning_rate = 0.04f;
	pattern_config.sensor_pattern_limit = 0.03f;
	pattern_config.sensor_pattern_min_motion = 0.5f;
	high_vision::Config uncorrected_config = pattern_config;
	uncorrected_config.sensor_pattern_correction = false;
	high_vision::Processor corrected(pattern_config);
	high_vision::Processor uncorrected(uncorrected_config);

	std::vector<float> fixed_pattern(pixels);
	std::vector<float> moving_truth(pixels);
	std::vector<float> patterned_frame(pixels);
	std::mt19937 pattern_random(0xf17edU);
	std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
	double pattern_mean = 0.0;
	for (std::size_t y = 0; y < height; ++y) {
		for (std::size_t x = 0; x < width; ++x) {
			const std::size_t i = y * width + x;
			fixed_pattern[i] =
				0.006f * uniform(pattern_random) +
				0.003f * std::sin(0.41f * static_cast<float>(y));
			pattern_mean += fixed_pattern[i];
		}
	}
	pattern_mean /= static_cast<double>(pixels);
	for (float &value : fixed_pattern)
		value -= static_cast<float>(pattern_mean);

	const int positions[] = {0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1};
	for (int index = 0; index < 360; ++index) {
		const int position =
			positions[index % (sizeof(positions) / sizeof(positions[0]))];
		for (std::size_t y = 0; y < height; ++y) {
			for (std::size_t x = 0; x < width; ++x) {
				const int sx = std::clamp(
					static_cast<int>(x) - position, 0,
					static_cast<int>(width) - 1);
				const std::size_t i = y * width + x;
				moving_truth[i] =
					truth[y * width + static_cast<std::size_t>(sx)];
				const float sigma = std::sqrt(
					pattern_config.read_noise *
							pattern_config.read_noise +
					pattern_config.shot_noise *
							pattern_config.shot_noise *
							moving_truth[i]);
				patterned_frame[i] = std::clamp(
					moving_truth[i] + fixed_pattern[i] +
						sigma * gaussian(pattern_random),
					0.0f, 1.0f);
			}
		}
		if (!corrected.process(patterned_frame.data(), width, output.data(),
				       width, width, height) ||
		    !uncorrected.process(patterned_frame.data(), width, output.data(),
					 width, width, height))
			return 1;
	}
	const double corrected_pattern_rmse = interior_rmse(
		corrected.belief(), moving_truth, width, height, 6);
	const double uncorrected_pattern_rmse = interior_rmse(
		uncorrected.belief(), moving_truth, width, height, 6);

	std::cout << std::fixed << std::setprecision(4)
		  << "{\n"
		  << "  \"grid\": \"128x96\",\n"
		  << "  \"milliseconds_per_frame\": " << elapsed_ms / 120.0
		  << ",\n"
		  << "  \"raw_shadow_rmse\": " << raw_rmse << ",\n"
		  << "  \"belief_shadow_rmse\": " << belief_rmse << ",\n"
		  << "  \"noise_reduction_db\": "
		  << 20.0 * std::log10(raw_rmse / belief_rmse) << ",\n"
		  << "  \"relative_exposure\": "
		  << processor.diagnostics().relative_exposure << ",\n"
		  << "  \"mean_support\": "
		  << processor.diagnostics().mean_support << ",\n"
		  << "  \"dark_object_recovery\": {\n"
		  << "    \"frame_1\": " << recovery[0] << ",\n"
		  << "    \"frame_2\": " << recovery[1] << ",\n"
		  << "    \"frame_4\": " << recovery[2] << ",\n"
		  << "    \"frame_8\": " << recovery[3] << "\n"
		  << "  },\n"
		  << "  \"likelihood_path\": {\n"
		  << "    \"belief_shadow_rmse\": "
		  << likelihood_belief_rmse << ",\n"
		  << "    \"noise_reduction_db\": "
		  << 20.0 *
			     std::log10(raw_rmse / likelihood_belief_rmse)
		  << ",\n"
		  << "    \"mean_support\": "
		  << likelihood_processor.diagnostics().mean_support
		  << ",\n"
		  << "    \"dark_object_recovery\": {\n"
		  << "      \"frame_1\": " << likelihood_recovery[0]
		  << ",\n"
		  << "      \"frame_2\": " << likelihood_recovery[1]
		  << ",\n"
		  << "      \"frame_4\": " << likelihood_recovery[2]
		  << ",\n"
		  << "      \"frame_8\": " << likelihood_recovery[3]
		  << "\n"
		  << "    }\n"
		  << "  },\n"
		  << "  \"sensor_pattern\": {\n"
		  << "    \"uncorrected_rmse\": " << uncorrected_pattern_rmse
		  << ",\n"
		  << "    \"corrected_rmse\": " << corrected_pattern_rmse
		  << ",\n"
		  << "    \"improvement_db\": "
		  << 20.0 * std::log10(uncorrected_pattern_rmse /
				       corrected_pattern_rmse)
		  << ",\n"
		  << "    \"estimated_rms\": "
		  << corrected.diagnostics().sensor_pattern_rms << ",\n"
		  << "    \"last_motion\": ["
		  << corrected.diagnostics().global_dx << ", "
		  << corrected.diagnostics().global_dy << "],\n"
		  << "    \"registration_confidence\": "
		  << corrected.diagnostics().registration_confidence << ",\n"
		  << "    \"updates\": "
		  << corrected.diagnostics().sensor_pattern_updates << "\n"
		  << "  }\n"
		  << "}\n";
}
