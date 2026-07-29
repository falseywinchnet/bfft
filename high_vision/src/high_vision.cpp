#include <high_vision/high_vision.hpp>

#ifdef HIGH_VISION_WITH_BFFT_MEYER
#include <bfft/meyer.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace high_vision {
namespace {

float clamp01(float value)
{
	return std::clamp(value, 0.0f, 1.0f);
}

bool is_night_mode(Mode mode)
{
	return mode == Mode::night_integrator ||
	       mode == Mode::night_likelihood ||
	       mode == Mode::night_moments;
}

float smoothstep(float edge0, float edge1, float value)
{
	if (!(edge1 > edge0))
		return value >= edge1 ? 1.0f : 0.0f;
	const float t = clamp01((value - edge0) / (edge1 - edge0));
	return t * t * (3.0f - 2.0f * t);
}

int bit_count(unsigned value)
{
	int count = 0;
	while (value) {
		value &= value - 1;
		++count;
	}
	return count;
}

float bilinear(const std::vector<float> &image, std::size_t width,
	       std::size_t height, float x, float y, float outside)
{
	if (x < 0.0f || y < 0.0f ||
	    x > static_cast<float>(width - 1) ||
	    y > static_cast<float>(height - 1))
		return outside;
	const std::size_t x0 = static_cast<std::size_t>(x);
	const std::size_t y0 = static_cast<std::size_t>(y);
	const std::size_t x1 = std::min(x0 + 1, width - 1);
	const std::size_t y1 = std::min(y0 + 1, height - 1);
	const float fx = x - static_cast<float>(x0);
	const float fy = y - static_cast<float>(y0);
	const float a = image[y0 * width + x0] * (1.0f - fx) +
			image[y0 * width + x1] * fx;
	const float b = image[y1 * width + x0] * (1.0f - fx) +
			image[y1 * width + x1] * fx;
	return a * (1.0f - fy) + b * fy;
}

void sanitize_entropy_config(EntropyControllerConfig &config)
{
	auto &budgets = config.budgets;
	budgets.erase(
		std::remove_if(
			budgets.begin(), budgets.end(),
			[](float value) {
				return !std::isfinite(value) || value < 1.0f;
			}),
		budgets.end());
	std::sort(budgets.begin(), budgets.end());
	budgets.erase(std::unique(budgets.begin(), budgets.end()), budgets.end());
	if (budgets.empty())
		budgets.push_back(1.0f);
	config.bisection_steps = std::clamp(config.bisection_steps, 8, 64);
	config.max_budget_steps_per_batch =
		std::clamp(config.max_budget_steps_per_batch, 1, 8);
	config.evidence_retention =
		std::clamp(config.evidence_retention, 0.0f, 1.0f);
}

bool valid_score_shape(const float *scores, std::size_t frames,
		       std::size_t shifts)
{
	return scores && frames && shifts &&
	       frames <= std::numeric_limits<std::size_t>::max() / shifts;
}

double mean_entropy_at(const float *scores, std::size_t frames,
		       std::size_t shifts, double inverse_temperature)
{
	double entropy_sum = 0.0;
	for (std::size_t frame = 0; frame < frames; ++frame) {
		const float *row = scores + frame * shifts;
		const float maximum =
			*std::max_element(row, row + shifts);
		double partition = 0.0;
		double weighted_score = 0.0;
		for (std::size_t shift = 0; shift < shifts; ++shift) {
			const double centered =
				static_cast<double>(row[shift] - maximum);
			const double weight = std::exp(std::max(
				inverse_temperature * centered, -700.0));
			partition += weight;
			weighted_score += weight * centered;
		}
		partition = std::max(partition, 1e-300);
		entropy_sum += std::log(partition) -
			       inverse_temperature * weighted_score / partition;
	}
	return entropy_sum / static_cast<double>(frames);
}

bool entropy_project(const float *scores, std::size_t frames,
		     std::size_t shifts, float requested_effective_shifts,
		     int bisection_steps, std::vector<float> &posterior,
		     EntropyProjectionDiagnostics *diagnostics)
{
	if (!valid_score_shape(scores, frames, shifts))
		return false;
	const float target = std::clamp(
		std::isfinite(requested_effective_shifts)
			? requested_effective_shifts
			: static_cast<float>(shifts),
		1.0f, static_cast<float>(shifts));
	const double target_entropy = std::log(static_cast<double>(target));
	double inverse_temperature = 0.0;
	if (target < static_cast<float>(shifts) * (1.0f - 1e-6f)) {
		double low = 0.0;
		double high = 1.0;
		while (mean_entropy_at(scores, frames, shifts, high) >
			       target_entropy &&
		       high < 1e8)
			high *= 2.0;
		for (int step = 0; step < bisection_steps; ++step) {
			const double middle = 0.5 * (low + high);
			if (mean_entropy_at(scores, frames, shifts, middle) >
			    target_entropy)
				low = middle;
			else
				high = middle;
		}
		inverse_temperature = 0.5 * (low + high);
	}

	posterior.resize(frames * shifts);
	double entropy_sum = 0.0;
	double frame_effective_sum = 0.0;
	double peak_sum = 0.0;
	for (std::size_t frame = 0; frame < frames; ++frame) {
		const float *row = scores + frame * shifts;
		float *probability = posterior.data() + frame * shifts;
		const float maximum =
			*std::max_element(row, row + shifts);
		double partition = 0.0;
		for (std::size_t shift = 0; shift < shifts; ++shift) {
			const double log_weight = std::max(
				inverse_temperature *
					static_cast<double>(row[shift] - maximum),
				-80.0);
			probability[shift] =
				static_cast<float>(std::exp(log_weight));
			partition += probability[shift];
		}
		partition = std::max(partition, 1e-30);
		double entropy = 0.0;
		float peak = 0.0f;
		for (std::size_t shift = 0; shift < shifts; ++shift) {
			probability[shift] =
				static_cast<float>(probability[shift] / partition);
			const double value = probability[shift];
			if (value > 0.0)
				entropy -= value * std::log(value);
			peak = std::max(peak, probability[shift]);
		}
		entropy_sum += entropy;
		frame_effective_sum += std::exp(entropy);
		peak_sum += peak;
	}

	if (diagnostics) {
		const double mean_entropy =
			entropy_sum / static_cast<double>(frames);
		diagnostics->target_effective_shifts = target;
		diagnostics->effective_shifts =
			static_cast<float>(std::exp(mean_entropy));
		diagnostics->mean_frame_effective_shifts = static_cast<float>(
			frame_effective_sum / static_cast<double>(frames));
		diagnostics->mean_peak_probability = static_cast<float>(
			peak_sum / static_cast<double>(frames));
		diagnostics->inverse_temperature =
			static_cast<float>(inverse_temperature);
		diagnostics->entropy_residual_nats =
			static_cast<float>(mean_entropy - target_entropy);
	}
	return true;
}

double predictive_score(const std::vector<float> &posterior,
			const float *scores, std::size_t frames,
			std::size_t shifts)
{
	double total = 0.0;
	for (std::size_t frame = 0; frame < frames; ++frame) {
		const float *probability = posterior.data() + frame * shifts;
		const float *row = scores + frame * shifts;
		double maximum = -std::numeric_limits<double>::infinity();
		for (std::size_t shift = 0; shift < shifts; ++shift) {
			const double value =
				std::log(std::max(
					static_cast<double>(probability[shift]),
					1e-30)) +
				row[shift];
			maximum = std::max(maximum, value);
		}
		double partition = 0.0;
		for (std::size_t shift = 0; shift < shifts; ++shift) {
			const double value =
				std::log(std::max(
					static_cast<double>(probability[shift]),
					1e-30)) +
				row[shift];
			partition += std::exp(value - maximum);
		}
		total += maximum + std::log(std::max(partition, 1e-300));
	}
	return total / static_cast<double>(frames);
}

} // namespace

struct EntropySupportController::Impl {
	EntropyControllerConfig cfg;
	EntropyControllerDiagnostics diag;
	std::vector<double> evidence;
	std::size_t selected_index = 0;
	std::vector<float> first_posterior;
	std::vector<float> second_posterior;

	explicit Impl(EntropyControllerConfig config) : cfg(std::move(config))
	{
		sanitize_entropy_config(cfg);
		reset();
	}

	void reset()
	{
		evidence.assign(cfg.budgets.size(), 0.0);
		selected_index = cfg.budgets.size() - 1;
		diag = {};
		diag.selected_budget = cfg.budgets[selected_index];
		diag.batch_best_budget = diag.selected_budget;
	}
};

EntropySupportController::EntropySupportController(
	EntropyControllerConfig config)
	: impl_(std::make_unique<Impl>(std::move(config)))
{
}

EntropySupportController::~EntropySupportController() = default;
EntropySupportController::EntropySupportController(
	EntropySupportController &&) noexcept = default;
EntropySupportController &EntropySupportController::operator=(
	EntropySupportController &&) noexcept = default;

void EntropySupportController::configure(
	const EntropyControllerConfig &config)
{
	impl_->cfg = config;
	sanitize_entropy_config(impl_->cfg);
	impl_->reset();
}

const EntropyControllerConfig &
EntropySupportController::config() const noexcept
{
	return impl_->cfg;
}

void EntropySupportController::reset()
{
	impl_->reset();
}

bool EntropySupportController::update(
	const float *first_scores, const float *second_scores,
	std::size_t frames, std::size_t shifts,
	std::size_t pixels_per_frame)
{
	if (!valid_score_shape(first_scores, frames, shifts) ||
	    !valid_score_shape(second_scores, frames, shifts) ||
	    !pixels_per_frame)
		return false;

	std::vector<double> batch_evidence(impl_->cfg.budgets.size());
	for (std::size_t index = 0; index < impl_->cfg.budgets.size(); ++index) {
		if (!entropy_project(
			    first_scores, frames, shifts,
			    impl_->cfg.budgets[index],
			    impl_->cfg.bisection_steps,
			    impl_->first_posterior, nullptr) ||
		    !entropy_project(
			    second_scores, frames, shifts,
			    impl_->cfg.budgets[index],
			    impl_->cfg.bisection_steps,
			    impl_->second_posterior, nullptr))
			return false;
		const double forward = predictive_score(
			impl_->first_posterior, second_scores, frames, shifts);
		const double reverse = predictive_score(
			impl_->second_posterior, first_scores, frames, shifts);
		batch_evidence[index] =
			0.5 * (forward + reverse) /
			static_cast<double>(pixels_per_frame);
	}

	const double batch_max = *std::max_element(
		batch_evidence.begin(), batch_evidence.end());
	for (std::size_t index = 0; index < impl_->evidence.size(); ++index)
		impl_->evidence[index] =
			impl_->cfg.evidence_retention * impl_->evidence[index] +
			batch_evidence[index] - batch_max;

	auto stable_argmax = [](const std::vector<double> &values,
				std::size_t preferred) {
		std::size_t best = std::min(preferred, values.size() - 1);
		for (std::size_t index = 0; index < values.size(); ++index)
			if (values[index] > values[best] + 1e-15)
				best = index;
		return best;
	};
	const std::size_t batch_best =
		stable_argmax(batch_evidence, impl_->selected_index);
	const std::size_t evidence_best =
		stable_argmax(impl_->evidence, impl_->selected_index);
	const std::size_t step = static_cast<std::size_t>(
		impl_->cfg.max_budget_steps_per_batch);
	std::size_t selected = impl_->selected_index;
	if (evidence_best > selected)
		selected = std::min(evidence_best, selected + step);
	else if (evidence_best < selected)
		selected = evidence_best + step < selected
				   ? selected - step
				   : evidence_best;
	if (selected != impl_->selected_index)
		++impl_->diag.support_transitions;
	impl_->selected_index = selected;
	++impl_->diag.batches;
	impl_->diag.selected_budget = impl_->cfg.budgets[selected];
	impl_->diag.batch_best_budget = impl_->cfg.budgets[batch_best];

	double largest = -std::numeric_limits<double>::infinity();
	double second = -std::numeric_limits<double>::infinity();
	for (double value : impl_->evidence) {
		if (value > largest) {
			second = largest;
			largest = value;
		} else if (value > second) {
			second = value;
		}
	}
	impl_->diag.cumulative_evidence_margin_per_pixel =
		static_cast<float>(
			std::isfinite(second) ? largest - second : 0.0);
	return true;
}

bool EntropySupportController::project_selected(
	const float *scores, std::size_t frames, std::size_t shifts,
	std::vector<float> &posterior)
{
	return entropy_project(
		scores, frames, shifts, impl_->diag.selected_budget,
		impl_->cfg.bisection_steps, posterior, &impl_->diag.projection);
}

bool EntropySupportController::project(
	const float *scores, std::size_t frames, std::size_t shifts,
	float target_effective_shifts, std::vector<float> &posterior,
	EntropyProjectionDiagnostics *diagnostics) const
{
	return entropy_project(
		scores, frames, shifts, target_effective_shifts,
		impl_->cfg.bisection_steps, posterior, diagnostics);
}

const EntropyControllerDiagnostics &
EntropySupportController::diagnostics() const noexcept
{
	return impl_->diag;
}

const std::vector<double> &
EntropySupportController::cumulative_evidence() const noexcept
{
	return impl_->evidence;
}

struct Processor::Impl {
	Config cfg;
	Diagnostics diag;
	std::unique_ptr<ExperimentalStage> stage;
	std::size_t width = 0;
	std::size_t height = 0;
	bool initialized = false;

	std::vector<float> raw_current;
	std::vector<float> current;
	std::vector<float> observation;
	std::vector<float> previous;
	std::vector<float> belief;
	std::vector<float> support;
	std::vector<float> variance;
	std::vector<float> coherent_innovation;
	std::vector<float> transported_belief;
	std::vector<float> transported_support;
	std::vector<float> transported_variance;
	std::vector<float> transported_innovation;
	std::vector<float> sensor_pattern;
	std::vector<float> sensor_residual_samples;
	std::vector<float> registration_current;
	std::vector<float> registration_reference;
#ifdef HIGH_VISION_WITH_BFFT_MEYER
	bfft_meyer_plan *registration_meyer_plan = nullptr;
	std::vector<double> meyer_input;
	std::vector<double> meyer_cartoon;
	std::vector<double> meyer_texture;
#endif
	std::vector<float> display;
	std::vector<float> recovered_radiance;
	std::vector<std::uint8_t> current_census;
	std::vector<std::uint8_t> previous_census;

	std::size_t tiles_x = 0;
	std::size_t tiles_y = 0;
	std::vector<float> flow_x;
	std::vector<float> flow_y;
	std::vector<float> flow_confidence;

	float relative_exposure = 1.0f;
	double telemetry_anchor = 0.0;
	float display_black = 0.0f;
	float display_white = 1.0f;
	std::array<float, 4097> moment_response_lut{};
	float moment_response_lut_power = -1.0f;
	std::uint64_t previous_timestamp_ns = 0;
	float moment_effective_fps = 30.0f;

	explicit Impl(Config config) : cfg(std::move(config))
	{
		sanitize_config();
	}

	~Impl()
	{
#ifdef HIGH_VISION_WITH_BFFT_MEYER
		bfft_meyer_plan_destroy(registration_meyer_plan);
#endif
	}

	void sanitize_config()
	{
		cfg.registration_radius = std::clamp(cfg.registration_radius, 0, 32);
		cfg.tile_size = std::clamp(cfg.tile_size, 8, 128);
		cfg.local_search_radius =
			std::clamp(cfg.local_search_radius, 0, 8);
		cfg.meyer_registration_passes =
			std::clamp(cfg.meyer_registration_passes, 1, 32);
		cfg.support_limit = std::max(cfg.support_limit, 1.0f);
		cfg.support_decay = std::clamp(cfg.support_decay, 0.8f, 1.0f);
		cfg.change_threshold =
			std::clamp(cfg.change_threshold, 0.005f, 1.0f);
		cfg.scene_cut_threshold =
			std::clamp(cfg.scene_cut_threshold, 0.05f, 1.0f);
		cfg.likelihood_release_low =
			std::max(cfg.likelihood_release_low, 0.0f);
		cfg.likelihood_release_high = std::max(
			cfg.likelihood_release_high,
			cfg.likelihood_release_low + 0.1f);
		cfg.likelihood_evidence_limit = std::max(
			cfg.likelihood_evidence_limit,
			cfg.likelihood_release_high);
		cfg.shadow_floor = std::clamp(cfg.shadow_floor, 0.0f, 0.25f);
		cfg.highlight_knee =
			std::clamp(cfg.highlight_knee, 0.5f, 0.999f);
		cfg.read_noise = std::max(cfg.read_noise, 1e-6f);
		cfg.shot_noise = std::max(cfg.shot_noise, 0.0f);
		cfg.moment_response_power =
			std::clamp(cfg.moment_response_power, 0.02f, 1.0f);
		cfg.moment_variance_gain =
			std::clamp(cfg.moment_variance_gain, 0.0f, 64.0f);
		cfg.moment_variance_floor =
			std::clamp(cfg.moment_variance_floor, 0.0f, 0.25f);
		cfg.moment_min_support =
			std::clamp(cfg.moment_min_support, 1.0f, 64.0f);
		cfg.moment_integration_seconds =
			std::clamp(cfg.moment_integration_seconds, 1.0f, 60.0f);
		cfg.sensor_pattern_learning_rate =
			std::clamp(cfg.sensor_pattern_learning_rate, 0.0f, 0.25f);
		cfg.sensor_pattern_limit =
			std::clamp(cfg.sensor_pattern_limit, 0.0f, 0.25f);
		cfg.sensor_pattern_min_motion =
			std::clamp(cfg.sensor_pattern_min_motion, 0.0f, 16.0f);
		cfg.black_percentile =
			std::clamp(cfg.black_percentile, 0.0f, 0.25f);
		cfg.white_percentile =
			std::clamp(cfg.white_percentile, 0.75f, 1.0f);
		cfg.tone_strength = std::clamp(cfg.tone_strength, 0.0f, 1.0f);
		cfg.local_contrast = std::clamp(cfg.local_contrast, 0.0f, 1.0f);
	}

	void allocate(std::size_t new_width, std::size_t new_height)
	{
		width = new_width;
		height = new_height;
		const std::size_t count = width * height;
		raw_current.assign(count, 0.0f);
		current.assign(count, 0.0f);
		observation.assign(count, 0.0f);
		previous.assign(count, 0.0f);
		belief.assign(count, 0.0f);
		support.assign(count, 0.0f);
		variance.assign(count, 0.0f);
		coherent_innovation.assign(count, 0.0f);
		transported_belief.assign(count, 0.0f);
		transported_support.assign(count, 0.0f);
		transported_variance.assign(count, 0.0f);
		transported_innovation.assign(count, 0.0f);
		sensor_pattern.assign(count, 0.0f);
		sensor_residual_samples.clear();
		sensor_residual_samples.reserve(count / 16 + 1);
		registration_current.assign(count, 0.0f);
		registration_reference.assign(count, 0.0f);
#ifdef HIGH_VISION_WITH_BFFT_MEYER
		bfft_meyer_plan_destroy(registration_meyer_plan);
		registration_meyer_plan = nullptr;
		meyer_input.assign(count, 0.0);
		meyer_cartoon.assign(count, 0.0);
		meyer_texture.assign(count, 0.0);
		if (cfg.meyer_registration &&
		    bfft_meyer_plan_create(
			    height, width, 0.05, 40.0,
			    cfg.meyer_registration_passes, 8, 1e-4, 4,
			    &registration_meyer_plan) == BFFT_OK) {
			if (bfft_meyer_plan_set_solver(
				    registration_meyer_plan, 1) != BFFT_OK) {
				bfft_meyer_plan_destroy(registration_meyer_plan);
				registration_meyer_plan = nullptr;
			}
		}
#endif
		display.assign(count, 0.0f);
		recovered_radiance.assign(count, 0.0f);
		current_census.assign(count, 0);
		previous_census.assign(count, 0);
		tiles_x = (width + static_cast<std::size_t>(cfg.tile_size) - 1) /
			  static_cast<std::size_t>(cfg.tile_size);
		tiles_y = (height + static_cast<std::size_t>(cfg.tile_size) - 1) /
			  static_cast<std::size_t>(cfg.tile_size);
		flow_x.assign(tiles_x * tiles_y, 0.0f);
		flow_y.assign(tiles_x * tiles_y, 0.0f);
			flow_confidence.assign(tiles_x * tiles_y, 1.0f);
		initialized = false;
		relative_exposure = 1.0f;
		telemetry_anchor = 0.0;
		display_black = 0.0f;
		display_white = 1.0f;
		moment_response_lut_power = -1.0f;
		previous_timestamp_ns = 0;
		moment_effective_fps = 30.0f;
		diag = {};
		if (stage)
			stage->reset(width, height);
	}

	float noise_sigma(float signal) const
	{
		const float read_variance = cfg.read_noise * cfg.read_noise;
		const float shot_variance =
			cfg.shot_noise * cfg.shot_noise * std::max(signal, 0.0f);
		return std::sqrt(read_variance + shot_variance);
	}

	void make_census(const std::vector<float> &source,
			 std::vector<std::uint8_t> &census)
	{
		std::fill(census.begin(), census.end(), 0);
		if (width < 3 || height < 3)
			return;
		for (std::size_t y = 1; y + 1 < height; ++y) {
			for (std::size_t x = 1; x + 1 < width; ++x) {
				const float c = source[y * width + x];
				unsigned bits = 0;
				bits |= source[y * width + x - 1] > c ? 1u : 0u;
				bits |= source[y * width + x + 1] > c ? 2u : 0u;
				bits |= source[(y - 1) * width + x] > c ? 4u : 0u;
				bits |= source[(y + 1) * width + x] > c ? 8u : 0u;
				bits |= source[(y - 1) * width + x - 1] > c ? 16u : 0u;
				bits |= source[(y + 1) * width + x + 1] > c ? 32u : 0u;
				census[y * width + x] =
					static_cast<std::uint8_t>(bits);
			}
		}
	}

	void pool_registration_witness(const std::vector<float> &source,
				       std::vector<float> &pooled)
	{
		if (width < 3 || height < 3) {
			pooled = source;
			return;
		}
		for (std::size_t y = 0; y < height; ++y) {
			for (std::size_t x = 0; x < width; ++x) {
				const std::size_t xl = x ? x - 1 : x;
				const std::size_t xr = std::min(x + 1, width - 1);
				const std::size_t yu = y ? y - 1 : y;
				const std::size_t yd = std::min(y + 1, height - 1);
				const float center = source[y * width + x];
				const float axial =
					source[y * width + xl] +
					source[y * width + xr] +
					source[yu * width + x] +
					source[yd * width + x];
				const float diagonal =
					source[yu * width + xl] +
					source[yu * width + xr] +
					source[yd * width + xl] +
					source[yd * width + xr];
				pooled[y * width + x] =
					(4.0f * center + 2.0f * axial + diagonal) /
					16.0f;
			}
		}
	}

	bool make_meyer_registration_witness(const std::vector<float> &source,
					     std::vector<float> &witness)
	{
#ifdef HIGH_VISION_WITH_BFFT_MEYER
		if (!cfg.meyer_registration || !registration_meyer_plan)
			return false;
		if (bfft_meyer_plan_set_passes(
			    registration_meyer_plan,
			    cfg.meyer_registration_passes) != BFFT_OK)
			return false;
		for (std::size_t i = 0; i < source.size(); ++i)
			meyer_input[i] =
				255.0 * std::clamp(static_cast<double>(source[i]),
						   0.0, 1.0);
		if (bfft_meyer_split(
			    registration_meyer_plan, meyer_input.data(),
			    meyer_cartoon.data(), meyer_texture.data()) != BFFT_OK)
			return false;
		for (std::size_t i = 0; i < source.size(); ++i)
			witness[i] =
				static_cast<float>(meyer_cartoon[i] / 255.0);
		return true;
#else
		(void)source;
		(void)witness;
		return false;
#endif
	}

	float match_error(int dx, int dy, std::size_t x0, std::size_t y0,
			  std::size_t x1, std::size_t y1, int stride) const
	{
		double error = 0.0;
		std::size_t samples = 0;
		for (std::size_t y = y0; y < y1;
		     y += static_cast<std::size_t>(stride)) {
			const int py = static_cast<int>(y) - dy;
			if (py < 1 || py + 1 >= static_cast<int>(height))
				continue;
			for (std::size_t x = x0; x < x1;
			     x += static_cast<std::size_t>(stride)) {
				const int px = static_cast<int>(x) - dx;
				if (px < 1 || px + 1 >= static_cast<int>(width))
					continue;
				const unsigned difference =
					current_census[y * width + x] ^
					previous_census[static_cast<std::size_t>(py) *
								 width +
							 static_cast<std::size_t>(px)];
				error += bit_count(difference) / 6.0;
				++samples;
			}
		}
		if (!samples)
			return 1.0f;
		return static_cast<float>(error / static_cast<double>(samples));
	}

	void estimate_motion()
	{
		if (is_night_mode(cfg.mode)) {
			// The accumulated scene belief is the highest-SNR witness already
			// available in the stream. Compare a small spatial gather of the
			// incoming frame against a matched gather of that belief instead
			// of asking two photon-starved frames to register each other.
			const bool current_meyer =
				make_meyer_registration_witness(
					current, registration_current);
			const bool reference_meyer =
				make_meyer_registration_witness(
					belief, registration_reference);
			diag.meyer_registration_applied =
				current_meyer && reference_meyer;
			if (!diag.meyer_registration_applied) {
				pool_registration_witness(
					current, registration_current);
				pool_registration_witness(
					belief, registration_reference);
			}
			make_census(registration_current, current_census);
			make_census(registration_reference, previous_census);
		} else {
			diag.meyer_registration_applied = false;
			make_census(current, current_census);
			make_census(previous, previous_census);
		}
		const int radius = cfg.registration_radius;
		const std::size_t margin =
			static_cast<std::size_t>(radius + cfg.local_search_radius + 2);
		const std::size_t x0 = std::min(margin, width);
		const std::size_t y0 = std::min(margin, height);
		const std::size_t x1 = width > margin ? width - margin : width;
		const std::size_t y1 = height > margin ? height - margin : height;

		int best_dx = 0;
		int best_dy = 0;
		float best = std::numeric_limits<float>::infinity();
		for (int dy = -radius; dy <= radius; ++dy) {
			for (int dx = -radius; dx <= radius; ++dx) {
				float error = match_error(dx, dy, x0, y0, x1, y1, 4);
				// Resolve textureless/tied frames toward zero motion.
				error += 0.0002f * static_cast<float>(dx * dx + dy * dy);
				if (error < best) {
					best = error;
					best_dx = dx;
					best_dy = dy;
				}
			}
		}
		diag.global_dx = static_cast<float>(best_dx);
		diag.global_dy = static_cast<float>(best_dy);
		diag.registration_error = std::isfinite(best) ? best : 1.0f;

			for (std::size_t ty = 0; ty < tiles_y; ++ty) {
				for (std::size_t tx = 0; tx < tiles_x; ++tx) {
				const std::size_t tile_x0 =
					std::max<std::size_t>(1, tx * cfg.tile_size);
				const std::size_t tile_y0 =
					std::max<std::size_t>(1, ty * cfg.tile_size);
				const std::size_t tile_x1 = std::min(
					width - 1, (tx + 1) * cfg.tile_size);
				const std::size_t tile_y1 = std::min(
					height - 1, (ty + 1) * cfg.tile_size);
				int local_dx = best_dx;
				int local_dy = best_dy;
				float local_best =
					std::numeric_limits<float>::infinity();
				const int local_radius = cfg.local_search_radius;
				for (int oy = -local_radius; oy <= local_radius; ++oy) {
					for (int ox = -local_radius; ox <= local_radius; ++ox) {
						const int dx = best_dx + ox;
						const int dy = best_dy + oy;
						float error = match_error(
							dx, dy, tile_x0, tile_y0,
							tile_x1, tile_y1, 2);
						error += 0.001f *
							 static_cast<float>(ox * ox + oy * oy);
						if (error < local_best) {
							local_best = error;
							local_dx = dx;
							local_dy = dy;
						}
					}
				}
				const std::size_t i = ty * tiles_x + tx;
				flow_x[i] = static_cast<float>(local_dx);
				flow_y[i] = static_cast<float>(local_dy);
				flow_confidence[i] =
					std::clamp(1.0f - local_best * 1.5f, 0.1f, 1.0f);
			}
		}
		double confidence_sum = 0.0;
		for (float confidence : flow_confidence)
			confidence_sum += confidence;
		diag.registration_confidence = static_cast<float>(
			confidence_sum /
			std::max<std::size_t>(flow_confidence.size(), 1));
	}

	void flow_at(float x, float y, float &dx, float &dy, float &confidence) const
	{
		if (tiles_x == 1 && tiles_y == 1) {
			dx = flow_x[0];
			dy = flow_y[0];
			confidence = flow_confidence[0];
			return;
		}
		const float gx = x / static_cast<float>(cfg.tile_size) - 0.5f;
		const float gy = y / static_cast<float>(cfg.tile_size) - 0.5f;
		const int ix0 = std::clamp(static_cast<int>(std::floor(gx)), 0,
					   static_cast<int>(tiles_x) - 1);
		const int iy0 = std::clamp(static_cast<int>(std::floor(gy)), 0,
					   static_cast<int>(tiles_y) - 1);
		const int ix1 = std::min(ix0 + 1, static_cast<int>(tiles_x) - 1);
		const int iy1 = std::min(iy0 + 1, static_cast<int>(tiles_y) - 1);
		const float fx = std::clamp(gx - std::floor(gx), 0.0f, 1.0f);
		const float fy = std::clamp(gy - std::floor(gy), 0.0f, 1.0f);
		auto interpolate = [&](const std::vector<float> &field) {
			const float a = field[static_cast<std::size_t>(iy0) * tiles_x +
					      static_cast<std::size_t>(ix0)] *
						(1.0f - fx) +
					field[static_cast<std::size_t>(iy0) * tiles_x +
					      static_cast<std::size_t>(ix1)] *
						fx;
			const float b = field[static_cast<std::size_t>(iy1) * tiles_x +
					      static_cast<std::size_t>(ix0)] *
						(1.0f - fx) +
					field[static_cast<std::size_t>(iy1) * tiles_x +
					      static_cast<std::size_t>(ix1)] *
						fx;
			return a * (1.0f - fy) + b * fy;
		};
		dx = interpolate(flow_x);
		dy = interpolate(flow_y);
		confidence = interpolate(flow_confidence);
	}

	void transport()
	{
		for (std::size_t y = 0; y < height; ++y) {
			for (std::size_t x = 0; x < width; ++x) {
				float dx, dy, confidence;
				flow_at(static_cast<float>(x), static_cast<float>(y),
					dx, dy, confidence);
				const bool night = is_night_mode(cfg.mode);
				if (night) {
					// Do not recursively deform a high-support radiance
					// estimate through independently noisy tile flows.
					// Camera translation defines the conservative gauge;
					// coherent per-pixel innovation below moves object
					// support by replacing it where the observation says
					// the old world point is no longer present.
					dx = diag.global_dx;
					dy = diag.global_dy;
				}
				const float px = static_cast<float>(x) - dx;
				const float py = static_cast<float>(y) - dy;
				const std::size_t i = y * width + x;
				transported_belief[i] =
					bilinear(belief, width, height, px, py, current[i]);
				transported_support[i] =
					bilinear(support, width, height, px, py, 0.0f) *
					(cfg.mode == Mode::night_moments
						 ? 1.0f
						 : cfg.support_decay) *
					(night ? 1.0f : confidence);
				transported_variance[i] =
					bilinear(variance, width, height, px, py, 0.0f);
				transported_innovation[i] = night
					? bilinear(coherent_innovation, width, height,
						   px, py, 0.0f)
					: 0.0f;
			}
		}
	}

	float estimate_exposure(const FrameMetadata &metadata)
	{
		const double telemetry = metadata.exposure_seconds > 0.0 &&
						 metadata.analog_gain > 0.0
					 ? metadata.exposure_seconds *
						   metadata.analog_gain
					 : 0.0;
		if (telemetry > 0.0) {
			if (!(telemetry_anchor > 0.0))
				telemetry_anchor = telemetry;
			return static_cast<float>(
				std::clamp(telemetry / telemetry_anchor, 1e-4, 1e4));
		}
		if (cfg.mode == Mode::night_moments && initialized) {
			// This mode promotes frame-population spread into radiance.
			// Inferring exposure from that same spread would divide away
			// the evidence and can create a false gain trajectory. Without
			// telemetry, retain the existing camera gauge.
			return relative_exposure;
		}

		// Without telemetry, exposure and a global illumination change are
		// not identifiable from arbitrary video. Do not continuously invent
		// an exposure trajectory from noisy frame ratios. Accept an update
		// only when registered pixels agree on a coherent multiplicative
		// step; otherwise retain the existing gauge.
		if (!initialized)
			return 1.0f;

		std::vector<float> log_ratios;
		log_ratios.reserve(width * height / 16 + 1);
		const float minimum_signal =
			std::max(0.015f, cfg.shadow_floor + cfg.read_noise);
		for (std::size_t y = 2; y + 2 < height; y += 4) {
			for (std::size_t x = 2; x + 2 < width; x += 4) {
				const std::size_t i = y * width + x;
				float dx, dy, confidence;
				flow_at(static_cast<float>(x), static_cast<float>(y),
					dx, dy, confidence);
				if (confidence < 0.45f)
					continue;
				if (is_night_mode(cfg.mode)) {
					dx = diag.global_dx;
					dy = diag.global_dy;
				}
				const float prior = bilinear(
					previous, width, height,
					static_cast<float>(x) - dx,
					static_cast<float>(y) - dy, -1.0f);
				const float measured = current[i];
				if (prior > minimum_signal && prior < 0.90f &&
				    measured > minimum_signal && measured < 0.98f) {
					const float ratio = measured / prior;
					if (ratio > 0.25f && ratio < 4.0f)
						log_ratios.push_back(std::log(ratio));
				}
			}
		}
		if (log_ratios.size() < 16)
			return relative_exposure;
		const std::size_t middle = log_ratios.size() / 2;
		std::nth_element(log_ratios.begin(), log_ratios.begin() + middle,
				 log_ratios.end());
		const float step_log =
			std::clamp(log_ratios[middle], std::log(0.25f),
				   std::log(4.0f));
		std::vector<float> deviations;
		deviations.reserve(log_ratios.size());
		for (float value : log_ratios)
			deviations.push_back(std::abs(value - step_log));
		const std::size_t deviation_middle = deviations.size() / 2;
		std::nth_element(deviations.begin(),
				 deviations.begin() + deviation_middle,
				 deviations.end());
		const float mad = deviations[deviation_middle];
		if (mad > 0.08f || std::abs(step_log) < std::log(1.04f))
			return relative_exposure;

		return std::clamp(
			relative_exposure * std::exp(0.75f * step_log),
			0.01f, 100.0f);
	}

	void update_sensor_pattern()
	{
		if (!is_night_mode(cfg.mode) ||
		    !cfg.sensor_pattern_correction ||
		    cfg.sensor_pattern_learning_rate <= 0.0f ||
		    cfg.sensor_pattern_limit <= 0.0f)
			return;
		const float motion =
			std::hypot(diag.global_dx, diag.global_dy);
		if (motion < cfg.sensor_pattern_min_motion ||
		    diag.registration_confidence < 0.45f)
			return;

		// The transported belief predicts scene radiance at the current
		// detector location. What repeats at a detector pixel after removing
		// that prediction is a sensor-coordinate nuisance candidate. Remove
		// its frame-global median first: an arbitrary illumination/exposure
		// offset is not identifiable as fixed-pattern noise.
		sensor_residual_samples.clear();
		for (std::size_t y = 2; y + 2 < height; y += 4) {
			for (std::size_t x = 2; x + 2 < width; x += 4) {
				const std::size_t i = y * width + x;
				const float predicted_code =
					relative_exposure * transported_belief[i];
				const float residual =
					raw_current[i] - sensor_pattern[i] -
					predicted_code;
				if (std::isfinite(residual))
					sensor_residual_samples.push_back(residual);
			}
		}
		if (sensor_residual_samples.size() < 16)
			return;
		const std::size_t middle = sensor_residual_samples.size() / 2;
		std::nth_element(sensor_residual_samples.begin(),
				 sensor_residual_samples.begin() + middle,
				 sensor_residual_samples.end());
		const float global_residual = sensor_residual_samples[middle];

		const float rate = cfg.sensor_pattern_learning_rate;
		const float limit = cfg.sensor_pattern_limit;
		double mean = 0.0;
		for (std::size_t i = 0; i < sensor_pattern.size(); ++i) {
			const float predicted_code =
				relative_exposure * transported_belief[i];
			const float residual =
				raw_current[i] - sensor_pattern[i] -
				predicted_code - global_residual;
			const float sigma =
				noise_sigma(std::max(transported_belief[i], 0.0f));
			const float robust_residual =
				std::clamp(residual, -3.0f * sigma, 3.0f * sigma);
			sensor_pattern[i] = std::clamp(
				sensor_pattern[i] + rate * robust_residual,
				-limit, limit);
			mean += sensor_pattern[i];
		}

		// Fix the gauge: the spatially constant component belongs to the
		// black/exposure offset, never to the detector pattern.
		const float field_mean = static_cast<float>(
			mean / std::max<std::size_t>(sensor_pattern.size(), 1));
		double energy = 0.0;
		for (std::size_t i = 0; i < sensor_pattern.size(); ++i) {
			sensor_pattern[i] = std::clamp(
				sensor_pattern[i] - field_mean, -limit, limit);
			current[i] = clamp01(raw_current[i] - sensor_pattern[i]);
			energy += static_cast<double>(sensor_pattern[i]) *
				  sensor_pattern[i];
		}
		diag.sensor_pattern_rms = static_cast<float>(
			std::sqrt(energy /
				  std::max<std::size_t>(sensor_pattern.size(), 1)));
		++diag.sensor_pattern_updates;
	}

	void initialize_belief()
	{
		for (std::size_t i = 0; i < current.size(); ++i) {
			observation[i] = current[i] / relative_exposure;
			belief[i] = observation[i];
			support[i] = 1.0f;
			variance[i] = cfg.read_noise * cfg.read_noise;
			coherent_innovation[i] = 0.0f;
		}
		previous = current;
		previous_census = current_census;
		initialized = true;
		diag.reset = true;
	}

	void fuse()
	{
		double support_sum = 0.0;
		double change_sum = 0.0;
		std::size_t clipped = 0;
		const bool night = is_night_mode(cfg.mode);
		const bool likelihood =
			cfg.mode == Mode::night_likelihood;
		const bool moments =
			cfg.mode == Mode::night_moments;
		const float support_limit = moments
			? std::max(cfg.moment_min_support,
				   cfg.moment_integration_seconds *
					   moment_effective_fps)
			: cfg.support_limit;
		const float decay = cfg.support_decay;
		for (std::size_t i = 0; i < current.size(); ++i) {
			const float encoded = current[i];
			const float obs = encoded / std::max(relative_exposure, 1e-6f);
			observation[i] = obs;
			const float shadow = smoothstep(
				cfg.shadow_floor, cfg.shadow_floor + 0.06f, encoded);
			const float highlight = 1.0f - smoothstep(
				cfg.highlight_knee, 0.999f, encoded);
			const float reliability = shadow * highlight;
			if (encoded >= 0.999f)
				++clipped;

			float prior = transported_support[i] *
				      (decay / std::max(cfg.support_decay, 1e-6f));
			const float predicted = transported_belief[i];
			float observation_sigma = noise_sigma(obs);
			if (moments &&
			    transported_support[i] >= cfg.moment_min_support) {
				// Fast 420v capture can expose a much wider temporal
				// population than the nominal read/shot model predicts.
				// Once a local population exists, use its measured spread
				// to distinguish ordinary frame noise from coherent change.
				observation_sigma = std::max(
					observation_sigma,
					std::sqrt(std::max(
						transported_variance[i], 0.0f)));
			}
			const float residual = obs - predicted;
			float retention = 1.0f;
			float change_probability = 0.0f;
			if (likelihood) {
				// A one-sided sequential generalized likelihood ratio is
				// the evidence bank. Under the transported hypothesis its
				// expected increment is negative, so ordinary noise erodes
				// the bank. A persistent dark occluder contributes positive
				// evidence even when its samples approach zero.
				const float mean_variance =
					std::max(transported_variance[i], 0.0f) /
					std::max(transported_support[i], 1.0f);
				const float signal = std::max(
					{std::abs(obs), std::abs(predicted),
					 cfg.shadow_floor + cfg.read_noise});
				const float null_variance =
					noise_sigma(predicted) *
						noise_sigma(predicted) +
					mean_variance;
				const float separation = std::max(
					cfg.change_threshold * signal,
					2.0f * std::sqrt(null_variance));
				const float darker =
					std::max(predicted - separation, 0.0f);
				const float brighter = predicted + separation;
				const float inverse_two_variance =
					0.5f /
					std::max(null_variance, 1e-12f);
				const float null_squared =
					residual * residual;
				const float down_residual = obs - darker;
				const float up_residual = obs - brighter;
				const float down_increment = std::clamp(
					(null_squared -
					 down_residual * down_residual) *
						inverse_two_variance,
					-4.0f, 8.0f);
				const float up_increment = std::clamp(
					(null_squared -
					 up_residual * up_residual) *
						inverse_two_variance,
					-4.0f, 8.0f);
				const bool down = down_increment > up_increment;
				const float increment =
					down ? down_increment : up_increment;
				float bank = down
					? std::max(-transported_innovation[i], 0.0f)
					: std::max(transported_innovation[i], 0.0f);
				bank = std::clamp(
					bank + increment, 0.0f,
					cfg.likelihood_evidence_limit);
				coherent_innovation[i] = down ? -bank : bank;
				const float sequential_probability = smoothstep(
					cfg.likelihood_release_low,
					cfg.likelihood_release_high, bank);

				// A very strong single-frame contradiction should not wait
				// for the sequential bank. Retain Night's calibrated
				// instantaneous safety valve, while the likelihood bank
				// handles weaker but persistent evidence on later frames.
				const float mean_sigma =
					std::sqrt(mean_variance);
				const float total_sigma = std::hypot(
					observation_sigma, mean_sigma);
				const float instant_scale =
					cfg.change_threshold * signal +
					3.0f * total_sigma;
				const float normalized =
					residual /
					std::max(instant_scale, 1e-6f);
				const float directional =
					normalized < 0.0f
						? -1.25f * normalized
						: normalized;
				const float instantaneous_probability =
					smoothstep(0.90f, 1.70f,
						   std::abs(directional));
				change_probability = std::max(
					sequential_probability,
					instantaneous_probability);
				retention =
					1.0f - 0.98f * change_probability;
			} else if (night) {
				if (moments &&
				    transported_support[i] <
					    cfg.moment_min_support) {
					// Bootstrap a population before asking its moments
					// to adjudicate change. Otherwise the first noisy
					// disagreement continually bankrupts support and the
					// empirical variance can never nucleate.
					coherent_innovation[i] = 0.0f;
					retention = 1.0f;
				} else {
				// The old absolute threshold made shadow changes nearly
				// invisible: a 4% radiance drop is enormous at 5% signal
				// but tiny next to an absolute threshold of 8–15%.
				//
				// Compare the observation with the uncertainty of the
				// accumulated mean, not with the population variance, and
				// express the user threshold as a fractional scene change.
				const float mean_sigma = std::sqrt(
					std::max(transported_variance[i], 0.0f) /
					std::max(transported_support[i], 1.0f));
				const float total_sigma = std::hypot(
					observation_sigma, mean_sigma);
				const float signal = std::max(
					{std::abs(obs), std::abs(predicted),
					 cfg.shadow_floor + cfg.read_noise});
				const float scale =
					cfg.change_threshold * signal +
					3.0f * total_sigma;
				float normalized =
					residual / std::max(scale, 1e-6f);

				// Noise changes sign; an object or occlusion keeps pushing
				// in one direction. This leaky signed fusor is therefore a
				// cheap temporal likelihood-ratio surrogate. Negative
				// evidence gets a modest advantage because a dark object
				// occluding a brighter accumulated background is the
				// failure mode most likely to be integrated away.
				float memory = transported_innovation[i];
				if (memory * normalized < 0.0f)
					memory *= 0.25f;
				memory = std::clamp(
					0.72f * memory + normalized, -4.0f, 4.0f);
				coherent_innovation[i] = memory;
				const float directional =
					normalized < 0.0f ? -1.25f * normalized
							 : normalized;
				const float release_score = std::max(
					std::abs(directional),
					0.65f * std::abs(memory));
				change_probability =
					smoothstep(0.65f, 1.40f, release_score);
				retention = 1.0f - 0.98f * change_probability;
				}
			} else {
				const float scale =
					cfg.change_threshold +
					3.0f * observation_sigma +
					2.0f * std::sqrt(std::max(
						transported_variance[i], 0.0f));
				const float change =
					std::abs(residual) /
					std::max(scale, 1e-6f);
				retention = 1.0f /
					(1.0f + change * change * change * change);
				coherent_innovation[i] = 0.0f;
			}
			// A changed region releases inherited support continuously. This
			// is what lets object-bound belief move and be replaced without
			// explicit masks, identities, or lifecycle management.
			prior *= retention;

			const float observation_weight =
				encoded >= 0.999f
					? 0.0f
					: (night ? 1.0f
						 : std::max(reliability, 0.01f));
			const float total = prior + observation_weight;
			if (total > 1e-8f) {
				const float next =
					(prior * predicted + observation_weight * obs) /
					total;
				const float delta = obs - next;
				variance[i] =
					(prior * (transported_variance[i] +
						  (predicted - next) *
							  (predicted - next)) +
					 observation_weight * delta * delta) /
					total;
				belief[i] = std::max(next, 0.0f);
				support[i] = std::min(total, support_limit);
			} else {
				belief[i] = std::max(predicted, 0.0f);
				support[i] = prior;
				variance[i] = transported_variance[i];
			}
			support_sum += support[i];
			change_sum += change_probability;
		}
		diag.mean_support = static_cast<float>(
			support_sum / std::max<std::size_t>(support.size(), 1));
		diag.mean_change_probability = static_cast<float>(
			change_sum / std::max<std::size_t>(support.size(), 1));
		diag.clipped_fraction = static_cast<float>(clipped) /
					std::max<std::size_t>(current.size(), 1);
		diag.moment_effective_fps =
			moments ? moment_effective_fps : 0.0f;
		diag.moment_window_frames =
			moments ? support_limit : 0.0f;
	}

	std::pair<float, float>
	signal_range(const std::vector<float> &signal) const
	{
		constexpr std::size_t bins = 2048;
		std::array<std::size_t, bins> histogram{};
		float maximum = 0.0f;
		for (std::size_t i = 0; i < signal.size(); ++i)
			if (support[i] > 0.05f && std::isfinite(signal[i]))
				maximum = std::max(maximum, signal[i]);
		if (!(maximum > 1e-8f))
			return {0.0f, 1.0f};
		std::size_t population = 0;
		for (std::size_t i = 0; i < signal.size(); ++i) {
			if (support[i] <= 0.05f || !std::isfinite(signal[i]))
				continue;
			const std::size_t bin = std::min<std::size_t>(
				static_cast<std::size_t>(
					signal[i] / maximum *
					static_cast<float>(bins - 1)),
				bins - 1);
			++histogram[bin];
			++population;
		}
		auto percentile = [&](float q) {
			const std::size_t wanted = static_cast<std::size_t>(
				q * static_cast<float>(population - 1));
			std::size_t seen = 0;
			for (std::size_t bin = 0; bin < bins; ++bin) {
				seen += histogram[bin];
				if (seen > wanted)
					return maximum * static_cast<float>(bin) /
					       static_cast<float>(bins - 1);
			}
			return maximum;
		};
		return {percentile(cfg.black_percentile),
			percentile(cfg.white_percentile)};
	}

	void tone_map(float *output, std::size_t output_stride)
	{
		const std::vector<float> *radiance = &belief;
		if (cfg.mode == Mode::night_moments) {
			const float variance_floor =
				cfg.moment_variance_floor *
				cfg.moment_variance_floor;
			constexpr float response_floor = 1e-5f;
			if (moment_response_lut_power !=
			    cfg.moment_response_power) {
				const float response_black = std::pow(
					response_floor,
					cfg.moment_response_power);
				for (std::size_t bin = 0;
				     bin < moment_response_lut.size(); ++bin) {
					const float value =
						static_cast<float>(bin) /
						static_cast<float>(
							moment_response_lut.size() -
							1);
					moment_response_lut[bin] = std::max(
						std::pow(value + response_floor,
							 cfg.moment_response_power) -
							response_black,
						0.0f);
				}
				moment_response_lut_power =
					cfg.moment_response_power;
			}
			double sigma_sum = 0.0;
			double lift_sum = 0.0;
			for (std::size_t i = 0; i < belief.size(); ++i) {
				const float empirical_sigma = std::sqrt(std::max(
					variance[i] - variance_floor, 0.0f));
				const float support_confidence = smoothstep(
					1.0f, cfg.moment_min_support, support[i]);
				const float coordinate = std::clamp(
					belief[i], 0.0f, 1.0f) *
					static_cast<float>(
						moment_response_lut.size() - 1);
				const std::size_t low =
					static_cast<std::size_t>(coordinate);
				const std::size_t high = std::min(
					low + 1,
					moment_response_lut.size() - 1);
				const float fraction =
					coordinate - static_cast<float>(low);
				const float response =
					moment_response_lut[low] *
						(1.0f - fraction) +
					moment_response_lut[high] * fraction;
				const float lift =
					cfg.moment_variance_gain * empirical_sigma *
					support_confidence;
				recovered_radiance[i] = response + lift;
				sigma_sum += empirical_sigma;
				lift_sum += lift;
			}
			const double population = static_cast<double>(
				std::max<std::size_t>(belief.size(), 1));
			diag.mean_temporal_sigma =
				static_cast<float>(sigma_sum / population);
			diag.mean_moment_lift =
				static_cast<float>(lift_sum / population);
			radiance = &recovered_radiance;
		} else {
			diag.mean_temporal_sigma = 0.0f;
			diag.mean_moment_lift = 0.0f;
		}

		auto range = signal_range(*radiance);
		if (diag.frame_index <= 1 || !(display_white > display_black)) {
			display_black = range.first;
			display_white = range.second;
		} else {
			constexpr float response = 0.12f;
			display_black += response * (range.first - display_black);
			display_white += response * (range.second - display_white);
		}
		const float span = std::max(display_white - display_black, 1e-5f);
		const float log_den = std::log1p(8.0f);
		for (std::size_t i = 0; i < radiance->size(); ++i) {
			const float normalized =
				std::max(((*radiance)[i] - display_black) / span,
					 0.0f);
			const float mapped = std::log1p(8.0f * normalized) / log_den;
			display[i] =
				clamp01(cfg.tone_strength * mapped +
					(1.0f - cfg.tone_strength) * current[i]);
		}

		for (std::size_t y = 0; y < height; ++y) {
			float *row = output + y * output_stride;
			for (std::size_t x = 0; x < width; ++x) {
				const std::size_t i = y * width + x;
				float value = display[i];
				if (cfg.local_contrast > 0.0f) {
					const std::size_t xl = x ? x - 1 : x;
					const std::size_t xr = std::min(x + 1, width - 1);
					const std::size_t yu = y ? y - 1 : y;
					const std::size_t yd = std::min(y + 1, height - 1);
					const float local =
						(display[y * width + xl] +
						 display[y * width + xr] +
						 display[yu * width + x] +
						 display[yd * width + x]) *
						0.25f;
					value += cfg.local_contrast * (value - local);
				}
				row[x] = clamp01(value);
			}
		}
	}

	bool process(const float *input, std::size_t input_stride, float *output,
		     std::size_t output_stride, std::size_t new_width,
		     std::size_t new_height, const FrameMetadata &metadata)
	{
		if (!input || !output || !new_width || !new_height ||
		    input_stride < new_width || output_stride < new_width)
			return false;
		if (new_width != width || new_height != height)
			allocate(new_width, new_height);

		const float black = static_cast<float>(metadata.black_level);
		const float white = static_cast<float>(metadata.white_level);
		const float sensor_span =
			white > black + 1e-8f ? white - black : 1.0f;
		for (std::size_t y = 0; y < height; ++y) {
			for (std::size_t x = 0; x < width; ++x) {
				const std::size_t i = y * width + x;
				raw_current[i] = clamp01(
					(input[y * input_stride + x] - black) /
					sensor_span);
				current[i] =
					clamp01(raw_current[i] - sensor_pattern[i]);
			}
		}

		diag.reset = false;
		++diag.frame_index;
		if (cfg.mode == Mode::night_moments) {
			if (metadata.timestamp_ns > previous_timestamp_ns &&
			    previous_timestamp_ns != 0) {
				const double seconds =
					static_cast<double>(
						metadata.timestamp_ns -
						previous_timestamp_ns) *
					1e-9;
				if (seconds >= 1.0 / 240.0 && seconds <= 1.0) {
					const float measured_fps = static_cast<float>(
						std::clamp(1.0 / seconds, 1.0, 120.0));
					moment_effective_fps +=
						0.08f *
						(measured_fps -
						 moment_effective_fps);
				}
			}
			if (metadata.timestamp_ns != 0)
				previous_timestamp_ns = metadata.timestamp_ns;
			diag.moment_effective_fps = moment_effective_fps;
			diag.moment_window_frames =
				cfg.moment_integration_seconds *
				moment_effective_fps;
		}
		if (cfg.mode == Mode::passthrough) {
			for (std::size_t y = 0; y < height; ++y)
				std::copy_n(input + y * input_stride, width,
					    output + y * output_stride);
			previous = current;
			return true;
		}

		if (!initialized) {
			make_census(current, current_census);
			relative_exposure = estimate_exposure(metadata);
			diag.relative_exposure = relative_exposure;
			initialize_belief();
		} else {
			estimate_motion();
			const float scene_cut_threshold =
				is_night_mode(cfg.mode)
					? std::max(cfg.scene_cut_threshold, 0.65f)
					: cfg.scene_cut_threshold;
			if (diag.registration_error > scene_cut_threshold) {
				relative_exposure = 1.0f;
				telemetry_anchor = 0.0;
				relative_exposure = estimate_exposure(metadata);
				diag.relative_exposure = relative_exposure;
				initialize_belief();
			} else {
				transport();
				relative_exposure = estimate_exposure(metadata);
				diag.relative_exposure = relative_exposure;
				update_sensor_pattern();
				fuse();
				previous = current;
			}
		}
		if (cfg.mode == Mode::experimental && stage) {
			stage->process(observation.data(), transported_support.data(),
				       belief.data(), width, height, metadata, diag);
			for (float &value : belief)
				value = std::max(
					std::isfinite(value) ? value : 0.0f, 0.0f);
		}
		tone_map(output, output_stride);
		return true;
	}
};

Processor::Processor(Config config)
	: impl_(std::make_unique<Impl>(std::move(config)))
{
}

Processor::~Processor() = default;
Processor::Processor(Processor &&) noexcept = default;
Processor &Processor::operator=(Processor &&) noexcept = default;

void Processor::configure(const Config &config)
{
	const int old_tile_size = impl_->cfg.tile_size;
	impl_->cfg = config;
	impl_->sanitize_config();
	if (impl_->initialized && impl_->cfg.tile_size != old_tile_size)
		impl_->allocate(impl_->width, impl_->height);
}

const Config &Processor::config() const noexcept
{
	return impl_->cfg;
}

void Processor::set_experimental_stage(std::unique_ptr<ExperimentalStage> stage)
{
	impl_->stage = std::move(stage);
	if (impl_->stage && impl_->width && impl_->height)
		impl_->stage->reset(impl_->width, impl_->height);
}

const ExperimentalStage *Processor::experimental_stage() const noexcept
{
	return impl_->stage.get();
}

void Processor::reset()
{
	const std::size_t width = impl_->width;
	const std::size_t height = impl_->height;
	if (width && height)
		impl_->allocate(width, height);
	else
		impl_->diag = {};
}

bool Processor::process(const float *input, std::size_t input_stride,
			float *output, std::size_t output_stride,
			std::size_t width, std::size_t height,
			const FrameMetadata &metadata)
{
	return impl_->process(input, input_stride, output, output_stride, width,
			      height, metadata);
}

const Diagnostics &Processor::diagnostics() const noexcept
{
	return impl_->diag;
}

const std::vector<float> &Processor::belief() const noexcept
{
	return impl_->belief;
}

const std::vector<float> &Processor::support() const noexcept
{
	return impl_->support;
}

} // namespace high_vision
