#include <obs-module.h>
#include <high_vision/high_vision.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace {

constexpr const char *kMode = "high_vision_mode";
constexpr const char *kRegistrationRadius = "high_vision_registration_radius";
constexpr const char *kTileSize = "high_vision_tile_size";
constexpr const char *kLocalSearch = "high_vision_local_search";
constexpr const char *kSupport = "high_vision_support";
constexpr const char *kDecay = "high_vision_decay";
constexpr const char *kChangeThreshold = "high_vision_change_threshold";
constexpr const char *kSceneCut = "high_vision_scene_cut";
constexpr const char *kMomentPower = "high_vision_moment_power";
constexpr const char *kMomentVarianceGain =
	"high_vision_moment_variance_gain";
constexpr const char *kMomentVarianceFloor =
	"high_vision_moment_variance_floor";
constexpr const char *kMomentMinSupport =
	"high_vision_moment_min_support";
constexpr const char *kMomentIntegrationSeconds =
	"high_vision_moment_integration_seconds";
constexpr const char *kToneStrength = "high_vision_tone_strength";
constexpr const char *kLocalContrast = "high_vision_local_contrast";
constexpr const char *kReset = "high_vision_reset";

struct HighVisionFilter {
	obs_source_t *source = nullptr;
	high_vision::Processor processor;
	std::mutex settings_mutex;
	std::mutex processing_mutex;
	high_vision::Config config;
	high_vision::Mode applied_mode = high_vision::Mode::synthetic_hdr;
	bool has_applied_mode = false;
	bool reset_requested = false;
	std::uint32_t work_width = 0;
	std::uint32_t work_height = 0;
	std::vector<float> input;
	std::vector<float> output;
	std::uint64_t frames = 0;
	std::uint64_t resets = 0;
	double total_ms = 0.0;
};

const char *mode_name(high_vision::Mode mode)
{
	switch (mode) {
	case high_vision::Mode::passthrough:
		return "bypass";
	case high_vision::Mode::synthetic_hdr:
		return "synthetic-hdr";
	case high_vision::Mode::night_integrator:
		return "night-integrator";
	case high_vision::Mode::night_likelihood:
		return "night-likelihood";
	case high_vision::Mode::night_moments:
		return "night-moments";
	case high_vision::Mode::experimental:
		return "experimental";
	}
	return "unknown";
}

std::pair<std::uint32_t, std::uint32_t>
processing_grid(std::uint32_t width, std::uint32_t height)
{
	// Photon evidence, detector-fixed noise, and subpixel motion all live on
	// the native sensor lattice. Resizing before inference changes the
	// measurement. OBS may scale the finished image after this filter.
	return {width, height};
}

std::uint8_t clamp_byte(float value)
{
	return static_cast<std::uint8_t>(
		std::clamp(std::lround(value), 0L, 255L));
}

bool supported(enum video_format format)
{
	switch (format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
	case VIDEO_FORMAT_Y800:
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUVA:
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY:
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
	case VIDEO_FORMAT_RGBA:
	case VIDEO_FORMAT_BGR3:
		return true;
	default:
		return false;
	}
}

bool planar_luma(enum video_format format)
{
	switch (format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
	case VIDEO_FORMAT_Y800:
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUVA:
		return true;
	default:
		return false;
	}
}

bool yuv_luma(enum video_format format)
{
	return planar_luma(format) || format == VIDEO_FORMAT_YUY2 ||
	       format == VIDEO_FORMAT_YVYU || format == VIDEO_FORMAT_UYVY;
}

std::pair<float, float> luma_code_range(const obs_source_frame *frame)
{
	if (!yuv_luma(frame->format) || frame->full_range)
		return {0.0f, 1.0f};
	float minimum = frame->color_range_min[0];
	float maximum = frame->color_range_max[0];
	if (!(maximum > minimum + 1e-6f)) {
		minimum = 16.0f / 255.0f;
		maximum = 235.0f / 255.0f;
	}
	return {std::clamp(minimum, 0.0f, 1.0f),
		std::clamp(maximum, 0.0f, 1.0f)};
}

float normalize_code_luma(const obs_source_frame *frame, float value)
{
	const auto [minimum, maximum] = luma_code_range(frame);
	return std::clamp((value - minimum) /
				  std::max(maximum - minimum, 1e-6f),
			  0.0f, 1.0f);
}

float denormalize_code_luma(const obs_source_frame *frame, float value)
{
	const auto [minimum, maximum] = luma_code_range(frame);
	return minimum + std::clamp(value, 0.0f, 1.0f) *
				 (maximum - minimum);
}

float read_encoded_luma(const obs_source_frame *frame, std::uint32_t x,
			std::uint32_t y)
{
	const std::uint8_t *row =
		frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0];
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
	case VIDEO_FORMAT_Y800:
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUVA:
		return normalize_code_luma(frame, row[x] / 255.0f);
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
		return normalize_code_luma(frame, row[x * 2] / 255.0f);
	case VIDEO_FORMAT_UYVY:
		return normalize_code_luma(frame, row[x * 2 + 1] / 255.0f);
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX: {
		const std::uint8_t *p = row + x * 4;
		return static_cast<float>(
			       0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2]) /
		       255.0f;
	}
	case VIDEO_FORMAT_RGBA: {
		const std::uint8_t *p = row + x * 4;
		return static_cast<float>(
			       0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) /
		       255.0f;
	}
	case VIDEO_FORMAT_BGR3: {
		const std::uint8_t *p = row + x * 3;
		return static_cast<float>(
			       0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2]) /
		       255.0f;
	}
	default:
		return 0.0f;
	}
}

float decode_luma(float value)
{
	return value <= 0.04045f
		       ? value / 12.92f
		       : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float encode_luma(float value)
{
	value = std::max(value, 0.0f);
	return value <= 0.0031308f
		       ? value * 12.92f
		       : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

void write_encoded_luma(obs_source_frame *frame, std::uint32_t x,
			std::uint32_t y, float encoded, bool monochrome)
{
	std::uint8_t *row =
		frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0];
	const float byte_value =
		denormalize_code_luma(frame, encoded) * 255.0f;
	if (planar_luma(frame->format)) {
		row[x] = clamp_byte(byte_value);
		return;
	}
	switch (frame->format) {
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
		row[x * 2] = clamp_byte(byte_value);
		break;
	case VIDEO_FORMAT_UYVY:
		row[x * 2 + 1] = clamp_byte(byte_value);
		break;
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX: {
		std::uint8_t *p = row + x * 4;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(byte_value);
			break;
		}
		const float old = static_cast<float>(
			0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2]);
		const float delta = byte_value - old;
		p[0] = clamp_byte(p[0] + delta);
		p[1] = clamp_byte(p[1] + delta);
		p[2] = clamp_byte(p[2] + delta);
		break;
	}
	case VIDEO_FORMAT_RGBA: {
		std::uint8_t *p = row + x * 4;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(byte_value);
			break;
		}
		const float old = static_cast<float>(
			0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]);
		const float delta = byte_value - old;
		p[0] = clamp_byte(p[0] + delta);
		p[1] = clamp_byte(p[1] + delta);
		p[2] = clamp_byte(p[2] + delta);
		break;
	}
	case VIDEO_FORMAT_BGR3: {
		std::uint8_t *p = row + x * 3;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(byte_value);
			break;
		}
		const float old = static_cast<float>(
			0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2]);
		const float delta = byte_value - old;
		p[0] = clamp_byte(p[0] + delta);
		p[1] = clamp_byte(p[1] + delta);
		p[2] = clamp_byte(p[2] + delta);
		break;
	}
	default:
		break;
	}
}

void neutralize_chroma(obs_source_frame *frame)
{
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_I40A: {
		const std::uint32_t height = (frame->height + 1) / 2;
		const std::uint32_t width = (frame->width + 1) / 2;
		for (std::uint32_t y = 0; y < height; ++y) {
			std::fill_n(frame->data[1] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[1],
				    width, std::uint8_t{128});
			std::fill_n(frame->data[2] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[2],
				    width, std::uint8_t{128});
		}
		break;
	}
	case VIDEO_FORMAT_NV12: {
		const std::uint32_t height = (frame->height + 1) / 2;
		const std::uint32_t width = ((frame->width + 1) / 2) * 2;
		for (std::uint32_t y = 0; y < height; ++y)
			std::fill_n(frame->data[1] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[1],
				    width, std::uint8_t{128});
		break;
	}
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I42A: {
		const std::uint32_t width = (frame->width + 1) / 2;
		for (std::uint32_t y = 0; y < frame->height; ++y) {
			std::fill_n(frame->data[1] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[1],
				    width, std::uint8_t{128});
			std::fill_n(frame->data[2] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[2],
				    width, std::uint8_t{128});
		}
		break;
	}
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_YUVA:
		for (std::uint32_t y = 0; y < frame->height; ++y) {
			std::fill_n(frame->data[1] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[1],
				    frame->width, std::uint8_t{128});
			std::fill_n(frame->data[2] +
					    static_cast<std::size_t>(y) *
						    frame->linesize[2],
				    frame->width, std::uint8_t{128});
		}
		break;
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY:
		for (std::uint32_t y = 0; y < frame->height; ++y) {
			std::uint8_t *row =
				frame->data[0] + static_cast<std::size_t>(y) *
							 frame->linesize[0];
			for (std::uint32_t x = 0; x < frame->width; x += 2) {
				const std::size_t pair =
					static_cast<std::size_t>(x / 2) * 4;
				if (frame->format == VIDEO_FORMAT_UYVY) {
					row[pair] = 128;
					row[pair + 2] = 128;
				} else {
					row[pair + 1] = 128;
					row[pair + 3] = 128;
				}
			}
		}
		break;
	default:
		break;
	}
}

const char *filter_name(void *)
{
	return "BFFT High Vision";
}

void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, kMode, 1);
	obs_data_set_default_int(settings, kRegistrationRadius, 6);
	obs_data_set_default_int(settings, kTileSize, 24);
	obs_data_set_default_int(settings, kLocalSearch, 2);
	obs_data_set_default_double(settings, kSupport, 24.0);
	obs_data_set_default_double(settings, kDecay, 0.985);
	obs_data_set_default_double(settings, kChangeThreshold, 0.08);
	obs_data_set_default_double(settings, kSceneCut, 0.24);
	obs_data_set_default_double(settings, kMomentPower, 0.10);
	obs_data_set_default_double(settings, kMomentVarianceGain, 6.0);
	obs_data_set_default_double(settings, kMomentVarianceFloor, 0.002);
	obs_data_set_default_double(settings, kMomentMinSupport, 4.0);
	obs_data_set_default_double(settings, kMomentIntegrationSeconds, 4.0);
	obs_data_set_default_double(settings, kToneStrength, 1.0);
	obs_data_set_default_double(settings, kLocalContrast, 0.15);
}

void filter_update(void *data, obs_data_t *settings)
{
	auto *filter = static_cast<HighVisionFilter *>(data);
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	filter->config.mode = static_cast<high_vision::Mode>(
		std::clamp(static_cast<int>(obs_data_get_int(settings, kMode)),
			   0, 4));
	filter->config.registration_radius = static_cast<int>(
		obs_data_get_int(settings, kRegistrationRadius));
	filter->config.tile_size =
		static_cast<int>(obs_data_get_int(settings, kTileSize));
	filter->config.local_search_radius =
		static_cast<int>(obs_data_get_int(settings, kLocalSearch));
	filter->config.support_limit =
		static_cast<float>(obs_data_get_double(settings, kSupport));
	filter->config.support_decay =
		static_cast<float>(obs_data_get_double(settings, kDecay));
	filter->config.change_threshold =
		static_cast<float>(obs_data_get_double(settings, kChangeThreshold));
	filter->config.scene_cut_threshold =
		static_cast<float>(obs_data_get_double(settings, kSceneCut));
	filter->config.moment_response_power =
		static_cast<float>(obs_data_get_double(settings, kMomentPower));
	filter->config.moment_variance_gain = static_cast<float>(
		obs_data_get_double(settings, kMomentVarianceGain));
	filter->config.moment_variance_floor = static_cast<float>(
		obs_data_get_double(settings, kMomentVarianceFloor));
	filter->config.moment_min_support = static_cast<float>(
		obs_data_get_double(settings, kMomentMinSupport));
	filter->config.moment_integration_seconds = static_cast<float>(
		obs_data_get_double(settings, kMomentIntegrationSeconds));
	filter->config.tone_strength =
		static_cast<float>(obs_data_get_double(settings, kToneStrength));
	filter->config.local_contrast =
		static_cast<float>(obs_data_get_double(settings, kLocalContrast));
}

bool reset_clicked(obs_properties_t *, obs_property_t *, void *data)
{
	auto *filter = static_cast<HighVisionFilter *>(data);
	if (!filter)
		return false;
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	filter->reset_requested = true;
	return true;
}

obs_properties_t *filter_properties(void *data)
{
	obs_properties_t *properties = obs_properties_create();
	obs_property_t *mode = obs_properties_add_list(
		properties, kMode, "Mode", OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode, "Bypass", 0);
	obs_property_list_add_int(mode, "Synthetic HDR", 1);
	obs_property_list_add_int(mode, "Night integrator (persistent)", 2);
	obs_property_list_add_int(
		mode, "Night likelihood (experimental)", 3);
	obs_property_list_add_int(
		mode, "Night moments (420v experimental)", 4);
	obs_properties_add_int_slider(properties, kRegistrationRadius,
				      "Camera registration radius", 0, 16, 1);
	obs_properties_add_int_slider(properties, kTileSize,
				      "Organic support tile size", 8, 64, 1);
	obs_properties_add_int_slider(properties, kLocalSearch,
				      "Local support search radius", 0, 6, 1);
	obs_properties_add_float_slider(properties, kSupport,
					"Maximum evidence support", 2.0, 120.0, 1.0);
	obs_properties_add_float_slider(properties, kDecay,
					"Evidence persistence", 0.90, 1.0, 0.001);
	obs_properties_add_float_slider(properties, kChangeThreshold,
					"Object-change threshold", 0.01, 0.30, 0.005);
	obs_properties_add_float_slider(properties, kSceneCut,
					"Scene-cut threshold", 0.08, 0.80, 0.01);
	obs_properties_add_float_slider(
		properties, kMomentPower, "Moment response power",
		0.02, 1.0, 0.01);
	obs_properties_add_float_slider(
		properties, kMomentVarianceGain, "Variance-as-signal gain",
		0.0, 32.0, 0.25);
	obs_properties_add_float_slider(
		properties, kMomentVarianceFloor, "Temporal noise floor",
		0.0, 0.05, 0.0005);
	obs_properties_add_float_slider(
		properties, kMomentMinSupport, "Moment bootstrap support",
		1.0, 32.0, 1.0);
	obs_properties_add_float_slider(
		properties, kMomentIntegrationSeconds,
		"Moment integration window (seconds)", 1.0, 60.0, 1.0);
	obs_properties_add_float_slider(properties, kToneStrength,
					"HDR tone-map strength", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(properties, kLocalContrast,
					"Local contrast", 0.0, 1.0, 0.01);
	obs_properties_add_button(properties, kReset, "Reset temporal belief",
				  reset_clicked);
	(void)data;
	return properties;
}

void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *filter = new HighVisionFilter;
	filter->source = source;
	filter_update(filter, settings);
	return filter;
}

void filter_destroy(void *data)
{
	auto *filter = static_cast<HighVisionFilter *>(data);
	blog(LOG_INFO,
	     "[BFFT High Vision] destroyed after %llu frames (%.2f ms/frame)",
	     static_cast<unsigned long long>(filter->frames),
	     filter->frames ? filter->total_ms / filter->frames : 0.0);
	delete filter;
}

obs_source_frame *filter_video(void *data, obs_source_frame *frame)
{
	auto *filter = static_cast<HighVisionFilter *>(data);
	if (!frame || !frame->data[0] || frame->width < 8 ||
	    frame->height < 8 || !supported(frame->format))
		return frame;
	std::lock_guard<std::mutex> processing_lock(filter->processing_mutex);

	high_vision::Config config;
	bool reset = false;
	{
		std::lock_guard<std::mutex> settings_lock(filter->settings_mutex);
		config = filter->config;
		reset = filter->reset_requested;
		filter->reset_requested = false;
	}
	filter->processor.configure(config);
	const bool mode_changed =
		!filter->has_applied_mode ||
		config.mode != filter->applied_mode;
	if (reset || mode_changed)
		filter->processor.reset();
	if (mode_changed) {
		blog(LOG_INFO, "[BFFT High Vision] mode=%s; temporal state reset",
		     mode_name(config.mode));
		filter->applied_mode = config.mode;
		filter->has_applied_mode = true;
	}
	if (config.mode == high_vision::Mode::passthrough)
		return frame;

	const auto [work_width, work_height] =
		processing_grid(frame->width, frame->height);
	if (work_width != filter->work_width ||
	    work_height != filter->work_height) {
		filter->work_width = work_width;
		filter->work_height = work_height;
		const std::size_t count =
			static_cast<std::size_t>(work_width) * work_height;
		filter->input.resize(count);
		filter->output.resize(count);
		filter->processor.reset();
		blog(LOG_INFO, "[BFFT High Vision] processing grid %ux%u",
		     work_width, work_height);
		const auto [range_minimum, range_maximum] =
			luma_code_range(frame);
		blog(LOG_INFO,
		     "[BFFT High Vision] input format=%d, range=%s "
		     "(Y %.4f..%.4f), trc=%u",
		     static_cast<int>(frame->format),
		     frame->full_range ? "full" : "limited", range_minimum,
		     range_maximum, static_cast<unsigned>(frame->trc));
	}

	const auto started = std::chrono::steady_clock::now();
	for (std::uint32_t y = 0; y < work_height; ++y) {
		for (std::uint32_t x = 0; x < work_width; ++x) {
			filter->input[static_cast<std::size_t>(y) * work_width + x] =
				decode_luma(read_encoded_luma(frame, x, y));
		}
	}

	high_vision::FrameMetadata metadata;
	metadata.timestamp_ns = frame->timestamp;
	if (!filter->processor.process(
		    filter->input.data(), work_width, filter->output.data(),
		    work_width, work_width, work_height, metadata))
		return frame;
	if (filter->processor.diagnostics().reset)
		++filter->resets;

	const bool monochrome =
		config.mode == high_vision::Mode::night_integrator ||
		config.mode == high_vision::Mode::night_likelihood ||
		config.mode == high_vision::Mode::night_moments;
	for (std::uint32_t y = 0; y < frame->height; ++y) {
		for (std::uint32_t x = 0; x < frame->width; ++x) {
			const float value = filter->output[
				static_cast<std::size_t>(y) * work_width + x];
			write_encoded_luma(frame, x, y, encode_luma(value),
					   monochrome);
		}
	}
	if (monochrome)
		neutralize_chroma(frame);

	const double elapsed_ms =
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - started)
			.count();
	++filter->frames;
	filter->total_ms += elapsed_ms;
	if (filter->frames % 300 == 0) {
		const auto &diagnostics = filter->processor.diagnostics();
		blog(LOG_INFO,
		     "[BFFT High Vision] mode=%s, %.2f ms/frame, "
		     "motion=(%.1f, %.1f), confidence=%.2f, exposure=%.3f, "
		     "support=%.1f, "
		     "change=%.2f%%, resets=%llu, clipped=%.2f%%, "
		     "tgfd=%s, fpn=%.4f/%llu, temporal=%.4f, lift=%.4f, "
		     "window=%.1fs/%.0ff@%.1ffps",
		     mode_name(config.mode),
		     filter->total_ms / filter->frames, diagnostics.global_dx,
		     diagnostics.global_dy, diagnostics.registration_confidence,
		     diagnostics.relative_exposure,
		     diagnostics.mean_support,
		     diagnostics.mean_change_probability * 100.0f,
		     static_cast<unsigned long long>(filter->resets),
		     diagnostics.clipped_fraction * 100.0f,
		     diagnostics.meyer_registration_applied ? "on" : "off",
		     diagnostics.sensor_pattern_rms,
		     static_cast<unsigned long long>(
			     diagnostics.sensor_pattern_updates),
		     diagnostics.mean_temporal_sigma,
		     diagnostics.mean_moment_lift,
		     config.moment_integration_seconds,
		     diagnostics.moment_window_frames,
		     diagnostics.moment_effective_fps);
	}
	return frame;
}

obs_source_info filter_info = {
	.id = "bfft_high_vision_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_ASYNC_VIDEO,
	.get_name = filter_name,
	.create = filter_create,
	.destroy = filter_destroy,
	.get_defaults = filter_defaults,
	.get_properties = filter_properties,
	.update = filter_update,
	.filter_video = filter_video,
};

} // namespace

void register_high_vision_filter()
{
	obs_register_source(&filter_info);
	blog(LOG_INFO, "[BFFT High Vision] temporal imaging framework registered");
}
