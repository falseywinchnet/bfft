#include <obs-module.h>
#include <high_vision/high_vision.hpp>

#include <algorithm>
#include <array>
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
constexpr const char *kToneStrength = "high_vision_tone_strength";
constexpr const char *kLocalContrast = "high_vision_local_contrast";
constexpr const char *kReset = "high_vision_reset";
constexpr std::uint32_t kMaxWorkLongSide = 512;

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
	std::vector<float> chroma_u;
	std::vector<float> chroma_v;
	std::vector<float> chroma_belief_u;
	std::vector<float> chroma_belief_v;
	std::vector<float> chroma_support;
	std::vector<float> transported_chroma_u;
	std::vector<float> transported_chroma_v;
	std::vector<float> transported_chroma_support;
	bool chroma_initialized = false;
	float chroma_black_u = 0.0f;
	float chroma_black_v = 0.0f;
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
	case high_vision::Mode::experimental:
		return "experimental";
	}
	return "unknown";
}

std::pair<std::uint32_t, std::uint32_t>
processing_grid(std::uint32_t width, std::uint32_t height)
{
	const double scale = std::min(
		1.0,
		static_cast<double>(kMaxWorkLongSide) /
			static_cast<double>(std::max(width, height)));
	auto aligned = [scale](std::uint32_t value) {
		const double scaled = static_cast<double>(value) * scale;
		return std::max<std::uint32_t>(
			8, static_cast<std::uint32_t>(
				   std::max(1.0, std::round(scaled / 8.0))) *
				   8);
	};
	return {aligned(width), aligned(height)};
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

bool yuv_chroma(enum video_format format)
{
	switch (format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUVA:
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY:
		return true;
	default:
		return false;
	}
}

std::pair<float, float> chroma_code_center_scale(
	const obs_source_frame *frame)
{
	if (frame->full_range)
		return {127.5f, 127.5f};
	float minimum = frame->color_range_min[1] * 255.0f;
	float maximum = frame->color_range_max[1] * 255.0f;
	if (!(maximum > minimum + 1.0f)) {
		minimum = 16.0f;
		maximum = 240.0f;
	}
	return {0.5f * (minimum + maximum),
		0.5f * (maximum - minimum)};
}

float decode_chroma_code(const obs_source_frame *frame, std::uint8_t code)
{
	const auto [center, scale] = chroma_code_center_scale(frame);
	return std::clamp(
		(static_cast<float>(code) - center) / std::max(scale, 1.0f),
		-1.0f, 1.0f);
}

std::uint8_t encode_chroma_code(const obs_source_frame *frame, float value)
{
	const auto [center, scale] = chroma_code_center_scale(frame);
	return clamp_byte(center + scale * std::clamp(value, -1.0f, 1.0f));
}

bool read_chroma(const obs_source_frame *frame, std::uint32_t x,
		 std::uint32_t y, float &u, float &v)
{
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_I40A: {
		const std::uint8_t *ur =
			frame->data[1] + static_cast<std::size_t>(y / 2) *
						 frame->linesize[1];
		const std::uint8_t *vr =
			frame->data[2] + static_cast<std::size_t>(y / 2) *
						 frame->linesize[2];
		u = decode_chroma_code(frame, ur[x / 2]);
		v = decode_chroma_code(frame, vr[x / 2]);
		return true;
	}
	case VIDEO_FORMAT_NV12: {
		const std::uint8_t *row =
			frame->data[1] + static_cast<std::size_t>(y / 2) *
						 frame->linesize[1];
		const std::size_t offset = static_cast<std::size_t>(x / 2) * 2;
		u = decode_chroma_code(frame, row[offset]);
		v = decode_chroma_code(frame, row[offset + 1]);
		return true;
	}
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I42A: {
		const std::uint8_t *ur =
			frame->data[1] + static_cast<std::size_t>(y) *
						 frame->linesize[1];
		const std::uint8_t *vr =
			frame->data[2] + static_cast<std::size_t>(y) *
						 frame->linesize[2];
		u = decode_chroma_code(frame, ur[x / 2]);
		v = decode_chroma_code(frame, vr[x / 2]);
		return true;
	}
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_YUVA: {
		const std::uint8_t *ur =
			frame->data[1] + static_cast<std::size_t>(y) *
						 frame->linesize[1];
		const std::uint8_t *vr =
			frame->data[2] + static_cast<std::size_t>(y) *
						 frame->linesize[2];
		u = decode_chroma_code(frame, ur[x]);
		v = decode_chroma_code(frame, vr[x]);
		return true;
	}
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY: {
		const std::uint8_t *row =
			frame->data[0] + static_cast<std::size_t>(y) *
						 frame->linesize[0];
		const std::size_t pair = static_cast<std::size_t>(x / 2) * 4;
		if (frame->format == VIDEO_FORMAT_UYVY) {
			u = decode_chroma_code(frame, row[pair]);
			v = decode_chroma_code(frame, row[pair + 2]);
		} else if (frame->format == VIDEO_FORMAT_YUY2) {
			u = decode_chroma_code(frame, row[pair + 1]);
			v = decode_chroma_code(frame, row[pair + 3]);
		} else {
			v = decode_chroma_code(frame, row[pair + 1]);
			u = decode_chroma_code(frame, row[pair + 3]);
		}
		return true;
	}
	default:
		u = v = 0.0f;
		return false;
	}
}

void write_chroma(obs_source_frame *frame, std::uint32_t x,
		  std::uint32_t y, float u, float v)
{
	const std::uint8_t uc = encode_chroma_code(frame, u);
	const std::uint8_t vc = encode_chroma_code(frame, v);
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_I40A:
		frame->data[1][static_cast<std::size_t>(y / 2) *
				       frame->linesize[1] +
			       x / 2] = uc;
		frame->data[2][static_cast<std::size_t>(y / 2) *
				       frame->linesize[2] +
			       x / 2] = vc;
		break;
	case VIDEO_FORMAT_NV12: {
		std::uint8_t *row =
			frame->data[1] + static_cast<std::size_t>(y / 2) *
						 frame->linesize[1];
		const std::size_t offset = static_cast<std::size_t>(x / 2) * 2;
		row[offset] = uc;
		row[offset + 1] = vc;
		break;
	}
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I42A:
		frame->data[1][static_cast<std::size_t>(y) *
				       frame->linesize[1] +
			       x / 2] = uc;
		frame->data[2][static_cast<std::size_t>(y) *
				       frame->linesize[2] +
			       x / 2] = vc;
		break;
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_YUVA:
		frame->data[1][static_cast<std::size_t>(y) *
				       frame->linesize[1] +
			       x] = uc;
		frame->data[2][static_cast<std::size_t>(y) *
				       frame->linesize[2] +
			       x] = vc;
		break;
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY: {
		std::uint8_t *row =
			frame->data[0] + static_cast<std::size_t>(y) *
						 frame->linesize[0];
		const std::size_t pair = static_cast<std::size_t>(x / 2) * 4;
		if (frame->format == VIDEO_FORMAT_UYVY) {
			row[pair] = uc;
			row[pair + 2] = vc;
		} else if (frame->format == VIDEO_FORMAT_YUY2) {
			row[pair + 1] = uc;
			row[pair + 3] = vc;
		} else {
			row[pair + 1] = vc;
			row[pair + 3] = uc;
		}
		break;
	}
	default:
		break;
	}
}

float sample_field(const std::vector<float> &field, std::uint32_t width,
		   std::uint32_t height, float x, float y, float fallback)
{
	if (field.empty() || x < 0.0f || y < 0.0f ||
	    x > static_cast<float>(width - 1) ||
	    y > static_cast<float>(height - 1))
		return fallback;
	const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(x));
	const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(y));
	const std::uint32_t x1 = std::min(x0 + 1, width - 1);
	const std::uint32_t y1 = std::min(y0 + 1, height - 1);
	const float fx = x - static_cast<float>(x0);
	const float fy = y - static_cast<float>(y0);
	const float a = field[static_cast<std::size_t>(y0) * width + x0] *
				(1.0f - fx) +
			field[static_cast<std::size_t>(y0) * width + x1] * fx;
	const float b = field[static_cast<std::size_t>(y1) * width + x0] *
				(1.0f - fx) +
			field[static_cast<std::size_t>(y1) * width + x1] * fx;
	return a * (1.0f - fy) + b * fy;
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
			std::uint32_t y, float encoded)
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

void reset_chroma_state(HighVisionFilter *filter, std::size_t count)
{
	filter->chroma_u.assign(count, 0.0f);
	filter->chroma_v.assign(count, 0.0f);
	filter->chroma_belief_u.assign(count, 0.0f);
	filter->chroma_belief_v.assign(count, 0.0f);
	filter->chroma_support.assign(count, 0.0f);
	filter->transported_chroma_u.assign(count, 0.0f);
	filter->transported_chroma_v.assign(count, 0.0f);
	filter->transported_chroma_support.assign(count, 0.0f);
	filter->chroma_initialized = false;
	filter->chroma_black_u = 0.0f;
	filter->chroma_black_v = 0.0f;
}

float smooth_unit(float value)
{
	value = std::clamp(value, 0.0f, 1.0f);
	return value * value * (3.0f - 2.0f * value);
}

void integrate_night_chroma(HighVisionFilter *filter,
			    const high_vision::Config &config)
{
	const std::uint32_t width = filter->work_width;
	const std::uint32_t height = filter->work_height;
	const std::size_t count = static_cast<std::size_t>(width) * height;
	if (!count)
		return;
	const auto &diagnostics = filter->processor.diagnostics();
	const auto &luma_support = filter->processor.support();

	// Estimate the global per-channel black error from the darkest available
	// luma population. A histogram keeps this allocation-free and robust to
	// isolated colored objects. This is intentionally a slow nuisance gauge;
	// it cannot chase ordinary scene color from frame to frame.
	constexpr std::size_t bins = 129;
	std::array<std::size_t, bins> histogram_u{};
	std::array<std::size_t, bins> histogram_v{};
	std::size_t dark_count = 0;
	const float black_gate = config.shadow_floor + 0.025f;
	for (std::size_t i = 0; i < count; ++i) {
		if (filter->input[i] > black_gate)
			continue;
		const auto bin = [](float value) {
			return std::min<std::size_t>(
				static_cast<std::size_t>(
					std::lround((std::clamp(value, -1.0f, 1.0f) +
						     1.0f) *
						    0.5f * (bins - 1))),
				bins - 1);
		};
		++histogram_u[bin(filter->chroma_u[i])];
		++histogram_v[bin(filter->chroma_v[i])];
		++dark_count;
	}
	if (dark_count >= std::max<std::size_t>(32, count / 128)) {
		const auto median = [dark_count](
					    const std::array<std::size_t, bins> &histogram) {
			const std::size_t wanted = dark_count / 2;
			std::size_t seen = 0;
			for (std::size_t bin = 0; bin < bins; ++bin) {
				seen += histogram[bin];
				if (seen > wanted)
					return 2.0f * static_cast<float>(bin) /
						       static_cast<float>(bins - 1) -
					       1.0f;
			}
			return 0.0f;
		};
		constexpr float black_response = 0.01f;
		filter->chroma_black_u += black_response *
			(median(histogram_u) - filter->chroma_black_u);
		filter->chroma_black_v += black_response *
			(median(histogram_v) - filter->chroma_black_v);
		filter->chroma_black_u =
			std::clamp(filter->chroma_black_u, -0.25f, 0.25f);
		filter->chroma_black_v =
			std::clamp(filter->chroma_black_v, -0.25f, 0.25f);
	}
	for (std::size_t i = 0; i < count; ++i) {
		filter->chroma_u[i] = std::clamp(
			filter->chroma_u[i] - filter->chroma_black_u,
			-1.0f, 1.0f);
		filter->chroma_v[i] = std::clamp(
			filter->chroma_v[i] - filter->chroma_black_v,
			-1.0f, 1.0f);
	}
	if (!filter->chroma_initialized || diagnostics.reset) {
		filter->chroma_belief_u = filter->chroma_u;
		filter->chroma_belief_v = filter->chroma_v;
		for (std::size_t i = 0; i < count; ++i) {
			const float signal = smooth_unit(
				(filter->input[i] - config.shadow_floor) / 0.12f);
			filter->chroma_support[i] =
				0.02f + signal * signal;
		}
		filter->chroma_initialized = true;
		return;
	}

	const float dx = diagnostics.global_dx;
	const float dy = diagnostics.global_dy;
	for (std::uint32_t y = 0; y < height; ++y) {
		for (std::uint32_t x = 0; x < width; ++x) {
			const std::size_t i =
				static_cast<std::size_t>(y) * width + x;
			const float px = static_cast<float>(x) - dx;
			const float py = static_cast<float>(y) - dy;
			filter->transported_chroma_u[i] = sample_field(
				filter->chroma_belief_u, width, height, px, py,
				filter->chroma_u[i]);
			filter->transported_chroma_v[i] = sample_field(
				filter->chroma_belief_v, width, height, px, py,
				filter->chroma_v[i]);
			filter->transported_chroma_support[i] = sample_field(
				filter->chroma_support, width, height, px, py,
				0.0f) *
				config.support_decay;
		}
	}

	for (std::size_t i = 0; i < count; ++i) {
		const float signal = smooth_unit(
			(filter->input[i] - config.shadow_floor) / 0.12f);
		const float observation_weight =
			0.02f + signal * signal;
		float prior = std::min(
			filter->transported_chroma_support[i],
			i < luma_support.size() ? std::max(luma_support[i], 1.0f)
					       : 1.0f);
		const float du =
			filter->chroma_u[i] - filter->transported_chroma_u[i];
		const float dv =
			filter->chroma_v[i] - filter->transported_chroma_v[i];
		const float residual = std::hypot(du, dv);
		const float threshold = 0.08f + 0.20f * (1.0f - signal);
		const float release = signal * smooth_unit(
			(residual - threshold) / std::max(threshold, 1e-4f));
		prior *= 1.0f - 0.95f * release;
		const float total = prior + observation_weight;
		filter->chroma_belief_u[i] =
			(prior * filter->transported_chroma_u[i] +
			 observation_weight * filter->chroma_u[i]) /
			std::max(total, 1e-6f);
		filter->chroma_belief_v[i] =
			(prior * filter->transported_chroma_v[i] +
			 observation_weight * filter->chroma_v[i]) /
			std::max(total, 1e-6f);
		filter->chroma_support[i] =
			std::min(total, config.support_limit);
	}
}

void write_night_chroma(HighVisionFilter *filter, obs_source_frame *frame)
{
	if (!filter->chroma_initialized || !yuv_chroma(frame->format))
		return;
	std::uint32_t step_x = 1;
	std::uint32_t step_y = 1;
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_NV12:
		step_x = 2;
		step_y = 2;
		break;
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY:
		step_x = 2;
		break;
	default:
		break;
	}
	for (std::uint32_t y = 0; y < frame->height; y += step_y) {
		const std::uint32_t wy = std::min(
			static_cast<std::uint32_t>(
				static_cast<std::uint64_t>(y) *
				filter->work_height / frame->height),
			filter->work_height - 1);
		for (std::uint32_t x = 0; x < frame->width; x += step_x) {
			const std::uint32_t wx = std::min(
				static_cast<std::uint32_t>(
					static_cast<std::uint64_t>(x) *
					filter->work_width / frame->width),
				filter->work_width - 1);
			const std::size_t i =
				static_cast<std::size_t>(wy) *
					filter->work_width +
				wx;
			// Chroma earns saturation only from its own accumulated
			// support. Unsupported shadow chroma therefore approaches
			// neutral instead of being painted onto enhanced luma.
			const float gain =
				smooth_unit(filter->chroma_support[i] / 8.0f);
			write_chroma(frame, x, y,
				     gain * filter->chroma_belief_u[i],
				     gain * filter->chroma_belief_v[i]);
		}
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
	obs_data_set_default_double(settings, kToneStrength, 1.0);
	obs_data_set_default_double(settings, kLocalContrast, 0.15);
}

void filter_update(void *data, obs_data_t *settings)
{
	auto *filter = static_cast<HighVisionFilter *>(data);
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	filter->config.mode = static_cast<high_vision::Mode>(
		std::clamp(static_cast<int>(obs_data_get_int(settings, kMode)),
			   0, 2));
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
	if (reset || mode_changed)
		filter->chroma_initialized = false;
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
		reset_chroma_state(filter, count);
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
		const std::uint32_t source_y = std::min(
			static_cast<std::uint32_t>(
				static_cast<std::uint64_t>(y) * frame->height /
				work_height),
			frame->height - 1);
		for (std::uint32_t x = 0; x < work_width; ++x) {
			const std::uint32_t source_x = std::min(
				static_cast<std::uint32_t>(
					static_cast<std::uint64_t>(x) * frame->width /
					work_width),
				frame->width - 1);
			filter->input[static_cast<std::size_t>(y) * work_width + x] =
				decode_luma(read_encoded_luma(frame, source_x, source_y));
			float u = 0.0f;
			float v = 0.0f;
			if (read_chroma(frame, source_x, source_y, u, v)) {
				const std::size_t i =
					static_cast<std::size_t>(y) *
						work_width +
					x;
				filter->chroma_u[i] = u;
				filter->chroma_v[i] = v;
			}
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
	if (config.mode == high_vision::Mode::night_integrator &&
	    yuv_chroma(frame->format))
		integrate_night_chroma(filter, config);
	else
		filter->chroma_initialized = false;

	for (std::uint32_t y = 0; y < frame->height; ++y) {
		const std::uint32_t work_y = std::min(
			static_cast<std::uint32_t>(
				static_cast<std::uint64_t>(y) * work_height /
				frame->height),
			work_height - 1);
		for (std::uint32_t x = 0; x < frame->width; ++x) {
			const std::uint32_t work_x = std::min(
				static_cast<std::uint32_t>(
					static_cast<std::uint64_t>(x) * work_width /
					frame->width),
				work_width - 1);
			const float value = filter->output[
				static_cast<std::size_t>(work_y) * work_width + work_x];
			write_encoded_luma(frame, x, y, encode_luma(value));
		}
	}
	if (config.mode == high_vision::Mode::night_integrator)
		write_night_chroma(filter, frame);

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
		     "tgfd=%s, fpn=%.4f/%llu, uv-black=(%.3f, %.3f)",
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
		     filter->chroma_black_u, filter->chroma_black_v);
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
