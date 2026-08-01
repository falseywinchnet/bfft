#include <obs-module.h>
#include <bfft/meyer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include "../src/detail/bruun_simd_backend.hpp"
namespace bruun {
#include "../src/detail/MAG_REPRESENT_KERNEL.hpp"
}
namespace effect_trig = bruun;

void register_high_vision_filter();

OBS_DECLARE_MODULE()
MODULE_EXPORT const char *obs_module_description(void)
{
	return "BFFT Cartoon and High Vision realtime filters";
}

namespace {

constexpr const char *kCartoon = "cartoon_gain_v2";
constexpr const char *kTexture = "texture_gain_v2";
constexpr const char *kShading = "shading_gain_v2";
constexpr const char *kShadeC = "shading_rof_c_v2";
constexpr const char *kEffectSweeps = "effect_sweeps_v1";
constexpr const char *kThreads = "threads";
constexpr const char *kMode = "mode";
constexpr const char *kRelief = "relief";
constexpr const char *kGloss = "gloss";
constexpr const char *kRecoveryGain = "recovery_gain";
constexpr const char *kInformationGain = "information_gain";
constexpr const char *kPhaseFolds = "phase_folds";

struct Filter {
	obs_source_t *source = nullptr;
	bfft_meyer_plan *plan = nullptr;
	uint32_t work_width = 0;
	uint32_t work_height = 0;
	uint32_t frame_width = 0;
	uint32_t frame_height = 0;
	int plan_threads = 0;

	// bfft_meyer_split permits image==cartoon. The decoded input plane is
	// therefore replaced in place by cartoon, leaving only texture resident.
	std::vector<double> input;
	std::vector<double> texture;
	// Effect scratch is allocated only by modes that consume it. No mode
	// needs more than these two planes.
	std::vector<double> scratch_a;
	std::vector<double> scratch_b;
	std::vector<uint32_t> work_source_x;
	std::vector<uint32_t> work_source_y;
	uint32_t crop_left = 0;
	uint32_t crop_top = 0;

	std::mutex settings_mutex;
	std::mutex processing_mutex;
	double cartoon_gain = 1.0;
	double texture_gain = 1.0;
	double shading_gain = 0.0;
	double shade_c = 0.02;
	int effect_sweeps = 8;
	int threads = 6;
	int mode = 0;
	double relief = 1.0;
	double gloss = 0.75;
	double recovery_gain = 5.0;
	double information_gain = 2.0;
	double phase_folds = 6.0;

	uint64_t frames = 0;
	double total_ms = 0.0;
	double input_ms = 0.0;
	double split_ms = 0.0;
	double effect_ms = 0.0;
	uint64_t plan_builds = 0;
};

bool is_power_of_two(uint32_t value)
{
	return value >= 8 && (value & (value - 1)) == 0;
}

uint32_t next_power_of_two(uint32_t value)
{
	uint32_t result = 8;
	while (result < value && result <= UINT32_MAX / 2)
		result *= 2;
	return result;
}

void choose_work_shape(uint32_t frame_width, uint32_t frame_height,
		       uint32_t &work_width, uint32_t &work_height)
{
	work_width = frame_width;
	work_height = frame_height;
	if (is_power_of_two(frame_width) || is_power_of_two(frame_height))
		return;

	const uint32_t padded_width = next_power_of_two(frame_width);
	const uint32_t padded_height = next_power_of_two(frame_height);
	const uint64_t width_candidate =
		static_cast<uint64_t>(padded_width) * frame_height;
	const uint64_t height_candidate =
		static_cast<uint64_t>(frame_width) * padded_height;
	const uint64_t source_area =
		static_cast<uint64_t>(frame_width) * frame_height;
	const bool video_sized = source_area >= 1280ULL * 720ULL;
	// Row FFTs consume contiguous image lines. Column FFTs must gather and
	// scatter strided lines around every transform. Above the cache crossover,
	// prefer rows while their reflected lattice is within the measured 4/3
	// area envelope; otherwise retain the smaller column lattice.
	if (width_candidate <= height_candidate ||
	    (video_sized && 3 * width_candidate <= 4 * height_candidate)) {
		work_width = padded_width;
	} else {
		work_height = padded_height;
	}
}

void update_resample_maps(Filter *filter, uint32_t frame_width,
			  uint32_t frame_height, uint32_t work_width,
			  uint32_t work_height)
{
	if (filter->frame_width == frame_width &&
	    filter->frame_height == frame_height &&
	    filter->work_width == work_width &&
	    filter->work_height == work_height &&
	    filter->work_source_x.size() == work_width &&
	    filter->work_source_y.size() == work_height)
		return;

	filter->frame_width = frame_width;
	filter->frame_height = frame_height;
	filter->work_source_x.resize(work_width);
	filter->work_source_y.resize(work_height);
	const int64_t left =
		static_cast<int64_t>(work_width - frame_width) / 2;
	const int64_t top =
		static_cast<int64_t>(work_height - frame_height) / 2;
	filter->crop_left = static_cast<uint32_t>(left);
	filter->crop_top = static_cast<uint32_t>(top);
	auto reflected_index = [](int64_t coordinate, uint32_t length) {
		if (coordinate < 0)
			return static_cast<uint32_t>(-coordinate - 1);
		if (coordinate >= static_cast<int64_t>(length))
			return static_cast<uint32_t>(
				2 * static_cast<int64_t>(length) -
				coordinate - 1);
		return static_cast<uint32_t>(coordinate);
	};
	for (uint32_t x = 0; x < work_width; ++x)
		filter->work_source_x[x] =
			reflected_index(static_cast<int64_t>(x) - left,
					frame_width);
	for (uint32_t y = 0; y < work_height; ++y)
		filter->work_source_y[y] =
			reflected_index(static_cast<int64_t>(y) - top,
					frame_height);
}

bool ensure_plan(Filter *filter, uint32_t frame_width, uint32_t frame_height,
		 int threads)
{
	uint32_t work_width, work_height;
	choose_work_shape(frame_width, frame_height, work_width, work_height);
	if (filter->plan && filter->work_width == work_width &&
	    filter->work_height == work_height &&
	    filter->plan_threads == threads) {
		update_resample_maps(filter, frame_width, frame_height,
				     work_width, work_height);
		return true;
	}

	// Destroy before replacement: a plan owns many image-sized buffers.
	bfft_meyer_plan_destroy(filter->plan);
	filter->plan = nullptr;
	filter->work_width = filter->work_height = 0;
	filter->frame_width = filter->frame_height = 0;
	filter->plan_threads = 0;
	std::vector<double>().swap(filter->input);
	std::vector<double>().swap(filter->texture);
	std::vector<double>().swap(filter->scratch_a);
	std::vector<double>().swap(filter->scratch_b);

	bfft_status status = bfft_meyer_plan_create(
		work_height, work_width, 0.05, 40.0, 1, 1, 0.0,
		threads, &filter->plan);
	if (status == BFFT_OK)
		status = bfft_meyer_plan_set_solver(filter->plan, 1);
	if (status != BFFT_OK) {
		blog(LOG_ERROR,
		     "[BFFT Cartoon] arbitrary-size FACR plan failed (%d) "
		     "for %ux%u",
		     static_cast<int>(status), work_width, work_height);
		bfft_meyer_plan_destroy(filter->plan);
		filter->plan = nullptr;
		return false;
	}

	filter->work_width = work_width;
	filter->work_height = work_height;
	filter->plan_threads = threads;
	++filter->plan_builds;
	const size_t count = static_cast<size_t>(work_width) * work_height;
	filter->input.resize(count);
	filter->texture.resize(count);
	update_resample_maps(filter, frame_width, frame_height, work_width,
			     work_height);
	blog(LOG_INFO,
	     "[BFFT Cartoon] native-pitch jump FACR grid %ux%u for %ux%u "
	     "input, %d threads",
	     work_width, work_height, frame_width, frame_height, threads);
	return true;
}

uint8_t clamp_byte(double value)
{
	return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

bool is_planar_luma(enum video_format format)
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

double read_luma(const obs_source_frame *frame, uint32_t x, uint32_t y)
{
	const uint8_t *row = frame->data[0] +
			     static_cast<size_t>(y) * frame->linesize[0];
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
	case VIDEO_FORMAT_Y800:
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUVA:
		return row[x];
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
		return row[x * 2];
	case VIDEO_FORMAT_UYVY:
		return row[x * 2 + 1];
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX: {
		const uint8_t *p = row + x * 4;
		return 0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
	}
	case VIDEO_FORMAT_RGBA: {
		const uint8_t *p = row + x * 4;
		return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
	}
	case VIDEO_FORMAT_BGR3: {
		const uint8_t *p = row + x * 3;
		return 0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
	}
	default:
		return 0.0;
	}
}

void write_luma(obs_source_frame *frame, uint32_t x, uint32_t y,
		double output, bool monochrome = false)
{
	uint8_t *row = frame->data[0] +
		       static_cast<size_t>(y) * frame->linesize[0];
	if (is_planar_luma(frame->format)) {
		row[x] = clamp_byte(output);
		return;
	}
	switch (frame->format) {
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
		row[x * 2] = clamp_byte(output);
		break;
	case VIDEO_FORMAT_UYVY:
		row[x * 2 + 1] = clamp_byte(output);
		break;
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX: {
		uint8_t *p = row + x * 4;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(output);
		} else {
			const double old =
				0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
			const double delta = output - old;
			p[0] = clamp_byte(p[0] + delta);
			p[1] = clamp_byte(p[1] + delta);
			p[2] = clamp_byte(p[2] + delta);
		}
		break;
	}
	case VIDEO_FORMAT_RGBA: {
		uint8_t *p = row + x * 4;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(output);
		} else {
			const double old =
				0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
			const double delta = output - old;
			p[0] = clamp_byte(p[0] + delta);
			p[1] = clamp_byte(p[1] + delta);
			p[2] = clamp_byte(p[2] + delta);
		}
		break;
	}
	case VIDEO_FORMAT_BGR3: {
		uint8_t *p = row + x * 3;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(output);
		} else {
			const double old =
				0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
			const double delta = output - old;
			p[0] = clamp_byte(p[0] + delta);
			p[1] = clamp_byte(p[1] + delta);
			p[2] = clamp_byte(p[2] + delta);
		}
		break;
	}
	default:
		break;
	}
}

void neutralize_chroma(obs_source_frame *frame)
{
	const uint32_t width = frame->width;
	const uint32_t height = frame->height;
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_I40A:
		for (uint32_t y = 0; y < (height + 1) / 2; ++y) {
			std::fill_n(frame->data[1] + static_cast<size_t>(y) *
						   frame->linesize[1],
				    (width + 1) / 2, uint8_t{128});
			std::fill_n(frame->data[2] + static_cast<size_t>(y) *
						   frame->linesize[2],
				    (width + 1) / 2, uint8_t{128});
		}
		break;
	case VIDEO_FORMAT_NV12:
		for (uint32_t y = 0; y < (height + 1) / 2; ++y)
			std::fill_n(frame->data[1] + static_cast<size_t>(y) *
						   frame->linesize[1],
				    width, uint8_t{128});
		break;
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I42A:
		for (uint32_t y = 0; y < height; ++y) {
			std::fill_n(frame->data[1] + static_cast<size_t>(y) *
						   frame->linesize[1],
				    (width + 1) / 2, uint8_t{128});
			std::fill_n(frame->data[2] + static_cast<size_t>(y) *
						   frame->linesize[2],
				    (width + 1) / 2, uint8_t{128});
		}
		break;
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_YUVA:
		for (uint32_t y = 0; y < height; ++y) {
			std::fill_n(frame->data[1] + static_cast<size_t>(y) *
						   frame->linesize[1],
				    width, uint8_t{128});
			std::fill_n(frame->data[2] + static_cast<size_t>(y) *
						   frame->linesize[2],
				    width, uint8_t{128});
		}
		break;
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY:
		for (uint32_t y = 0; y < height; ++y) {
			uint8_t *row = frame->data[0] +
				       static_cast<size_t>(y) * frame->linesize[0];
			for (uint32_t x = 0; x < width; ++x) {
				const uint32_t offset =
					frame->format == VIDEO_FORMAT_UYVY
						? 0
						: 1;
				row[x * 2 + offset] = 128;
			}
		}
		break;
	default:
		break;
	}
}

bool supported(enum video_format format)
{
	switch (format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_UYVY:
	case VIDEO_FORMAT_RGBA:
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
	case VIDEO_FORMAT_Y800:
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_BGR3:
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I40A:
	case VIDEO_FORMAT_I42A:
	case VIDEO_FORMAT_YUVA:
		return true;
	default:
		return false;
	}
}

void read_work_input(Filter *filter, const obs_source_frame *frame)
{
	const uint32_t ww = filter->work_width;
	const uint32_t wh = filter->work_height;
	const bool exact_width = ww == frame->width;
	if (is_planar_luma(frame->format)) {
		for (uint32_t wy = 0; wy < wh; ++wy) {
			const uint8_t *row =
				frame->data[0] +
				static_cast<size_t>(
					filter->work_source_y[wy]) *
					frame->linesize[0];
			double *dst = filter->input.data() +
				      static_cast<size_t>(wy) * ww;
			if (exact_width) {
				for (uint32_t wx = 0; wx < ww; ++wx)
					dst[wx] = row[wx];
			} else {
				for (uint32_t wx = 0; wx < ww; ++wx)
					dst[wx] = row[filter->work_source_x[wx]];
			}
		}
		return;
	}

	for (uint32_t wy = 0; wy < wh; ++wy) {
		const uint32_t sy = filter->work_source_y[wy];
		double *dst = filter->input.data() +
			      static_cast<size_t>(wy) * ww;
		if (exact_width) {
			for (uint32_t wx = 0; wx < ww; ++wx)
				dst[wx] = read_luma(frame, wx, sy);
		} else {
			for (uint32_t wx = 0; wx < ww; ++wx)
				dst[wx] = read_luma(
					frame, filter->work_source_x[wx], sy);
		}
	}
}

double *ensure_effect_plane(std::vector<double> &plane, size_t count)
{
	if (plane.size() != count)
		plane.resize(count);
	return plane.data();
}

void write_work_output(Filter *filter, obs_source_frame *frame,
		       const double *output, bool monochrome)
{
	const uint32_t ww = filter->work_width;
	if (monochrome)
		neutralize_chroma(frame);
	if (is_planar_luma(frame->format)) {
		for (uint32_t y = 0; y < frame->height; ++y) {
			uint8_t *row =
				frame->data[0] +
				static_cast<size_t>(y) * frame->linesize[0];
			const double *src =
				output +
				static_cast<size_t>(filter->crop_top + y) * ww +
				filter->crop_left;
			for (uint32_t x = 0; x < frame->width; ++x)
				row[x] = clamp_byte(src[x]);
		}
		return;
	}

	for (uint32_t y = 0; y < frame->height; ++y) {
		const double *src =
			output +
			static_cast<size_t>(filter->crop_top + y) * ww +
			filter->crop_left;
		for (uint32_t x = 0; x < frame->width; ++x)
			write_luma(frame, x, y, src[x], monochrome);
	}
}

const char *filter_name(void *)
{
	return "BFFT Cartoon";
}

void filter_update(void *data, obs_data_t *settings)
{
	auto *filter = static_cast<Filter *>(data);
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	filter->cartoon_gain = obs_data_get_double(settings, kCartoon);
	filter->texture_gain = obs_data_get_double(settings, kTexture);
	filter->shading_gain = obs_data_get_double(settings, kShading);
	filter->shade_c = obs_data_get_double(settings, kShadeC);
	filter->effect_sweeps =
		static_cast<int>(obs_data_get_int(settings, kEffectSweeps));
	filter->threads = static_cast<int>(obs_data_get_int(settings, kThreads));
	filter->mode = static_cast<int>(obs_data_get_int(settings, kMode));
	filter->relief = obs_data_get_double(settings, kRelief);
	filter->gloss = obs_data_get_double(settings, kGloss);
	filter->recovery_gain =
		obs_data_get_double(settings, kRecoveryGain);
	filter->information_gain =
		obs_data_get_double(settings, kInformationGain);
	filter->phase_folds = obs_data_get_double(settings, kPhaseFolds);
}

void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_double(settings, kCartoon, 1.0);
	obs_data_set_default_double(settings, kTexture, 1.0);
	obs_data_set_default_double(settings, kShading, 0.0);
	obs_data_set_default_double(settings, kShadeC, 0.02);
	obs_data_set_default_int(settings, kEffectSweeps, 8);
	obs_data_set_default_int(settings, kThreads, 6);
	obs_data_set_default_int(settings, kMode, 0);
	obs_data_set_default_double(settings, kRelief, 1.0);
	obs_data_set_default_double(settings, kGloss, 0.75);
	obs_data_set_default_double(settings, kRecoveryGain, 5.0);
	obs_data_set_default_double(settings, kInformationGain, 2.0);
	obs_data_set_default_double(settings, kPhaseFolds, 6.0);
}

obs_properties_t *filter_properties(void *)
{
	obs_properties_t *props = obs_properties_create();
	obs_property_t *mode = obs_properties_add_list(
		props, kMode, "Display mode", OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode, "Cartoon + texture", 0);
	// Preserve value 3 so existing Fine chrome scenes remain selected.
	obs_property_list_add_int(mode, "Fine chrome", 3);
	obs_property_list_add_int(mode, "Recursive recovery", 20);
	obs_property_list_add_int(mode, "Layer interference", 21);
	obs_property_list_add_int(mode, "Information caustics", 22);
	obs_properties_add_float_slider(
		props, kCartoon, "Cartoon gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kTexture, "Texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kShading, "Shading gain (added)", -1.0, 4.0, 0.05);
	obs_properties_add_float_slider(
		props, kShadeC, "TV projection constant", 0.004, 0.2, 0.002);
	obs_properties_add_int_slider(
		props, kEffectSweeps, "TV effect sweeps", 4, 16, 1);
	obs_properties_add_int_slider(
		props, kThreads, "CPU threads", 1, 8, 1);
	obs_properties_add_float_slider(
		props, kRelief, "Chrome relief depth", 0.1, 4.0, 0.05);
	obs_properties_add_float_slider(
		props, kGloss, "Chrome gloss", 0.0, 1.0, 0.05);
	obs_properties_add_float_slider(
		props, kRecoveryGain, "Recovery boost", 0.0, 10.0, 0.1);
	obs_properties_add_float_slider(
		props, kInformationGain, "Information gain", 0.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kPhaseFolds, "Information phase folds", 1.0, 16.0,
		0.25);
	return props;
}

void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *filter = new Filter;
	filter->source = source;
	filter_update(filter, settings);
	return filter;
}

void filter_destroy(void *data)
{
	auto *filter = static_cast<Filter *>(data);
	blog(LOG_INFO,
	     "[BFFT Cartoon] destroyed after %llu frames; %llu plan build(s)",
	     static_cast<unsigned long long>(filter->frames),
	     static_cast<unsigned long long>(filter->plan_builds));
	bfft_meyer_plan_destroy(filter->plan);
	delete filter;
}

obs_source_frame *filter_video(void *data, obs_source_frame *frame)
{
	auto *filter = static_cast<Filter *>(data);
	if (!frame || !frame->data[0] || frame->width < 8 ||
	    frame->height < 8 || !supported(frame->format))
		return frame;

	std::lock_guard<std::mutex> processing_lock(filter->processing_mutex);
	double cartoon_gain, texture_gain, shading_gain, shade_c, relief, gloss;
	double recovery_gain, information_gain, phase_folds;
	int effect_sweeps, threads, mode;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		cartoon_gain = filter->cartoon_gain;
		texture_gain = filter->texture_gain;
		shading_gain = filter->shading_gain;
		shade_c = filter->shade_c;
		effect_sweeps = std::clamp(filter->effect_sweeps, 4, 16);
		threads = std::clamp(filter->threads, 1, 8);
		switch (filter->mode) {
		case 3:
		case 20:
		case 21:
		case 22:
			mode = filter->mode;
			break;
		default:
			mode = 0;
			break;
		}
		relief = filter->relief;
		gloss = filter->gloss;
		recovery_gain = filter->recovery_gain;
		information_gain = filter->information_gain;
		phase_folds = filter->phase_folds;
	}
	if (mode == 0 && std::abs(cartoon_gain - 1.0) < 1e-12 &&
	    std::abs(texture_gain - 1.0) < 1e-12 &&
	    std::abs(shading_gain) < 1e-12)
		return frame;

	if (!ensure_plan(filter, frame->width, frame->height, threads))
		return frame;

	const auto started = std::chrono::steady_clock::now();
	const uint32_t ww = filter->work_width, wh = filter->work_height;
	const size_t count = static_cast<size_t>(ww) * wh;
	read_work_input(filter, frame);
	const auto input_done = std::chrono::steady_clock::now();

	if (bfft_meyer_split(filter->plan, filter->input.data(),
			     filter->input.data(),
			     filter->texture.data()) != BFFT_OK)
		return frame;
	const auto split_done = std::chrono::steady_clock::now();

	if (mode == 0) {
		double *smooth = nullptr;
		if (std::abs(shading_gain) >= 1e-12) {
			smooth = ensure_effect_plane(filter->scratch_a, count);
			if (bfft_meyer_rof(filter->plan, filter->input.data(), smooth,
					   shade_c, 0.0, effect_sweeps,
					   0.0) != BFFT_OK)
				return frame;
		}
		// The jump split is complementary, so recomposition needs no model
		// residual plane. Reuse the cartoon/input plane for the final field.
		const bool signed_detail = std::abs(cartoon_gain) < 1e-12;
		for (size_t i = 0; i < count; ++i) {
			const double cartoon = filter->input[i];
			double value = cartoon_gain * cartoon +
				       texture_gain * filter->texture[i];
			if (std::abs(shading_gain) >= 1e-12)
				value += shading_gain *
					 (cartoon - smooth[i]);
			filter->input[i] = signed_detail ? 128.0 + value : value;
		}
		write_work_output(filter, frame, filter->input.data(),
				  signed_detail);
	} else if (mode == 3) {
		// Fine chrome: one accurate outer-map correction,
		// u_jump - ROF(u_jump, lambda). The complementary jump split makes
		// f-v exactly u, so no subtraction plane is needed.
		double *defect = ensure_effect_plane(filter->scratch_a, count);
		double *chrome_output =
			ensure_effect_plane(filter->scratch_b, count);
		if (bfft_meyer_rof(filter->plan, filter->input.data(), defect,
				   shade_c, 0.0, effect_sweeps,
				   0.0) != BFFT_OK)
			return frame;

		double energy = 0.0;
		for (size_t i = 0; i < count; ++i) {
			const double d = filter->input[i] - defect[i];
			defect[i] = d;
			energy += d * d;
		}
		const double rms =
			std::sqrt(energy / std::max<size_t>(count, 1));
		const double inv_scale =
			1.0 / std::max(3.0 * rms, 1e-6);

		// Shade once on the native-pitch work lattice.
		for (uint32_t wy = 0; wy < wh; ++wy) {
			for (uint32_t wx = 0; wx < ww; ++wx) {
				const size_t i = static_cast<size_t>(wy) * ww + wx;
				const double h = std::clamp(
					defect[i] * inv_scale,
					-1.0, 1.0);
				const uint32_t xl = wx ? wx - 1 : ww - 1;
				const uint32_t xr = wx + 1 < ww ? wx + 1 : 0;
				const uint32_t yu = wy ? wy - 1 : wh - 1;
				const uint32_t yd = wy + 1 < wh ? wy + 1 : 0;
				const double dx =
					(defect[
						 static_cast<size_t>(wy) * ww + xr] -
					 defect[
						 static_cast<size_t>(wy) * ww + xl]) *
					inv_scale;
				const double dy =
					(defect[
						 static_cast<size_t>(yd) * ww + wx] -
					 defect[
						 static_cast<size_t>(yu) * ww + wx]) *
					inv_scale;
				double nx = -relief * 2.5 * dx;
				double ny = -relief * 2.5 * dy;
				double nz = 1.0;
				const double nlen =
					std::sqrt(nx * nx + ny * ny + 1.0);
				nx /= nlen;
				ny /= nlen;
				nz /= nlen;
				const int ox = static_cast<int>(
					std::lround(nx * relief * 8.0));
				const int oy = static_cast<int>(
					std::lround(ny * relief * 8.0));
				const uint32_t sx = static_cast<uint32_t>(
					std::clamp(static_cast<int>(wx) + ox,
						   0, static_cast<int>(ww) - 1));
				const uint32_t sy = static_cast<uint32_t>(
					std::clamp(static_cast<int>(wy) + oy,
						   0, static_cast<int>(wh) - 1));
				const size_t displaced_index =
					static_cast<size_t>(sy) * ww + sx;
				const double displaced =
					filter->input[displaced_index] +
					filter->texture[displaced_index];
				const double light = std::max(
					0.0, -0.35 * nx - 0.45 * ny +
						     0.82 * nz);
				const double specular = std::pow(
					light, 8.0 + gloss * 72.0);
				double phase = 10.0 * ny + 3.0 * h;
				phase -= std::floor(
						 phase /
						 effect_trig::bruun_tau) *
					 effect_trig::bruun_tau;
				double environment_sine, environment_cosine;
				effect_trig::bruun_table256_poly3_sincos(
					phase, &environment_sine,
					&environment_cosine);
				const double environment =
					0.5 + 0.5 * environment_sine;
				const double chrome =
					20.0 + 85.0 * light +
					75.0 * environment +
					100.0 * gloss * specular;
				const double output =
					(0.35 - 0.2 * gloss) * displaced +
					chrome;
				chrome_output[i] = output;
			}
		}
		write_work_output(filter, frame, chrome_output, true);
	} else if (mode == 20) {
		// Repeat the split directly on the first cartoon. Both calls use
		// image==cartoon, so the complete two-stage path needs one extra
		// texture plane and no state copy.
		double *recursive_texture =
			ensure_effect_plane(filter->scratch_a, count);
		if (bfft_meyer_split(filter->plan, filter->input.data(),
				     filter->input.data(),
				     recursive_texture) != BFFT_OK)
			return frame;
		recovery_gain = std::clamp(recovery_gain, 0.0, 10.0);
		for (size_t i = 0; i < count; ++i) {
			// Former stage settings cartoon=1, texture=1+boost.
			filter->texture[i] = filter->input[i] +
				(1.0 + recovery_gain) * recursive_texture[i];
		}
		write_work_output(filter, frame, filter->texture.data(), false);
	} else if (mode == 21) {
		// The new split has no unassigned residual. Its informative second
		// layer is the cartoon-side TV defect: show its signed coupling to
		// material texture instead.
		double *smooth = ensure_effect_plane(filter->scratch_a, count);
		if (bfft_meyer_rof(filter->plan, filter->input.data(), smooth,
				   shade_c, 0.0, effect_sweeps, 0.0) != BFFT_OK)
			return frame;
		information_gain = std::clamp(information_gain, 0.0, 6.0);
		for (size_t i = 0; i < count; ++i) {
			const double defect = filter->input[i] - smooth[i];
			const double texture = filter->texture[i];
			const double magnitude = std::sqrt(
				defect * defect + texture * texture);
			const double coupling =
				2.0 * defect * texture /
				std::max(magnitude, 1e-6);
			filter->input[i] =
				128.0 + information_gain * coupling;
		}
		write_work_output(filter, frame, filter->input.data(), true);
	} else {
		// Information caustics. Material texture plus the cartoon-side TV
		// defect supplies geometry; their phase supplies the carrier.
		double *field = ensure_effect_plane(filter->scratch_a, count);
		double *caustic_output =
			ensure_effect_plane(filter->scratch_b, count);
		if (bfft_meyer_rof(filter->plan, filter->input.data(), field,
				   shade_c, 0.0, effect_sweeps, 0.0) != BFFT_OK)
			return frame;
		double energy = 0.0;
		for (size_t i = 0; i < count; ++i) {
			const double defect = filter->input[i] - field[i];
			field[i] = filter->texture[i] + defect;
			energy += field[i] * field[i];
		}
		const double inv_scale =
			1.0 / std::max(
				      3.0 * std::sqrt(
						    energy /
						    std::max<size_t>(count, 1)),
				      1e-6);
		information_gain = std::clamp(information_gain, 0.0, 6.0);
		phase_folds = std::clamp(phase_folds, 1.0, 16.0);
		for (uint32_t y = 0; y < wh; ++y) {
			const uint32_t yu = y ? y - 1 : wh - 1;
			const uint32_t yd = y + 1 < wh ? y + 1 : 0;
			for (uint32_t x = 0; x < ww; ++x) {
				const uint32_t xl = x ? x - 1 : ww - 1;
				const uint32_t xr = x + 1 < ww ? x + 1 : 0;
				const size_t i = static_cast<size_t>(y) * ww + x;
				const double texture = filter->texture[i];
				const double defect = field[i] - texture;
				const double magnitude =
					std::sqrt(defect * defect +
						  texture * texture);
				double phase =
					effect_trig::bruun_phase_atan2(
						defect, texture) *
					phase_folds;
				phase -= std::floor(
						 phase /
						 effect_trig::bruun_tau) *
					 effect_trig::bruun_tau;
				double carrier, carrier_cosine;
				effect_trig::bruun_table256_poly3_sincos(
					phase, &carrier, &carrier_cosine);

				const double dx =
					(field[
						 static_cast<size_t>(y) * ww + xr] -
					 field[
						 static_cast<size_t>(y) * ww + xl]) *
					inv_scale;
				const double dy =
					(field[
						 static_cast<size_t>(yd) * ww + x] -
					 field[
						 static_cast<size_t>(yu) * ww + x]) *
					inv_scale;
				double nx = -relief * dx;
				double ny = -relief * dy;
				const double nlen =
					std::sqrt(nx * nx + ny * ny + 1.0);
				nx /= nlen;
				ny /= nlen;
				const int ox = static_cast<int>(
					std::lround(nx * relief * 6.0));
				const int oy = static_cast<int>(
					std::lround(ny * relief * 6.0));
				const uint32_t sx = static_cast<uint32_t>(
					std::clamp(static_cast<int>(x) + ox,
						   0, static_cast<int>(ww) - 1));
				const uint32_t sy = static_cast<uint32_t>(
					std::clamp(static_cast<int>(y) + oy,
						   0, static_cast<int>(wh) - 1));
				const size_t displaced_index =
					static_cast<size_t>(sy) * ww + sx;
				const double displaced =
					filter->input[displaced_index] +
					filter->texture[displaced_index];
				const double presence =
					magnitude / (magnitude + 12.0);
				caustic_output[i] =
					displaced +
					information_gain * presence *
						(28.0 * carrier -
						 12.0 * nx - 10.0 * ny);
			}
		}
		write_work_output(filter, frame, caustic_output, false);
	}

	const auto effect_done = std::chrono::steady_clock::now();
	const double elapsed_ms = std::chrono::duration<double, std::milli>(
					  effect_done - started)
					  .count();
	filter->frames++;
	filter->total_ms += elapsed_ms;
	filter->input_ms +=
		std::chrono::duration<double, std::milli>(input_done - started)
			.count();
	filter->split_ms +=
		std::chrono::duration<double, std::milli>(split_done - input_done)
			.count();
	filter->effect_ms +=
		std::chrono::duration<double, std::milli>(effect_done - split_done)
			.count();
	if (filter->frames % 300 == 0) {
		blog(LOG_INFO,
		     "[BFFT Cartoon] %.2f ms/frame (input %.2f, split %.2f, "
		     "effect %.2f; %.1f fps capacity)",
		     filter->total_ms / filter->frames,
		     filter->input_ms / filter->frames,
		     filter->split_ms / filter->frames,
		     filter->effect_ms / filter->frames,
		     1000.0 * filter->frames / filter->total_ms);
	}
	return frame;
}

obs_source_info filter_info = {
	.id = "bfft_cartoon_filter",
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

bool obs_module_load(void)
{
	obs_register_source(&filter_info);
	register_high_vision_filter();
	blog(LOG_INFO, "[BFFT Cartoon] loaded");
	return true;
}
