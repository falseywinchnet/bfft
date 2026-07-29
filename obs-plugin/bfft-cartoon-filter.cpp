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
// Native-pitch plans have roughly nine times the old 512x288 area. A new
// key intentionally avoids inheriting an old 12-24-pass saved setting.
constexpr const char *kPasses = "native_passes_v1";
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
	int plan_passes = 0;
	int plan_threads = 0;

	std::vector<double> input;
	std::vector<double> cartoon;
	std::vector<double> texture;
	std::vector<double> projected_cartoon;
	std::vector<double> difference;
	std::vector<double> output;
	std::vector<double> recursive_texture;
	std::vector<uint32_t> work_source_x;
	std::vector<uint32_t> work_source_y;
	std::vector<uint32_t> frame_work_x;
	std::vector<uint32_t> frame_work_y;

	std::mutex settings_mutex;
	std::mutex processing_mutex;
	double cartoon_gain = 1.0;
	double texture_gain = 1.0;
	double shading_gain = 0.0;
	double shade_c = 0.02;
	int passes = 2;
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
	uint64_t pass_updates = 0;
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
	if (height_candidate <= width_candidate) {
		work_height = padded_height;
	} else {
		work_width = padded_width;
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
	filter->frame_work_x.resize(frame_width);
	filter->frame_work_y.resize(frame_height);
	const int64_t left =
		static_cast<int64_t>(work_width - frame_width) / 2;
	const int64_t top =
		static_cast<int64_t>(work_height - frame_height) / 2;
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
	for (uint32_t x = 0; x < frame_width; ++x)
		filter->frame_work_x[x] =
			static_cast<uint32_t>(left) + x;
	for (uint32_t y = 0; y < frame_height; ++y)
		filter->frame_work_y[y] =
			static_cast<uint32_t>(top) + y;
}

bool ensure_plan(Filter *filter, uint32_t frame_width, uint32_t frame_height,
		 int passes, int threads)
{
	uint32_t work_width, work_height;
	choose_work_shape(frame_width, frame_height, work_width, work_height);
	if (filter->plan && filter->work_width == work_width &&
	    filter->work_height == work_height &&
	    filter->plan_threads == threads) {
		if (filter->plan_passes != passes) {
			const bfft_status status =
				bfft_meyer_plan_set_passes(filter->plan, passes);
			if (status != BFFT_OK) {
				blog(LOG_ERROR,
				     "[BFFT Cartoon] pass update failed (%d)",
				     static_cast<int>(status));
				return false;
			}
			filter->plan_passes = passes;
			++filter->pass_updates;
		}
		update_resample_maps(filter, frame_width, frame_height,
				     work_width, work_height);
		return true;
	}

	// Destroy before replacement: a plan owns many image-sized buffers.
	bfft_meyer_plan_destroy(filter->plan);
	filter->plan = nullptr;
	filter->work_width = filter->work_height = 0;
	filter->frame_width = filter->frame_height = 0;
	filter->plan_passes = filter->plan_threads = 0;

	bfft_status status = bfft_meyer_plan_create(
		work_height, work_width, 0.05, 40.0, passes, 32, 1e-4,
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
	filter->plan_passes = passes;
	filter->plan_threads = threads;
	++filter->plan_builds;
	const size_t count = static_cast<size_t>(work_width) * work_height;
	filter->input.resize(count);
	filter->cartoon.resize(count);
	filter->texture.resize(count);
	filter->projected_cartoon.resize(count);
	filter->difference.resize(count);
	filter->output.resize(count);
	filter->recursive_texture.resize(count);
	update_resample_maps(filter, frame_width, frame_height, work_width,
			     work_height);
	blog(LOG_INFO,
	     "[BFFT Cartoon] native-pitch FACR grid %ux%u for %ux%u input, "
	     "%d passes, %d threads",
	     work_width, work_height, frame_width, frame_height, passes,
	     threads);
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
	if (is_planar_luma(frame->format)) {
		for (uint32_t wy = 0; wy < wh; ++wy) {
			const uint8_t *row =
				frame->data[0] +
				static_cast<size_t>(
					filter->work_source_y[wy]) *
					frame->linesize[0];
			double *dst = filter->input.data() +
				      static_cast<size_t>(wy) * ww;
			for (uint32_t wx = 0; wx < ww; ++wx)
				dst[wx] = row[filter->work_source_x[wx]];
		}
		return;
	}

	for (uint32_t wy = 0; wy < wh; ++wy) {
		const uint32_t sy = filter->work_source_y[wy];
		double *dst = filter->input.data() +
			      static_cast<size_t>(wy) * ww;
		for (uint32_t wx = 0; wx < ww; ++wx)
			dst[wx] = read_luma(
				frame, filter->work_source_x[wx], sy);
	}
}

void write_work_output(Filter *filter, obs_source_frame *frame,
		       bool monochrome)
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
				filter->output.data() +
				static_cast<size_t>(
					filter->frame_work_y[y]) *
					ww;
			for (uint32_t x = 0; x < frame->width; ++x)
				row[x] =
					clamp_byte(src[filter->frame_work_x[x]]);
		}
		return;
	}

	for (uint32_t y = 0; y < frame->height; ++y) {
		const double *src =
			filter->output.data() +
			static_cast<size_t>(filter->frame_work_y[y]) * ww;
		for (uint32_t x = 0; x < frame->width; ++x)
			write_luma(frame, x, y,
				   src[filter->frame_work_x[x]], monochrome);
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
	filter->passes = static_cast<int>(obs_data_get_int(settings, kPasses));
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
	obs_data_set_default_int(settings, kPasses, 2);
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
		props, kPasses, "Native quality / passes", 1, 8, 1);
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
	     "[BFFT Cartoon] destroyed after %llu frames; %llu plan build(s), "
	     "%llu allocation-free quality update(s)",
	     static_cast<unsigned long long>(filter->frames),
	     static_cast<unsigned long long>(filter->plan_builds),
	     static_cast<unsigned long long>(filter->pass_updates));
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
	int passes, threads, mode;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		cartoon_gain = filter->cartoon_gain;
		texture_gain = filter->texture_gain;
		shading_gain = filter->shading_gain;
		shade_c = filter->shade_c;
		passes = std::clamp(filter->passes, 1, 8);
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

	const int solve_passes =
		mode == 0 && std::abs(shading_gain) < 1e-12
			? passes
			: std::min(passes, 8);
	if (!ensure_plan(filter, frame->width, frame->height, solve_passes,
			 threads))
		return frame;

	const auto started = std::chrono::steady_clock::now();
	const uint32_t ww = filter->work_width, wh = filter->work_height;
	const size_t count = static_cast<size_t>(ww) * wh;
	read_work_input(filter, frame);
	const auto input_done = std::chrono::steady_clock::now();

	if (bfft_meyer_split(filter->plan, filter->input.data(),
			     filter->cartoon.data(),
			     filter->texture.data()) != BFFT_OK)
		return frame;
	const auto split_done = std::chrono::steady_clock::now();

	if (mode == 0) {
		if (std::abs(shading_gain) >= 1e-12 &&
		    bfft_meyer_rof(filter->plan, filter->cartoon.data(),
				   filter->projected_cartoon.data(), shade_c,
				   0.0, 8, 0.0) != BFFT_OK)
			return frame;
		// With no cartoon carrier, the remaining fields are signed detail.
		// Display them around neutral gray and remove source chroma; writing
		// near-zero Y while retaining NV12 U/V is what produced red flats.
		const bool signed_detail = std::abs(cartoon_gain) < 1e-12;
		for (size_t i = 0; i < count; ++i) {
			const double residual =
				filter->input[i] - filter->cartoon[i] -
				filter->texture[i];
			double value =
				residual + cartoon_gain * filter->cartoon[i] +
				texture_gain * filter->texture[i];
			if (std::abs(shading_gain) >= 1e-12)
				value += shading_gain *
					 (filter->cartoon[i] -
					  filter->projected_cartoon[i]);
			filter->output[i] =
				signed_detail ? 128.0 + value : value;
		}
		write_work_output(filter, frame, signed_detail);
	} else if (mode == 3) {
		// Fine chrome: one accurate outer-map correction,
		// u_TGFD - ROF(f - v_TGFD, lambda).
		for (size_t i = 0; i < count; ++i)
			filter->difference[i] =
				filter->input[i] - filter->texture[i];
		if (bfft_meyer_rof(filter->plan, filter->difference.data(),
				   filter->projected_cartoon.data(), shade_c,
				   0.0, std::clamp(passes, 8, 24),
				   0.0) != BFFT_OK)
			return frame;

		double energy = 0.0;
		for (size_t i = 0; i < count; ++i) {
			const double d = filter->cartoon[i] -
					 filter->projected_cartoon[i];
			filter->difference[i] = d;
			energy += d * d;
		}
		const double rms =
			std::sqrt(energy / std::max<size_t>(count, 1));
		const double inv_scale =
			1.0 / std::max(3.0 * rms, 1e-6);

		// The height field is work-grid sized. Shade it once there, then
		// expand; the previous full-frame loop repeated identical sin/pow
		// work for every source pixel mapped to the same work cell.
		for (uint32_t wy = 0; wy < wh; ++wy) {
			for (uint32_t wx = 0; wx < ww; ++wx) {
				const size_t i = static_cast<size_t>(wy) * ww + wx;
				const double h = std::clamp(
					filter->difference[i] * inv_scale,
					-1.0, 1.0);
				const uint32_t xl = wx ? wx - 1 : ww - 1;
				const uint32_t xr = wx + 1 < ww ? wx + 1 : 0;
				const uint32_t yu = wy ? wy - 1 : wh - 1;
				const uint32_t yd = wy + 1 < wh ? wy + 1 : 0;
				const double dx =
					(filter->difference[
						 static_cast<size_t>(wy) * ww + xr] -
					 filter->difference[
						 static_cast<size_t>(wy) * ww + xl]) *
					inv_scale;
				const double dy =
					(filter->difference[
						 static_cast<size_t>(yd) * ww + wx] -
					 filter->difference[
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
				const double displaced =
					filter->input[
						static_cast<size_t>(sy) * ww + sx];
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
				filter->output[i] = output;
			}
		}
		write_work_output(filter, frame, true);
	} else if (mode == 20) {
		// Repeat the decomposition on the residual-bearing cartoon state
		// g = f - v. This is the useful two-filter "recovery" behavior
		// without an intervening quantized OBS round trip.
		for (size_t i = 0; i < count; ++i)
			filter->difference[i] =
				filter->input[i] - filter->texture[i];
		if (bfft_meyer_split(filter->plan, filter->difference.data(),
				     filter->projected_cartoon.data(),
				     filter->recursive_texture.data()) != BFFT_OK)
			return frame;
		recovery_gain = std::clamp(recovery_gain, 0.0, 10.0);
		for (size_t i = 0; i < count; ++i)
			filter->output[i] =
				filter->difference[i] +
				recovery_gain * filter->recursive_texture[i];
		write_work_output(filter, frame, false);
	} else if (mode == 21) {
		// Quadrature-like cross term between the explicit texture and the
		// model residual. It is positive where they reinforce, negative
		// where they oppose, and zero where either channel is absent.
		information_gain = std::clamp(information_gain, 0.0, 6.0);
		for (size_t i = 0; i < count; ++i) {
			const double residual =
				filter->input[i] - filter->cartoon[i] -
				filter->texture[i];
			const double texture = filter->texture[i];
			const double magnitude = std::sqrt(
				residual * residual + texture * texture);
			const double coupling =
				2.0 * residual * texture /
				std::max(magnitude, 1e-6);
			filter->output[i] =
				128.0 + information_gain * coupling;
		}
		write_work_output(filter, frame, true);
	} else {
		// Information caustics. The non-cartoon field supplies surface
		// geometry; the phase between texture and residual supplies a
		// bounded optical carrier. Quiet regions remain unchanged.
		double energy = 0.0;
		for (size_t i = 0; i < count; ++i) {
			const double residual =
				filter->input[i] - filter->cartoon[i] -
				filter->texture[i];
			filter->difference[i] =
				filter->texture[i] + residual;
			energy += filter->difference[i] *
				  filter->difference[i];
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
				const double residual =
					filter->input[i] -
					filter->cartoon[i] -
					filter->texture[i];
				const double texture = filter->texture[i];
				const double magnitude =
					std::sqrt(residual * residual +
						  texture * texture);
				double phase =
					effect_trig::bruun_phase_atan2(
						residual, texture) *
					phase_folds;
				phase -= std::floor(
						 phase /
						 effect_trig::bruun_tau) *
					 effect_trig::bruun_tau;
				double carrier, carrier_cosine;
				effect_trig::bruun_table256_poly3_sincos(
					phase, &carrier, &carrier_cosine);

				const double dx =
					(filter->difference[
						 static_cast<size_t>(y) * ww + xr] -
					 filter->difference[
						 static_cast<size_t>(y) * ww + xl]) *
					inv_scale;
				const double dy =
					(filter->difference[
						 static_cast<size_t>(yd) * ww + x] -
					 filter->difference[
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
				const double displaced =
					filter->input[
						static_cast<size_t>(sy) * ww + sx];
				const double presence =
					magnitude / (magnitude + 12.0);
				filter->output[i] =
					displaced +
					information_gain * presence *
						(28.0 * carrier -
						 12.0 * nx - 10.0 * ny);
			}
		}
		write_work_output(filter, frame, false);
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
