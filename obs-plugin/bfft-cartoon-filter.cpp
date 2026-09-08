#include <obs-module.h>
#include <bfft/meyer.h>

#include <algorithm>
#include <array>
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
	return "BFFT Cartoon sync/async and High Vision realtime filters";
}

namespace {

constexpr const char *kCartoon = "cartoon_gain_v2";
constexpr const char *kTexture = "texture_gain_v2";
constexpr const char *kShading = "shading_gain_v2";
constexpr const char *kShadeC = "shading_rof_c_v2";
constexpr const char *kEffectSweeps = "effect_sweeps_v1";
constexpr const char *kThreads = "threads";
constexpr const char *kMode = "mode";
constexpr const char *kJumpLattice = "jump_lattice_v1";
constexpr const char *kDecomposition = "decomposition_v1";
constexpr const char *kFusedPasses = "fused_passes_v1";
constexpr const char *kRelief = "relief";
constexpr const char *kGloss = "gloss";
constexpr const char *kRecoveryGain = "recovery_gain";
constexpr const char *kInformationGain = "information_gain";
constexpr const char *kPhaseFolds = "phase_folds";

const char *kSyncEffect = R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d correction_image;
uniform float monochrome;

sampler_state linear_sampler {
	Filter = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
};

VertData VSDefault(VertData v_in)
{
	VertData v_out;
	v_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	v_out.uv = v_in.uv;
	return v_out;
}

float4 PSBfftCartoon(VertData v_in) : TARGET
{
	float4 source = image.Sample(linear_sampler, v_in.uv);
	float correction = correction_image.Sample(linear_sampler, v_in.uv).r;
	if (monochrome > 0.5)
		return float4(correction, correction, correction, source.a);
	source.rgb = saturate(source.rgb + correction);
	return source;
}

technique Draw {
	pass {
		vertex_shader = VSDefault(v_in);
		pixel_shader = PSBfftCartoon(v_in);
	}
}
)";

struct Filter {
	obs_source_t *source = nullptr;
	bool synchronous = false;
	bfft_meyer_plan *plan = nullptr;
	uint32_t work_width = 0;
	uint32_t work_height = 0;
	uint32_t frame_width = 0;
	uint32_t frame_height = 0;
	uint32_t content_width = 0;
	uint32_t content_height = 0;
	int plan_threads = 0;

	// bfft_meyer_split permits image==cartoon. The decoded input plane is
	// therefore replaced in place by cartoon, leaving only texture resident.
	std::vector<double> input;
	std::vector<double> texture;
	// Effect scratch is allocated only by modes that consume it. No mode
	// needs more than these two planes.
	std::vector<double> scratch_a;
	std::vector<double> scratch_b;
	struct Sample {
		uint32_t lo = 0;
		uint32_t hi = 0;
		double weight = 0.0;
	};
	std::vector<Sample> source_sample_x;
	std::vector<Sample> source_sample_y;
	std::vector<Sample> output_sample_x;
	std::vector<Sample> output_sample_y;

	std::mutex settings_mutex;
	std::mutex processing_mutex;
	double cartoon_gain = 1.0;
	double texture_gain = 0.0;
	double shading_gain = 0.0;
	double shade_c = 0.02;
	int effect_sweeps = 8;
	int threads = 6;
	int mode = 0;
	int jump_lattice = 512;
	int decomposition = 0;
	int fused_passes = 16;
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

	// The new macOS fast-capture source is GPU/synchronous.  Its bridge stages
	// only the aspect-correct reduced lattice, processes the previous surface,
	// then applies a floating-point luma correction to the native-resolution
	// source.  Full-resolution chroma therefore never takes the CPU round trip.
	gs_effect_t *sync_effect = nullptr;
	gs_texrender_t *sync_analysis = nullptr;
	std::array<gs_stagesurf_t *, 2> sync_stage{};
	std::array<bool, 2> sync_written{};
	size_t sync_stage_index = 0;
	gs_texture_t *sync_correction_texture = nullptr;
	std::vector<float> sync_correction;
	uint32_t sync_target_width = 0;
	uint32_t sync_target_height = 0;
	uint32_t sync_analysis_width = 0;
	uint32_t sync_analysis_height = 0;
	bool sync_ready = false;
	bool sync_monochrome = false;
};

int normalize_lattice(int value)
{
	if (value >= 2048)
		return 2048;
	if (value >= 1024)
		return 1024;
	if (value >= 512)
		return 512;
	return 256;
}

void choose_work_shape(uint32_t frame_width, uint32_t frame_height,
		       uint32_t long_side, uint32_t &work_width,
		       uint32_t &work_height, uint32_t &content_width,
		       uint32_t &content_height)
{
	// The semismooth finite-flow chart is defined on the complete periodic
	// spectrum, so both canvas axes must be powers of two.  Wide video uses a
	// half-height canvas: for 16:9 this retains 89% of the requested linear
	// detail while halving the jump cost.  Near-square sources use a square.
	const bool landscape = frame_width >= frame_height;
	const uint64_t long_extent = landscape ? frame_width : frame_height;
	const uint64_t short_extent = landscape ? frame_height : frame_width;
	const bool wide = 2 * long_extent >= 3 * short_extent;
	work_width = landscape ? long_side : (wide ? long_side / 2 : long_side);
	work_height = landscape ? (wide ? long_side / 2 : long_side) : long_side;

	const double scale = std::min(
		static_cast<double>(work_width) / frame_width,
		static_cast<double>(work_height) / frame_height);
	content_width = std::clamp(
		static_cast<uint32_t>(std::lround(frame_width * scale)),
		uint32_t{8}, work_width);
	content_height = std::clamp(
		static_cast<uint32_t>(std::lround(frame_height * scale)),
		uint32_t{8}, work_height);
}

uint32_t reflected_index(int64_t coordinate, uint32_t length)
{
	const int64_t period = 2 * static_cast<int64_t>(length);
	int64_t folded = coordinate % period;
	if (folded < 0)
		folded += period;
	if (folded >= static_cast<int64_t>(length))
		folded = period - folded - 1;
	return static_cast<uint32_t>(folded);
}

Filter::Sample linear_sample(double coordinate, uint32_t length)
{
	coordinate = std::clamp(coordinate, 0.0,
				static_cast<double>(length - 1));
	const uint32_t lo = static_cast<uint32_t>(std::floor(coordinate));
	const uint32_t hi = std::min(lo + 1, length - 1);
	return {lo, hi, coordinate - lo};
}

void update_resample_maps(Filter *filter, uint32_t frame_width,
			  uint32_t frame_height, uint32_t work_width,
			  uint32_t work_height, uint32_t content_width,
			  uint32_t content_height)
{
	if (filter->frame_width == frame_width &&
	    filter->frame_height == frame_height &&
	    filter->work_width == work_width &&
	    filter->work_height == work_height &&
	    filter->content_width == content_width &&
	    filter->content_height == content_height &&
	    filter->source_sample_x.size() == work_width &&
	    filter->source_sample_y.size() == work_height &&
	    filter->output_sample_x.size() == frame_width &&
	    filter->output_sample_y.size() == frame_height)
		return;

	filter->frame_width = frame_width;
	filter->frame_height = frame_height;
	filter->content_width = content_width;
	filter->content_height = content_height;
	filter->source_sample_x.resize(work_width);
	filter->source_sample_y.resize(work_height);
	filter->output_sample_x.resize(frame_width);
	filter->output_sample_y.resize(frame_height);
	const int64_t left = (work_width - content_width) / 2;
	const int64_t top = (work_height - content_height) / 2;

	for (uint32_t x = 0; x < work_width; ++x) {
		const uint32_t cx = reflected_index(
			static_cast<int64_t>(x) - left, content_width);
		const double sx = (cx + 0.5) * frame_width / content_width - 0.5;
		filter->source_sample_x[x] = linear_sample(sx, frame_width);
	}
	for (uint32_t y = 0; y < work_height; ++y) {
		const uint32_t cy = reflected_index(
			static_cast<int64_t>(y) - top, content_height);
		const double sy = (cy + 0.5) * frame_height / content_height - 0.5;
		filter->source_sample_y[y] = linear_sample(sy, frame_height);
	}
	for (uint32_t x = 0; x < frame_width; ++x) {
		const double wx = left +
			(x + 0.5) * content_width / frame_width - 0.5;
		filter->output_sample_x[x] = linear_sample(wx, work_width);
	}
	for (uint32_t y = 0; y < frame_height; ++y) {
		const double wy = top +
			(y + 0.5) * content_height / frame_height - 0.5;
		filter->output_sample_y[y] = linear_sample(wy, work_height);
	}
}

bool ensure_plan(Filter *filter, uint32_t frame_width, uint32_t frame_height,
		 int threads, int jump_lattice)
{
	uint32_t work_width, work_height, content_width, content_height;
	const uint32_t long_side =
		static_cast<uint32_t>(normalize_lattice(jump_lattice));
	choose_work_shape(frame_width, frame_height, long_side, work_width,
			  work_height, content_width, content_height);
	if (filter->plan && filter->work_width == work_width &&
	    filter->work_height == work_height &&
	    filter->plan_threads == threads) {
		update_resample_maps(filter, frame_width, frame_height,
				     work_width, work_height, content_width,
				     content_height);
		return true;
	}

	// Destroy before replacement: a plan owns many image-sized buffers.
	bfft_meyer_plan_destroy(filter->plan);
	filter->plan = nullptr;
	filter->work_width = filter->work_height = 0;
	filter->frame_width = filter->frame_height = 0;
	filter->content_width = filter->content_height = 0;
	filter->plan_threads = 0;
	std::vector<double>().swap(filter->input);
	std::vector<double>().swap(filter->texture);
	std::vector<double>().swap(filter->scratch_a);
	std::vector<double>().swap(filter->scratch_b);

	bfft_status status = bfft_meyer_plan_create(
		work_height, work_width, 0.05, 40.0, 1, 1, 0.0,
		threads, &filter->plan);
	if (status != BFFT_OK) {
		blog(LOG_ERROR,
		     "[BFFT Cartoon] spectral finite-flow plan failed (%d) "
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
			     work_height, content_width, content_height);
	blog(LOG_INFO,
	     "[BFFT Cartoon] finite-flow spectral grid %ux%u, content %ux%u "
	     "for %ux%u input, %d threads",
	     work_width, work_height, content_width, content_height,
	     frame_width, frame_height, threads);
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
	for (uint32_t wy = 0; wy < wh; ++wy) {
		const Filter::Sample sy = filter->source_sample_y[wy];
		double *dst = filter->input.data() +
			      static_cast<size_t>(wy) * ww;
		for (uint32_t wx = 0; wx < ww; ++wx) {
			const Filter::Sample sx = filter->source_sample_x[wx];
			const double a = read_luma(frame, sx.lo, sy.lo);
			const double b = read_luma(frame, sx.hi, sy.lo);
			const double c = read_luma(frame, sx.lo, sy.hi);
			const double d = read_luma(frame, sx.hi, sy.hi);
			const double upper = a + sx.weight * (b - a);
			const double lower = c + sx.weight * (d - c);
			dst[wx] = upper + sy.weight * (lower - upper);
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
	for (uint32_t y = 0; y < frame->height; ++y) {
		const Filter::Sample sy = filter->output_sample_y[y];
		const double *upper = output + static_cast<size_t>(sy.lo) * ww;
		const double *lower = output + static_cast<size_t>(sy.hi) * ww;
		for (uint32_t x = 0; x < frame->width; ++x) {
			const Filter::Sample sx = filter->output_sample_x[x];
			const double a = upper[sx.lo] +
				sx.weight * (upper[sx.hi] - upper[sx.lo]);
			const double b = lower[sx.lo] +
				sx.weight * (lower[sx.hi] - lower[sx.lo]);
			write_luma(frame, x, y, a + sy.weight * (b - a),
				   monochrome);
		}
	}
}

bfft_status split_frame(Filter *filter, const double *image, double *cartoon,
			double *texture, int decomposition, int fused_passes)
{
	if (decomposition == 1) {
		const bfft_status configured = bfft_meyer_plan_set_passes(
			filter->plan, std::clamp(fused_passes, 1, 64));
		if (configured != BFFT_OK)
			return configured;
		// Gilles' model leaves a texture-side survivor w.  Expose the same
		// exact two-product contract as finite-flow by folding that survivor
		// into effective cartoon: (f-v, v).  Keep u in temporary storage so
		// image may alias cartoon without destroying f before the fold.
		double *model_cartoon = ensure_effect_plane(
			filter->scratch_b,
			static_cast<size_t>(filter->work_width) * filter->work_height);
		const bfft_status status = bfft_meyer_split_legacy(
			filter->plan, image, model_cartoon, texture);
		if (status != BFFT_OK)
			return status;
		for (size_t i = 0;
		     i < static_cast<size_t>(filter->work_width) *
				 filter->work_height;
		     ++i)
			cartoon[i] = image[i] - texture[i];
		return BFFT_OK;
	}
	// The realtime finite-flow endpoint validated in the surgical audit:
	// prefix 4, horizon 18, two exact refreshes, three jumps (19 map-cost).
	return bfft_meyer_split_flow_jump(filter->plan, image, cartoon, texture,
					  4, 18, 2, 3);
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
	filter->jump_lattice =
		static_cast<int>(obs_data_get_int(settings, kJumpLattice));
	filter->decomposition =
		static_cast<int>(obs_data_get_int(settings, kDecomposition));
	filter->fused_passes =
		static_cast<int>(obs_data_get_int(settings, kFusedPasses));
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
	obs_data_set_default_double(settings, kTexture, 0.0);
	obs_data_set_default_double(settings, kShading, 0.0);
	obs_data_set_default_double(settings, kShadeC, 0.02);
	obs_data_set_default_int(settings, kEffectSweeps, 8);
	obs_data_set_default_int(settings, kThreads, 6);
	obs_data_set_default_int(settings, kMode, 0);
	obs_data_set_default_int(settings, kJumpLattice, 512);
	obs_data_set_default_int(settings, kDecomposition, 0);
	obs_data_set_default_int(settings, kFusedPasses, 16);
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
	obs_property_list_add_int(mode, "Fine chrome (optional TVS)", 3);
	obs_property_list_add_int(mode, "Recursive recovery", 20);
	obs_property_list_add_int(mode, "Layer interference", 21);
	obs_property_list_add_int(mode, "Information caustics", 22);
	obs_property_t *decomposition = obs_properties_add_list(
		props, kDecomposition, "Decomposition engine",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(
		decomposition, "Finite-flow fast (4, 18, 2, 3)", 0);
	obs_property_list_add_int(decomposition, "Fused passes", 1);
	obs_properties_add_int_slider(
		props, kFusedPasses, "Fused pass count", 1, 64, 1);
	obs_property_t *lattice = obs_properties_add_list(
		props, kJumpLattice, "Spectral work lattice",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(lattice, "Preview (256 class)", 256);
	obs_property_list_add_int(lattice, "Detailed (512 class, default)", 512);
	obs_property_list_add_int(lattice, "High detail (1024 class)", 1024);
	obs_property_list_add_int(lattice, "Extreme detail (2048 class)", 2048);
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

void destroy_sync_graphics(Filter *filter)
{
	if (filter->sync_correction_texture) {
		gs_texture_destroy(filter->sync_correction_texture);
		filter->sync_correction_texture = nullptr;
	}
	for (auto *&stage : filter->sync_stage) {
		if (stage)
			gs_stagesurface_destroy(stage);
		stage = nullptr;
	}
	if (filter->sync_analysis) {
		gs_texrender_destroy(filter->sync_analysis);
		filter->sync_analysis = nullptr;
	}
	if (filter->sync_effect) {
		gs_effect_destroy(filter->sync_effect);
		filter->sync_effect = nullptr;
	}
}

void *filter_create_impl(obs_data_t *settings, obs_source_t *source,
			 bool synchronous)
{
	auto *filter = new Filter;
	filter->source = source;
	filter->synchronous = synchronous;
	filter_update(filter, settings);
	if (synchronous) {
		char *errors = nullptr;
		obs_enter_graphics();
		filter->sync_effect = gs_effect_create(
			kSyncEffect, "bfft-cartoon-sync.effect", &errors);
		obs_leave_graphics();
		if (errors) {
			blog(LOG_ERROR, "[BFFT Cartoon] sync shader: %s", errors);
			bfree(errors);
		}
		if (!filter->sync_effect) {
			delete filter;
			return nullptr;
		}
	}
	return filter;
}

void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	return filter_create_impl(settings, source, false);
}

void *sync_filter_create(obs_data_t *settings, obs_source_t *source)
{
	return filter_create_impl(settings, source, true);
}

void filter_destroy(void *data)
{
	auto *filter = static_cast<Filter *>(data);
	blog(LOG_INFO,
	     "[BFFT Cartoon] destroyed %s filter after %llu frames; "
	     "%llu plan build(s)",
	     filter->synchronous ? "sync" : "async",
	     static_cast<unsigned long long>(filter->frames),
	     static_cast<unsigned long long>(filter->plan_builds));
	bfft_meyer_plan_destroy(filter->plan);
	if (filter->synchronous) {
		obs_enter_graphics();
		destroy_sync_graphics(filter);
		obs_leave_graphics();
	}
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
	int effect_sweeps, threads, mode, jump_lattice, decomposition;
	int fused_passes;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		cartoon_gain = filter->cartoon_gain;
		texture_gain = filter->texture_gain;
		shading_gain = filter->shading_gain;
		shade_c = filter->shade_c;
		effect_sweeps = std::clamp(filter->effect_sweeps, 4, 16);
		threads = std::clamp(filter->threads, 1, 8);
		jump_lattice = normalize_lattice(filter->jump_lattice);
		decomposition = filter->decomposition == 1 ? 1 : 0;
		fused_passes = std::clamp(filter->fused_passes, 1, 64);
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

	if (!ensure_plan(filter, frame->width, frame->height, threads,
			 jump_lattice))
		return frame;

	const auto started = std::chrono::steady_clock::now();
	const uint32_t ww = filter->work_width, wh = filter->work_height;
	const size_t count = static_cast<size_t>(ww) * wh;
	read_work_input(filter, frame);
	const auto input_done = std::chrono::steady_clock::now();

	if (split_frame(filter, filter->input.data(), filter->input.data(),
			filter->texture.data(), decomposition,
			fused_passes) != BFFT_OK)
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
		// Both engine choices expose exact complementary products, so
		// recomposition needs no residual plane. Reuse cartoon/input for the
		// final field.
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
		// Fine chrome / TVS: one optional outer-map correction,
		// cartoon - ROF(cartoon, lambda). The complementary split contract
		// means no third residual plane is needed.
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
		if (split_frame(filter, filter->input.data(), filter->input.data(),
				recursive_texture, decomposition,
				fused_passes) != BFFT_OK)
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

bool sync_output_is_monochrome(Filter *filter)
{
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	if (filter->mode == 3 || filter->mode == 21)
		return true;
	return filter->mode == 0 && std::abs(filter->cartoon_gain) < 1e-12;
}

bool ensure_sync_graphics(Filter *filter, uint32_t target_width,
			  uint32_t target_height)
{
	int jump_lattice;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		jump_lattice = normalize_lattice(filter->jump_lattice);
	}
	uint32_t work_width, work_height, analysis_width, analysis_height;
	choose_work_shape(target_width, target_height,
			  static_cast<uint32_t>(jump_lattice), work_width,
			  work_height, analysis_width, analysis_height);
	if (filter->sync_analysis && filter->sync_target_width == target_width &&
	    filter->sync_target_height == target_height &&
	    filter->sync_analysis_width == analysis_width &&
	    filter->sync_analysis_height == analysis_height)
		return true;

	if (filter->sync_correction_texture) {
		gs_texture_destroy(filter->sync_correction_texture);
		filter->sync_correction_texture = nullptr;
	}
	for (auto *&stage : filter->sync_stage) {
		if (stage)
			gs_stagesurface_destroy(stage);
		stage = nullptr;
	}
	if (filter->sync_analysis)
		gs_texrender_destroy(filter->sync_analysis);
	filter->sync_analysis = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	filter->sync_stage[0] =
		gs_stagesurface_create(analysis_width, analysis_height, GS_RGBA);
	filter->sync_stage[1] =
		gs_stagesurface_create(analysis_width, analysis_height, GS_RGBA);
	filter->sync_correction_texture = gs_texture_create(
		analysis_width, analysis_height, GS_R32F, 1, nullptr, GS_DYNAMIC);
	filter->sync_correction.assign(
		static_cast<size_t>(analysis_width) * analysis_height, 0.0f);
	filter->sync_target_width = target_width;
	filter->sync_target_height = target_height;
	filter->sync_analysis_width = analysis_width;
	filter->sync_analysis_height = analysis_height;
	filter->sync_written = {false, false};
	filter->sync_stage_index = 0;
	filter->sync_ready = false;
	blog(LOG_INFO,
	     "[BFFT Cartoon] sync bridge %ux%u -> %ux%u reduced lattice",
	     target_width, target_height, analysis_width, analysis_height);
	return filter->sync_analysis && filter->sync_stage[0] &&
	       filter->sync_stage[1] && filter->sync_correction_texture;
}

void stage_sync_target(Filter *filter)
{
	obs_source_t *target = obs_filter_get_target(filter->source);
	if (!target)
		return;
	obs_source_t *parent = obs_filter_get_parent(filter->source);
	gs_texrender_reset(filter->sync_analysis);
	gs_viewport_push();
	gs_projection_push();
	gs_matrix_push();
	if (gs_texrender_begin(filter->sync_analysis,
			       filter->sync_analysis_width,
			       filter->sync_analysis_height)) {
		vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_matrix_identity();
		gs_ortho(0.0f, static_cast<float>(filter->sync_target_width),
			 0.0f, static_cast<float>(filter->sync_target_height),
			 -100.0f, 100.0f);
		const uint32_t flags = parent ? obs_source_get_output_flags(parent) : 0;
		const bool custom = (flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
		const bool async = (flags & OBS_SOURCE_ASYNC) != 0;
		if (target == parent && !custom && !async)
			obs_source_default_render(target);
		else
			obs_source_video_render(target);
		gs_texrender_end(filter->sync_analysis);
		gs_stage_texture(filter->sync_stage[filter->sync_stage_index],
				 gs_texrender_get_texture(filter->sync_analysis));
		filter->sync_written[filter->sync_stage_index] = true;
	}
	gs_matrix_pop();
	gs_projection_pop();
	gs_viewport_pop();
}

void process_sync_surface(Filter *filter, size_t read_index)
{
	if (!filter->sync_written[read_index])
		return;
	uint8_t *mapped = nullptr;
	uint32_t stride = 0;
	if (!gs_stagesurface_map(filter->sync_stage[read_index], &mapped, &stride))
		return;

	obs_source_frame frame{};
	frame.data[0] = mapped;
	frame.linesize[0] = stride;
	frame.width = filter->sync_analysis_width;
	frame.height = filter->sync_analysis_height;
	frame.format = VIDEO_FORMAT_RGBA;
	frame.full_range = true;
	const size_t count = static_cast<size_t>(frame.width) * frame.height;
	for (uint32_t y = 0; y < frame.height; ++y) {
		for (uint32_t x = 0; x < frame.width; ++x) {
			const size_t i = static_cast<size_t>(y) * frame.width + x;
			filter->sync_correction[i] =
				static_cast<float>(read_luma(&frame, x, y));
		}
	}

	filter_video(filter, &frame);
	const bool monochrome = sync_output_is_monochrome(filter);
	for (uint32_t y = 0; y < frame.height; ++y) {
		for (uint32_t x = 0; x < frame.width; ++x) {
			const size_t i = static_cast<size_t>(y) * frame.width + x;
			const float processed =
				static_cast<float>(read_luma(&frame, x, y));
			filter->sync_correction[i] = monochrome
				? processed / 255.0f
				: (processed - filter->sync_correction[i]) / 255.0f;
		}
	}
	gs_stagesurface_unmap(filter->sync_stage[read_index]);
	gs_texture_set_image(filter->sync_correction_texture,
			     reinterpret_cast<const uint8_t *>(
				     filter->sync_correction.data()),
			     filter->sync_analysis_width * sizeof(float), false);
	filter->sync_monochrome = monochrome;
	filter->sync_ready = true;
}

void sync_filter_render(void *data, gs_effect_t *)
{
	auto *filter = static_cast<Filter *>(data);
	obs_source_t *target = obs_filter_get_target(filter->source);
	if (!target) {
		obs_source_skip_video_filter(filter->source);
		return;
	}
	const uint32_t width = obs_source_get_base_width(target);
	const uint32_t height = obs_source_get_base_height(target);
	if (!width || !height || !ensure_sync_graphics(filter, width, height)) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	const size_t read_index = (filter->sync_stage_index + 1) % 2;
	process_sync_surface(filter, read_index);
	stage_sync_target(filter);
	filter->sync_stage_index = read_index;
	if (!filter->sync_ready) {
		obs_source_skip_video_filter(filter->source);
		return;
	}
	if (!obs_source_process_filter_begin(filter->source, GS_RGBA,
					     OBS_NO_DIRECT_RENDERING))
		return;
	gs_effect_set_texture(
		gs_effect_get_param_by_name(filter->sync_effect,
					    "correction_image"),
		filter->sync_correction_texture);
	gs_effect_set_float(
		gs_effect_get_param_by_name(filter->sync_effect, "monochrome"),
		filter->sync_monochrome ? 1.0f : 0.0f);
	obs_source_process_filter_end(filter->source, filter->sync_effect, width,
				      height);
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

obs_source_info sync_filter_info = {
	.id = "bfft_cartoon_filter_sync",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = filter_name,
	.create = sync_filter_create,
	.destroy = filter_destroy,
	.get_defaults = filter_defaults,
	.get_properties = filter_properties,
	.update = filter_update,
	.video_render = sync_filter_render,
};

} // namespace

bool obs_module_load(void)
{
	obs_register_source(&filter_info);
	obs_register_source(&sync_filter_info);
	register_high_vision_filter();
	blog(LOG_INFO, "[BFFT Cartoon] loaded");
	return true;
}
