#include <obs-module.h>
#include <bfft/meyer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

OBS_DECLARE_MODULE()
MODULE_EXPORT const char *obs_module_description(void)
{
	return "Realtime BFFT Meyer cartoon and texture decomposition";
}

namespace {

// v2 keys intentionally avoid inheriting the first prototype's persisted
// subtractive controls, whose zero points had different meanings.
constexpr const char *kCartoon = "cartoon_gain_v2";
constexpr const char *kTexture = "texture_gain_v2";
constexpr const char *kShading = "shading_gain_v2";
constexpr const char *kShadeC = "shading_rof_c_v2";
constexpr const char *kPasses = "passes";
constexpr const char *kThreads = "threads";
constexpr const char *kMode = "mode";
constexpr const char *kRelief = "relief";
constexpr const char *kGloss = "gloss";
constexpr const char *kLCartoon = "oklch_l_cartoon";
constexpr const char *kLTexture = "oklch_l_texture";
constexpr const char *kCCartoon = "oklch_c_cartoon";
constexpr const char *kCTexture = "oklch_c_texture";
constexpr const char *kHCartoon = "oklch_h_cartoon";
constexpr const char *kHTexture = "oklch_h_texture";
constexpr const char *kLDelayMs = "oklch_l_delay_ms_v2";
constexpr const char *kCDelayMs = "oklch_c_delay_ms_v2";
constexpr const char *kHDelayMs = "oklch_h_delay_ms_v2";
constexpr const char *kTemporalMix = "temporal_mix";
constexpr const char *kEchoTaps = "echo_taps";
constexpr const char *kEchoDecay = "echo_decay";
constexpr const char *kHueRotation = "hue_rotation";
constexpr const char *kChromaScale = "chroma_scale";
constexpr const char *kPinwheelTurns = "pinwheel_turns";
constexpr const char *kChromaDelayWeight = "chroma_delay_weight";
constexpr const char *kLeadDelayMs = "lead_scene_delay_ms";
constexpr const char *kLeadStrength = "lead_cartoon_strength";
constexpr const char *kLeadThreshold = "lead_motion_threshold";
constexpr const char *kLeadFeather = "lead_motion_feather";
constexpr const char *kLeadHueRotation = "lead_hue_rotation";
constexpr const char *kLeadChromaScale = "lead_chroma_scale";
constexpr const char *kTileSize = "glass_tile_size";
constexpr const char *kTileDelayMs = "glass_tile_delay_ms";
constexpr const char *kTileRefraction = "glass_tile_refraction";
constexpr const char *kTileBevel = "glass_tile_bevel";
constexpr const char *kTileRandomness = "glass_tile_randomness";
constexpr const char *kQTexture[4] = {
	"hue_q1_texture", "hue_q2_texture",
	"hue_q3_texture", "hue_q4_texture",
};
constexpr const char *kQDelayMs[4] = {
	"hue_q1_delay_ms", "hue_q2_delay_ms",
	"hue_q3_delay_ms", "hue_q4_delay_ms",
};
constexpr const char *kQHueRotation[4] = {
	"hue_q1_rotation", "hue_q2_rotation",
	"hue_q3_rotation", "hue_q4_rotation",
};
constexpr const char *kQPayload[4] = {
	"hue_q1_payload", "hue_q2_payload",
	"hue_q3_payload", "hue_q4_payload",
};
constexpr uint32_t kMaxWorkWidth = 512;
constexpr uint32_t kMaxWorkHeight = 256;
constexpr size_t kMaxHistoryFrames = 40;
constexpr uint64_t kHistoryMarginNs = 100'000'000ULL;
constexpr double kTau = 6.283185307179586476925286766559;

struct ColorHistoryFrame {
	uint64_t timestamp = 0;
	std::vector<uint16_t> lch;
};

struct Filter {
	obs_source_t *source = nullptr;
	bfft_meyer_plan *plan = nullptr;
	uint32_t work_width = 0;
	uint32_t work_height = 0;
	int plan_passes = 0;
	int plan_threads = 0;

	std::vector<double> input;
	std::vector<double> cartoon;
	std::vector<double> texture;
	std::vector<double> reference_cartoon;
	std::vector<double> difference;
	std::vector<double> work_lch;
	std::vector<double> color_output;
	std::vector<double> live_cartoon_lch;
	std::deque<ColorHistoryFrame> color_history;
	int history_mode = -1;

	std::mutex settings_mutex;
	std::mutex processing_mutex;
	double cartoon_gain = 1.0;
	double texture_gain = 1.0;
	double shading_gain = 0.0;
	double shade_c = 0.02;
	int passes = 12;
	int threads = 4;
	int mode = 0;
	double relief = 1.0;
	double gloss = 0.75;
	double l_cartoon = 1.0;
	double l_texture = 1.0;
	double c_cartoon = 1.0;
	double c_texture = 1.0;
	double h_cartoon = 1.0;
	double h_texture = 1.0;
	double component_delay_ms[3] = {0.0, 0.0, 0.0};
	double q_texture[4] = {1.0, 1.0, 1.0, 1.0};
	double q_delay_ms[4] = {0.0, 0.0, 0.0, 0.0};
	double q_hue_rotation[4] = {0.0, 0.0, 0.0, 0.0};
	int q_payload[4] = {0, 0, 0, 0};
	double temporal_mix = 1.0;
	int echo_taps = 4;
	double echo_decay = 0.65;
	double hue_rotation = 0.0;
	double chroma_scale = 1.0;
	double pinwheel_turns = 2.0;
	double chroma_delay_weight = 1.0;
	double lead_delay_ms = 180.0;
	double lead_strength = 1.0;
	double lead_threshold = 0.03;
	double lead_feather = 12.0;
	double lead_hue_rotation = 0.0;
	double lead_chroma_scale = 1.0;
	int tile_size = 32;
	double tile_delay_ms = 240.0;
	double tile_refraction = 8.0;
	double tile_bevel = 0.5;
	double tile_randomness = 0.35;

	uint64_t frames = 0;
	double total_ms = 0.0;
	uint64_t plan_builds = 0;
	uint64_t pass_updates = 0;
};

uint32_t floor_power_of_two(uint32_t value, uint32_t ceiling)
{
	value = std::min(value, ceiling);
	uint32_t result = 8;
	while (result <= value / 2)
		result *= 2;
	return result;
}

uint32_t nearest_power_of_two(uint32_t value, uint32_t ceiling)
{
	uint32_t lo = floor_power_of_two(value, ceiling);
	uint32_t hi = std::min(lo * 2, ceiling);
	return (value - lo < hi - value) ? lo : hi;
}

uint8_t clamp_byte(double value)
{
	return static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

bool ensure_plan(Filter *filter, uint32_t frame_width, uint32_t frame_height,
		 int passes, int threads)
{
	const uint32_t work_width =
		nearest_power_of_two(frame_width, kMaxWorkWidth);
	const uint32_t work_height =
		nearest_power_of_two(frame_height, kMaxWorkHeight);
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
		return true;
	}

	// A 512x256 Meyer engine owns dozens of image-sized double buffers and
	// per-lane FFT workspaces.  Release an incompatible engine before making
	// its replacement so a camera format/thread change never doubles that
	// peak.  Quality changes take the allocation-free path above.
	bfft_meyer_plan_destroy(filter->plan);
	filter->plan = nullptr;
	filter->work_width = 0;
	filter->work_height = 0;
	filter->plan_passes = 0;
	filter->plan_threads = 0;

	const bfft_status status = bfft_meyer_plan_create(
		work_height, work_width, 0.05, 40.0, passes, 32, 1e-4,
		threads, &filter->plan);
	if (status != BFFT_OK) {
		blog(LOG_ERROR, "[BFFT Cartoon] plan creation failed (%d)",
		     static_cast<int>(status));
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
	filter->reference_cartoon.resize(count);
	filter->difference.resize(count);
	filter->work_lch.resize(count * 3);
	filter->color_output.resize(count * 3);
	filter->live_cartoon_lch.resize(count * 3);
	filter->color_history.clear();
	filter->history_mode = -1;
	blog(LOG_INFO, "[BFFT Cartoon] processing grid %ux%u, %d passes, %d threads",
	     work_width, work_height, passes, threads);
	return true;
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
			break;
		}
		const double old = 0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
		const double delta = output - old;
		p[0] = clamp_byte(p[0] + delta);
		p[1] = clamp_byte(p[1] + delta);
		p[2] = clamp_byte(p[2] + delta);
		break;
	}
	case VIDEO_FORMAT_RGBA: {
		uint8_t *p = row + x * 4;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(output);
			break;
		}
		const double old = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
		const double delta = output - old;
		p[0] = clamp_byte(p[0] + delta);
		p[1] = clamp_byte(p[1] + delta);
		p[2] = clamp_byte(p[2] + delta);
		break;
	}
	case VIDEO_FORMAT_BGR3: {
		uint8_t *p = row + x * 3;
		if (monochrome) {
			p[0] = p[1] = p[2] = clamp_byte(output);
			break;
		}
		const double old = 0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
		const double delta = output - old;
		p[0] = clamp_byte(p[0] + delta);
		p[1] = clamp_byte(p[1] + delta);
		p[2] = clamp_byte(p[2] + delta);
		break;
	}
	default:
		break;
	}
}

double srgb_decode(double c)
{
	return c <= 0.04045 ? c / 12.92
			    : std::pow((std::abs(c) + 0.055) / 1.055, 2.4);
}

double srgb_encode(double c)
{
	return c <= 0.0031308 ? 12.92 * c
			      : 1.055 * std::pow(std::max(c, 0.0), 1.0 / 2.4) -
					0.055;
}

void rgb_to_oklch(const double rgb[3], double lch[3])
{
	const double r = srgb_decode(rgb[0]);
	const double g = srgb_decode(rgb[1]);
	const double b = srgb_decode(rgb[2]);
	const double l = std::cbrt(0.4122214708 * r + 0.5363325363 * g +
				  0.0514459929 * b);
	const double m = std::cbrt(0.2119034982 * r + 0.6806995451 * g +
				  0.1073969566 * b);
	const double s = std::cbrt(0.0883024619 * r + 0.2817188376 * g +
				  0.6299787005 * b);
	const double L = 0.2104542553 * l + 0.7936177850 * m -
			 0.0040720468 * s;
	const double a = 1.9779984951 * l - 2.4285922050 * m +
			 0.4505937099 * s;
	const double bb = 0.0259040371 * l + 0.7827717662 * m -
			  0.8086757660 * s;
	lch[0] = L;
	lch[1] = std::hypot(a, bb);
	double h = std::atan2(bb, a) / kTau;
	lch[2] = h < 0.0 ? h + 1.0 : h;
}

void oklch_to_rgb(const double lch[3], double rgb[3])
{
	const double angle = kTau * (lch[2] - std::floor(lch[2]));
	const double a = std::max(lch[1], 0.0) * std::cos(angle);
	const double b = std::max(lch[1], 0.0) * std::sin(angle);
	const double l_ = lch[0] + 0.3963377774 * a + 0.2158037573 * b;
	const double m_ = lch[0] - 0.1055613458 * a - 0.0638541728 * b;
	const double s_ = lch[0] - 0.0894841775 * a - 1.2914855480 * b;
	const double l = l_ * l_ * l_;
	const double m = m_ * m_ * m_;
	const double s = s_ * s_ * s_;
	rgb[0] = std::clamp(srgb_encode(
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
		0.0, 1.0);
	rgb[1] = std::clamp(srgb_encode(
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
		0.0, 1.0);
	rgb[2] = std::clamp(srgb_encode(
		-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
		0.0, 1.0);
}

void read_rgb(const obs_source_frame *frame, uint32_t x, uint32_t y,
	      double rgb[3])
{
	const uint8_t *row = frame->data[0] +
			     static_cast<size_t>(y) * frame->linesize[0];
	switch (frame->format) {
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX: {
		const uint8_t *p = row + x * 4;
		rgb[0] = p[2] / 255.0;
		rgb[1] = p[1] / 255.0;
		rgb[2] = p[0] / 255.0;
		return;
	}
	case VIDEO_FORMAT_RGBA: {
		const uint8_t *p = row + x * 4;
		rgb[0] = p[0] / 255.0;
		rgb[1] = p[1] / 255.0;
		rgb[2] = p[2] / 255.0;
		return;
	}
	case VIDEO_FORMAT_BGR3: {
		const uint8_t *p = row + x * 3;
		rgb[0] = p[2] / 255.0;
		rgb[1] = p[1] / 255.0;
		rgb[2] = p[0] / 255.0;
		return;
	}
	default:
		break;
	}

	double yy = 0.0, uu = 0.5, vv = 0.5;
	switch (frame->format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_I40A:
		yy = row[x] / 255.0;
		uu = frame->data[1][static_cast<size_t>(y / 2) *
					   frame->linesize[1] + x / 2] /
		     255.0;
		vv = frame->data[2][static_cast<size_t>(y / 2) *
					   frame->linesize[2] + x / 2] /
		     255.0;
		break;
	case VIDEO_FORMAT_NV12: {
		yy = row[x] / 255.0;
		const uint8_t *uv = frame->data[1] +
				    static_cast<size_t>(y / 2) *
					    frame->linesize[1] +
				    (x / 2) * 2;
		uu = uv[0] / 255.0;
		vv = uv[1] / 255.0;
		break;
	}
	case VIDEO_FORMAT_I422:
	case VIDEO_FORMAT_I42A:
		yy = row[x] / 255.0;
		uu = frame->data[1][static_cast<size_t>(y) *
					   frame->linesize[1] + x / 2] /
		     255.0;
		vv = frame->data[2][static_cast<size_t>(y) *
					   frame->linesize[2] + x / 2] /
		     255.0;
		break;
	case VIDEO_FORMAT_I444:
	case VIDEO_FORMAT_YUVA:
		yy = row[x] / 255.0;
		uu = frame->data[1][static_cast<size_t>(y) *
					   frame->linesize[1] + x] /
		     255.0;
		vv = frame->data[2][static_cast<size_t>(y) *
					   frame->linesize[2] + x] /
		     255.0;
		break;
	case VIDEO_FORMAT_YUY2:
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY: {
		const uint8_t *p = row + (x / 2) * 4;
		if (frame->format == VIDEO_FORMAT_UYVY) {
			yy = p[(x & 1) ? 3 : 1] / 255.0;
			uu = p[0] / 255.0;
			vv = p[2] / 255.0;
		} else {
			yy = p[(x & 1) ? 2 : 0] / 255.0;
			uu = p[frame->format == VIDEO_FORMAT_YUY2 ? 1 : 3] /
			     255.0;
			vv = p[frame->format == VIDEO_FORMAT_YUY2 ? 3 : 1] /
			     255.0;
		}
		break;
	}
	case VIDEO_FORMAT_Y800:
		yy = row[x] / 255.0;
		break;
	default:
		rgb[0] = rgb[1] = rgb[2] = 0.0;
		return;
	}
	const double yuv[3] = {yy, uu, vv};
	for (int c = 0; c < 3; ++c)
		rgb[c] = std::clamp(
			frame->color_matrix[c * 4 + 0] * yuv[0] +
				frame->color_matrix[c * 4 + 1] * yuv[1] +
				frame->color_matrix[c * 4 + 2] * yuv[2] +
				frame->color_matrix[c * 4 + 3],
			0.0, 1.0);
}

bool invert_color_matrix(const obs_source_frame *frame, double inv[9])
{
	const double a = frame->color_matrix[0], b = frame->color_matrix[1];
	const double c = frame->color_matrix[2], d = frame->color_matrix[4];
	const double e = frame->color_matrix[5], f = frame->color_matrix[6];
	const double g = frame->color_matrix[8], h = frame->color_matrix[9];
	const double i = frame->color_matrix[10];
	const double det = a * (e * i - f * h) -
			   b * (d * i - f * g) + c * (d * h - e * g);
	if (std::abs(det) < 1e-12)
		return false;
	const double s = 1.0 / det;
	inv[0] = (e * i - f * h) * s;
	inv[1] = (c * h - b * i) * s;
	inv[2] = (b * f - c * e) * s;
	inv[3] = (f * g - d * i) * s;
	inv[4] = (a * i - c * g) * s;
	inv[5] = (c * d - a * f) * s;
	inv[6] = (d * h - e * g) * s;
	inv[7] = (b * g - a * h) * s;
	inv[8] = (a * e - b * d) * s;
	return true;
}

void rgb_to_frame_yuv(const obs_source_frame *frame, const double inv[9],
		      const double rgb[3], uint8_t yuv[3])
{
	const double shifted[3] = {
		rgb[0] - frame->color_matrix[3],
		rgb[1] - frame->color_matrix[7],
		rgb[2] - frame->color_matrix[11],
	};
	for (int c = 0; c < 3; ++c) {
		double v = inv[c * 3 + 0] * shifted[0] +
			   inv[c * 3 + 1] * shifted[1] +
			   inv[c * 3 + 2] * shifted[2];
		double lo = frame->color_range_min[c];
		double hi = frame->color_range_max[c];
		if (!(hi > lo)) {
			lo = 0.0;
			hi = 1.0;
		}
		v = std::clamp(v, lo, hi);
		yuv[c] = clamp_byte(v * 255.0);
	}
}

const double *work_pixel(const std::vector<double> &rgb, uint32_t ww,
			 uint32_t wh, uint32_t fw, uint32_t fh,
			 uint32_t x, uint32_t y)
{
	const uint32_t wx = std::min(
		static_cast<uint32_t>((static_cast<uint64_t>(x) * ww) / fw),
		ww - 1);
	const uint32_t wy = std::min(
		static_cast<uint32_t>((static_cast<uint64_t>(y) * wh) / fh),
		wh - 1);
	return rgb.data() + (static_cast<size_t>(wy) * ww + wx) * 3;
}

void write_color_frame(obs_source_frame *frame,
		       const std::vector<double> &rgb, uint32_t ww, uint32_t wh)
{
	const uint32_t fw = frame->width, fh = frame->height;
	if (frame->format == VIDEO_FORMAT_BGRA ||
	    frame->format == VIDEO_FORMAT_BGRX ||
	    frame->format == VIDEO_FORMAT_RGBA ||
	    frame->format == VIDEO_FORMAT_BGR3) {
		for (uint32_t y = 0; y < fh; ++y) {
			uint8_t *row = frame->data[0] +
				       static_cast<size_t>(y) * frame->linesize[0];
			for (uint32_t x = 0; x < fw; ++x) {
				const double *p = work_pixel(
					rgb, ww, wh, fw, fh, x, y);
				if (frame->format == VIDEO_FORMAT_RGBA) {
					uint8_t *d = row + x * 4;
					d[0] = clamp_byte(p[0] * 255.0);
					d[1] = clamp_byte(p[1] * 255.0);
					d[2] = clamp_byte(p[2] * 255.0);
				} else if (frame->format == VIDEO_FORMAT_BGR3) {
					uint8_t *d = row + x * 3;
					d[0] = clamp_byte(p[2] * 255.0);
					d[1] = clamp_byte(p[1] * 255.0);
					d[2] = clamp_byte(p[0] * 255.0);
				} else {
					uint8_t *d = row + x * 4;
					d[0] = clamp_byte(p[2] * 255.0);
					d[1] = clamp_byte(p[1] * 255.0);
					d[2] = clamp_byte(p[0] * 255.0);
				}
			}
		}
		return;
	}

	double inv[9];
	if (!invert_color_matrix(frame, inv))
		return;
	for (uint32_t y = 0; y < fh; ++y) {
		uint8_t *row = frame->data[0] +
			       static_cast<size_t>(y) * frame->linesize[0];
		for (uint32_t x = 0; x < fw; ++x) {
			const double *p = work_pixel(rgb, ww, wh, fw, fh, x, y);
			uint8_t yuv[3];
			rgb_to_frame_yuv(frame, inv, p, yuv);
			switch (frame->format) {
			case VIDEO_FORMAT_I420:
			case VIDEO_FORMAT_I40A:
				row[x] = yuv[0];
				if (!(x & 1) && !(y & 1)) {
					frame->data[1][static_cast<size_t>(y / 2) *
								   frame->linesize[1] +
							   x / 2] = yuv[1];
					frame->data[2][static_cast<size_t>(y / 2) *
								   frame->linesize[2] +
							   x / 2] = yuv[2];
				}
				break;
			case VIDEO_FORMAT_NV12:
				row[x] = yuv[0];
				if (!(x & 1) && !(y & 1)) {
					uint8_t *uv =
						frame->data[1] +
						static_cast<size_t>(y / 2) *
							frame->linesize[1] +
						x;
					uv[0] = yuv[1];
					uv[1] = yuv[2];
				}
				break;
			case VIDEO_FORMAT_I422:
			case VIDEO_FORMAT_I42A:
				row[x] = yuv[0];
				if (!(x & 1)) {
					frame->data[1][static_cast<size_t>(y) *
								   frame->linesize[1] +
							   x / 2] = yuv[1];
					frame->data[2][static_cast<size_t>(y) *
								   frame->linesize[2] +
							   x / 2] = yuv[2];
				}
				break;
			case VIDEO_FORMAT_I444:
			case VIDEO_FORMAT_YUVA:
				row[x] = yuv[0];
				frame->data[1][static_cast<size_t>(y) *
							   frame->linesize[1] +
						   x] = yuv[1];
				frame->data[2][static_cast<size_t>(y) *
							   frame->linesize[2] +
						   x] = yuv[2];
				break;
			case VIDEO_FORMAT_Y800:
				row[x] = yuv[0];
				break;
			case VIDEO_FORMAT_YUY2:
			case VIDEO_FORMAT_YVYU:
			case VIDEO_FORMAT_UYVY: {
				uint8_t *d = row + (x / 2) * 4;
				if (frame->format == VIDEO_FORMAT_UYVY) {
					d[(x & 1) ? 3 : 1] = yuv[0];
					if (!(x & 1)) {
						d[0] = yuv[1];
						d[2] = yuv[2];
					}
				} else {
					d[(x & 1) ? 2 : 0] = yuv[0];
					if (!(x & 1)) {
						d[frame->format ==
								  VIDEO_FORMAT_YUY2
							  ? 1
							  : 3] = yuv[1];
						d[frame->format ==
								  VIDEO_FORMAT_YUY2
							  ? 3
							  : 1] = yuv[2];
					}
				}
				break;
			}
			default:
				break;
			}
		}
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
		for (uint32_t y = 0; y < height; ++y) {
			uint8_t *row = frame->data[0] +
				       static_cast<size_t>(y) * frame->linesize[0];
			for (uint32_t x = 0; x < width; ++x)
				row[x * 2 + 1] = 128;
		}
		break;
	case VIDEO_FORMAT_YVYU:
	case VIDEO_FORMAT_UYVY:
		for (uint32_t y = 0; y < height; ++y) {
			uint8_t *row = frame->data[0] +
				       static_cast<size_t>(y) * frame->linesize[0];
			for (uint32_t x = 0; x < width; ++x)
				row[x * 2 + (frame->format == VIDEO_FORMAT_UYVY ? 0 : 1)] =
					128;
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

bool split_recompose_plane(Filter *filter, double cartoon_gain,
			   double texture_gain)
{
	if (bfft_meyer_split(filter->plan, filter->input.data(),
			     filter->cartoon.data(),
			     filter->texture.data()) != BFFT_OK)
		return false;
	for (size_t i = 0; i < filter->input.size(); ++i) {
		const double residual = filter->input[i] - filter->cartoon[i] -
					filter->texture[i];
		filter->difference[i] =
			residual + cartoon_gain * filter->cartoon[i] +
			texture_gain * filter->texture[i];
	}
	return true;
}

const ColorHistoryFrame *history_at(
	const std::deque<ColorHistoryFrame> &history, uint64_t timestamp)
{
	if (history.empty())
		return nullptr;
	for (auto it = history.rbegin(); it != history.rend(); ++it) {
		if (it->timestamp <= timestamp)
			return &*it;
	}
	return &history.front();
}

uint16_t encode_history_component(double value, int component)
{
	double normalized;
	if (component == 0)
		normalized = (value + 0.5) * 0.5;
	else if (component == 1)
		normalized = value;
	else
		normalized = value - std::floor(value);
	return static_cast<uint16_t>(std::lround(
		std::clamp(normalized, 0.0, 1.0) * 65535.0));
}

double decode_history_component(uint16_t value, int component)
{
	const double normalized = value / 65535.0;
	return component == 0 ? normalized * 2.0 - 0.5 : normalized;
}

double blend_component(double current, double delayed, double mix,
		       int component)
{
	mix = std::clamp(mix, 0.0, 1.0);
	if (component == 2) {
		double delta = delayed - current;
		delta -= std::round(delta);
		double out = current + mix * delta;
		return out - std::floor(out);
	}
	return current + mix * (delayed - current);
}

bool process_color_mode(Filter *filter, obs_source_frame *frame, int mode,
			double l_cartoon, double l_texture,
			double c_cartoon, double c_texture,
			double h_cartoon, double h_texture,
			const double component_delay_ms[3],
			const double q_texture[4],
			const double q_delay_ms[4],
			const double q_hue_rotation[4],
			const int q_payload[4], double temporal_mix,
			int echo_taps, double echo_decay,
			double hue_rotation, double chroma_scale,
			double pinwheel_turns, double chroma_delay_weight,
			double lead_delay_ms, double lead_strength,
			double lead_threshold, double lead_feather,
			double lead_hue_rotation, double lead_chroma_scale,
			int tile_size, double tile_delay_ms,
			double tile_refraction, double tile_bevel,
			double tile_randomness)
{
	const uint32_t ww = filter->work_width, wh = filter->work_height;
	const size_t count = static_cast<size_t>(ww) * wh;
	for (uint32_t wy = 0; wy < wh; ++wy) {
		const uint32_t sy = std::min(
			static_cast<uint32_t>((static_cast<uint64_t>(wy) *
					      frame->height) /
					     wh),
			frame->height - 1);
		for (uint32_t wx = 0; wx < ww; ++wx) {
			const uint32_t sx = std::min(
				static_cast<uint32_t>((static_cast<uint64_t>(wx) *
						      frame->width) /
						     ww),
				frame->width - 1);
			const size_t i = static_cast<size_t>(wy) * ww + wx;
			double rgb[3], lch[3];
			read_rgb(frame, sx, sy, rgb);
			rgb_to_oklch(rgb, lch);
			for (int c = 0; c < 3; ++c) {
				filter->work_lch[i * 3 + c] = lch[c];
				filter->color_output[i * 3 + c] = lch[c];
			}
		}
	}

	if (mode != 6 && mode != 8) {
		const double scales[3] = {255.0, 510.0, 255.0};
		const double cartoon_gains[3] = {
			l_cartoon, c_cartoon, h_cartoon};
		const double texture_gains[3] = {
			l_texture, c_texture, h_texture};
		for (int c = 0; c < 3; ++c) {
			for (size_t i = 0; i < count; ++i)
				filter->input[i] =
					filter->work_lch[i * 3 + c] * scales[c];
			if (!split_recompose_plane(
				    filter, cartoon_gains[c], texture_gains[c]))
				return false;
			for (size_t i = 0; i < count; ++i) {
				double value;
				if (mode == 12) {
					value =
						(filter->input[i] -
						 filter->texture[i] +
						 (cartoon_gains[c] - 1.0) *
							 filter->cartoon[i]) /
						scales[c];
				} else {
					value =
						filter->difference[i] / scales[c];
				}
				if (c == 1)
					value = std::max(value, 0.0);
				if (c == 2)
					value -= std::floor(value);
				if (mode == 12) {
					filter->live_cartoon_lch[i * 3 + c] =
						value;
					filter->color_output[i * 3 + c] =
						filter->work_lch[i * 3 + c];
				} else {
					filter->color_output[i * 3 + c] =
						value;
				}
			}
		}
	} else {
		// Each hue quadrant gets its own masked lightness solve. Outside
		// the active quadrant the field is held at its mean, preventing
		// unrelated hues from contributing texture to that solve.
		for (int q = 0; q < 4; ++q) {
			double mean = 0.0;
			size_t members = 0;
			for (size_t i = 0; i < count; ++i) {
				const int sector = std::min(
					static_cast<int>(
						filter->work_lch[i * 3 + 2] * 4.0),
					3);
				if (sector == q) {
					mean += filter->work_lch[i * 3];
					++members;
				}
			}
			if (!members)
				continue;
			mean /= static_cast<double>(members);
			for (size_t i = 0; i < count; ++i) {
				const int sector = std::min(
					static_cast<int>(
						filter->work_lch[i * 3 + 2] * 4.0),
					3);
				filter->input[i] =
					255.0 * (sector == q
							 ? filter->work_lch[i * 3]
							 : mean);
			}
			if (!split_recompose_plane(
				    filter, l_cartoon, q_texture[q]))
				return false;
			for (size_t i = 0; i < count; ++i) {
				const int sector = std::min(
					static_cast<int>(
						filter->work_lch[i * 3 + 2] * 4.0),
					3);
				if (sector != q)
					continue;
				double payload;
				switch (std::clamp(q_payload[q], 0, 3)) {
				case 1:
					payload =
						0.5 +
						(filter->input[i] -
						 filter->cartoon[i] -
						 filter->texture[i]) /
							255.0;
					break;
				case 2:
					payload = filter->work_lch[i * 3 + 1] /
						  0.4;
					break;
				case 3:
					payload =
						0.5 + filter->texture[i] / 255.0;
					break;
				default:
					payload = filter->difference[i] / 255.0;
					break;
				}
				filter->color_output[i * 3] =
					std::clamp(payload, 0.0, 1.0);
			}
		}
	}

	if (filter->history_mode != mode) {
		filter->color_history.clear();
		filter->history_mode = mode;
	}
	uint64_t now = frame->timestamp;
	if (!now) {
		now = static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now().time_since_epoch())
				.count());
	}
	ColorHistoryFrame snapshot;
	snapshot.timestamp = now;
	snapshot.lch.resize(count * 3);
	for (size_t i = 0; i < count; ++i)
		for (int c = 0; c < 3; ++c)
			snapshot.lch[i * 3 + c] = encode_history_component(
				filter->color_output[i * 3 + c], c);
	filter->color_history.emplace_back(std::move(snapshot));

	double max_delay_ms = 0.0;
	for (int c = 0; c < 3; ++c)
		max_delay_ms = std::max(
			max_delay_ms,
			std::clamp(component_delay_ms[c], 0.0, 500.0));
	for (int q = 0; q < 4; ++q)
		max_delay_ms = std::max(
			max_delay_ms, std::clamp(q_delay_ms[q], 0.0, 500.0));
	if (mode == 12)
		max_delay_ms = std::max(
			max_delay_ms, std::clamp(lead_delay_ms, 0.0, 500.0));
	if (mode == 13)
		max_delay_ms = std::max(
			max_delay_ms, std::clamp(tile_delay_ms, 0.0, 500.0));
	const uint64_t max_delay_ns =
		static_cast<uint64_t>(max_delay_ms * 1'000'000.0);
	const uint64_t keep_ns = max_delay_ns + kHistoryMarginNs;
	while (filter->color_history.size() > 1 &&
	       (filter->color_history.size() > kMaxHistoryFrames ||
		(now > keep_ns &&
		 filter->color_history.front().timestamp < now - keep_ns)))
		filter->color_history.pop_front();

	auto state_for_delay = [&](double milliseconds) {
		const uint64_t lag = static_cast<uint64_t>(
			std::clamp(milliseconds, 0.0, 500.0) * 1'000'000.0);
		return history_at(filter->color_history,
				  now > lag ? now - lag : 0);
	};
	auto state_value = [](const ColorHistoryFrame *state, size_t pixel,
			      int component) {
		return decode_history_component(
			state->lch[pixel * 3 + component], component);
	};
	std::copy(filter->color_output.begin(), filter->color_output.end(),
		  filter->work_lch.begin());
	temporal_mix = std::clamp(temporal_mix, 0.0, 1.0);

	if (mode == 12) {
		const ColorHistoryFrame *delayed =
			state_for_delay(lead_delay_ms);
		const double lead_hue_shift = lead_hue_rotation / 360.0;
		lead_strength = std::clamp(lead_strength, 0.0, 1.5);
		lead_threshold = std::clamp(lead_threshold, 0.0, 0.3);
		lead_feather = std::clamp(lead_feather, 1.0, 40.0);
		lead_chroma_scale =
			std::clamp(lead_chroma_scale, 0.0, 3.0);
		for (size_t i = 0; i < count; ++i) {
			double delayed_lch[3];
			double current_lch[3];
			double lead_lch[3];
			for (int c = 0; c < 3; ++c) {
				delayed_lch[c] =
					state_value(delayed, i, c);
				current_lch[c] =
					filter->color_output[i * 3 + c];
				lead_lch[c] =
					filter->live_cartoon_lch[i * 3 + c];
			}
			lead_lch[1] =
				std::max(0.0, lead_lch[1] * lead_chroma_scale);
			lead_lch[2] += lead_hue_shift;
			lead_lch[2] -= std::floor(lead_lch[2]);

			const double dl =
				current_lch[0] - delayed_lch[0];
			const double dc =
				current_lch[1] - delayed_lch[1];
			double dh = current_lch[2] - delayed_lch[2];
			dh -= std::round(dh);
			const double hue_arc =
				kTau * dh *
				std::min(current_lch[1], delayed_lch[1]);
			const double motion = std::sqrt(
				dl * dl + 0.5 * dc * dc +
				0.25 * hue_arc * hue_arc);
			double mask = std::clamp(
				(motion - lead_threshold) * lead_feather,
				0.0, 1.0);
			mask = mask * mask * (3.0 - 2.0 * mask);
			const double alpha =
				std::clamp(mask * lead_strength, 0.0, 1.0);
			for (int c = 0; c < 3; ++c)
				filter->work_lch[i * 3 + c] =
					blend_component(
						delayed_lch[c], lead_lch[c],
						alpha, c);
		}
	} else if (mode == 5) {
		const ColorHistoryFrame *states[3];
		for (int c = 0; c < 3; ++c)
			states[c] = state_for_delay(component_delay_ms[c]);
		for (size_t i = 0; i < count; ++i)
			for (int c = 0; c < 3; ++c)
				filter->work_lch[i * 3 + c] = blend_component(
					filter->color_output[i * 3 + c],
					state_value(states[c], i, c),
					temporal_mix, c);
	} else if (mode == 6) {
		const ColorHistoryFrame *states[4];
		for (int q = 0; q < 4; ++q)
			states[q] = state_for_delay(q_delay_ms[q]);
		for (size_t i = 0; i < count; ++i) {
			const int q = std::min(static_cast<int>(
				filter->color_output[i * 3 + 2] * 4.0), 3);
			for (int c = 0; c < 3; ++c)
				filter->work_lch[i * 3 + c] = blend_component(
					filter->color_output[i * 3 + c],
					state_value(states[q], i, c),
					temporal_mix, c);
		}
	} else if (mode == 7 || mode == 8) {
		echo_taps = std::clamp(echo_taps, 1, 8);
		echo_decay = std::clamp(echo_decay, 0.0, 1.0);
		for (size_t i = 0; i < count; ++i) {
			const int q = std::min(static_cast<int>(
				filter->color_output[i * 3 + 2] * 4.0), 3);
			for (int c = 0; c < 3; ++c) {
				const double endpoint =
					mode == 7 ? component_delay_ms[c]
						  : q_delay_ms[q];
				const double current =
					filter->color_output[i * 3 + c];
				double sum = 0.0, weight_sum = 0.0;
				for (int tap = 0; tap < echo_taps; ++tap) {
					const double fraction =
						echo_taps == 1
							? 0.0
							: static_cast<double>(tap) /
								  (echo_taps - 1);
					const ColorHistoryFrame *state =
						state_for_delay(endpoint * fraction);
					double value = state_value(state, i, c);
					if (c == 2) {
						double delta = value - current;
						value = current +
							(delta - std::round(delta));
					}
					const double weight =
						std::pow(echo_decay, tap);
					sum += weight * value;
					weight_sum += weight;
				}
				filter->work_lch[i * 3 + c] = blend_component(
					current, sum / std::max(weight_sum, 1e-12),
					temporal_mix, c);
			}
		}
	} else if (mode >= 9 && mode <= 11 && max_delay_ms > 0.0) {
		for (size_t i = 0; i < count; ++i) {
			const double hue =
				filter->color_output[i * 3 + 2] -
				std::floor(filter->color_output[i * 3 + 2]);
			double lag_ms = 0.0;
			if (mode == 9) {
				const double sector_position = hue * 4.0;
				const int q0 =
					std::min(static_cast<int>(sector_position), 3);
				const int q1 = (q0 + 1) & 3;
				const double t = sector_position - q0;
				lag_ms = (1.0 - t) * q_delay_ms[q0] +
					 t * q_delay_ms[q1];
			} else if (mode == 10) {
				const double chroma =
					filter->color_output[i * 3 + 1];
				const double depth = std::clamp(
					chroma * chroma_delay_weight / 0.4,
					0.0, 1.0);
				lag_ms = max_delay_ms * depth;
			} else {
				const double x = static_cast<double>(i % ww) /
							 std::max(ww - 1, 1u) -
						 0.5;
				const double y = static_cast<double>(i / ww) /
							 std::max(wh - 1, 1u) -
						 0.5;
				double phase =
					hue + pinwheel_turns *
						       std::atan2(y, x) / kTau;
				phase -= std::floor(phase);
				lag_ms = max_delay_ms * phase;
			}
			const ColorHistoryFrame *state = state_for_delay(lag_ms);
			for (int c = 0; c < 3; ++c)
				filter->work_lch[i * 3 + c] = blend_component(
					filter->color_output[i * 3 + c],
					state_value(state, i, c),
					temporal_mix, c);
		}
	} else if (mode == 13) {
		tile_size = std::clamp(tile_size, 8, 128);
		tile_delay_ms = std::clamp(tile_delay_ms, 0.0, 500.0);
		tile_refraction = std::clamp(tile_refraction, 0.0, 24.0);
		tile_bevel = std::clamp(tile_bevel, 0.0, 1.0);
		tile_randomness = std::clamp(tile_randomness, 0.0, 1.0);
		for (uint32_t y = 0; y < wh; ++y) {
			for (uint32_t x = 0; x < ww; ++x) {
				const size_t i = static_cast<size_t>(y) * ww + x;
				const uint32_t tx = x / tile_size;
				const uint32_t ty = y / tile_size;
				uint32_t hash = tx * 0x9e3779b9u ^
						ty * 0x85ebca6bu;
				hash ^= hash >> 16;
				hash *= 0x7feb352du;
				hash ^= hash >> 15;
				const double random_phase =
					(hash & 0xffffu) / 65535.0;
				const double ordered_phase =
					((tx + 2u * ty) & 3u) / 3.0;
				const double phase =
					(1.0 - tile_randomness) *
						ordered_phase +
					tile_randomness * random_phase;
				const ColorHistoryFrame *state =
					state_for_delay(tile_delay_ms * phase);

				const double u =
					(2.0 * ((x % tile_size) + 0.5) /
						 tile_size) -
					1.0;
				const double v =
					(2.0 * ((y % tile_size) + 0.5) /
						 tile_size) -
					1.0;
				const double lens =
					std::max(0.0, 1.0 - u * u - v * v);
				const int x0 = static_cast<int>(tx * tile_size);
				const int y0 = static_cast<int>(ty * tile_size);
				const int x1 = std::min(
					x0 + tile_size - 1,
					static_cast<int>(ww) - 1);
				const int y1 = std::min(
					y0 + tile_size - 1,
					static_cast<int>(wh) - 1);
				const int sx = std::clamp(
					static_cast<int>(x) +
						static_cast<int>(std::lround(
							u * tile_refraction *
							lens)),
					x0, x1);
				const int sy = std::clamp(
					static_cast<int>(y) +
						static_cast<int>(std::lround(
							v * tile_refraction *
							lens)),
					y0, y1);
				const size_t sample =
					static_cast<size_t>(sy) * ww + sx;
				for (int c = 0; c < 3; ++c)
					filter->work_lch[i * 3 + c] =
						blend_component(
							filter->color_output
								[i * 3 + c],
							state_value(
								state, sample,
								c),
							temporal_mix, c);
				const double edge =
					std::max(std::abs(u), std::abs(v));
				filter->work_lch[i * 3] +=
					tile_bevel *
					(0.075 * std::pow(edge, 12.0) -
					 0.012 * std::pow(1.0 - edge, 3.0));
			}
		}
	}

	const double hue_shift = hue_rotation / 360.0;
	for (size_t i = 0; i < count; ++i) {
		if (mode != 12)
			filter->work_lch[i * 3 + 1] =
				std::max(0.0,
					 filter->work_lch[i * 3 + 1] *
						 std::clamp(
							 chroma_scale,
							 0.0, 3.0));
		double sector_shift = 0.0;
		if (mode == 6 || mode == 8 || mode == 9) {
			const int q = std::min(static_cast<int>(
				filter->color_output[i * 3 + 2] * 4.0), 3);
			sector_shift =
				std::clamp(q_hue_rotation[q], -180.0, 180.0) /
				360.0;
		}
		double hue = filter->work_lch[i * 3 + 2] +
			     (mode == 12 ? 0.0 : hue_shift) + sector_shift;
		filter->work_lch[i * 3 + 2] = hue - std::floor(hue);
	}

	for (size_t i = 0; i < count; ++i) {
		double lch[3] = {
			filter->work_lch[i * 3 + 0],
			filter->work_lch[i * 3 + 1],
			filter->work_lch[i * 3 + 2],
		};
		double rgb[3];
		oklch_to_rgb(lch, rgb);
		for (int c = 0; c < 3; ++c)
			filter->color_output[i * 3 + c] = rgb[c];
	}
	write_color_frame(frame, filter->color_output, ww, wh);
	return true;
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
	filter->l_cartoon = obs_data_get_double(settings, kLCartoon);
	filter->l_texture = obs_data_get_double(settings, kLTexture);
	filter->c_cartoon = obs_data_get_double(settings, kCCartoon);
	filter->c_texture = obs_data_get_double(settings, kCTexture);
	filter->h_cartoon = obs_data_get_double(settings, kHCartoon);
	filter->h_texture = obs_data_get_double(settings, kHTexture);
	filter->component_delay_ms[0] =
		obs_data_get_double(settings, kLDelayMs);
	filter->component_delay_ms[1] =
		obs_data_get_double(settings, kCDelayMs);
	filter->component_delay_ms[2] =
		obs_data_get_double(settings, kHDelayMs);
	for (int q = 0; q < 4; ++q)
		filter->q_texture[q] =
			obs_data_get_double(settings, kQTexture[q]);
	for (int q = 0; q < 4; ++q)
		filter->q_delay_ms[q] =
			obs_data_get_double(settings, kQDelayMs[q]);
	for (int q = 0; q < 4; ++q)
		filter->q_hue_rotation[q] =
			obs_data_get_double(settings, kQHueRotation[q]);
	for (int q = 0; q < 4; ++q)
		filter->q_payload[q] =
			static_cast<int>(obs_data_get_int(settings, kQPayload[q]));
	filter->temporal_mix =
		obs_data_get_double(settings, kTemporalMix);
	filter->echo_taps =
		static_cast<int>(obs_data_get_int(settings, kEchoTaps));
	filter->echo_decay = obs_data_get_double(settings, kEchoDecay);
	filter->hue_rotation = obs_data_get_double(settings, kHueRotation);
	filter->chroma_scale = obs_data_get_double(settings, kChromaScale);
	filter->pinwheel_turns =
		obs_data_get_double(settings, kPinwheelTurns);
	filter->chroma_delay_weight =
		obs_data_get_double(settings, kChromaDelayWeight);
	filter->lead_delay_ms = obs_data_get_double(settings, kLeadDelayMs);
	filter->lead_strength = obs_data_get_double(settings, kLeadStrength);
	filter->lead_threshold = obs_data_get_double(settings, kLeadThreshold);
	filter->lead_feather = obs_data_get_double(settings, kLeadFeather);
	filter->lead_hue_rotation =
		obs_data_get_double(settings, kLeadHueRotation);
	filter->lead_chroma_scale =
		obs_data_get_double(settings, kLeadChromaScale);
	filter->tile_size =
		static_cast<int>(obs_data_get_int(settings, kTileSize));
	filter->tile_delay_ms = obs_data_get_double(settings, kTileDelayMs);
	filter->tile_refraction =
		obs_data_get_double(settings, kTileRefraction);
	filter->tile_bevel = obs_data_get_double(settings, kTileBevel);
	filter->tile_randomness =
		obs_data_get_double(settings, kTileRandomness);
}

void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_double(settings, kCartoon, 1.0);
	obs_data_set_default_double(settings, kTexture, 1.0);
	obs_data_set_default_double(settings, kShading, 0.0);
	obs_data_set_default_double(settings, kShadeC, 0.02);
	obs_data_set_default_int(settings, kPasses, 12);
	obs_data_set_default_int(settings, kThreads, 4);
	obs_data_set_default_int(settings, kMode, 0);
	obs_data_set_default_double(settings, kRelief, 1.0);
	obs_data_set_default_double(settings, kGloss, 0.75);
	obs_data_set_default_double(settings, kLCartoon, 1.0);
	obs_data_set_default_double(settings, kLTexture, 1.0);
	obs_data_set_default_double(settings, kCCartoon, 1.0);
	obs_data_set_default_double(settings, kCTexture, 1.0);
	obs_data_set_default_double(settings, kHCartoon, 1.0);
	obs_data_set_default_double(settings, kHTexture, 1.0);
	obs_data_set_default_double(settings, kLDelayMs, 0.0);
	obs_data_set_default_double(settings, kCDelayMs, 0.0);
	obs_data_set_default_double(settings, kHDelayMs, 0.0);
	for (int q = 0; q < 4; ++q)
		obs_data_set_default_double(settings, kQTexture[q], 1.0);
	for (int q = 0; q < 4; ++q)
		obs_data_set_default_double(settings, kQDelayMs[q], 0.0);
	for (int q = 0; q < 4; ++q)
		obs_data_set_default_double(settings, kQHueRotation[q], 0.0);
	for (int q = 0; q < 4; ++q)
		obs_data_set_default_int(settings, kQPayload[q], 0);
	obs_data_set_default_double(settings, kTemporalMix, 1.0);
	obs_data_set_default_int(settings, kEchoTaps, 4);
	obs_data_set_default_double(settings, kEchoDecay, 0.65);
	obs_data_set_default_double(settings, kHueRotation, 0.0);
	obs_data_set_default_double(settings, kChromaScale, 1.0);
	obs_data_set_default_double(settings, kPinwheelTurns, 2.0);
	obs_data_set_default_double(settings, kChromaDelayWeight, 1.0);
	obs_data_set_default_double(settings, kLeadDelayMs, 180.0);
	obs_data_set_default_double(settings, kLeadStrength, 1.0);
	obs_data_set_default_double(settings, kLeadThreshold, 0.03);
	obs_data_set_default_double(settings, kLeadFeather, 12.0);
	obs_data_set_default_double(settings, kLeadHueRotation, 0.0);
	obs_data_set_default_double(settings, kLeadChromaScale, 1.0);
	obs_data_set_default_int(settings, kTileSize, 32);
	obs_data_set_default_double(settings, kTileDelayMs, 240.0);
	obs_data_set_default_double(settings, kTileRefraction, 8.0);
	obs_data_set_default_double(settings, kTileBevel, 0.5);
	obs_data_set_default_double(settings, kTileRandomness, 0.35);
}

obs_properties_t *filter_properties(void *)
{
	obs_properties_t *props = obs_properties_create();
	obs_property_t *mode = obs_properties_add_list(
		props, kMode, "Display mode", OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode, "Cartoon + texture", 0);
	obs_property_list_add_int(mode, "Difference field", 1);
	obs_property_list_add_int(mode, "Liquid chrome relief", 2);
	obs_property_list_add_int(mode, "Fine chrome", 3);
	obs_property_list_add_int(mode, "OKLCH independent", 4);
	obs_property_list_add_int(mode, "OKLCH individual delays", 5);
	obs_property_list_add_int(mode, "Four hue sectors delayed", 6);
	obs_property_list_add_int(mode, "OKLCH echo prism", 7);
	obs_property_list_add_int(mode, "Hue-sector echoes", 8);
	obs_property_list_add_int(mode, "Continuous hue time prism", 9);
	obs_property_list_add_int(mode, "Chroma comet trails", 10);
	obs_property_list_add_int(mode, "Hue-time pinwheel", 11);
	obs_property_list_add_int(mode, "Live cartoon / delayed scene", 12);
	obs_property_list_add_int(mode, "Tiled time glass", 13);
	obs_properties_add_float_slider(
		props, kCartoon, "Cartoon gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kTexture, "Texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kShading, "Shading gain (added)", -1.0, 4.0, 0.05);
	obs_properties_add_float_slider(
		props, kShadeC, "Shading TV constant", 0.004, 0.2, 0.002);
	obs_properties_add_int_slider(
		props, kPasses, "Quality / passes", 4, 24, 1);
	obs_properties_add_int_slider(
		props, kThreads, "CPU threads", 1, 8, 1);
	obs_properties_add_float_slider(
		props, kRelief, "Difference / relief depth", 0.1, 4.0, 0.05);
	obs_properties_add_float_slider(
		props, kGloss, "Chrome gloss", 0.0, 1.0, 0.05);
	obs_properties_add_float_slider(
		props, kLCartoon, "OKLCH L cartoon gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kLTexture, "OKLCH L texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kCCartoon, "OKLCH C cartoon gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kCTexture, "OKLCH C texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kHCartoon, "OKLCH H cartoon gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kHTexture, "OKLCH H texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kLDelayMs, "OKLCH L delay (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kCDelayMs, "OKLCH C delay (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kHDelayMs, "OKLCH H delay (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kQTexture[0], "Hue sector 1 texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kQTexture[1], "Hue sector 2 texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kQTexture[2], "Hue sector 3 texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kQTexture[3], "Hue sector 4 texture gain", -2.0, 6.0, 0.05);
	obs_properties_add_float_slider(
		props, kQDelayMs[0], "Hue sector 1 delay (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kQDelayMs[1], "Hue sector 2 delay (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kQDelayMs[2], "Hue sector 3 delay (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kQDelayMs[3], "Hue sector 4 delay (ms)", 0.0, 500.0, 1.0);
	for (int q = 0; q < 4; ++q) {
		const char *rotation_labels[4] = {
			"Hue sector 1 rotation", "Hue sector 2 rotation",
			"Hue sector 3 rotation", "Hue sector 4 rotation",
		};
		obs_properties_add_float_slider(
			props, kQHueRotation[q], rotation_labels[q],
			-180.0, 180.0, 1.0);
		const char *payload_labels[4] = {
			"Hue sector 1 payload", "Hue sector 2 payload",
			"Hue sector 3 payload", "Hue sector 4 payload",
		};
		obs_property_t *payload = obs_properties_add_list(
			props, kQPayload[q], payload_labels[q],
			OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(payload, "Processed lightness", 0);
		obs_property_list_add_int(payload, "Meyer residual", 1);
		obs_property_list_add_int(payload, "Chroma data", 2);
		obs_property_list_add_int(payload, "Texture field", 3);
	}
	obs_properties_add_float_slider(
		props, kTemporalMix, "Delayed / current mix", 0.0, 1.0, 0.01);
	obs_properties_add_int_slider(
		props, kEchoTaps, "Echo taps", 1, 8, 1);
	obs_properties_add_float_slider(
		props, kEchoDecay, "Echo decay", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(
		props, kHueRotation, "Hue rotation (degrees)", -180.0, 180.0, 1.0);
	obs_properties_add_float_slider(
		props, kChromaScale, "Chroma scale", 0.0, 3.0, 0.01);
	obs_properties_add_float_slider(
		props, kPinwheelTurns, "Pinwheel turns", 0.0, 8.0, 0.05);
	obs_properties_add_float_slider(
		props, kChromaDelayWeight, "Chroma delay weight", 0.0, 4.0, 0.05);
	obs_properties_add_float_slider(
		props, kLeadDelayMs, "Delayed scene (ms)", 0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kLeadStrength, "Live cartoon strength", 0.0, 1.5, 0.01);
	obs_properties_add_float_slider(
		props, kLeadThreshold, "Foreshadow motion threshold",
		0.0, 0.3, 0.002);
	obs_properties_add_float_slider(
		props, kLeadFeather, "Foreshadow mask feather", 1.0, 40.0, 0.5);
	obs_properties_add_float_slider(
		props, kLeadHueRotation, "Live cartoon hue rotation",
		-180.0, 180.0, 1.0);
	obs_properties_add_float_slider(
		props, kLeadChromaScale, "Live cartoon chroma scale",
		0.0, 3.0, 0.01);
	obs_properties_add_int_slider(
		props, kTileSize, "Glass tile size", 8, 128, 1);
	obs_properties_add_float_slider(
		props, kTileDelayMs, "Glass delay depth (ms)",
		0.0, 500.0, 1.0);
	obs_properties_add_float_slider(
		props, kTileRefraction, "Glass refraction", 0.0, 24.0, 0.25);
	obs_properties_add_float_slider(
		props, kTileBevel, "Glass edge bevel", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(
		props, kTileRandomness, "Glass tile randomness",
		0.0, 1.0, 0.01);
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

	// OBS currently serializes asynchronous-filter delivery, but the Meyer
	// plan intentionally reuses internal workspaces.  Keep that contract
	// explicit so a future source scheduling change cannot overlap frames
	// and corrupt the plan.
	std::lock_guard<std::mutex> processing_lock(filter->processing_mutex);

	double cartoon_gain;
	double texture_gain;
	double shading_gain;
	double shade_c;
	int passes;
	int threads;
	int mode;
	double relief;
	double gloss;
	double l_cartoon;
	double l_texture;
	double c_cartoon;
	double c_texture;
	double h_cartoon;
	double h_texture;
	double component_delay_ms[3];
	double q_texture[4];
	double q_delay_ms[4];
	double q_hue_rotation[4];
	int q_payload[4];
	double temporal_mix;
	int echo_taps;
	double echo_decay;
	double hue_rotation;
	double chroma_scale;
	double pinwheel_turns;
	double chroma_delay_weight;
	double lead_delay_ms;
	double lead_strength;
	double lead_threshold;
	double lead_feather;
	double lead_hue_rotation;
	double lead_chroma_scale;
	int tile_size;
	double tile_delay_ms;
	double tile_refraction;
	double tile_bevel;
	double tile_randomness;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		cartoon_gain = filter->cartoon_gain;
		texture_gain = filter->texture_gain;
		shading_gain = filter->shading_gain;
		shade_c = filter->shade_c;
		passes = std::clamp(filter->passes, 4, 24);
		threads = std::clamp(filter->threads, 1, 8);
		mode = filter->mode;
		relief = filter->relief;
		gloss = filter->gloss;
		l_cartoon = filter->l_cartoon;
		l_texture = filter->l_texture;
		c_cartoon = filter->c_cartoon;
		c_texture = filter->c_texture;
		h_cartoon = filter->h_cartoon;
		h_texture = filter->h_texture;
		for (int c = 0; c < 3; ++c)
			component_delay_ms[c] = std::clamp(
				filter->component_delay_ms[c], 0.0, 500.0);
		for (int q = 0; q < 4; ++q)
			q_texture[q] = filter->q_texture[q];
		for (int q = 0; q < 4; ++q)
			q_delay_ms[q] =
				std::clamp(filter->q_delay_ms[q], 0.0, 500.0);
		for (int q = 0; q < 4; ++q)
			q_hue_rotation[q] =
				std::clamp(filter->q_hue_rotation[q],
					   -180.0, 180.0);
		for (int q = 0; q < 4; ++q)
			q_payload[q] = std::clamp(filter->q_payload[q], 0, 3);
		temporal_mix = filter->temporal_mix;
		echo_taps = filter->echo_taps;
		echo_decay = filter->echo_decay;
		hue_rotation = filter->hue_rotation;
		chroma_scale = filter->chroma_scale;
		pinwheel_turns = filter->pinwheel_turns;
		chroma_delay_weight = filter->chroma_delay_weight;
		lead_delay_ms =
			std::clamp(filter->lead_delay_ms, 0.0, 500.0);
		lead_strength = filter->lead_strength;
		lead_threshold = filter->lead_threshold;
		lead_feather = filter->lead_feather;
		lead_hue_rotation = filter->lead_hue_rotation;
		lead_chroma_scale = filter->lead_chroma_scale;
		tile_size = filter->tile_size;
		tile_delay_ms = filter->tile_delay_ms;
		tile_refraction = filter->tile_refraction;
		tile_bevel = filter->tile_bevel;
		tile_randomness = filter->tile_randomness;
	}
	if (mode < 4 && !filter->color_history.empty()) {
		filter->color_history.clear();
		filter->history_mode = -1;
	}
	if (std::abs(cartoon_gain - 1.0) < 1e-12 &&
	    std::abs(texture_gain - 1.0) < 1e-12 &&
	    std::abs(shading_gain) < 1e-12 && mode == 0)
		return frame;
	// Relief modes prioritize the 30 fps contract. Eight TGFD passes followed
	// by eight sweeps of the single terminal TV solve leave useful headroom.
	const int solve_passes =
		(mode == 0 && std::abs(shading_gain) < 1e-12)
			? passes
			: std::min(passes, 8);
	if (!ensure_plan(filter, frame->width, frame->height, solve_passes,
			 threads))
		return frame;

	const auto started = std::chrono::steady_clock::now();
	const uint32_t ww = filter->work_width;
	const uint32_t wh = filter->work_height;
	if (mode >= 4 && mode <= 13) {
		if (!process_color_mode(
			    filter, frame, mode, l_cartoon, l_texture,
			    c_cartoon, c_texture, h_cartoon, h_texture,
			    component_delay_ms, q_texture, q_delay_ms,
			    q_hue_rotation, q_payload,
			    temporal_mix, echo_taps, echo_decay,
			    hue_rotation, chroma_scale, pinwheel_turns,
			    chroma_delay_weight, lead_delay_ms,
			    lead_strength, lead_threshold, lead_feather,
			    lead_hue_rotation, lead_chroma_scale,
			    tile_size, tile_delay_ms, tile_refraction,
			    tile_bevel, tile_randomness))
			return frame;
		const double elapsed_ms =
			std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - started)
				.count();
		filter->frames++;
		filter->total_ms += elapsed_ms;
		if (filter->frames % 300 == 0) {
			blog(LOG_INFO,
			     "[BFFT Cartoon] %.2f ms/frame average "
			     "(%.1f fps capacity)",
			     filter->total_ms / filter->frames,
			     1000.0 * filter->frames / filter->total_ms);
		}
		return frame;
	}

	for (uint32_t wy = 0; wy < wh; ++wy) {
		const uint32_t sy = std::min(
			static_cast<uint32_t>((static_cast<uint64_t>(wy) *
					      frame->height) /
					     wh),
			frame->height - 1);
		for (uint32_t wx = 0; wx < ww; ++wx) {
			const uint32_t sx = std::min(
				static_cast<uint32_t>((static_cast<uint64_t>(wx) *
						      frame->width) /
						     ww),
				frame->width - 1);
			filter->input[static_cast<size_t>(wy) * ww + wx] =
				read_luma(frame, sx, sy);
		}
	}

	if (bfft_meyer_split(filter->plan, filter->input.data(),
			     filter->cartoon.data(),
			     filter->texture.data()) != BFFT_OK)
		return frame;

	if (mode == 0) {
		if (std::abs(shading_gain) >= 1e-12 &&
		    bfft_meyer_rof(filter->plan, filter->cartoon.data(),
				   filter->reference_cartoon.data(), shade_c, 0.0,
				   8, 0.0) != BFFT_OK)
			return frame;
		for (uint32_t y = 0; y < frame->height; ++y) {
		const uint32_t wy = std::min(
			static_cast<uint32_t>((static_cast<uint64_t>(y) * wh) /
					     frame->height),
			wh - 1);
		for (uint32_t x = 0; x < frame->width; ++x) {
			const uint32_t wx = std::min(
				static_cast<uint32_t>((static_cast<uint64_t>(x) * ww) /
						     frame->width),
				ww - 1);
			const size_t i = static_cast<size_t>(wy) * ww + wx;
			const double original = read_luma(frame, x, y);
			const double residual =
				filter->input[i] - filter->cartoon[i] -
				filter->texture[i];
			double output = residual +
					cartoon_gain * filter->cartoon[i] +
					texture_gain * filter->texture[i];
			if (std::abs(shading_gain) >= 1e-12)
				output += shading_gain *
					  (filter->cartoon[i] -
					   filter->reference_cartoon[i]);
			// Preserve the source sample at identity exactly, including
			// the small resampling mismatch between the OBS and work grids.
			if (std::abs(cartoon_gain - 1.0) < 1e-12 &&
			    std::abs(texture_gain - 1.0) < 1e-12 &&
			    std::abs(shading_gain) < 1e-12)
				output = original;
			write_luma(frame, x, y, output);
		}
	}
	} else {
		const size_t count = static_cast<size_t>(ww) * wh;
		const double *projection_input = filter->cartoon.data();
		int projection_sweeps = 8;
		if (mode == 3) {
			// Fine chrome uses the one-step outer-map defect:
			//   u_TGFD - ROF(f - v_TGFD, lambda).
			// Keep the older u - ROF(u) field untouched for modes 1 and 2.
			for (size_t i = 0; i < count; ++i)
				filter->difference[i] =
					filter->input[i] - filter->texture[i];
			projection_input = filter->difference.data();
			projection_sweeps = std::clamp(passes, 8, 24);
		}
		if (bfft_meyer_rof(filter->plan, projection_input,
				   filter->reference_cartoon.data(), shade_c, 0.0,
				   projection_sweeps, 0.0) != BFFT_OK)
			return frame;

		double energy = 0.0;
		for (size_t i = 0; i < count; ++i) {
			const double d =
				filter->cartoon[i] - filter->reference_cartoon[i];
			filter->difference[i] = d;
			energy += d * d;
		}
		const double rms = std::sqrt(energy / std::max<size_t>(count, 1));
		const double inv_scale = 1.0 / std::max(3.0 * rms, 1e-6);
		if (mode == 2 || mode == 3)
			neutralize_chroma(frame);

		for (uint32_t y = 0; y < frame->height; ++y) {
			const uint32_t wy = std::min(
				static_cast<uint32_t>((static_cast<uint64_t>(y) * wh) /
						     frame->height),
				wh - 1);
			for (uint32_t x = 0; x < frame->width; ++x) {
				const uint32_t wx = std::min(
					static_cast<uint32_t>(
						(static_cast<uint64_t>(x) * ww) /
						frame->width),
					ww - 1);
				const size_t i = static_cast<size_t>(wy) * ww + wx;
				const double h = std::clamp(
					filter->difference[i] * inv_scale, -1.0, 1.0);
				if (mode == 1) {
					write_luma(frame, x, y,
						   128.0 + 110.0 * relief * h, true);
					continue;
				}

				const uint32_t xl = wx ? wx - 1 : ww - 1;
				const uint32_t xr = wx + 1 < ww ? wx + 1 : 0;
				const uint32_t yu = wy ? wy - 1 : wh - 1;
				const uint32_t yd = wy + 1 < wh ? wy + 1 : 0;
				const double dx =
					(filter->difference[static_cast<size_t>(wy) * ww + xr] -
					 filter->difference[static_cast<size_t>(wy) * ww + xl]) *
					inv_scale;
				const double dy =
					(filter->difference[static_cast<size_t>(yd) * ww + wx] -
					 filter->difference[static_cast<size_t>(yu) * ww + wx]) *
					inv_scale;
				double nx = -relief * 2.5 * dx;
				double ny = -relief * 2.5 * dy;
				double nz = 1.0;
				const double nlen = std::sqrt(nx * nx + ny * ny + 1.0);
				nx /= nlen;
				ny /= nlen;
				nz /= nlen;

				const int ox = static_cast<int>(std::lround(nx * relief * 8.0));
				const int oy = static_cast<int>(std::lround(ny * relief * 8.0));
				const uint32_t sx = static_cast<uint32_t>(std::clamp(
					static_cast<int>(wx) + ox, 0,
					static_cast<int>(ww) - 1));
				const uint32_t sy = static_cast<uint32_t>(std::clamp(
					static_cast<int>(wy) + oy, 0,
					static_cast<int>(wh) - 1));
				const double displaced =
					filter->input[static_cast<size_t>(sy) * ww + sx];

				const double light = std::max(
					0.0, -0.35 * nx - 0.45 * ny + 0.82 * nz);
				const double specular = std::pow(
					std::max(0.0, light), 8.0 + gloss * 72.0);
				const double environment =
					0.5 + 0.5 * std::sin(10.0 * ny + 3.0 * h);
				const double chrome =
					20.0 + 85.0 * light +
					75.0 * environment + 100.0 * gloss * specular;
				const double output =
					(0.35 - 0.2 * gloss) * displaced + chrome;
				write_luma(frame, x, y, output, true);
			}
		}
	}

	const double elapsed_ms =
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - started)
			.count();
	filter->frames++;
	filter->total_ms += elapsed_ms;
	if (filter->frames % 300 == 0) {
		blog(LOG_INFO,
		     "[BFFT Cartoon] %.2f ms/frame average (%.1f fps capacity)",
		     filter->total_ms / filter->frames,
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
	blog(LOG_INFO, "[BFFT Cartoon] loaded v0.1.1 (allocation-stable)");
	return true;
}
