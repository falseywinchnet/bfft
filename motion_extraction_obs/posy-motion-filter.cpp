#include <obs-module.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("posy-motion", "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Live Posy-style motion extraction and acceleration visualization";
}

namespace {

constexpr const char *K_MODE = "pm_mode";
constexpr const char *K_DELAY = "pm_delay";
constexpr const char *K_GAIN = "pm_gain";
constexpr const char *K_BLUR = "pm_blur";
constexpr const char *K_GLOW = "pm_glow";
constexpr const char *K_RDELAY = "pm_rdelay";
constexpr const char *K_GDELAY = "pm_gdelay";
constexpr const char *K_BDELAY = "pm_bdelay";
constexpr const char *K_EXPONENT = "pm_exponent";
constexpr const char *K_THRESHOLD = "pm_threshold";
constexpr const char *K_ACCEL = "pm_accel";
constexpr const char *K_WARP = "pm_warp";
constexpr const char *K_MEMORY = "pm_memory";
constexpr const char *K_CAPTURE = "pm_capture";
constexpr const char *K_CLEAR = "pm_clear";
constexpr const char *K_RESET = "pm_reset";

enum Mode {
	BYPASS,
	SIGNED_DELAY,
	ABS_DELAY,
	FROZEN_BASELINE,
	MOTION_OVERLAY,
	LARGE_FEATURES,
	GLOW_OVERLAY,
	RGB_TIME,
	RECOLOR,
	COLOR_EXTRACT,
	ACCELERATION_ENERGY,
};

struct Settings {
	int mode = SIGNED_DELAY;
	int delay = 2;
	int rdelay = 0, gdelay = 3, bdelay = 6;
	int blur = 8, glow = 12, memory_mb = 160;
	float gain = 4.0f, exponent = 1.0f, threshold = 0.02f;
	float acceleration = 0.7f, warp = 10.0f;
};

struct Snapshot {
	std::uint64_t number = 0;
	std::vector<std::uint8_t> y;
};

struct Filter {
	obs_source_t *source = nullptr;
	std::mutex settings_mutex, processing_mutex;
	Settings settings;
	bool capture_requested = false, clear_requested = false;
	bool reset_requested = false;
	std::uint32_t width = 0, height = 0;
	std::uint64_t number = 0, frames = 0;
	std::deque<Snapshot> history;
	std::vector<std::uint8_t> current, previous, previous2, baseline;
	std::vector<float> a, b, tmp, response;
	double total_ms = 0.0;
};

std::uint8_t byte(float x)
{
	return static_cast<std::uint8_t>(
		std::clamp(std::lround(x), 0L, 255L));
}

bool planar(enum video_format f)
{
	return f == VIDEO_FORMAT_I420 || f == VIDEO_FORMAT_NV12 ||
	       f == VIDEO_FORMAT_Y800 || f == VIDEO_FORMAT_I444 ||
	       f == VIDEO_FORMAT_I422 || f == VIDEO_FORMAT_I40A ||
	       f == VIDEO_FORMAT_I42A || f == VIDEO_FORMAT_YUVA;
}

bool supported(enum video_format f)
{
	return planar(f) || f == VIDEO_FORMAT_YUY2 || f == VIDEO_FORMAT_YVYU ||
	       f == VIDEO_FORMAT_UYVY || f == VIDEO_FORMAT_BGRA ||
	       f == VIDEO_FORMAT_BGRX || f == VIDEO_FORMAT_RGBA ||
	       f == VIDEO_FORMAT_BGR3;
}

float read_y(const obs_source_frame *f, std::uint32_t x, std::uint32_t y)
{
	const auto *row = f->data[0] + static_cast<std::size_t>(y) *
					      f->linesize[0];
	if (planar(f->format))
		return row[x] / 255.0f;
	if (f->format == VIDEO_FORMAT_YUY2 || f->format == VIDEO_FORMAT_YVYU)
		return row[2 * x] / 255.0f;
	if (f->format == VIDEO_FORMAT_UYVY)
		return row[2 * x + 1] / 255.0f;
	const int n = f->format == VIDEO_FORMAT_BGR3 ? 3 : 4;
	const auto *p = row + n * x;
	if (f->format == VIDEO_FORMAT_RGBA)
		return (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) /
		       255.0f;
	return (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]) / 255.0f;
}

void write_y(obs_source_frame *f, std::uint32_t x, std::uint32_t y, float v,
	     bool mono)
{
	auto *row = f->data[0] + static_cast<std::size_t>(y) * f->linesize[0];
	const auto q = byte(255.0f * v);
	if (planar(f->format)) {
		row[x] = q;
		return;
	}
	if (f->format == VIDEO_FORMAT_YUY2 || f->format == VIDEO_FORMAT_YVYU) {
		row[2 * x] = q;
		return;
	}
	if (f->format == VIDEO_FORMAT_UYVY) {
		row[2 * x + 1] = q;
		return;
	}
	const int n = f->format == VIDEO_FORMAT_BGR3 ? 3 : 4;
	auto *p = row + n * x;
	if (mono) {
		p[0] = p[1] = p[2] = q;
		return;
	}
	const float old = read_y(f, x, y) * 255.0f;
	const float d = q - old;
	p[0] = byte(p[0] + d);
	p[1] = byte(p[1] + d);
	p[2] = byte(p[2] + d);
}

void write_rgb(obs_source_frame *f, std::uint32_t x, std::uint32_t y,
	       float r, float g, float b)
{
	r = std::clamp(r, 0.0f, 1.0f);
	g = std::clamp(g, 0.0f, 1.0f);
	b = std::clamp(b, 0.0f, 1.0f);
	auto *row = f->data[0] + static_cast<std::size_t>(y) * f->linesize[0];
	if (f->format == VIDEO_FORMAT_BGRA || f->format == VIDEO_FORMAT_BGRX) {
		auto *p = row + 4 * x;
		p[0] = byte(255 * b); p[1] = byte(255 * g); p[2] = byte(255 * r);
		return;
	}
	if (f->format == VIDEO_FORMAT_RGBA) {
		auto *p = row + 4 * x;
		p[0] = byte(255 * r); p[1] = byte(255 * g); p[2] = byte(255 * b);
		return;
	}
	if (f->format == VIDEO_FORMAT_BGR3) {
		auto *p = row + 3 * x;
		p[0] = byte(255 * b); p[1] = byte(255 * g); p[2] = byte(255 * r);
		return;
	}
	const float yy = 0.2126f * r + 0.7152f * g + 0.0722f * b;
	const float u = (b - yy) / 1.8556f;
	const float v = (r - yy) / 1.5748f;
	const float yc = f->full_range ? 255.0f * yy : 16.0f + 219.0f * yy;
	const float uc = 128.0f + (f->full_range ? 255.0f : 224.0f) * u;
	const float vc = 128.0f + (f->full_range ? 255.0f : 224.0f) * v;
	if (planar(f->format))
		row[x] = byte(yc);
	else
		write_y(f, x, y, yc / 255.0f, false);
	// Chroma is intentionally sampled once per destination cell. Motion
	// evidence remains native-resolution in Y; only the source format's
	// inherent chroma lattice is subsampled.
	if (f->format == VIDEO_FORMAT_NV12 && !(x & 1) && !(y & 1)) {
		auto *uv = f->data[1] + static_cast<std::size_t>(y / 2) *
						f->linesize[1] + x;
		uv[0] = byte(uc); uv[1] = byte(vc);
	} else if ((f->format == VIDEO_FORMAT_I420 ||
		    f->format == VIDEO_FORMAT_I40A) &&
		   !(x & 1) && !(y & 1)) {
		f->data[1][static_cast<std::size_t>(y / 2) * f->linesize[1] +
			   x / 2] = byte(uc);
		f->data[2][static_cast<std::size_t>(y / 2) * f->linesize[2] +
			   x / 2] = byte(vc);
	} else if (f->format == VIDEO_FORMAT_I444 ||
		   f->format == VIDEO_FORMAT_YUVA) {
		f->data[1][static_cast<std::size_t>(y) * f->linesize[1] + x] =
			byte(uc);
		f->data[2][static_cast<std::size_t>(y) * f->linesize[2] + x] =
			byte(vc);
	}
}

void neutralize(obs_source_frame *f)
{
	if (f->format == VIDEO_FORMAT_NV12) {
		for (std::uint32_t y = 0; y < (f->height + 1) / 2; ++y)
			std::fill_n(f->data[1] + static_cast<std::size_t>(y) *
						    f->linesize[1],
				    f->width, std::uint8_t{128});
	} else if (f->format == VIDEO_FORMAT_I420 ||
		   f->format == VIDEO_FORMAT_I40A) {
		for (std::uint32_t y = 0; y < (f->height + 1) / 2; ++y) {
			std::fill_n(f->data[1] + static_cast<std::size_t>(y) *
						    f->linesize[1],
				    (f->width + 1) / 2, std::uint8_t{128});
			std::fill_n(f->data[2] + static_cast<std::size_t>(y) *
						    f->linesize[2],
				    (f->width + 1) / 2, std::uint8_t{128});
		}
	}
}

void box_blur(const std::vector<float> &src, std::vector<float> &dst,
	      std::vector<float> &tmp, int w, int h, int radius)
{
	if (radius <= 0) {
		dst = src;
		return;
	}
	tmp.resize(src.size());
	dst.resize(src.size());
	for (int y = 0; y < h; ++y) {
		float sum = 0;
		for (int x = -radius; x <= radius; ++x)
			sum += src[static_cast<std::size_t>(y) * w +
				   std::clamp(x, 0, w - 1)];
		for (int x = 0; x < w; ++x) {
			tmp[static_cast<std::size_t>(y) * w + x] =
				sum / (2 * radius + 1);
			sum += src[static_cast<std::size_t>(y) * w +
				   std::clamp(x + radius + 1, 0, w - 1)];
			sum -= src[static_cast<std::size_t>(y) * w +
				   std::clamp(x - radius, 0, w - 1)];
		}
	}
	for (int x = 0; x < w; ++x) {
		float sum = 0;
		for (int y = -radius; y <= radius; ++y)
			sum += tmp[static_cast<std::size_t>(
					   std::clamp(y, 0, h - 1)) *
					   w +
				   x];
		for (int y = 0; y < h; ++y) {
			dst[static_cast<std::size_t>(y) * w + x] =
				sum / (2 * radius + 1);
			sum += tmp[static_cast<std::size_t>(
					   std::clamp(y + radius + 1, 0, h - 1)) *
					   w +
				   x];
			sum -= tmp[static_cast<std::size_t>(
					   std::clamp(y - radius, 0, h - 1)) *
					   w +
				   x];
		}
	}
}

const std::vector<std::uint8_t> &reference(Filter *f, int delay)
{
	if (delay <= 1 && !f->previous.empty())
		return f->previous;
	const std::uint64_t target =
		f->number > static_cast<std::uint64_t>(delay)
			? f->number - static_cast<std::uint64_t>(delay)
			: 0;
	const Snapshot *best = nullptr;
	for (const auto &s : f->history) {
		if (!best || std::llabs(static_cast<long long>(s.number) -
				       static_cast<long long>(target)) <
				     std::llabs(static_cast<long long>(best->number) -
				       static_cast<long long>(target)))
			best = &s;
	}
	return best ? best->y : f->current;
}

void retain(Filter *f, const Settings &s)
{
	const std::size_t pixels = f->current.size();
	const int max_delay =
		std::max({s.delay * (s.mode == ACCELERATION_ENERGY ? 2 : 1),
			  s.rdelay, s.gdelay, s.bdelay, 2});
	const std::size_t budget =
		static_cast<std::size_t>(std::max(s.memory_mb, 16)) << 20;
	const int slots = std::max<int>(
		2, std::min<std::size_t>(max_delay + 1, budget / pixels));
	const int stride = std::max(1, (max_delay + slots - 1) / slots);
	if (f->number % static_cast<std::uint64_t>(stride) == 0)
		f->history.push_back({f->number, f->current});
	while (static_cast<int>(f->history.size()) > slots)
		f->history.pop_front();
}

void reset_state(Filter *f)
{
	f->history.clear();
	f->previous.clear();
	f->previous2.clear();
	f->baseline.clear();
	f->number = 0;
}

const char *name(void *) { return "Posy Motion Extraction (Live)"; }

void defaults(obs_data_t *d)
{
	obs_data_set_default_int(d, K_MODE, SIGNED_DELAY);
	obs_data_set_default_int(d, K_DELAY, 2);
	obs_data_set_default_double(d, K_GAIN, 4.0);
	obs_data_set_default_int(d, K_BLUR, 8);
	obs_data_set_default_int(d, K_GLOW, 12);
	obs_data_set_default_int(d, K_RDELAY, 0);
	obs_data_set_default_int(d, K_GDELAY, 3);
	obs_data_set_default_int(d, K_BDELAY, 6);
	obs_data_set_default_double(d, K_EXPONENT, 1.0);
	obs_data_set_default_double(d, K_THRESHOLD, 0.02);
	obs_data_set_default_double(d, K_ACCEL, 0.7);
	obs_data_set_default_double(d, K_WARP, 10.0);
	obs_data_set_default_int(d, K_MEMORY, 160);
}

void update(void *data, obs_data_t *d)
{
	auto *f = static_cast<Filter *>(data);
	std::lock_guard<std::mutex> lock(f->settings_mutex);
	f->settings.mode = static_cast<int>(obs_data_get_int(d, K_MODE));
	f->settings.delay = static_cast<int>(obs_data_get_int(d, K_DELAY));
	f->settings.gain = static_cast<float>(obs_data_get_double(d, K_GAIN));
	f->settings.blur = static_cast<int>(obs_data_get_int(d, K_BLUR));
	f->settings.glow = static_cast<int>(obs_data_get_int(d, K_GLOW));
	f->settings.rdelay = static_cast<int>(obs_data_get_int(d, K_RDELAY));
	f->settings.gdelay = static_cast<int>(obs_data_get_int(d, K_GDELAY));
	f->settings.bdelay = static_cast<int>(obs_data_get_int(d, K_BDELAY));
	f->settings.exponent =
		static_cast<float>(obs_data_get_double(d, K_EXPONENT));
	f->settings.threshold =
		static_cast<float>(obs_data_get_double(d, K_THRESHOLD));
	f->settings.acceleration =
		static_cast<float>(obs_data_get_double(d, K_ACCEL));
	f->settings.warp = static_cast<float>(obs_data_get_double(d, K_WARP));
	f->settings.memory_mb = static_cast<int>(obs_data_get_int(d, K_MEMORY));
}

bool capture(obs_properties_t *, obs_property_t *, void *data)
{
	auto *f = static_cast<Filter *>(data);
	if (!f)
		return false;
	std::lock_guard<std::mutex> lock(f->settings_mutex);
	f->capture_requested = true;
	return true;
}
bool clear(obs_properties_t *, obs_property_t *, void *data)
{
	auto *f = static_cast<Filter *>(data);
	if (!f)
		return false;
	std::lock_guard<std::mutex> lock(f->settings_mutex);
	f->clear_requested = true;
	return true;
}
bool reset(obs_properties_t *, obs_property_t *, void *data)
{
	auto *f = static_cast<Filter *>(data);
	if (!f)
		return false;
	std::lock_guard<std::mutex> lock(f->settings_mutex);
	f->reset_requested = true;
	return true;
}

obs_properties_t *properties(void *data)
{
	auto *p = obs_properties_create();
	auto *m = obs_properties_add_list(p, K_MODE, "Live mode",
					  OBS_COMBO_TYPE_LIST,
					  OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(m, "Bypass", BYPASS);
	obs_property_list_add_int(m, "Signed delayed extraction", SIGNED_DELAY);
	obs_property_list_add_int(m, "Absolute motion (deer / wildlife)",
				  ABS_DELAY);
	obs_property_list_add_int(m, "Frozen baseline / changes over time",
				  FROZEN_BASELINE);
	obs_property_list_add_int(m, "Highlight motion over picture",
				  MOTION_OVERLAY);
	obs_property_list_add_int(m, "Large features / wind direction",
				  LARGE_FEATURES);
	obs_property_list_add_int(m, "Motion glow", GLOW_OVERLAY);
	obs_property_list_add_int(m, "RGB channel time shift", RGB_TIME);
	obs_property_list_add_int(m, "Two-color motion", RECOLOR);
	obs_property_list_add_int(m, "Color-preserving motion", COLOR_EXTRACT);
	obs_property_list_add_int(m, "Acceleration energy: brightness + warp",
				  ACCELERATION_ENERGY);
	obs_properties_add_int_slider(p, K_DELAY, "Delay (frames)", 1, 1800, 1);
	obs_properties_add_float_slider(p, K_GAIN, "Motion gain", 0.1, 32.0,
					0.1);
	obs_properties_add_int_slider(p, K_BLUR, "Large-feature blur radius", 0,
				      64, 1);
	obs_properties_add_int_slider(p, K_GLOW, "Glow radius", 0, 64, 1);
	obs_properties_add_int_slider(p, K_RDELAY, "Red delay", 0, 600, 1);
	obs_properties_add_int_slider(p, K_GDELAY, "Green delay", 0, 600, 1);
	obs_properties_add_int_slider(p, K_BDELAY, "Blue delay", 0, 600, 1);
	obs_properties_add_float_slider(p, K_EXPONENT,
					"Speed/acceleration exponent", 0.1, 8.0,
					0.05);
	obs_properties_add_float_slider(p, K_THRESHOLD, "Motion noise floor",
					0.0, 0.25, 0.0025);
	obs_properties_add_float_slider(p, K_ACCEL,
					"Acceleration vs velocity", 0.0, 1.0,
					0.01);
	obs_properties_add_float_slider(p, K_WARP, "Acceleration distortion",
					0.0, 64.0, 0.5);
	obs_properties_add_int_slider(p, K_MEMORY, "Delay memory budget (MB)",
				      16, 512, 16);
	obs_properties_add_button(p, K_CAPTURE, "Capture frozen baseline",
				  capture);
	obs_properties_add_button(p, K_CLEAR, "Clear frozen baseline", clear);
	obs_properties_add_button(p, K_RESET, "Reset delay history", reset);
	(void)data;
	return p;
}

void *create(obs_data_t *d, obs_source_t *source)
{
	auto *f = new Filter;
	f->source = source;
	update(f, d);
	return f;
}

void destroy(void *data)
{
	auto *f = static_cast<Filter *>(data);
	blog(LOG_INFO, "[Posy Motion] %llu frames, %.2f ms/frame",
	     static_cast<unsigned long long>(f->frames),
	     f->frames ? f->total_ms / f->frames : 0.0);
	delete f;
}

obs_source_frame *video(void *data, obs_source_frame *frame)
{
	auto *f = static_cast<Filter *>(data);
	if (!frame || !frame->data[0] || !supported(frame->format))
		return frame;
	std::lock_guard<std::mutex> process_lock(f->processing_mutex);
	Settings s;
	bool do_capture, do_clear, do_reset;
	{
		std::lock_guard<std::mutex> lock(f->settings_mutex);
		s = f->settings;
		do_capture = std::exchange(f->capture_requested, false);
		do_clear = std::exchange(f->clear_requested, false);
		do_reset = std::exchange(f->reset_requested, false);
	}
	const std::size_t n = static_cast<std::size_t>(frame->width) *
			      frame->height;
	if (f->width != frame->width || f->height != frame->height) {
		f->width = frame->width;
		f->height = frame->height;
		f->current.resize(n);
		f->a.resize(n);
		f->b.resize(n);
		f->response.resize(n);
		reset_state(f);
		blog(LOG_INFO, "[Posy Motion] native grid %ux%u, format=%d",
		     f->width, f->height, static_cast<int>(frame->format));
	}
	if (do_reset)
		reset_state(f);
	if (do_clear)
		f->baseline.clear();
	const auto started = std::chrono::steady_clock::now();
	for (std::uint32_t y = 0; y < f->height; ++y)
		for (std::uint32_t x = 0; x < f->width; ++x)
			f->current[static_cast<std::size_t>(y) * f->width + x] =
				byte(255.0f * read_y(frame, x, y));
	if (do_capture || (s.mode == FROZEN_BASELINE && f->baseline.empty()))
		f->baseline = f->current;
	retain(f, s);

	const auto &delayed = reference(f, std::max(s.delay, 1));
	const auto &delayed2 = reference(f, std::max(s.delay * 2, 2));
	const float floor = std::clamp(s.threshold, 0.0f, 0.25f);
	for (std::size_t i = 0; i < n; ++i) {
		const float c = f->current[i] / 255.0f;
		const float d = delayed[i] / 255.0f;
		f->a[i] = c;
		f->b[i] = d;
		f->response[i] = std::max(std::abs(c - d) - floor, 0.0f);
	}

	bool mono = false;
	if (s.mode == SIGNED_DELAY || s.mode == FROZEN_BASELINE ||
	    s.mode == LARGE_FEATURES || s.mode == ABS_DELAY ||
	    s.mode == RECOLOR || s.mode == COLOR_EXTRACT) {
		const std::vector<std::uint8_t> *ref = &delayed;
		if (s.mode == FROZEN_BASELINE && !f->baseline.empty())
			ref = &f->baseline;
		if (s.mode == LARGE_FEATURES) {
			box_blur(f->a, f->response, f->tmp, f->width, f->height,
				 std::clamp(s.blur, 0, 64));
			box_blur(f->b, f->a, f->tmp, f->width, f->height,
				 std::clamp(s.blur, 0, 64));
		}
		for (std::uint32_t y = 0; y < f->height; ++y) {
			for (std::uint32_t x = 0; x < f->width; ++x) {
				const std::size_t i =
					static_cast<std::size_t>(y) * f->width + x;
				float diff;
				if (s.mode == LARGE_FEATURES)
					diff = f->response[i] - f->a[i];
				else
					diff = (f->current[i] - (*ref)[i]) /
					       255.0f;
				float out;
				if (s.mode == ABS_DELAY)
					out = s.gain * std::abs(diff);
				else if (s.mode == COLOR_EXTRACT)
					out = 0.5f + s.gain * diff;
				else if (s.mode == RECOLOR)
					out = 0.5f + 0.5f * std::tanh(s.gain * diff);
				else
					out = 0.5f + s.gain * diff;
				if (s.mode == RECOLOR) {
					const float e = std::tanh(s.gain * diff);
					write_rgb(frame, x, y, 0.5f + 0.5f * e,
						  0.12f * std::abs(e),
						  0.5f - 0.5f * e);
				} else {
					write_y(frame, x, y,
						std::clamp(out, 0.0f, 1.0f),
						s.mode != COLOR_EXTRACT);
				}
			}
		}
		mono = s.mode != COLOR_EXTRACT && s.mode != RECOLOR;
	} else if (s.mode == MOTION_OVERLAY || s.mode == GLOW_OVERLAY) {
		if (s.mode == GLOW_OVERLAY)
			box_blur(f->response, f->a, f->tmp, f->width, f->height,
				 std::clamp(s.glow, 0, 64));
		for (std::uint32_t y = 0; y < f->height; ++y)
			for (std::uint32_t x = 0; x < f->width; ++x) {
				const std::size_t i =
					static_cast<std::size_t>(y) * f->width + x;
				const float e = s.mode == GLOW_OVERLAY ? f->a[i]
								       : f->response[i];
				write_y(frame, x, y,
					std::clamp(f->current[i] / 255.0f +
							   s.gain * e,
						   0.0f, 1.0f),
					false);
			}
	} else if (s.mode == RGB_TIME) {
		const auto &rr = reference(f, s.rdelay);
		const auto &gg = reference(f, s.gdelay);
		const auto &bb = reference(f, s.bdelay);
		for (std::uint32_t y = 0; y < f->height; ++y)
			for (std::uint32_t x = 0; x < f->width; ++x) {
				const std::size_t i =
					static_cast<std::size_t>(y) * f->width + x;
				write_rgb(frame, x, y, rr[i] / 255.0f,
					  gg[i] / 255.0f, bb[i] / 255.0f);
			}
	} else if (s.mode == ACCELERATION_ENERGY) {
		for (std::size_t i = 0; i < n; ++i) {
			const float c = f->current[i] / 255.0f;
			const float d1 = delayed[i] / 255.0f;
			const float d2 = delayed2[i] / 255.0f;
			const float velocity = std::abs(c - d1);
			const float acceleration = std::abs(c - 2 * d1 + d2);
			const float e = std::max(
				(1 - s.acceleration) * velocity +
					s.acceleration * acceleration - floor,
				0.0f);
			f->response[i] = std::pow(std::min(e * s.gain, 1.0f),
						 std::max(s.exponent, 0.1f));
		}
		for (std::uint32_t y = 0; y < f->height; ++y)
			for (std::uint32_t x = 0; x < f->width; ++x) {
				const std::size_t i =
					static_cast<std::size_t>(y) * f->width + x;
				const int xl = x ? x - 1 : x, xr =
					x + 1 < f->width ? x + 1 : x;
				const int yu = y ? y - 1 : y, yd =
					y + 1 < f->height ? y + 1 : y;
				const float dx =
					f->response[static_cast<std::size_t>(y) *
							    f->width +
						    xr] -
					f->response[static_cast<std::size_t>(y) *
							    f->width +
						    xl];
				const float dy =
					f->response[static_cast<std::size_t>(yd) *
							    f->width +
						    x] -
					f->response[static_cast<std::size_t>(yu) *
							    f->width +
						    x];
				const int sx = std::clamp(
					static_cast<int>(x + s.warp * dx), 0,
					static_cast<int>(f->width) - 1);
				const int sy = std::clamp(
					static_cast<int>(y + s.warp * dy), 0,
					static_cast<int>(f->height) - 1);
				const float carrier =
					f->current[static_cast<std::size_t>(sy) *
							   f->width +
						   sx] /
					255.0f;
				write_y(frame, x, y,
					std::clamp(carrier + f->response[i], 0.0f,
						   1.0f),
					false);
			}
	}
	if (mono)
		neutralize(frame);
	f->previous2 = std::move(f->previous);
	f->previous = f->current;
	f->current.resize(n);
	++f->number;
	++f->frames;
	f->total_ms += std::chrono::duration<double, std::milli>(
			       std::chrono::steady_clock::now() - started)
			       .count();
	return frame;
}

obs_source_info info = {
	.id = "posy_motion_live_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_ASYNC_VIDEO,
	.get_name = name,
	.create = create,
	.destroy = destroy,
	.get_defaults = defaults,
	.get_properties = properties,
	.update = update,
	.filter_video = video,
};

} // namespace

bool obs_module_load(void)
{
	obs_register_source(&info);
	blog(LOG_INFO, "[Posy Motion] independent live filter registered");
	return true;
}
