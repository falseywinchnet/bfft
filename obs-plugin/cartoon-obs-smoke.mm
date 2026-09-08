#include <obs.h>
#include <util/base.h>
#include <util/platform.h>

#import <AppKit/AppKit.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char *CARTOON_ID = "bfft_cartoon_filter";
constexpr const char *SYNC_CARTOON_ID = "bfft_cartoon_filter_sync";
constexpr const char *SOURCE_ID = "bfft_cartoon_smoke_source";
constexpr const char *SYNC_SOURCE_ID = "bfft_cartoon_smoke_sync_source";
constexpr const char *TAP_ID = "bfft_cartoon_smoke_tap";
constexpr uint32_t WIDTH = 640;
constexpr uint32_t HEIGHT = 360;

struct Tap {
	std::mutex mutex;
	std::vector<uint8_t> pixels;
	uint64_t frames = 0;
	uint64_t timestamp = 0;
};

struct SyncSource {
	gs_texture_t *texture = nullptr;
	std::vector<uint8_t> pixels;
};

std::vector<uint8_t> synthetic_frame();

const char *source_name(void *) { return "BFFT cartoon smoke source"; }
void *source_create(obs_data_t *, obs_source_t *) { return new int(1); }
void source_destroy(void *data) { delete static_cast<int *>(data); }

const char *tap_name(void *) { return "BFFT cartoon smoke tap"; }
void *tap_create(obs_data_t *, obs_source_t *) { return new Tap; }
void tap_destroy(void *data) { delete static_cast<Tap *>(data); }
obs_source_frame *tap_video(void *data, obs_source_frame *frame)
{
	auto *tap = static_cast<Tap *>(data);
	if (!frame || frame->format != VIDEO_FORMAT_BGRA)
		return frame;
	std::lock_guard<std::mutex> lock(tap->mutex);
	tap->pixels.resize(static_cast<size_t>(frame->width) * frame->height * 4);
	for (uint32_t y = 0; y < frame->height; ++y)
		std::memcpy(tap->pixels.data() + static_cast<size_t>(y) *
					 frame->width * 4,
			    frame->data[0] + static_cast<size_t>(y) *
					 frame->linesize[0],
			    static_cast<size_t>(frame->width) * 4);
	++tap->frames;
	tap->timestamp = frame->timestamp;
	return frame;
}

const char *sync_source_name(void *) { return "BFFT cartoon sync smoke source"; }
void *sync_source_create(obs_data_t *, obs_source_t *)
{
	auto *source = new SyncSource;
	source->pixels = synthetic_frame();
	const uint8_t *pixels = source->pixels.data();
	obs_enter_graphics();
	source->texture =
		gs_texture_create(WIDTH, HEIGHT, GS_RGBA, 1, &pixels, GS_DYNAMIC);
	obs_leave_graphics();
	if (!source->texture) {
		delete source;
		return nullptr;
	}
	return source;
}
void sync_source_destroy(void *data)
{
	auto *source = static_cast<SyncSource *>(data);
	obs_enter_graphics();
	gs_texture_destroy(source->texture);
	obs_leave_graphics();
	delete source;
}
void sync_source_render(void *data, gs_effect_t *)
{
	auto *source = static_cast<SyncSource *>(data);
	obs_source_draw(source->texture, 0, 0, WIDTH, HEIGHT, false);
}
uint32_t sync_source_width(void *) { return WIDTH; }
uint32_t sync_source_height(void *) { return HEIGHT; }

void register_smoke_sources()
{
	obs_source_info source{};
	source.id = SOURCE_ID;
	source.type = OBS_SOURCE_TYPE_INPUT;
	source.output_flags = OBS_SOURCE_ASYNC_VIDEO;
	source.get_name = source_name;
	source.create = source_create;
	source.destroy = source_destroy;
	obs_register_source(&source);

	obs_source_info tap{};
	tap.id = TAP_ID;
	tap.type = OBS_SOURCE_TYPE_FILTER;
	tap.output_flags = OBS_SOURCE_ASYNC_VIDEO;
	tap.get_name = tap_name;
	tap.create = tap_create;
	tap.destroy = tap_destroy;
	tap.filter_video = tap_video;
	obs_register_source(&tap);

	obs_source_info sync_source{};
	sync_source.id = SYNC_SOURCE_ID;
	sync_source.type = OBS_SOURCE_TYPE_INPUT;
	sync_source.output_flags = OBS_SOURCE_VIDEO;
	sync_source.get_name = sync_source_name;
	sync_source.create = sync_source_create;
	sync_source.destroy = sync_source_destroy;
	sync_source.video_render = sync_source_render;
	sync_source.get_width = sync_source_width;
	sync_source.get_height = sync_source_height;
	obs_register_source(&sync_source);
}

bool registered(const char *wanted)
{
	for (size_t i = 0;; ++i) {
		const char *id = nullptr;
		if (!obs_enum_source_types(i, &id))
			return false;
		if (id && std::strcmp(id, wanted) == 0)
			return true;
	}
}

std::vector<uint8_t> synthetic_frame()
{
	std::vector<uint8_t> pixels(static_cast<size_t>(WIDTH) * HEIGHT * 4);
	for (uint32_t y = 0; y < HEIGHT; ++y) {
		for (uint32_t x = 0; x < WIDTH; ++x) {
			const size_t i = 4 * (static_cast<size_t>(y) * WIDTH + x);
			const bool disk =
				(static_cast<int>(x) - 185) *
					(static_cast<int>(x) - 185) +
				(static_cast<int>(y) - 170) *
					(static_cast<int>(y) - 170) <
				115 * 115;
			const bool box = x > 355 && x < 565 && y > 70 && y < 285;
			const double weave = 22.0 * std::sin(0.23 * x + 0.09 * y) +
				18.0 * std::sin(0.06 * x - 0.31 * y);
			const double base = 48.0 + 0.18 * x + 0.11 * y +
				(disk ? 82.0 : 0.0) + (box ? 55.0 : 0.0) + weave;
			pixels[i + 0] = static_cast<uint8_t>(std::clamp(
				std::lround(base - (disk ? 25.0 : 0.0)), 0L, 255L));
			pixels[i + 1] = static_cast<uint8_t>(std::clamp(
				std::lround(base + (box ? 18.0 : 0.0)), 0L, 255L));
			pixels[i + 2] = static_cast<uint8_t>(std::clamp(
				std::lround(base + (disk ? 30.0 : 0.0)), 0L, 255L));
			pixels[i + 3] = 255;
		}
	}
	return pixels;
}

void update_cartoon(obs_source_t *filter, int decomposition, int mode)
{
	obs_data_t *settings = obs_source_get_settings(filter);
	obs_data_set_int(settings, "decomposition_v1", decomposition);
	obs_data_set_int(settings, "fused_passes_v1", 16);
	obs_data_set_int(settings, "jump_lattice_v1", 256);
	obs_data_set_int(settings, "threads", 6);
	obs_data_set_int(settings, "mode", mode);
	obs_data_set_double(settings, "cartoon_gain_v2", 1.0);
	obs_data_set_double(settings, "texture_gain_v2", 0.0);
	obs_data_set_double(settings, "shading_gain_v2", 0.0);
	obs_source_update(filter, settings);
	obs_data_release(settings);
}

bool process(obs_source_t *source, Tap *tap, std::vector<uint8_t> &input,
	     std::vector<uint8_t> &output, double &elapsed_ms, uint64_t sequence)
{
	(void)sequence;
	obs_source_frame frame{};
	frame.data[0] = input.data();
	frame.linesize[0] = WIDTH * 4;
	frame.width = WIDTH;
	frame.height = HEIGHT;
	frame.timestamp = os_gettime_ns();
	frame.format = VIDEO_FORMAT_BGRA;
	frame.full_range = true;
	const auto started = std::chrono::steady_clock::now();
	obs_source_output_video(source, &frame);
	for (int attempt = 0; attempt < 500; ++attempt) {
		{
			std::lock_guard<std::mutex> lock(tap->mutex);
			if (tap->timestamp == frame.timestamp) {
				output = tap->pixels;
				elapsed_ms = std::chrono::duration<double, std::milli>(
					std::chrono::steady_clock::now() - started)
						     .count();
				return true;
			}
		}
		os_sleep_ms(1);
	}
	return false;
}

double mean_absolute_difference(const std::vector<uint8_t> &a,
				const std::vector<uint8_t> &b)
{
	if (a.size() != b.size() || a.empty())
		return 0.0;
	double sum = 0.0;
	for (size_t i = 0; i < a.size(); i += 4)
		for (int channel = 0; channel < 3; ++channel)
			sum += std::abs(int(a[i + channel]) - int(b[i + channel]));
	return sum / (3.0 * a.size() / 4.0);
}

bool write_ppm(const std::string &path, const std::vector<uint8_t> &pixels)
{
	std::FILE *file = std::fopen(path.c_str(), "wb");
	if (!file)
		return false;
	std::fprintf(file, "P6\n%u %u\n255\n", WIDTH, HEIGHT);
	for (size_t i = 0; i < pixels.size(); i += 4) {
		const uint8_t rgb[3] = {pixels[i + 2], pixels[i + 1], pixels[i]};
		std::fwrite(rgb, 1, 3, file);
	}
	std::fclose(file);
	return true;
}

bool process_sync(obs_source_t *source, std::vector<uint8_t> &output,
		  double &elapsed_ms)
{
	bool ok = false;
	obs_enter_graphics();
	gs_texrender_t *render = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	gs_stagesurf_t *stage = gs_stagesurface_create(WIDTH, HEIGHT, GS_RGBA);
	const auto started = std::chrono::steady_clock::now();
	for (int i = 0; i < 4; ++i) {
		gs_texrender_reset(render);
		if (!gs_texrender_begin(render, WIDTH, HEIGHT))
			continue;
		vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_matrix_identity();
		gs_ortho(0.0f, static_cast<float>(WIDTH), 0.0f,
			 static_cast<float>(HEIGHT), -100.0f, 100.0f);
		obs_source_video_render(source);
		gs_texrender_end(render);
	}
	elapsed_ms = std::chrono::duration<double, std::milli>(
			     std::chrono::steady_clock::now() - started)
			     .count() /
		     4.0;
	gs_stage_texture(stage, gs_texrender_get_texture(render));
	uint8_t *mapped = nullptr;
	uint32_t stride = 0;
	if (gs_stagesurface_map(stage, &mapped, &stride)) {
		output.resize(static_cast<size_t>(WIDTH) * HEIGHT * 4);
		for (uint32_t y = 0; y < HEIGHT; ++y)
			std::memcpy(output.data() + static_cast<size_t>(y) * WIDTH * 4,
				    mapped + static_cast<size_t>(y) * stride,
				    static_cast<size_t>(WIDTH) * 4);
		gs_stagesurface_unmap(stage);
		ok = true;
	}
	gs_stagesurface_destroy(stage);
	gs_texrender_destroy(render);
	obs_leave_graphics();
	return ok;
}

} // namespace

int main(int argc, char **argv)
{
	@autoreleasepool {
		[NSApplication sharedApplication];
		const char *module = argc > 1 ? argv[1] :
			"build-obs-cartoon/bfft-cartoon.plugin/Contents/MacOS/bfft-cartoon";
		const char *out_prefix = argc > 2 ? argv[2] :
			"build-obs-cartoon/cartoon-smoke";
		if (!obs_startup("en-US", nullptr, nullptr))
			return 2;
		obs_video_info video{};
		video.graphics_module =
			"/Applications/OBS.app/Contents/Frameworks/libobs-metal.dylib";
		video.fps_num = 30;
		video.fps_den = 1;
		video.base_width = WIDTH;
		video.base_height = HEIGHT;
		video.output_width = WIDTH;
		video.output_height = HEIGHT;
		video.output_format = VIDEO_FORMAT_BGRA;
		video.colorspace = VIDEO_CS_709;
		video.range = VIDEO_RANGE_FULL;
		video.scale_type = OBS_SCALE_BILINEAR;
		if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS) {
			obs_shutdown();
			return 3;
		}
		obs_module_t *loaded = nullptr;
		const int opened = obs_open_module(&loaded, module, "/private/tmp");
		if (opened != MODULE_SUCCESS || !obs_init_module(loaded)) {
			std::fprintf(stderr, "module load failed: %d\n", opened);
			obs_shutdown();
			return 4;
		}
		register_smoke_sources();
			if (!registered(CARTOON_ID) || !registered(SYNC_CARTOON_ID)) {
			obs_shutdown();
			return 5;
		}
		obs_source_t *source = obs_source_create_private(
			SOURCE_ID, "cartoon-smoke-source", nullptr);
			obs_source_t *tap_filter = obs_source_create_private(
				TAP_ID, "cartoon-smoke-tap", nullptr);
			obs_data_t *cartoon_settings = obs_data_create();
			obs_data_set_int(cartoon_settings, "jump_lattice_v1", 256);
			obs_source_t *cartoon = obs_source_create_private(
				CARTOON_ID, "cartoon-smoke-filter", cartoon_settings);
			obs_data_release(cartoon_settings);
		if (!source || !tap_filter || !cartoon) {
			obs_shutdown();
			return 6;
		}
		// OBS prepends on add and executes async filters from the end toward
		// the start. Add the processor first so the tap observes its output.
		obs_source_filter_add(source, cartoon);
		obs_source_filter_add(source, tap_filter);
		obs_source_inc_showing(source);
		obs_source_inc_active(source);
		auto *tap = static_cast<Tap *>(obs_obj_get_data(tap_filter));
		std::vector<uint8_t> input = synthetic_frame();
		std::vector<uint8_t> bootstrap, settle, flow, fused, chrome;
		double bootstrap_ms = 0.0, flow_ms = 0.0, fused_ms = 0.0;
		double settle_ms = 0.0, chrome_ms = 0.0;
		update_cartoon(cartoon, 0, 0);
		// OBS uses the first async frame to bootstrap its cache. Discard that
		// unfiltered delivery. Its async selector also retains one previous
		// frame across a settings update, so discard one settle frame per mode.
		const bool bootstrap_ok = process(
			source, tap, input, bootstrap, bootstrap_ms, 0);
		const bool flow_settled = process(
			source, tap, input, settle, settle_ms, 1);
		const bool flow_ok = process(source, tap, input, flow, flow_ms, 2);
		update_cartoon(cartoon, 1, 0);
		const bool fused_settled = process(
			source, tap, input, settle, settle_ms, 3);
		const bool fused_ok = process(source, tap, input, fused, fused_ms, 4);
		update_cartoon(cartoon, 0, 3);
		const bool chrome_settled = process(
			source, tap, input, settle, settle_ms, 5);
		const bool chrome_ok = process(source, tap, input, chrome, chrome_ms, 6);
		const double flow_delta = mean_absolute_difference(input, flow);
		const double engine_delta = mean_absolute_difference(flow, fused);
		const double effect_delta = mean_absolute_difference(flow, chrome);
			const bool images_ok = bootstrap_ok && flow_settled && flow_ok &&
			fused_settled && fused_ok && chrome_settled && chrome_ok &&
			flow_delta > 0.1 && engine_delta > 0.02 &&
			effect_delta > 1.0;
			if (images_ok) {
			write_ppm(std::string(out_prefix) + "-input.ppm", input);
			write_ppm(std::string(out_prefix) + "-flow-fast.ppm", flow);
			write_ppm(std::string(out_prefix) + "-fused16.ppm", fused);
			write_ppm(std::string(out_prefix) + "-fine-chrome.ppm", chrome);
			}

			obs_source_t *sync_source = obs_source_create_private(
				SYNC_SOURCE_ID, "cartoon-sync-smoke-source", nullptr);
			obs_data_t *sync_settings = obs_data_create();
			obs_data_set_int(sync_settings, "jump_lattice_v1", 256);
			obs_source_t *sync_cartoon = obs_source_create_private(
				SYNC_CARTOON_ID, "cartoon-sync-smoke-filter", sync_settings);
			obs_data_release(sync_settings);
			bool sync_ok = sync_source && sync_cartoon;
			std::vector<uint8_t> sync_output;
			double sync_ms = 0.0;
			if (sync_ok) {
				update_cartoon(sync_cartoon, 0, 0);
				obs_source_filter_add(sync_source, sync_cartoon);
				obs_source_inc_showing(sync_source);
				obs_source_inc_active(sync_source);
				sync_ok = process_sync(sync_source, sync_output, sync_ms);
				obs_source_dec_active(sync_source);
				obs_source_dec_showing(sync_source);
				obs_source_filter_remove(sync_source, sync_cartoon);
			}
			const double sync_delta = sync_ok
				? mean_absolute_difference(input, sync_output)
				: 0.0;
			sync_ok = sync_ok && sync_delta > 0.1;
			if (sync_cartoon)
				obs_source_release(sync_cartoon);
			if (sync_source)
				obs_source_release(sync_source);
		obs_source_dec_active(source);
		obs_source_dec_showing(source);
		obs_source_filter_remove(source, cartoon);
		obs_source_filter_remove(source, tap_filter);
		obs_source_release(cartoon);
		obs_source_release(tap_filter);
		obs_source_release(source);
		obs_shutdown();
			std::printf("finite-flow %.3f ms, fused16 %.3f ms, chrome %.3f ms; "
				    "input/flow MAD %.3f, flow/fused MAD %.3f, "
				    "flow/chrome MAD %.3f; sync %.3f ms, MAD %.3f\n",
				    flow_ms, fused_ms, chrome_ms, flow_delta, engine_delta,
				    effect_delta, sync_ms, sync_delta);
			return images_ok && sync_ok ? 0 : 7;
	}
}
