#include <obs.h>
#include <util/base.h>

#import <AppKit/AppKit.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char *FILTER_ID = "convstar_support_toys";
constexpr const char *SOURCE_ID = "convstar_support_smoke_source";
uint32_t g_width = 320;
uint32_t g_height = 180;

struct SyntheticSource {
	gs_texture_t *texture = nullptr;
	std::vector<uint8_t> pixels;
};

std::vector<uint8_t> make_pixels()
{
	std::vector<uint8_t> pixels(static_cast<size_t>(g_width) * g_height * 4);
	for (uint32_t y = 0; y < g_height; ++y) {
		for (uint32_t x = 0; x < g_width; ++x) {
			const size_t q = 4 * (static_cast<size_t>(y) * g_width + x);
			const double nx = (static_cast<double>(x) + 0.5) / g_width;
			const double ny = (static_cast<double>(y) + 0.5) / g_height;
			const double dx = nx - 0.30;
			const double dy = ny - 0.50;
			const bool disk =
				dx * dx + dy * dy < 0.18 * 0.18;
			const uint32_t tile_x = std::max(1u, g_width / 16);
			const uint32_t tile_y = std::max(1u, g_height / 8);
			const bool stripe = ((x / tile_x + y / tile_y) & 1U) != 0;
			const double wave = 25.0 * std::sin(38.0 * nx + 8.0 * ny) +
					    15.0 * std::sin(10.0 * nx - 31.0 * ny);
			const double base = 55.0 + 90.0 * nx + (disk ? 75.0 : 0.0) +
					    (stripe ? 28.0 : 0.0) + wave;
			pixels[q] = static_cast<uint8_t>(std::clamp(
				std::lround(base + (disk ? 22.0 : 0.0)), 0L, 255L));
			pixels[q + 1] = static_cast<uint8_t>(std::clamp(
				std::lround(base + (stripe ? 15.0 : 0.0)), 0L, 255L));
			pixels[q + 2] = static_cast<uint8_t>(std::clamp(
				std::lround(base - (disk ? 20.0 : 0.0)), 0L, 255L));
			pixels[q + 3] = 255;
		}
	}
	return pixels;
}

const char *source_name(void *) { return "Conv* support smoke source"; }
void *source_create(obs_data_t *, obs_source_t *)
{
	auto *source = new SyntheticSource;
	source->pixels = make_pixels();
	const uint8_t *pixels = source->pixels.data();
	obs_enter_graphics();
	source->texture =
		gs_texture_create(g_width, g_height, GS_RGBA, 1, &pixels, GS_DYNAMIC);
	obs_leave_graphics();
	if (!source->texture) {
		delete source;
		return nullptr;
	}
	return source;
}
void source_destroy(void *data)
{
	auto *source = static_cast<SyntheticSource *>(data);
	obs_enter_graphics();
	gs_texture_destroy(source->texture);
	obs_leave_graphics();
	delete source;
}
void source_render(void *data, gs_effect_t *)
{
	auto *source = static_cast<SyntheticSource *>(data);
	obs_source_draw(source->texture, 0, 0, g_width, g_height, false);
}
uint32_t source_width(void *) { return g_width; }
uint32_t source_height(void *) { return g_height; }

void register_source()
{
	obs_source_info info{};
	info.id = SOURCE_ID;
	info.type = OBS_SOURCE_TYPE_INPUT;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = source_name;
	info.create = source_create;
	info.destroy = source_destroy;
	info.video_render = source_render;
	info.get_width = source_width;
	info.get_height = source_height;
	obs_register_source(&info);
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

bool render_frames(obs_source_t *source, std::vector<uint8_t> &pixels,
		   double &mean_ms)
{
	bool ok = false;
	obs_enter_graphics();
	gs_texrender_t *render = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	gs_stagesurf_t *stage = gs_stagesurface_create(g_width, g_height, GS_RGBA);
	constexpr int warmup_frames = 4;
	constexpr int measured_frames = 20;
	auto started = std::chrono::steady_clock::now();
	for (int frame = 0; frame < warmup_frames + measured_frames; ++frame) {
		if (frame == warmup_frames)
			started = std::chrono::steady_clock::now();
		gs_texrender_reset(render);
		if (!gs_texrender_begin(render, g_width, g_height))
			continue;
		vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_matrix_identity();
		gs_ortho(0.0f, static_cast<float>(g_width), 0.0f,
			 static_cast<float>(g_height), -100.0f, 100.0f);
		obs_source_video_render(source);
		gs_texrender_end(render);
	}
	mean_ms = std::chrono::duration<double, std::milli>(
			  std::chrono::steady_clock::now() - started)
			  .count() /
		  measured_frames;
	gs_stage_texture(stage, gs_texrender_get_texture(render));
	uint8_t *mapped = nullptr;
	uint32_t stride = 0;
	if (gs_stagesurface_map(stage, &mapped, &stride)) {
		pixels.resize(static_cast<size_t>(g_width) * g_height * 4);
		for (uint32_t y = 0; y < g_height; ++y)
			std::memcpy(pixels.data() + static_cast<size_t>(y) * g_width * 4,
				    mapped + static_cast<size_t>(y) * stride,
				    static_cast<size_t>(g_width) * 4);
		gs_stagesurface_unmap(stage);
		ok = true;
	}
	gs_stagesurface_destroy(stage);
	gs_texrender_destroy(render);
	obs_leave_graphics();
	return ok;
}

// OKLCH embedded as (L, C cos h, C sin h), so hue distance wraps exactly.
struct OklchCoordinate {
	double l;
	double chroma_x;
	double chroma_y;
};

double srgb_to_linear(uint8_t byte)
{
	const double value = byte / 255.0;
	return value <= 0.04045 ? value / 12.92 :
		std::pow((value + 0.055) / 1.055, 2.4);
}

OklchCoordinate to_oklch_coordinate(const uint8_t *pixel)
{
	const double r = srgb_to_linear(pixel[0]);
	const double g = srgb_to_linear(pixel[1]);
	const double b = srgb_to_linear(pixel[2]);
	const double lp = std::cbrt(std::max(0.0,
		0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b));
	const double mp = std::cbrt(std::max(0.0,
		0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b));
	const double sp = std::cbrt(std::max(0.0,
		0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b));
	return {
		0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp,
		1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp,
		0.0259040371 * lp + 0.7827717662 * mp - 0.8086757660 * sp};
}

double mean_oklch_distance(const std::vector<uint8_t> &a,
			   const std::vector<uint8_t> &b)
{
	if (a.size() != b.size())
		return 0.0;
	double sum = 0.0;
	for (size_t i = 0; i < a.size(); i += 4) {
		const OklchCoordinate first = to_oklch_coordinate(a.data() + i);
		const OklchCoordinate second = to_oklch_coordinate(b.data() + i);
		const double dl = first.l - second.l;
		const double dcx = first.chroma_x - second.chroma_x;
		const double dcy = first.chroma_y - second.chroma_y;
		sum += 100.0 * std::sqrt(dl * dl + dcx * dcx + dcy * dcy);
	}
	return sum / (a.size() / 4.0);
}

void write_ppm(const std::string &path, const std::vector<uint8_t> &pixels)
{
	std::FILE *file = std::fopen(path.c_str(), "wb");
	if (!file)
		return;
	std::fprintf(file, "P6\n%u %u\n255\n", g_width, g_height);
	for (size_t i = 0; i < pixels.size(); i += 4)
		std::fwrite(pixels.data() + i, 1, 3, file);
	std::fclose(file);
}

} // namespace

int main(int argc, char **argv)
{
	@autoreleasepool {
		[NSApplication sharedApplication];
		const char *module = argc > 1 ? argv[1]
			: "build-convstar-obs/convstar-support.plugin/Contents/MacOS/convstar-support";
		const char *capture = argc > 2 ? argv[2]
			: "build-convstar-obs/convstar-support-smoke.ppm";
		const int mode = argc > 3 ? std::clamp(std::atoi(argv[3]), 0, 13) : 2;
		g_width = argc > 4 ? std::clamp(std::atoi(argv[4]), 64, 3840) : 320;
		g_height = argc > 5 ? std::clamp(std::atoi(argv[5]), 64, 2160) : 180;
		const int resolution = argc > 6 ? std::clamp(std::atoi(argv[6]), 0, 2160) : 360;
		const double deformation = argc > 7 ?
			std::clamp(std::atof(argv[7]), 0.0, 3.0) : 2.0;
		const double hue_perversity = argc > 8 ?
			std::clamp(std::atof(argv[8]), 0.0, 2.0) : 1.0;
		const double lightness_perversity = argc > 9 ?
			std::clamp(std::atof(argv[9]), 0.0, 2.0) : 1.0;
		const bool use_opengl = argc > 10 &&
			std::strcmp(argv[10], "opengl") == 0;
		const double dip_reverse = argc > 11 ?
			std::clamp(std::atof(argv[11]), 0.0, 1.0) : 0.72;
		if (!obs_startup("en-US", nullptr, nullptr))
			return 2;
		obs_video_info video{};
		video.graphics_module = use_opengl
			? "/Applications/OBS.app/Contents/Frameworks/libobs-opengl.dylib"
			: "/Applications/OBS.app/Contents/Frameworks/libobs-metal.dylib";
		video.fps_num = 30;
		video.fps_den = 1;
		video.base_width = g_width;
		video.base_height = g_height;
		video.output_width = g_width;
		video.output_height = g_height;
		video.output_format = VIDEO_FORMAT_RGBA;
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
			obs_shutdown();
			return 4;
		}
		register_source();
		if (!registered(FILTER_ID)) {
			obs_shutdown();
			return 5;
		}
		obs_data_t *settings = obs_data_create();
		obs_data_set_int(settings, "convstar_toy_mode", mode);
		obs_data_set_double(settings, "convstar_toy_amount", 1.0);
		obs_data_set_double(settings, "convstar_support_deformation", deformation);
		obs_data_set_double(settings, "convstar_toy_speed", 0.0);
		obs_data_set_int(settings, "convstar_support_resolution", resolution);
		obs_data_set_double(settings, "convstar_oklch_hue_perversity",
				    hue_perversity);
		obs_data_set_double(settings, "convstar_oklch_lightness_perversity",
				    lightness_perversity);
		obs_data_set_double(settings, "convstar_dip_reverse", dip_reverse);
		obs_source_t *source = obs_source_create_private(
			SOURCE_ID, "convstar-smoke-source", nullptr);
		obs_source_t *filter = obs_source_create_private(
			FILTER_ID, "convstar-smoke-filter", settings);
		obs_data_release(settings);
		if (!source || !filter) {
			obs_shutdown();
			return 6;
		}
		obs_source_filter_add(source, filter);
		obs_source_inc_showing(source);
		obs_source_inc_active(source);
		std::vector<uint8_t> output;
		double mean_ms = 0.0;
		const bool rendered = render_frames(source, output, mean_ms);
		const auto input = make_pixels();
		const double delta = mean_oklch_distance(input, output);
		const bool ok = rendered && output.size() == input.size() && delta > 0.1;
		if (ok)
			write_ppm(capture, output);
		obs_source_dec_active(source);
		obs_source_dec_showing(source);
		obs_source_filter_remove(source, filter);
		obs_source_release(filter);
		obs_source_release(source);
		obs_shutdown();
		std::printf("DIP-CONV Gaussian comb smoke %.3f ms/frame, mean cyclic OKLCH dE %.3f\n",
			    mean_ms, delta);
		return ok ? 0 : 7;
	}
}
