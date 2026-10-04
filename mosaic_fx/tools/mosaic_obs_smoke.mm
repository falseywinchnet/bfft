// Actual libobs/Metal regression for Mosaic FX (RVFX smoke harness pattern).
//   mosaic-fx-smoke MODULE LIBOBS_METAL OUT_DIR [WIDTH HEIGHT]
// Verifies registration and shader compilation, GPU output against the CPU
// reference renderer on the exact published tile pack, static stability,
// locality of re-tiling under motion, glint animation, exact zero-mix
// bypass, HDR passthrough, and records frame timings.
#include "mosaic/mosaic.hpp"

#include <obs.h>
#include <util/base.h>
#import <AppKit/AppKit.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
uint32_t W = 1280, H = 720;
int errors = 0;
log_handler_t logger = nullptr;
void* logger_arg = nullptr;
void log_hook(int level, const char* message, va_list args, void*)
{
    if (level == LOG_ERROR) ++errors;
    if (logger) logger(level, message, args, logger_arg);
}

struct Source {
    gs_texture_t* texture = nullptr;
    gs_color_space space = GS_CS_SRGB;
    double disc = -1, blob = -1;
    long long noise = -1;
};
std::vector<uint8_t> scene_pixels(double disc_fraction, double blob = -1, long long noise = 0)
{
    std::vector<uint8_t> px(std::size_t(W) * H * 4, 255);
    const double cx = disc_fraction * W, cy = 0.5 * H, r = 0.16 * H;
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            uint8_t* p = &px[(std::size_t(y) * W + x) * 4];
            double R = 40 + 120.0 * x / W, G = 70 + 70.0 * y / H, B = 150;
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r) { R = 235; G = 190; B = 50; }
            if (x > 0.72 * W && x < 0.95 * W && y > 0.15 * H && y < 0.55 * H) {
                const unsigned h = (unsigned(x / 4) * 73856093u) ^ (unsigned(y / 4) * 19349663u);
                R = 40 + h % 200; G = 40 + (h >> 8) % 200; B = 40 + (h >> 16) % 200;
            }
            if (y > 0.75 * H && y < 0.82 * H) { R = 30; G = 140; B = 120; }
            if (blob >= 0) {
                // A second moving object, like a person crossing the frame.
                const double bx = 0.5 * W, by = blob * H, rx = 0.07 * W, ry = 0.22 * H;
                if ((x - bx) * (x - bx) / (rx * rx) + (y - by) * (y - by) / (ry * ry) < 1) { R = 200; G = 120; B = 90; }
            }
            p[0] = uint8_t(R); p[1] = uint8_t(G); p[2] = uint8_t(B);
        }
    if (noise > 0) {
        // Camera-like sensor noise, +-3 codes per channel, new every frame.
        uint32_t state = uint32_t(noise) * 2654435761u + 12345u;
        for (std::size_t i = 0; i < px.size(); ++i) {
            if ((i & 3) == 3) continue;
            state = state * 1664525u + 1013904223u;
            px[i] = uint8_t(std::clamp(int(px[i]) + int(state >> 29) - 3, 0, 255));
        }
    }
    return px;
}
void upload_scene(Source* s, double disc, double blob = -1, long long noise = 0)
{
    if (disc == s->disc && blob == s->blob && noise == s->noise) return;
    auto px = scene_pixels(disc, blob, noise);
    const uint8_t* bytes = px.data();
    obs_enter_graphics();
    if (!s->texture) s->texture = gs_texture_create(W, H, GS_RGBA, 1, &bytes, GS_DYNAMIC);
    else gs_texture_set_image(s->texture, bytes, W * 4, false);
    obs_leave_graphics();
    s->disc = disc;
    s->blob = blob;
    s->noise = noise;
}
void* create(obs_data_t* settings, obs_source_t*)
{
    auto* s = new Source;
    upload_scene(s, obs_data_get_double(settings, "disc"));
    return s;
}
void destroy(void* data)
{
    auto* s = static_cast<Source*>(data);
    obs_enter_graphics();
    if (s->texture) gs_texture_destroy(s->texture);
    obs_leave_graphics();
    delete s;
}
void source_update(void* data, obs_data_t* settings)
{
    auto* s = static_cast<Source*>(data);
    s->space = static_cast<gs_color_space>(obs_data_get_int(settings, "space"));
    upload_scene(s, obs_data_get_double(settings, "disc"), obs_data_get_double(settings, "blob"),
                 obs_data_get_int(settings, "noise"));
}
void source_defaults(obs_data_t* s)
{
    obs_data_set_default_double(s, "disc", 0.30);
    obs_data_set_default_double(s, "blob", -1);
}
void draw(void* data, gs_effect_t*) { obs_source_draw(static_cast<Source*>(data)->texture, 0, 0, W, H, false); }
const char* name(void*) { return "Mosaic regression source"; }
gs_color_space source_space(void* data, size_t, const gs_color_space*) { return static_cast<Source*>(data)->space; }
uint32_t width(void*) { return W; }
uint32_t height(void*) { return H; }
void register_source()
{
    obs_source_info i{};
    i.id = "mosaic_test_source";
    i.type = OBS_SOURCE_TYPE_INPUT;
    i.output_flags = OBS_SOURCE_VIDEO;
    i.create = create; i.destroy = destroy; i.update = source_update; i.get_defaults = source_defaults;
    i.video_get_color_space = source_space; i.video_render = draw; i.get_name = name;
    i.get_width = width; i.get_height = height;
    obs_register_source(&i);
}

struct Renderer {
    gs_texrender_t* out = nullptr;
    gs_stagesurf_t* stage = nullptr;
    std::vector<uint8_t> pixels;
    double elapsed = 0;
    Renderer()
    {
        obs_enter_graphics();
        out = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        stage = gs_stagesurface_create(W, H, GS_RGBA);
        obs_leave_graphics();
        pixels.resize(std::size_t(W) * H * 4);
    }
    ~Renderer()
    {
        obs_enter_graphics();
        gs_stagesurface_destroy(stage);
        gs_texrender_destroy(out);
        obs_leave_graphics();
    }
    bool render(obs_source_t* source, bool read = true)
    {
        obs_enter_graphics();
        const auto start = std::chrono::steady_clock::now();
        gs_texrender_reset(out);
        if (!gs_texrender_begin(out, W, H)) { obs_leave_graphics(); return false; }
        gs_blend_state_push();
        gs_blend_function_separate(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA, GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
        vec4 clear;
        vec4_zero(&clear);
        gs_clear(GS_CLEAR_COLOR, &clear, 0, 0);
        gs_matrix_identity();
        gs_ortho(0, float(W), 0, float(H), -100, 100);
        obs_source_video_render(source);
        gs_blend_state_pop();
        gs_texrender_end(out);
        bool ok = true;
        if (!read) {
            elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            obs_leave_graphics();
            return true;
        }
        gs_stage_texture(stage, gs_texrender_get_texture(out));
        uint8_t* data = nullptr;
        uint32_t stride = 0;
        if (gs_stagesurface_map(stage, &data, &stride)) {
            for (uint32_t y = 0; y < H; ++y) std::memcpy(pixels.data() + std::size_t(y) * W * 4, data + std::size_t(y) * stride, std::size_t(W) * 4);
            gs_stagesurface_unmap(stage);
        } else ok = false;
        elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        obs_leave_graphics();
        return ok;
    }
};

void check(bool value, const char* why)
{
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(10);
    }
}
void save(const std::string& path, const std::vector<uint8_t>& px)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (std::size_t i = 0; i < px.size(); i += 4) std::fwrite(px.data() + i, 1, 3, f);
    std::fclose(f);
}
bool load_pack(const std::string& path, mosaic::GpuPack& p)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t ints[5];
    float floats[4];
    bool ok = std::fread(ints, sizeof ints, 1, f) == 1 && std::fread(floats, sizeof floats, 1, f) == 1;
    if (ok) {
        constexpr int T = mosaic::GpuPack::kTilesPerRow;
        p.rows = ints[0]; p.cells_w = ints[1]; p.cells_h = ints[2];
        p.count = uint32_t(ints[3]); p.max_cell = uint32_t(ints[4]);
        p.analysis_w = floats[0]; p.analysis_h = floats[1]; p.cell = floats[2]; p.half_scale = floats[3];
        p.geometry.resize(std::size_t(2 * T) * p.rows * 4);
        p.colors.resize(std::size_t(T) * p.rows * 4);
        p.cells.resize(std::size_t(p.cells_w) * p.cells_h * 4);
        ok = std::fread(p.geometry.data(), 2, p.geometry.size(), f) == p.geometry.size() &&
             std::fread(p.colors.data(), 2, p.colors.size(), f) == p.colors.size() &&
             std::fread(p.cells.data(), 2, p.cells.size(), f) == p.cells.size();
    }
    std::fclose(f);
    return ok;
}
void set(obs_source_t* src, const char* key, double v)
{
    auto* s = obs_source_get_settings(src);
    obs_data_set_double(s, key, v);
    obs_source_update(src, s);
    obs_data_release(s);
}
void set_int(obs_source_t* src, const char* key, long long v)
{
    auto* s = obs_source_get_settings(src);
    obs_data_set_int(s, key, v);
    obs_source_update(src, s);
    obs_data_release(s);
}
// Render until the output is unchanged for `quiet` consecutive frames.
int settle(Renderer& r, obs_source_t* source, int quiet, int limit, std::vector<double>* times = nullptr)
{
    std::vector<uint8_t> last;
    int same = 0;
    for (int frame = 0; frame < limit; ++frame) {
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        check(r.render(source), "render");
        if (times) times->push_back(r.elapsed);
        same = r.pixels == last ? same + 1 : 0;
        last = r.pixels;
        if (same >= quiet) return frame;
    }
    return -1;
}
std::vector<char> slurp(const std::string& path)
{
    std::vector<char> data;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        char buf[65536];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) data.insert(data.end(), buf, buf + n);
        std::fclose(f);
    }
    return data;
}
struct Diff { double mean = 0; std::size_t over = 0; int worst = 0; };
Diff diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int threshold, uint32_t x0 = 0, uint32_t x1 = 0)
{
    if (!x1) x1 = W;
    Diff d;
    std::size_t n = 0;
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = x0; x < x1; ++x) {
            const std::size_t i = (std::size_t(y) * W + x) * 4;
            int m = 0;
            for (int c = 0; c < 3; ++c) {
                const int e = std::abs(int(a[i + c]) - int(b[i + c]));
                d.mean += e;
                m = std::max(m, e);
            }
            d.worst = std::max(d.worst, m);
            d.over += m > threshold;
            ++n;
        }
    d.mean /= double(n * 3);
    return d;
}
} // namespace

int main(int argc, char** argv)
{
    @autoreleasepool {
        if (argc < 4) return 2;
        if (argc > 5) { W = uint32_t(std::atoi(argv[4])); H = uint32_t(std::atoi(argv[5])); }
        const std::string out = argv[3];
        const std::string dump = out + "/pack.bin";
        const bool dynamic = argc > 6 && std::string(argv[6]) == "dynamic";
        std::remove(dump.c_str());
        setenv("MOSAIC_FX_DUMP", dump.c_str(), 1);
        const std::string stats_path = out + "/stats.json";
        std::remove(stats_path.c_str());
        setenv("MOSAIC_FX_STATS", stats_path.c_str(), 1);
        [NSApplication sharedApplication];
        base_get_log_handler(&logger, &logger_arg);
        base_set_log_handler(log_hook, nullptr);
        check(obs_startup("en-US", nullptr, nullptr), "OBS startup");
        obs_video_info v{};
        v.graphics_module = argv[2];
        v.fps_num = 60; v.fps_den = 1;
        v.base_width = W; v.base_height = H; v.output_width = W; v.output_height = H;
        v.output_format = VIDEO_FORMAT_RGBA; v.gpu_conversion = true;
        v.colorspace = VIDEO_CS_709; v.range = VIDEO_RANGE_FULL;
        check(obs_reset_video(&v) == OBS_VIDEO_SUCCESS, "Metal startup");
        obs_module_t* module = nullptr;
        check(obs_open_module(&module, argv[1], "/tmp") == MODULE_SUCCESS && obs_init_module(module), "module load");
        register_source();
        auto* source = obs_source_create_private("mosaic_test_source", "mosaic-source", nullptr);
        auto* settings = obs_data_create();
        if (!dynamic) {
            obs_data_set_double(settings, "mz_sweep", 0);
            obs_data_set_double(settings, "mz_twinkle", 0);
            obs_data_set_double(settings, "mz_hz", 30);
        }
        auto* filter = obs_source_create_private("mosaic_fx_filter", "mosaic-filter", settings);
        obs_data_release(settings);
        check(source && filter, "filter registration/create");
        if (dynamic) {
            // Camera-like load with default filter settings: continuous motion
            // and fresh sensor noise every frame, paced at 30 fps.
            auto r_owner = std::make_unique<Renderer>();
            Renderer& r = *r_owner;
            auto run = [&](int frames, std::vector<double>& times) {
                for (int i = 0; i < frames; ++i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const double t = i / 30.0;
                    auto* sset = obs_source_get_settings(source);
                    obs_data_set_double(sset, "disc", 0.30 + 0.10 * std::sin(6.2831853 * t / 4));
                    obs_data_set_double(sset, "blob", 0.5 + 0.25 * std::sin(6.2831853 * t / 3));
                    obs_data_set_int(sset, "noise", i + 1);
                    obs_source_update(source, sset);
                    obs_data_release(sset);
                    check(r.render(source), "dynamic draw");
                    times.push_back(r.elapsed);
                    std::this_thread::sleep_until(t0 + std::chrono::microseconds(33333));
                }
            };
            auto summary = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                double mean = 0;
                for (double x : v) mean += x / v.size();
                return std::array<double, 3>{mean, v[v.size() * 95 / 100], v.back()};
            };
            std::vector<double> base, mosaic_times;
            run(150, base);
            obs_source_filter_add(source, filter);
            int waited = 0;
            for (; waited < 600; ++waited) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                check(r.render(source), "warmup draw");
                if (FILE* f = std::fopen(dump.c_str(), "rb")) { std::fclose(f); break; }
            }
            check(waited < 600, "worker publishes under motion");
            run(300, mosaic_times);
            save(out + "/dynamic_last.ppm", r.pixels);
            obs_source_filter_remove(source, filter);
            const auto b = summary(base), m = summary(mosaic_times);
            std::printf("dynamic %ux%u render+readback: unfiltered mean %.3f p95 %.3f max %.3f ms; "
                        "mosaic mean %.3f p95 %.3f max %.3f ms\n", W, H, b[0], b[1], b[2], m[0], m[1], m[2]);
            obs_source_release(filter);
            filter = nullptr;
            // libobs destroys released sources asynchronously.
            FILE* sf = nullptr;
            for (int i = 0; i < 300 && !sf; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                sf = std::fopen(stats_path.c_str(), "rb");
            }
            check(sf != nullptr, "plugin stats written on destroy");
            char buf[4096] = {0};
            std::fread(buf, 1, sizeof buf - 1, sf);
            std::fclose(sf);
            std::printf("plugin stats %s\n", buf);
            check(errors == 0, "no libobs errors");
            r_owner.reset(); // graphics objects must go before shutdown
            obs_source_release(source);
            obs_shutdown();
            return 0;
        }
        {
            Renderer r;
            std::vector<double> base_times;
            check(settle(r, source, 3, 50, &base_times) >= 0, "baseline");
            const auto baseline = r.pixels;
            double base_ms = 0;
            for (double t : base_times) base_ms += t / base_times.size();

            obs_source_filter_add(source, filter);
            // Analysis is asynchronous: wait for the first published tile pack.
            int waited = 0;
            for (; waited < 600; ++waited) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                check(r.render(source), "warmup draw");
                if (FILE* f = std::fopen(dump.c_str(), "rb")) { std::fclose(f); break; }
            }
            std::printf("first tile pack after %d frames\n", waited);
            check(waited < 600, "worker publishes a tile pack");
            std::vector<double> warm;
            const int frames = settle(r, source, 12, 600, &warm);
            check(frames >= 0, "mosaic settles on a static source");
            const auto mosaic_frame = r.pixels;
            save(out + "/smoke_mosaic.ppm", mosaic_frame);
            const Diff vs_source = diff(mosaic_frame, baseline, 24);
            std::printf("settled after %d frames; mean change from source %.2f codes\n", frames, vs_source.mean);
            check(vs_source.mean > 6, "mosaic visibly changes the image");

            // GPU vs CPU reference on the exact published pack.
            mosaic::GpuPack pack;
            check(load_pack(dump, pack), "published pack dump");
            std::vector<mosaic::Tile> tiles;
            mosaic::Grid grid;
            mosaic::unpack(pack, tiles, grid);
            mosaic::RenderParams p;
            p.scale = float(W) / pack.analysis_w;
            p.sweep = 0;
            p.twinkle = 0;
            const uint32_t grout = 0xff2e3438u;
            p.grout[0] = (grout & 0xff) / 255.f; p.grout[1] = ((grout >> 8) & 0xff) / 255.f; p.grout[2] = ((grout >> 16) & 0xff) / 255.f;
            p.facet = 0.24f; p.glint = 1.1f; p.shininess = 90.f; p.metal = 0.55f; p.bevel = 0.30f; p.diffuse = 0.35f;
            p.bevel_px = 1.8f; p.grout_px = 1.3f;
            std::vector<uint8_t> cpu(std::size_t(W) * H * 4);
            mosaic::render(tiles, grid, p, cpu.data(), int(W), int(H), std::size_t(W) * 4);
            save(out + "/smoke_cpu.ppm", cpu);
            const Diff gc = diff(mosaic_frame, cpu, 24);
            std::printf("tiles %u (max %u per cell); GPU/CPU mean %.4f codes, worst %d, pixels >24: %zu (%.4f%%)\n",
                        pack.count, pack.max_cell, gc.mean, gc.worst, gc.over, 100.0 * gc.over / (W * H));
            check(pack.count > 5000, "plausible tile count");
            check(gc.mean < 1.0 && gc.over < W * H / 200, "GPU shader agrees with CPU reference");

            // Static source: every frame identical (glint motion disabled).
            std::vector<double> steady;
            for (int i = 0; i < 40; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                check(r.render(source), "steady draw");
                steady.push_back(r.elapsed);
                if (r.pixels != mosaic_frame) {
                    const Diff d = diff(r.pixels, mosaic_frame, 0);
                    std::printf("static frame %d changed: %zu pixels, mean %.4f, worst %d\n", i, d.over, d.mean, d.worst);
                }
                check(r.pixels == mosaic_frame, "static scene keeps identical tiles and colors");
            }

            // Throughput: 120 submitted frames, one final synchronizing readback.
            auto throughput = [&]() {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 120; ++i) check(r.render(source, false), "throughput draw");
                check(r.render(source), "throughput sync");
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 121;
            };
            set(filter, "mz_sweep", 0.25);
            set(filter, "mz_twinkle", 0.35);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            throughput();
            const double mosaic_tp = throughput();
            obs_source_filter_remove(source, filter);
            throughput();
            const double base_tp = throughput();
            obs_source_filter_add(source, filter);
            set(filter, "mz_sweep", 0);
            set(filter, "mz_twinkle", 0);
            check(settle(r, source, 12, 600) >= 0, "resettle after throughput");
            check(r.pixels == mosaic_frame, "throughput phase left tiles unchanged");
            std::printf("Metal %ux%u throughput per frame: unfiltered %.3f ms, mosaic %.3f ms, added %.3f ms\n",
                        W, H, base_tp, mosaic_tp, mosaic_tp - base_tp);

            // Moving the disc re-tiles near it; the far right stays identical.
            const auto before = slurp(dump);
            set(source, "disc", 0.34);
            int republish = 0;
            for (; republish < 600; ++republish) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                check(r.render(source), "motion draw");
                if (slurp(dump) != before) break;
            }
            std::printf("motion republished after %d frames\n", republish);
            check(republish < 600, "motion publishes a new tile pack");
            check(settle(r, source, 12, 600) >= 0, "settles after motion");
            const auto moved = r.pixels;
            const Diff far = diff(moved, mosaic_frame, 0, uint32_t(0.62 * W), W);
            const Diff near = diff(moved, mosaic_frame, 24, uint32_t(0.10 * W), uint32_t(0.55 * W));
            std::printf("after motion: far-right changed pixels %zu (%.4f%%, mean %.4f, worst %d), near-disc changed >24: %zu\n",
                        far.over, 100.0 * far.over / (0.38 * W * H), far.mean, far.worst, near.over);
            check(near.over > W * H / 200, "motion re-tiles near the object");
            check(far.over < W * H / 500, "motion leaves distant tiles untouched");

            // Glint animation changes pixels without new analysis.
            set(filter, "mz_sweep", 1.0);
            set(filter, "mz_twinkle", 0.35);
            check(r.render(source), "glint frame a");
            const auto ga = r.pixels;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            check(r.render(source), "glint frame b");
            const Diff glint = diff(ga, r.pixels, 8);
            std::printf("glint animation: %zu pixels moved > 8 codes in 300 ms\n", glint.over);
            check(glint.over > 500, "glint animates");

            // Zero mix is an exact bypass.
            set(filter, "mz_mix", 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            check(r.render(source), "bypass draw");
            obs_source_filter_remove(source, filter);
            check(r.render(source), "unfiltered draw");
            const auto unfiltered = r.pixels;
            obs_source_filter_add(source, filter);
            check(r.render(source), "bypass draw 2");
            check(r.pixels == unfiltered, "zero mix bypass exact");
            set(filter, "mz_mix", 1);

            // Floating/HDR sources pass through untouched.
            for (auto space : {GS_CS_SRGB_16F, GS_CS_709_EXTENDED}) {
                set_int(source, "space", space);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                obs_source_filter_remove(source, filter);
                check(r.render(source), "HDR baseline");
                const auto plain = r.pixels;
                obs_source_filter_add(source, filter);
                check(r.render(source), "HDR filtered");
                check(r.pixels == plain, "floating/HDR passthrough exact");
            }
            set_int(source, "space", GS_CS_SRGB);

            auto stats = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                double mean = 0;
                for (double t : v) mean += t / v.size();
                return std::make_pair(mean, v[v.size() * 95 / 100]);
            };
            const auto st = stats(steady);
            std::printf("Metal %ux%u render+readback: baseline mean %.3f ms; mosaic steady mean %.3f ms, p95 %.3f ms\n",
                        W, H, base_ms, st.first, st.second);
            check(errors == 0, "no libobs errors");
            std::puts("Mosaic FX OBS registration, shader, CPU/GPU equivalence, static stability, motion locality, "
                      "glint, zero-mix bypass and HDR passthrough passed");
            obs_source_filter_remove(source, filter);
        }
        obs_source_release(filter);
        obs_source_release(source);
        obs_shutdown();
        return 0;
    }
}
