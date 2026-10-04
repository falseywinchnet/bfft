// Mosaic FX OBS filter.
//
// Render thread: capture the source; at the analysis rate, downsample it on
// the GPU (straight RGB, alpha 1) and read it back one frame late (deferred
// on Metal, as in RVFX); hand the newest frame to the worker without waiting;
// adopt newly published tile data by buffer swap; draw.
//
// Worker thread: block prefilter.  Woken blocks run the Meyer cartoon and
// region-limited re-tiling at most `retile_hz` times a second; otherwise only
// woken (or globally shifted) tiles are re-measured.  The Posterizer Mark IV
// palette re-learns while warming up, every fourth wake, and on a slow
// refresh.  Geometry and colors publish separately.
//
// GPU: geometry (2 RGBA16 texels/tile) and cell tables upload only when tile
// placement changes, colors (1 texel/tile) when colors change.  "Locate"
// resolves each pixel's tile, grout distance, bevel direction and facet tilt
// into an RGBA16 owner cache only after a geometry change; "Draw" shades
// every frame from three fetches (source, owner cache, tile color).
#include "mosaic/cartoon.hpp"
#include "mosaic/mosaic.hpp"
#include "rvfx/engine.hpp"

#include <obs-module.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

OBS_DECLARE_MODULE()

MODULE_EXPORT const char* obs_module_description(void)
{
    return "Mosaic FX: deterministic contour-row tessellation with persistent tiles and glint";
}

namespace {

const char* kShader = R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d tile_geo;
uniform texture2d tile_color;
uniform texture2d cell_data;
uniform texture2d locate_image;
uniform float2 geo_tex_size;
uniform float2 color_tex_size;
uniform float2 cell_tex_size;
uniform float2 analysis_size;
uniform float cell_size;
uniform float half_scale;
uniform float px_scale;
uniform float grout_half;
uniform float bevel_px;
uniform float bevel;
uniform float facet;
uniform float diffuse;
uniform float glint;
uniform float shininess;
uniform float metal;
uniform float twinkle;
uniform float time;
uniform float3 light_dir;
uniform float3 half_dir;
uniform float3 grout_color;
uniform float mix_amount;

sampler_state point_sampler { Filter=Point; AddressU=Clamp; AddressV=Clamp; };
sampler_state linear_sampler { Filter=Linear; AddressU=Clamp; AddressV=Clamp; };
struct VertData { float4 pos : POSITION; float2 uv : TEXCOORD0; };
VertData VSDefault(VertData v_in) {
    VertData v_out;
    v_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
    v_out.uv = v_in.uv;
    return v_out;
}

// Analysis frame: straight (un-premultiplied) RGB, alpha 1.
float4 PSDownsample(VertData v_in) : TARGET {
    float4 c = image.Sample(linear_sampler, v_in.uv);
    float3 rgb = c.a > 0.0 ? saturate(c.rgb / c.a) : float3(0.0, 0.0, 0.0);
    return float4(rgb, 1.0);
}

float4 fetch_geo(float index, float part) {
    float row = floor(index / 1024.0);
    float col = index - row * 1024.0;
    return tile_geo.Sample(point_sampler, float2(col * 2.0 + part + 0.5, row + 0.5) / geo_tex_size);
}

// Two signed unit components in [-1,1] as 8+8 bits of one 16-bit channel.
float pack2(float2 v) {
    float2 q = floor(saturate(v * 0.5 + 0.5) * 255.0 + 0.5);
    return (q.x * 256.0 + q.y) / 65535.0;
}
float2 unpack2(float x) {
    float u = floor(x * 65535.0 + 0.5);
    float hi = floor(u / 256.0);
    return float2(hi, u - hi * 256.0) / 127.5 - 1.0;
}

// After a geometry change only: owning tile, distance to its grout boundary
// in output pixels, bevel direction (signed tile axis) and facet tilt.
// RGBA16: (index / 65535, edge / 64, pack2(direction), pack2(tilt)).
float4 PSLocate(VertData v_in) : TARGET {
    float2 q = v_in.uv * analysis_size;
    float2 cell = floor(q / cell_size);
    float best = 1000000.0;
    float second = 1000000.0;
    float best_index = -1.0;
    float second_index = -1.0;
    float2 best_uv = float2(0.0, 0.0);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            float2 c = cell + float2(float(i), float(j));
            if (c.x >= 0.0 && c.y >= 0.0 && c.x < cell_tex_size.x && c.y < cell_tex_size.y) {
                float4 cd = cell_data.Sample(point_sampler, (c + 0.5) / cell_tex_size);
                float start = floor(cd.x * 65535.0 + 0.5) + floor(cd.y * 65535.0 + 0.5) * 65536.0;
                float count = floor(cd.z * 65535.0 + 0.5);
                for (int k = 0; k < 32; k++) {
                    if (float(k) >= count)
                        break;
                    float index = start + float(k);
                    float4 a = fetch_geo(index, 0.0);
                    float4 b = fetch_geo(index, 1.0);
                    float2 cs = a.zw * 2.0 - 1.0;
                    float2 d = q - a.xy * analysis_size;
                    float u = d.x * cs.x + d.y * cs.y;
                    float v = d.y * cs.x - d.x * cs.y;
                    float2 hh = b.xy * half_scale;
                    float rho = max(abs(u) / hh.x, abs(v) / hh.y);
                    if (rho < best) {
                        second = best;
                        second_index = best_index;
                        best = rho;
                        best_index = index;
                        best_uv = float2(u, v);
                    } else if (rho < second) {
                        second = rho;
                        second_index = index;
                    }
                }
            }
        }
    }
    if (best_index < 0.0 || best >= 1.0)
        return float4(1.0, 0.0, 0.5, 0.5);
    float4 a = fetch_geo(best_index, 0.0);
    float4 b = fetch_geo(best_index, 1.0);
    float2 cs = a.zw * 2.0 - 1.0;
    float2 hh = b.xy * half_scale;
    float mt = hh.x - abs(best_uv.x);
    float mn = hh.y - abs(best_uv.y);
    float edge = min(mt, mn) * px_scale;
    if (second_index >= 0.0) {
        float2 oh = fetch_geo(second_index, 1.0).xy * half_scale;
        float rate = 1.0 / min(hh.x, hh.y) + 1.0 / min(oh.x, oh.y);
        edge = min(edge, (second - best) / rate * px_scale);
    }
    float2 dir = mt < mn ? cs * (best_uv.x < 0.0 ? -1.0 : 1.0)
                         : float2(-cs.y, cs.x) * (best_uv.y < 0.0 ? -1.0 : 1.0);
    return float4(best_index / 65535.0, min(edge / 64.0, 1.0), pack2(dir), pack2(b.zw * 2.0 - 1.0));
}

// Every frame: three fetches (source, owner cache, tile color).
float4 PSShade(VertData v_in) : TARGET {
    float4 src = image.Sample(point_sampler, v_in.uv);
    float4 loc = locate_image.Sample(point_sampler, v_in.uv);
    float index = floor(loc.r * 65535.0 + 0.5);
    float3 color = grout_color;
    if (index < 65535.0) {
        float row = floor(index / 1024.0);
        float4 t = tile_color.Sample(point_sampler, float2(index - row * 1024.0 + 0.5, row + 0.5) / color_tex_size);
        float edge = loc.g * 64.0;
        float cover = saturate(edge - grout_half + 0.5);
        float bev = bevel * (1.0 - saturate((edge - grout_half) / bevel_px));
        float2 n2 = unpack2(loc.a) * facet + unpack2(loc.b) * bev;
        float3 n = normalize(float3(n2.x, n2.y, 1.0));
        float shade = 1.0 + diffuse * (dot(n, light_dir) - light_dir.z);
        float spec = glint * pow(max(dot(n, half_dir), 0.0), shininess);
        spec = spec * (1.0 - twinkle * (0.5 + 0.5 * sin(6.2831853 * (t.w + 0.37 * time))));
        float3 tint = float3(1.0 - metal, 1.0 - metal, 1.0 - metal) + t.rgb * metal;
        float3 tile = t.rgb * shade + tint * spec;
        color = grout_color + (tile - grout_color) * cover;
    }
    color = saturate(color);
    float3 source_rgb = src.a > 0.0 ? saturate(src.rgb / src.a) : float3(0.0, 0.0, 0.0);
    float3 result = source_rgb + (color - source_rgb) * mix_amount;
    return float4(result * src.a, src.a);
}

technique Downsample { pass { vertex_shader = VSDefault(v_in); pixel_shader = PSDownsample(v_in); } }
technique Locate { pass { vertex_shader = VSDefault(v_in); pixel_shader = PSLocate(v_in); } }
technique Draw { pass { vertex_shader = VSDefault(v_in); pixel_shader = PSShade(v_in); } }
)";

constexpr int kShaderCellLimit = 32; // matches the shader's per-cell loop bound

using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Settings {
    mosaic::Config config;
    mosaic::RenderParams look;
    int analysis_width = 960;
    double hz = 15, retile_hz = 6;
    int lattice = 512;
    int palette_colors = 32;
    int threads = 4;
    float mix = 1;
    bool freeze = false;
};

bool same_geometry(const mosaic::Config& a, const mosaic::Config& b)
{
    return a.target_tiles == b.target_tiles && a.size_classes == b.size_classes && a.size_ratio == b.size_ratio &&
           a.entropy_low == b.entropy_low && a.entropy_high == b.entropy_high &&
           a.edge_threshold == b.edge_threshold && a.frame_edges == b.frame_edges &&
           a.contour_rows == b.contour_rows && a.outline_contrast == b.outline_contrast &&
           (a.outline > 0) == (b.outline > 0) && a.spacing == b.spacing && a.elongation == b.elongation;
}

// Worker -> render thread hand-off.  The worker copies into it; the render
// thread swaps the buffers out (O(1) under the lock).
struct Published {
    std::uint64_t version = 0, geometry_version = 0;
    int rows = 1, cells_w = 1, cells_h = 1;
    float analysis_w = 1, analysis_h = 1, cell = 1, half_scale = 1;
    std::vector<std::uint16_t> geometry, cells, colors;
};

// Counts reported to the OBS log every 30 s and, with MOSAIC_FX_STATS set,
// written as JSON when the filter is destroyed.
struct Profile {
    std::mutex mutex;
    std::uint64_t samples = 0, wakes = 0, retiles = 0, deferred = 0, publishes = 0, geometry_publishes = 0;
    double meyer_ms = 0, worker_ms = 0, worker_max = 0;
    std::uint64_t frames = 0, locates = 0;
    double render_ms = 0, render_max = 0;
    std::string json()
    {
        std::lock_guard<std::mutex> lock(mutex);
        char buf[700];
        auto per = [](double v, std::uint64_t n) { return n ? v / double(n) : 0.0; };
        std::snprintf(buf, sizeof buf,
                      "{\"worker\":{\"samples\":%llu,\"wakes\":%llu,\"retiles\":%llu,\"deferred\":%llu,"
                      "\"publishes\":%llu,\"geometry_publishes\":%llu,\"meyer_ms_per_retile\":%.3f,"
                      "\"ms_per_sample\":%.3f,\"max_ms\":%.3f},\"render\":{\"frames\":%llu,\"locates\":%llu,"
                      "\"cpu_ms_per_frame\":%.3f,\"max_ms\":%.3f}}",
                      (unsigned long long)samples, (unsigned long long)wakes, (unsigned long long)retiles,
                      (unsigned long long)deferred, (unsigned long long)publishes,
                      (unsigned long long)geometry_publishes, per(meyer_ms, retiles), per(worker_ms, samples),
                      worker_max, (unsigned long long)frames, (unsigned long long)locates, per(render_ms, frames),
                      render_max);
        return buf;
    }
    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex);
        samples = wakes = retiles = deferred = publishes = geometry_publishes = frames = locates = 0;
        meyer_ms = worker_ms = worker_max = render_ms = render_max = 0;
    }
};

class Worker {
public:
    explicit Worker(Profile& profile) : profile_(profile) { thread_ = std::thread([this] { run(); }); }
    ~Worker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    // Latest job wins; the render thread never waits for analysis.
    void submit(const std::uint8_t* data, int w, int h, std::size_t stride, const Settings& s)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.resize(std::size_t(w) * h * 4);
        if (stride == std::size_t(w) * 4) std::memcpy(pending_.data(), data, pending_.size());
        else
            for (int y = 0; y < h; ++y)
                std::memcpy(&pending_[std::size_t(y) * w * 4], data + std::size_t(y) * stride, std::size_t(w) * 4);
        pending_w_ = w;
        pending_h_ = h;
        pending_settings_ = s;
        has_job_ = true;
        cv_.notify_one();
    }
    bool idle() const { return !busy_.load() && !has_job_.load(); }
    // Swaps newer published data into `out` (whose old buffers go back).
    bool take(Published& out)
    {
        std::lock_guard<std::mutex> lock(publish_mutex_);
        if (published_.version == out.version) return false;
        const bool geometry = published_.geometry_version != out.geometry_version;
        out.version = published_.version;
        out.geometry_version = published_.geometry_version;
        out.rows = published_.rows;
        out.colors.swap(published_.colors);
        if (geometry) {
            out.cells_w = published_.cells_w;
            out.cells_h = published_.cells_h;
            out.analysis_w = published_.analysis_w;
            out.analysis_h = published_.analysis_h;
            out.cell = published_.cell;
            out.half_scale = published_.half_scale;
            out.geometry.swap(published_.geometry);
            out.cells.swap(published_.cells);
        }
        return true;
    }

private:
    void run()
    {
        std::vector<std::uint8_t> frame;
        for (;;) {
            Settings s;
            int w = 0, h = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || has_job_; });
                if (stop_) return;
                frame.swap(pending_);
                w = pending_w_;
                h = pending_h_;
                s = pending_settings_;
                busy_ = true;
                has_job_ = false;
            }
            analyze(frame, w, h, s);
            busy_ = false;
        }
    }

    void learn_palette(const std::vector<std::uint8_t>& rgba, int w, int h, const Settings& s, bool wake,
                       bool& changed)
    {
        if (s.palette_colors < 2) {
            if (!palette_.empty() || palette_colors_ != 0) {
                palette_.clear();
                palette_colors_ = 0;
                changed = true;
            }
            return;
        }
        if (palette_colors_ != s.palette_colors) {
            rvfx::Config pc;
            pc.posterize_only = true;
            pc.glyph_layer = false;
            pc.palette_colors = std::uint32_t(s.palette_colors);
            pc.trace_width = 320;
            pc.family_priority = 1.0f;
            engine_.set_config(pc);
            engine_.reset();
            palette_colors_ = s.palette_colors;
            palette_frames_ = 0;
            wakes_ = 0;
        }
        // Warm up, then every fourth wake, plus a slow refresh.
        if (wake) ++wakes_;
        const bool due = palette_frames_ < 4 || (wake && wakes_ % 4 == 1) || palette_frames_ % 60 == 0;
        ++palette_frames_;
        if (!due) return;
        rvfx::FrameView view;
        view.data = rgba.data();
        view.width = std::uint32_t(w);
        view.height = std::uint32_t(h);
        view.stride = std::ptrdiff_t(w) * 4;
        view.format = rvfx::PixelFormat::RGBA;
        engine_.process(view);
        const auto& nodes = engine_.palette();
        // Mark IV keeps node identities; adopt only clearly moved nodes so
        // centroid creep never nudges every tile's color.
        if (nodes.size() != palette_.size()) {
            palette_.resize(nodes.size());
            for (std::size_t i = 0; i < nodes.size(); ++i) palette_[i] = {nodes[i].l, nodes[i].a, nodes[i].b};
            changed = true;
            return;
        }
        const float tolerance = palette_frames_ <= 4 ? 0.006f : 0.02f;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const float dl = nodes[i].l - palette_[i].l, da = nodes[i].a - palette_[i].a, db = nodes[i].b - palette_[i].b;
            if (dl * dl + da * da + db * db > tolerance * tolerance) {
                palette_[i] = {nodes[i].l, nodes[i].a, nodes[i].b};
                changed = true;
            }
        }
    }

    void analyze(std::vector<std::uint8_t>& rgba, int w, int h, const Settings& s)
    {
        const auto t0 = Clock::now();
        if (!configured_ || !same_geometry(config_, s.config)) {
            mosaic_.reset();
            packed_ = {};
        }
        const bool color_settings = configured_ && (config_.outline != s.config.outline ||
                                                     config_.lightness_jitter != s.config.lightness_jitter ||
                                                     config_.chroma_jitter != s.config.chroma_jitter);
        config_ = s.config;
        configured_ = true;
        mosaic_.set_config(s.config);

        const std::size_t stride = std::size_t(w) * 4;
        const bool wake = mosaic_.wants_structure(rgba.data(), w, h, stride);
        bool palette_changed = false;
        learn_palette(rgba, w, h, s, wake, palette_changed);

        const bool due = mosaic_.tiles().empty() ||
                         std::chrono::duration<double>(t0 - last_retile_).count() >= 1.0 / std::max(0.5, s.retile_hz);
        const mosaic::Stats* st = nullptr;
        double meyer = 0;
        if (wake && due) {
            const auto tm = Clock::now();
            if (!cartoon_.run(rgba.data(), w, h, stride, s.lattice, s.threads, structure_)) {
                blog(LOG_WARNING, "[Mosaic FX] Meyer cartoon failed for %dx%d", w, h);
                return;
            }
            meyer = ms_between(tm, Clock::now());
            st = &mosaic_.process_swap(rgba.data(), w, h, stride, structure_, palette_);
            last_retile_ = t0;
        } else {
            // Geometry deferred by the re-tiling rate keeps its woken blocks pending.
            st = &mosaic_.update_colors(rgba.data(), w, h, stride, palette_, wake);
        }
        if (st->max_cell_tiles > kShaderCellLimit && !warned_cells_) {
            blog(LOG_WARNING, "[Mosaic FX] %u tiles in one grid cell exceed the shader bound %d; reduce the size range",
                 st->max_cell_tiles, kShaderCellLimit);
            warned_cells_ = true;
        }
        const bool geometry = st->geometry || packed_.order.size() != mosaic_.tiles().size();
        const bool colors = geometry || st->recolored || palette_changed || color_settings;
        if (colors) {
            mosaic::pack(mosaic_, packed_, geometry);
            dump(packed_);
            std::lock_guard<std::mutex> lock(publish_mutex_);
            published_.rows = packed_.rows;
            published_.colors.assign(packed_.colors.begin(), packed_.colors.end());
            if (geometry) {
                published_.cells_w = packed_.cells_w;
                published_.cells_h = packed_.cells_h;
                published_.analysis_w = packed_.analysis_w;
                published_.analysis_h = packed_.analysis_h;
                published_.cell = packed_.cell;
                published_.half_scale = packed_.half_scale;
                published_.geometry.assign(packed_.geometry.begin(), packed_.geometry.end());
                published_.cells.assign(packed_.cells.begin(), packed_.cells.end());
                ++published_.geometry_version;
            }
            ++published_.version;
        }
        const double total = ms_between(t0, Clock::now());
        std::lock_guard<std::mutex> lock(profile_.mutex);
        ++profile_.samples;
        profile_.wakes += wake;
        profile_.retiles += meyer > 0;
        profile_.deferred += wake && !(meyer > 0);
        profile_.publishes += colors;
        profile_.geometry_publishes += geometry;
        profile_.meyer_ms += meyer;
        profile_.worker_ms += total;
        profile_.worker_max = std::max(profile_.worker_max, total);
    }

    // Test hook: MOSAIC_FX_DUMP=path writes each published pack.
    void dump(const mosaic::GpuPack& p) const
    {
        const char* path = std::getenv("MOSAIC_FX_DUMP");
        if (!path || !*path) return;
        const std::string tmp = std::string(path) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const std::int32_t ints[5] = {p.rows, p.cells_w, p.cells_h, std::int32_t(p.count), std::int32_t(p.max_cell)};
        const float floats[4] = {p.analysis_w, p.analysis_h, p.cell, p.half_scale};
        std::fwrite(ints, sizeof ints, 1, f);
        std::fwrite(floats, sizeof floats, 1, f);
        std::fwrite(p.geometry.data(), 2, p.geometry.size(), f);
        std::fwrite(p.colors.data(), 2, p.colors.size(), f);
        std::fwrite(p.cells.data(), 2, p.cells.size(), f);
        std::fclose(f);
        std::rename(tmp.c_str(), path);
    }

    Profile& profile_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::atomic<bool> has_job_{false}, busy_{false};
    std::vector<std::uint8_t> pending_;
    int pending_w_ = 0, pending_h_ = 0;
    Settings pending_settings_;

    std::mutex publish_mutex_;
    Published published_;

    // Worker-owned state.
    mosaic::Mosaic mosaic_;
    mosaic::Cartoon cartoon_;
    rvfx::Engine engine_;
    mosaic::Config config_;
    mosaic::GpuPack packed_;
    bool configured_ = false, warned_cells_ = false;
    int palette_colors_ = -1;
    std::uint64_t palette_frames_ = 0, wakes_ = 0;
    std::vector<mosaic::Lab> palette_;
    std::vector<float> structure_;
    Clock::time_point last_retile_{};
};

struct Params {
    gs_eparam_t *image, *tile_geo, *tile_color, *cell_data, *locate_image, *geo_tex_size, *color_tex_size,
        *cell_tex_size, *analysis_size, *cell_size, *half_scale, *px_scale, *grout_half, *bevel_px, *bevel, *facet,
        *diffuse, *glint, *shininess, *metal, *twinkle, *time, *light_dir, *half_dir, *grout_color, *mix_amount;
};

Params params_of(gs_effect_t* e)
{
    auto P = [&](const char* name) { return gs_effect_get_param_by_name(e, name); };
    return Params{P("image"), P("tile_geo"), P("tile_color"), P("cell_data"), P("locate_image"), P("geo_tex_size"),
                  P("color_tex_size"), P("cell_tex_size"), P("analysis_size"), P("cell_size"), P("half_scale"),
                  P("px_scale"), P("grout_half"), P("bevel_px"), P("bevel"), P("facet"), P("diffuse"), P("glint"),
                  P("shininess"), P("metal"), P("twinkle"), P("time"), P("light_dir"), P("half_dir"),
                  P("grout_color"), P("mix_amount")};
}

struct Filter {
    obs_source_t* source = nullptr;
    std::mutex mutex;
    Settings pending;
    Profile profile;
    Worker worker{profile};
    // The tile search uses its own effect object: in OBS 32.2.1 Metal,
    // alternating techniques of one effect that bind different texture sets
    // crashed the following draw (stale texture parameter).
    gs_effect_t* effect = nullptr;
    gs_effect_t* locate_effect = nullptr;
    Params p{}, lp{};
    gs_texrender_t* capture = nullptr;
    gs_texrender_t* analysis = nullptr;
    gs_texrender_t* locate = nullptr; // per-pixel owner cache, RGBA16
    std::array<gs_stagesurf_t*, 2> stages{};
    std::array<bool, 2> written{};
    int slot = 0, aw = 0, ah = 0;
    std::uint32_t width = 0, height = 0;
    bool deferred_readback = false, pending_readback = false;
    gs_texture_t* geo_tex = nullptr;
    gs_texture_t* color_tex = nullptr;
    gs_texture_t* cell_tex = nullptr;
    int tex_rows = 0;
    Published gpu;                       // adopted data (buffers swapped in)
    std::uint64_t uploaded_geometry = 0, located_geometry = ~std::uint64_t(0);
    std::uint32_t located_w = 0, located_h = 0;
    std::uint64_t last_frame = 0;
    bool has_frame = false;
    Clock::time_point start = Clock::now(), last_sample{}, last_report = Clock::now();
};

void release_analysis(Filter* f)
{
    for (auto& s : f->stages) {
        if (s) gs_stagesurface_destroy(s);
        s = nullptr;
    }
    f->written = {false, false};
    f->pending_readback = false;
    f->slot = 0;
}

void destroy(void* data)
{
    auto* f = static_cast<Filter*>(data);
    if (!f) return;
    if (const char* path = std::getenv("MOSAIC_FX_STATS")) {
        if (FILE* out = std::fopen(path, "wb")) {
            const std::string j = f->profile.json();
            std::fwrite(j.data(), 1, j.size(), out);
            std::fclose(out);
        }
    }
    obs_enter_graphics();
    release_analysis(f);
    if (f->geo_tex) gs_texture_destroy(f->geo_tex);
    if (f->color_tex) gs_texture_destroy(f->color_tex);
    if (f->cell_tex) gs_texture_destroy(f->cell_tex);
    if (f->capture) gs_texrender_destroy(f->capture);
    if (f->analysis) gs_texrender_destroy(f->analysis);
    if (f->locate) gs_texrender_destroy(f->locate);
    if (f->effect) gs_effect_destroy(f->effect);
    if (f->locate_effect) gs_effect_destroy(f->locate_effect);
    obs_leave_graphics();
    delete f;
}

float clampf(obs_data_t* s, const char* key, float lo, float hi)
{
    return std::clamp(float(obs_data_get_double(s, key)), lo, hi);
}

void update(void* data, obs_data_t* s)
{
    auto* f = static_cast<Filter*>(data);
    Settings a;
    a.config.target_tiles = std::uint32_t(std::clamp<long long>(obs_data_get_int(s, "mz_tiles"), 1000, 60000));
    a.config.size_classes = int(std::clamp<long long>(obs_data_get_int(s, "mz_classes"), 1, 4));
    // Keep the largest/smallest span within the shader's per-cell bound.
    const float span = std::pow(clampf(s, "mz_ratio", 1.05f, 2.0f), float(a.config.size_classes - 1));
    a.config.size_ratio = span > 3.f ? std::pow(3.f, 1.f / float(a.config.size_classes - 1))
                                     : clampf(s, "mz_ratio", 1.05f, 2.0f);
    a.config.edge_threshold = clampf(s, "mz_edge", 0.005f, 0.2f);
    a.config.outline = clampf(s, "mz_outline", 0, 1);
    a.config.contour_rows = int(std::clamp<long long>(obs_data_get_int(s, "mz_rows"), 0, 64));
    a.config.frame_edges = obs_data_get_bool(s, "mz_frame_rows");
    a.config.lightness_jitter = clampf(s, "mz_jitter", 0, 0.15f);
    a.config.recolor_threshold = clampf(s, "mz_recolor", 0.002f, 0.3f);
    a.config.restructure_threshold = clampf(s, "mz_restructure", 0.005f, 0.3f);
    a.config.prefilter_luma = clampf(s, "mz_prefilter", 0.5f, 40.f);
    a.look.grout_px = clampf(s, "mz_grout", 0, 6);
    const auto grout = std::uint32_t(obs_data_get_int(s, "mz_grout_color"));
    a.look.grout[0] = float(grout & 0xff) / 255.f;
    a.look.grout[1] = float((grout >> 8) & 0xff) / 255.f;
    a.look.grout[2] = float((grout >> 16) & 0xff) / 255.f;
    a.look.bevel = clampf(s, "mz_bevel", 0, 1.5f);
    a.look.glint = clampf(s, "mz_glint", 0, 4);
    a.look.facet = clampf(s, "mz_facet", 0, 1);
    a.look.shininess = clampf(s, "mz_shininess", 4, 400);
    a.look.metal = clampf(s, "mz_metal", 0, 1);
    a.look.sweep = clampf(s, "mz_sweep", -3, 3);
    a.look.twinkle = clampf(s, "mz_twinkle", 0, 1);
    a.look.light_azimuth = clampf(s, "mz_azimuth", -360, 360) * 3.14159265f / 180.f;
    a.look.diffuse = clampf(s, "mz_diffuse", 0, 1);
    a.analysis_width = int(std::clamp<long long>(obs_data_get_int(s, "mz_analysis"), 320, 1920));
    a.hz = std::clamp(obs_data_get_double(s, "mz_hz"), 1.0, 60.0);
    a.retile_hz = std::clamp(obs_data_get_double(s, "mz_retile_hz"), 0.5, 60.0);
    a.lattice = int(obs_data_get_int(s, "mz_lattice"));
    a.palette_colors = int(std::clamp<long long>(obs_data_get_int(s, "mz_colors"), 0, 64));
    a.threads = int(std::clamp<long long>(obs_data_get_int(s, "mz_threads"), 1, 16));
    a.mix = clampf(s, "mz_mix", 0, 1);
    a.freeze = obs_data_get_bool(s, "mz_freeze");
    std::lock_guard<std::mutex> lock(f->mutex);
    f->pending = a;
}

void* create(obs_data_t* s, obs_source_t* source)
{
    auto* f = new Filter;
    f->source = source;
    update(f, s);
    obs_enter_graphics();
    for (gs_effect_t** e : {&f->effect, &f->locate_effect}) {
        char* errors = nullptr;
        *e = gs_effect_create(kShader, e == &f->effect ? "mosaic-fx.effect" : "mosaic-fx-locate.effect", &errors);
        if (errors) {
            blog(LOG_WARNING, "[Mosaic FX] %s", errors);
            bfree(errors);
        }
    }
    f->deferred_readback = gs_get_device_type() == GS_DEVICE_METAL;
    if (f->effect) f->p = params_of(f->effect);
    if (f->locate_effect) f->lp = params_of(f->locate_effect);
    f->capture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    f->analysis = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    f->locate = gs_texrender_create(GS_RGBA16, GS_ZS_NONE);
    obs_leave_graphics();
    if (!f->effect || !f->locate_effect || !f->capture || !f->analysis || !f->locate) {
        destroy(f);
        return nullptr;
    }
    return f;
}

bool resources(Filter* f, std::uint32_t w, std::uint32_t h, int requested)
{
    const int aw = std::min(int(w), requested);
    const int ah = std::max(8, int(double(aw) * h / w + 0.5));
    if (f->width == w && f->height == h && f->aw == aw && f->ah == ah && f->stages[0] && f->stages[1]) return true;
    release_analysis(f);
    f->width = w;
    f->height = h;
    f->aw = aw;
    f->ah = ah;
    f->stages[0] = gs_stagesurface_create(aw, ah, GS_RGBA);
    f->stages[1] = gs_stagesurface_create(aw, ah, GS_RGBA);
    return f->stages[0] && f->stages[1];
}

void set_vec2(gs_eparam_t* p, float x, float y)
{
    vec2 v;
    vec2_set(&v, x, y);
    gs_effect_set_vec2(p, &v);
}

void set_vec3(gs_eparam_t* p, const float* a)
{
    vec3 v;
    vec3_set(&v, a[0], a[1], a[2]);
    gs_effect_set_vec3(p, &v);
}

bool capture(Filter* f, obs_source_t* target)
{
    gs_texrender_reset(f->capture);
    bool ok = false;
    gs_viewport_push();
    gs_projection_push();
    gs_matrix_push();
    if (gs_texrender_begin(f->capture, f->width, f->height)) {
        vec4 clear;
        vec4_zero(&clear);
        gs_clear(GS_CLEAR_COLOR, &clear, 0, 0);
        gs_matrix_identity();
        gs_ortho(0, float(f->width), 0, float(f->height), -100, 100);
        gs_blend_state_push();
        gs_blend_function_separate(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA, GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
        auto* parent = obs_filter_get_parent(f->source);
        const auto flags = parent ? obs_source_get_output_flags(parent) : 0;
        if (target == parent && !(flags & (OBS_SOURCE_CUSTOM_DRAW | OBS_SOURCE_ASYNC)))
            obs_source_default_render(target);
        else
            obs_source_video_render(target);
        gs_blend_state_pop();
        gs_texrender_end(f->capture);
        ok = true;
    }
    gs_matrix_pop();
    gs_projection_pop();
    gs_viewport_pop();
    return ok;
}

void stage(Filter* f)
{
    gs_stage_texture(f->stages[f->slot], gs_texrender_get_texture(f->analysis));
    f->written[f->slot] = true;
    f->slot = 1 - f->slot;
}

void sample(Filter* f)
{
    gs_texrender_reset(f->analysis);
    gs_viewport_push();
    gs_projection_push();
    gs_matrix_push();
    gs_blend_state_push();
    gs_enable_blending(false);
    if (gs_texrender_begin(f->analysis, f->aw, f->ah)) {
        gs_matrix_identity();
        gs_ortho(0, float(f->aw), 0, float(f->ah), -100, 100);
        auto* texture = gs_texrender_get_texture(f->capture);
        gs_effect_set_texture(f->p.image, texture);
        while (gs_effect_loop(f->effect, "Downsample")) gs_draw_sprite(texture, 0, f->aw, f->ah);
        gs_texrender_end(f->analysis);
        if (f->deferred_readback) f->pending_readback = true;
        else stage(f);
    }
    gs_blend_state_pop();
    gs_matrix_pop();
    gs_projection_pop();
    gs_viewport_pop();
}

gs_texture_t* fit_texture(gs_texture_t* tex, std::uint32_t w, std::uint32_t h)
{
    if (tex && gs_texture_get_width(tex) == w && gs_texture_get_height(tex) == h) return tex;
    if (tex) gs_texture_destroy(tex);
    return gs_texture_create(w, h, GS_RGBA16, 1, nullptr, GS_DYNAMIC);
}

// Adopts the newest published data: colors always, geometry and cells only
// when placement changed.
void upload(Filter* f)
{
    if (!f->worker.take(f->gpu)) return;
    constexpr std::uint32_t T = mosaic::GpuPack::kTilesPerRow;
    const auto& g = f->gpu;
    // Texture rows grow in steps of 8 so small count changes reuse textures.
    const int rows = std::max(8, ((g.rows + 7) / 8) * 8);
    if (rows != f->tex_rows) {
        f->tex_rows = rows;
        f->uploaded_geometry = 0; // geometry must be re-sent into the new texture
    }
    f->color_tex = fit_texture(f->color_tex, T, std::uint32_t(rows));
    f->geo_tex = fit_texture(f->geo_tex, 2 * T, std::uint32_t(rows));
    if (!f->color_tex || !f->geo_tex) return;
    auto& colors = f->gpu.colors;
    colors.resize(std::size_t(T) * rows * 4, 0);
    gs_texture_set_image(f->color_tex, reinterpret_cast<const std::uint8_t*>(colors.data()), T * 8, false);
    if (g.geometry_version != f->uploaded_geometry) {
        auto& geometry = f->gpu.geometry;
        geometry.resize(std::size_t(2 * T) * rows * 4, 0);
        gs_texture_set_image(f->geo_tex, reinterpret_cast<const std::uint8_t*>(geometry.data()), 2 * T * 8, false);
        f->cell_tex = fit_texture(f->cell_tex, std::uint32_t(g.cells_w), std::uint32_t(g.cells_h));
        if (!f->cell_tex) return;
        gs_texture_set_image(f->cell_tex, reinterpret_cast<const std::uint8_t*>(f->gpu.cells.data()),
                             std::uint32_t(g.cells_w) * 8, false);
        f->uploaded_geometry = g.geometry_version;
    }
}

void draw_passthrough(Filter* f, std::uint32_t w, std::uint32_t h)
{
    gs_effect_t* base = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    auto* texture = gs_texrender_get_texture(f->capture);
    gs_effect_set_texture(gs_effect_get_param_by_name(base, "image"), texture);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
    while (gs_effect_loop(base, "Draw")) gs_draw_sprite(texture, 0, w, h);
    gs_blend_state_pop();
}

void locate(Filter* f, std::uint32_t w, std::uint32_t h)
{
    const auto& g = f->gpu;
    const Params& lp = f->lp;
    gs_texrender_reset(f->locate);
    gs_viewport_push();
    gs_projection_push();
    gs_matrix_push();
    gs_blend_state_push();
    gs_enable_blending(false);
    gs_effect_set_texture(lp.tile_geo, f->geo_tex);
    gs_effect_set_texture(lp.cell_data, f->cell_tex);
    set_vec2(lp.geo_tex_size, float(2 * mosaic::GpuPack::kTilesPerRow), float(f->tex_rows));
    set_vec2(lp.cell_tex_size, float(g.cells_w), float(g.cells_h));
    set_vec2(lp.analysis_size, g.analysis_w, g.analysis_h);
    gs_effect_set_float(lp.cell_size, g.cell);
    gs_effect_set_float(lp.half_scale, g.half_scale);
    gs_effect_set_float(lp.px_scale, float(w) / g.analysis_w);
    if (gs_texrender_begin(f->locate, w, h)) {
        gs_matrix_identity();
        gs_ortho(0, float(w), 0, float(h), -100, 100);
        while (gs_effect_loop(f->locate_effect, "Locate")) gs_draw_sprite(gs_texrender_get_texture(f->capture), 0, w, h);
        gs_texrender_end(f->locate);
        f->located_geometry = f->uploaded_geometry;
        f->located_w = w;
        f->located_h = h;
        std::lock_guard<std::mutex> lock(f->profile.mutex);
        ++f->profile.locates;
    }
    gs_blend_state_pop();
    gs_matrix_pop();
    gs_projection_pop();
    gs_viewport_pop();
}

void finish_frame(Filter* f, Clock::time_point t0)
{
    const auto now = Clock::now();
    const double ms = ms_between(t0, now);
    {
        std::lock_guard<std::mutex> lock(f->profile.mutex);
        ++f->profile.frames;
        f->profile.render_ms += ms;
        f->profile.render_max = std::max(f->profile.render_max, ms);
    }
    if (std::chrono::duration<double>(now - f->last_report).count() >= 30) {
        f->last_report = now;
        blog(LOG_INFO, "[Mosaic FX] 30 s profile %s", f->profile.json().c_str());
        f->profile.reset();
    }
}

void render(void* data, gs_effect_t*)
{
    const auto t0 = Clock::now();
    auto* f = static_cast<Filter*>(data);
    auto* target = obs_filter_get_target(f->source);
    if (!target) {
        obs_source_skip_video_filter(f->source);
        return;
    }
    Settings s;
    {
        std::lock_guard<std::mutex> lock(f->mutex);
        s = f->pending;
    }
    // Tile colors are display-referred sRGB: bypass floating/HDR spaces.
    const gs_color_space preferred[] = {GS_CS_SRGB, GS_CS_SRGB_16F, GS_CS_709_EXTENDED};
    const auto space = obs_source_get_color_space(target, 3, preferred);
    if (space != GS_CS_SRGB || gs_get_color_space() != GS_CS_SRGB || s.mix == 0) {
        obs_source_skip_video_filter(f->source);
        return;
    }
    const auto w = obs_source_get_base_width(target), h = obs_source_get_base_height(target);
    if (!w || !h || !resources(f, w, h, s.analysis_width)) {
        obs_source_skip_video_filter(f->source);
        return;
    }
    const bool srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(false);
    const bool linear = gs_set_linear_srgb(false);
    const std::uint64_t frame = obs_get_video_frame_time();
    if (!f->has_frame || frame != f->last_frame) {
        f->has_frame = true;
        f->last_frame = frame;
        // Metal's gs_stage_texture waits for completion at submission: copy
        // the previous frame's analysis texture before this frame's capture.
        if (f->pending_readback) {
            if (!s.freeze) stage(f);
            f->pending_readback = false;
        }
        if (!capture(f, target)) {
            f->has_frame = false;
            gs_set_linear_srgb(linear);
            gs_enable_framebuffer_srgb(srgb);
            obs_source_skip_video_filter(f->source);
            return;
        }
        if (!s.freeze) {
            // Read only a preceding frame's submission, newest first.
            for (int k = 0; k < 2; ++k) {
                const int i = (f->slot + 1 + k) % 2;
                if (!f->written[i]) continue;
                f->written[i] = false;
                std::uint8_t* mapped = nullptr;
                std::uint32_t stride = 0;
                if (f->worker.idle() && gs_stagesurface_map(f->stages[i], &mapped, &stride)) {
                    f->worker.submit(mapped, f->aw, f->ah, stride, s);
                    gs_stagesurface_unmap(f->stages[i]);
                }
            }
            if (f->worker.idle() && std::chrono::duration<double>(t0 - f->last_sample).count() >= 1.0 / s.hz) {
                sample(f);
                f->last_sample = t0;
            }
        } else {
            f->written = {false, false};
        }
        upload(f);
    }
    if (!f->uploaded_geometry || !f->geo_tex || !f->color_tex || !f->cell_tex) {
        draw_passthrough(f, w, h);
        gs_set_linear_srgb(linear);
        gs_enable_framebuffer_srgb(srgb);
        finish_frame(f, t0);
        return;
    }
    // Tile ownership depends only on placement: color-only updates reuse the
    // owner cache.
    if (f->located_geometry != f->uploaded_geometry || f->located_w != w || f->located_h != h) locate(f, w, h);

    s.look.time = std::chrono::duration<float>(t0 - f->start).count();
    float L[3], H[3];
    mosaic::light_vectors(s.look, L, H);
    const Params& p = f->p;
    gs_effect_set_texture(p.image, gs_texrender_get_texture(f->capture));
    gs_effect_set_texture(p.locate_image, gs_texrender_get_texture(f->locate));
    gs_effect_set_texture(p.tile_color, f->color_tex);
    set_vec2(p.color_tex_size, float(mosaic::GpuPack::kTilesPerRow), float(f->tex_rows));
    gs_effect_set_float(p.grout_half, 0.5f * s.look.grout_px);
    gs_effect_set_float(p.bevel_px, std::max(0.05f, s.look.bevel_px));
    gs_effect_set_float(p.bevel, s.look.bevel);
    gs_effect_set_float(p.facet, s.look.facet);
    gs_effect_set_float(p.diffuse, s.look.diffuse);
    gs_effect_set_float(p.glint, s.look.glint);
    gs_effect_set_float(p.shininess, s.look.shininess);
    gs_effect_set_float(p.metal, s.look.metal);
    gs_effect_set_float(p.twinkle, s.look.twinkle);
    gs_effect_set_float(p.time, s.look.time);
    set_vec3(p.light_dir, L);
    set_vec3(p.half_dir, H);
    set_vec3(p.grout_color, s.look.grout);
    gs_effect_set_float(p.mix_amount, s.mix);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
    while (gs_effect_loop(f->effect, "Draw")) gs_draw_sprite(gs_texrender_get_texture(f->capture), 0, w, h);
    gs_blend_state_pop();
    gs_set_linear_srgb(linear);
    gs_enable_framebuffer_srgb(srgb);
    finish_frame(f, t0);
}

gs_color_space color_space(void* data, size_t count, const gs_color_space* preferred)
{
    auto* target = obs_filter_get_target(static_cast<Filter*>(data)->source);
    return target ? obs_source_get_color_space(target, count, preferred) : GS_CS_SRGB;
}

const char* name(void*) { return "Mosaic FX"; }

void defaults(obs_data_t* s)
{
    const mosaic::Config c;
    const mosaic::RenderParams r;
    obs_data_set_default_int(s, "mz_tiles", c.target_tiles);
    obs_data_set_default_int(s, "mz_classes", c.size_classes);
    obs_data_set_default_double(s, "mz_ratio", c.size_ratio);
    obs_data_set_default_double(s, "mz_edge", 0.025);
    obs_data_set_default_double(s, "mz_outline", c.outline);
    obs_data_set_default_int(s, "mz_rows", c.contour_rows);
    obs_data_set_default_bool(s, "mz_frame_rows", c.frame_edges);
    obs_data_set_default_double(s, "mz_jitter", c.lightness_jitter);
    obs_data_set_default_double(s, "mz_recolor", c.recolor_threshold);
    obs_data_set_default_double(s, "mz_restructure", c.restructure_threshold);
    obs_data_set_default_double(s, "mz_prefilter", c.prefilter_luma);
    obs_data_set_default_double(s, "mz_grout", r.grout_px);
    obs_data_set_default_int(s, "mz_grout_color", 0xff2e3438); // 0xAABBGGRR: warm dark grout
    obs_data_set_default_double(s, "mz_bevel", r.bevel);
    obs_data_set_default_double(s, "mz_glint", r.glint);
    obs_data_set_default_double(s, "mz_facet", r.facet);
    obs_data_set_default_double(s, "mz_shininess", r.shininess);
    obs_data_set_default_double(s, "mz_metal", r.metal);
    obs_data_set_default_double(s, "mz_sweep", r.sweep);
    obs_data_set_default_double(s, "mz_twinkle", r.twinkle);
    obs_data_set_default_double(s, "mz_azimuth", r.light_azimuth * 180.0 / 3.14159265);
    obs_data_set_default_double(s, "mz_diffuse", r.diffuse);
    obs_data_set_default_int(s, "mz_analysis", 960);
    obs_data_set_default_double(s, "mz_hz", 15);
    obs_data_set_default_double(s, "mz_retile_hz", 6);
    obs_data_set_default_int(s, "mz_lattice", 512);
    obs_data_set_default_int(s, "mz_colors", 32);
    obs_data_set_default_int(s, "mz_threads", 4);
    obs_data_set_default_double(s, "mz_mix", 1);
    obs_data_set_default_bool(s, "mz_freeze", false);
}

obs_properties_t* properties(void*)
{
    auto* p = obs_properties_create();
    auto* tiles = obs_properties_create();
    obs_properties_add_int_slider(tiles, "mz_tiles", "Tile budget", 1000, 60000, 500);
    obs_properties_add_int_slider(tiles, "mz_classes", "Tile sizes", 1, 4, 1);
    obs_properties_add_float_slider(tiles, "mz_ratio", "Size step between classes", 1.05, 2.0, 0.05);
    obs_properties_add_int_slider(tiles, "mz_colors", "Tile colors (0 = true color)", 0, 64, 1);
    obs_properties_add_float_slider(tiles, "mz_edge", "Edge sensitivity (lower = more outlines)", 0.005, 0.2, 0.005);
    obs_properties_add_float_slider(tiles, "mz_outline", "Outline darkness", 0, 1, 0.05);
    obs_properties_add_int_slider(tiles, "mz_rows", "Contour rows (0 = everywhere)", 0, 64, 1);
    obs_properties_add_bool(tiles, "mz_frame_rows", "Rows follow the frame border");
    obs_properties_add_float_slider(tiles, "mz_jitter", "Tile shade variation", 0, 0.15, 0.005);
    obs_properties_add_group(p, "mz_tiles_group", "Tiles", OBS_GROUP_NORMAL, tiles);

    auto* look = obs_properties_create();
    obs_properties_add_float_slider(look, "mz_grout", "Grout width (px)", 0, 6, 0.1);
    obs_properties_add_color(look, "mz_grout_color", "Grout color");
    obs_properties_add_float_slider(look, "mz_bevel", "Bevel", 0, 1.5, 0.05);
    obs_properties_add_float_slider(look, "mz_diffuse", "Relief shading", 0, 1, 0.05);
    obs_properties_add_float_slider(look, "mz_glint", "Glint strength", 0, 4, 0.05);
    obs_properties_add_float_slider(look, "mz_facet", "Glint scatter (tile tilt)", 0, 1, 0.01);
    obs_properties_add_float_slider(look, "mz_shininess", "Glint sharpness", 4, 400, 1);
    obs_properties_add_float_slider(look, "mz_metal", "Glint takes tile color", 0, 1, 0.05);
    obs_properties_add_float_slider(look, "mz_sweep", "Light sweep (rad/s)", -3, 3, 0.05);
    obs_properties_add_float_slider(look, "mz_twinkle", "Twinkle", 0, 1, 0.05);
    obs_properties_add_float_slider(look, "mz_azimuth", "Light direction (deg)", -180, 180, 1);
    obs_properties_add_group(p, "mz_look_group", "Look", OBS_GROUP_NORMAL, look);

    auto* time = obs_properties_create();
    obs_properties_add_float_slider(time, "mz_recolor", "Recolor threshold (Oklab)", 0.002, 0.3, 0.002);
    obs_properties_add_float_slider(time, "mz_restructure", "Re-tiling threshold", 0.005, 0.3, 0.005);
    obs_properties_add_float_slider(time, "mz_prefilter", "Change wake threshold (8-bit luma)", 0.5, 40, 0.5);
    obs_properties_add_float_slider(time, "mz_hz", "Analysis samples per second", 1, 60, 1);
    obs_properties_add_float_slider(time, "mz_retile_hz", "Re-tiling passes per second (max)", 0.5, 30, 0.5);
    obs_properties_add_int_slider(time, "mz_analysis", "Analysis width (px)", 320, 1920, 32);
    auto* lattice = obs_properties_add_list(time, "mz_lattice", "Cartoon lattice", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(lattice, "256 (fastest)", 256);
    obs_property_list_add_int(lattice, "512 (default)", 512);
    obs_property_list_add_int(lattice, "1024 (detailed)", 1024);
    obs_properties_add_int_slider(time, "mz_threads", "Cartoon CPU threads", 1, 16, 1);
    obs_properties_add_bool(time, "mz_freeze", "Freeze tiles");
    obs_properties_add_group(p, "mz_time_group", "Motion and analysis", OBS_GROUP_NORMAL, time);

    obs_properties_add_float_slider(p, "mz_mix", "Effect mix", 0, 1, 0.01);
    return p;
}

} // namespace

bool obs_module_load(void)
{
    obs_source_info info{};
    info.id = "mosaic_fx_filter";
    info.type = OBS_SOURCE_TYPE_FILTER;
    info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
    info.get_name = name;
    info.create = create;
    info.destroy = destroy;
    info.update = update;
    info.get_defaults = defaults;
    info.get_properties = properties;
    info.video_render = render;
    info.video_get_color_space = color_space;
    obs_register_source(&info);
    blog(LOG_INFO, "[Mosaic FX] filter registered");
    return true;
}
