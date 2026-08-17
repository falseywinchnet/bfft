#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rvfx {

enum class PixelFormat : std::uint8_t { RGBA, BGRA, NV12, I420 };
enum class EffectMode : std::uint8_t { Phosphor, LiquidMetal, EmbossSheen };
enum class GlyphMotion : std::uint8_t { Falling, Arcing, Mixed };
enum class CommandKind : std::uint8_t {
    Trace, Glyph, GlyphTrail, EmbossShadow, Sheen
};

struct FrameView {
    const std::uint8_t* data = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::ptrdiff_t stride = 0;
    PixelFormat format = PixelFormat::RGBA;
    const std::uint8_t* plane1 = nullptr;
    const std::uint8_t* plane2 = nullptr;
    std::ptrdiff_t stride1 = 0;
    std::ptrdiff_t stride2 = 0;
    bool full_range = true;
};

struct MutableFrameView {
    std::uint8_t* data = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::ptrdiff_t stride = 0;
    PixelFormat format = PixelFormat::RGBA;
    std::uint8_t* plane1 = nullptr;
    std::uint8_t* plane2 = nullptr;
    std::ptrdiff_t stride1 = 0;
    std::ptrdiff_t stride2 = 0;
    bool full_range = true;
};

struct Config {
    std::uint32_t trace_width = 480;
    std::uint32_t palette_colors = 8;
    std::uint32_t palette_samples = 4096;
    std::uint32_t segments_per_frame = 2048;
    float detail_priority = 1.5f;
    float population_exponent = 0.70f;
    float lightness_weight = 1.0f;
    float chroma_weight = 1.0f;
    float hue_weight = 1.0f;
    float alpha_weight = 0.7f;
    float node_separation = 1.0f;
    std::uint32_t minimum_leaf = 8;
    std::uint32_t bifurcation_refinement = 4;
    float prior_learning_rate = 0.14f;
    float trace_speed = 0.075f;
    float trace_persistence = 0.86f;
    float glow = 0.65f;
    std::uint32_t glyph_particles = 256;
    std::uint32_t glyph_spawn_per_frame = 6;
    bool glyph_layer = true;
    bool posterize_only = false;
    GlyphMotion glyph_motion = GlyphMotion::Mixed;
    float frame_budget_ms = 1000.0f / 30.0f;
    EffectMode effect = EffectMode::Phosphor;
};

struct PaletteColor {
    float l = 0.0f, a = 0.0f, b = 0.0f, alpha = 1.0f;
    std::uint8_t r = 0, g = 0, blue = 0, opacity = 255;
};

struct TraceSegment {
    std::uint64_t id = 0;
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    std::uint16_t left_color = 0, right_color = 0;
    std::uint32_t age = 0;
};

struct DrawCommand {
    CommandKind kind = CommandKind::Trace;
    std::uint64_t source_id = 0;
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    float width = 1.0f;
    float opacity = 1.0f;
    float glow = 0.0f;
    std::uint8_t r = 255, g = 255, b = 255;
    std::uint8_t glyph = 0;
};

struct FrameStats {
    double palette_ms = 0.0;
    double posterize_ms = 0.0;
    double trace_ms = 0.0;
    double effects_ms = 0.0;
    double total_ms = 0.0;
    std::uint32_t active_segments = 0;
    std::uint32_t live_glyphs = 0;
    std::uint32_t changed_cells = 0;
    std::uint32_t reused_cells = 0;
    std::uint32_t emitted_commands = 0;
    bool within_budget = false;
};

class Engine {
public:
    explicit Engine(Config config = {});
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void set_config(const Config& config);
    const Config& config() const noexcept;
    void reset();

    // Updates temporal palette priors, the posterized ownership lattice,
    // stable trace edges, and the bounded reveal/effects command stream.
    const FrameStats& process(const FrameView& frame);

    // Composites the current posterization and commands. Source and
    // destination may alias, which is the OBS filter path.
    void render(const FrameView& source, const MutableFrameView& destination) const;

    const std::vector<PaletteColor>& palette() const noexcept;
    const std::vector<std::uint16_t>& labels() const noexcept;
    const std::vector<TraceSegment>& active_segments() const noexcept;
    const std::vector<DrawCommand>& commands() const noexcept;
    // Off-hot-path diagnostic/export representation of the current stable
    // trace runs and selected effect commands.
    std::string svg_snapshot() const;
    std::uint32_t lattice_width() const noexcept;
    std::uint32_t lattice_height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rvfx
