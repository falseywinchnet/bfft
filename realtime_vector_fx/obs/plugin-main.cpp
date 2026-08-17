#include "rvfx/engine.hpp"
#include <obs-module.h>
#include <memory>
#include <mutex>

OBS_DECLARE_MODULE()

void rvfx_register_gpu_filter();

MODULE_EXPORT const char* obs_module_description(void) {
    return "Persistent real-time posterization, incremental tracing, and phosphor/metal/sheen effects";
}

namespace {
struct Filter { obs_source_t* source{}; std::mutex mutex; rvfx::Config config; rvfx::Engine engine; };
constexpr const char* K_COLORS="rvfx_colors";
constexpr const char* K_TRACE_WIDTH="rvfx_trace_width";
constexpr const char* K_SEGMENTS="rvfx_segments";
constexpr const char* K_PERSISTENCE="rvfx_persistence";
constexpr const char* K_EFFECT="rvfx_effect";
constexpr const char* K_GLYPHS="rvfx_glyphs";
constexpr const char* K_GLYPH_MOTION="rvfx_glyph_motion";

void* create(obs_data_t* settings,obs_source_t* source) {
    auto* f=new Filter; f->source=source;
    f->config.palette_colors=static_cast<std::uint32_t>(obs_data_get_int(settings,K_COLORS));
    f->config.trace_width=static_cast<std::uint32_t>(obs_data_get_int(settings,K_TRACE_WIDTH));
    f->config.segments_per_frame=static_cast<std::uint32_t>(obs_data_get_int(settings,K_SEGMENTS));
    f->config.trace_persistence=static_cast<float>(obs_data_get_double(settings,K_PERSISTENCE));
    f->config.effect=static_cast<rvfx::EffectMode>(obs_data_get_int(settings,K_EFFECT));
    f->config.glyph_particles=static_cast<std::uint32_t>(obs_data_get_int(settings,K_GLYPHS));
    f->config.glyph_layer=f->config.glyph_particles>0;
    f->config.glyph_motion=static_cast<rvfx::GlyphMotion>(obs_data_get_int(settings,K_GLYPH_MOTION));
    f->engine.set_config(f->config); return f;
}
void destroy(void* data){delete static_cast<Filter*>(data);}
void update(void* data,obs_data_t* s){auto* f=static_cast<Filter*>(data);std::lock_guard<std::mutex> lock(f->mutex);
    f->config.palette_colors=static_cast<std::uint32_t>(obs_data_get_int(s,K_COLORS));
    f->config.trace_width=static_cast<std::uint32_t>(obs_data_get_int(s,K_TRACE_WIDTH));
    f->config.segments_per_frame=static_cast<std::uint32_t>(obs_data_get_int(s,K_SEGMENTS));
    f->config.trace_persistence=static_cast<float>(obs_data_get_double(s,K_PERSISTENCE));
    f->config.effect=static_cast<rvfx::EffectMode>(obs_data_get_int(s,K_EFFECT));
    f->config.glyph_particles=static_cast<std::uint32_t>(obs_data_get_int(s,K_GLYPHS));
    f->config.glyph_layer=f->config.glyph_particles>0;
    f->config.glyph_motion=static_cast<rvfx::GlyphMotion>(obs_data_get_int(s,K_GLYPH_MOTION));
    f->engine.set_config(f->config);}
bool supported(video_format format){return format==VIDEO_FORMAT_RGBA||format==VIDEO_FORMAT_BGRA||
    format==VIDEO_FORMAT_BGRX||format==VIDEO_FORMAT_NV12||format==VIDEO_FORMAT_I420;}
obs_source_frame* video(void* data,obs_source_frame* frame){auto* f=static_cast<Filter*>(data);
    if(!frame||!frame->data[0]||!supported(frame->format))return frame;
    std::lock_guard<std::mutex> lock(f->mutex);rvfx::PixelFormat fmt=rvfx::PixelFormat::BGRA;
    if(frame->format==VIDEO_FORMAT_RGBA)fmt=rvfx::PixelFormat::RGBA;
    else if(frame->format==VIDEO_FORMAT_NV12)fmt=rvfx::PixelFormat::NV12;
    else if(frame->format==VIDEO_FORMAT_I420)fmt=rvfx::PixelFormat::I420;
    rvfx::FrameView in{frame->data[0],frame->width,frame->height,static_cast<std::ptrdiff_t>(frame->linesize[0]),fmt};
    in.plane1=frame->data[1];in.plane2=frame->data[2];in.stride1=frame->linesize[1];in.stride2=frame->linesize[2];in.full_range=frame->full_range;
    f->engine.process(in);rvfx::MutableFrameView out{frame->data[0],frame->width,frame->height,static_cast<std::ptrdiff_t>(frame->linesize[0]),fmt};
    out.plane1=frame->data[1];out.plane2=frame->data[2];out.stride1=frame->linesize[1];out.stride2=frame->linesize[2];out.full_range=frame->full_range;
    f->engine.render(in,out);return frame;}
const char* name(void*){return "Realtime Vector FX";}
void defaults(obs_data_t* s){obs_data_set_default_int(s,K_COLORS,8);obs_data_set_default_int(s,K_TRACE_WIDTH,480);
    obs_data_set_default_int(s,K_SEGMENTS,2048);obs_data_set_default_int(s,K_EFFECT,0);
    obs_data_set_default_double(s,K_PERSISTENCE,.86);obs_data_set_default_int(s,K_GLYPHS,256);
    obs_data_set_default_int(s,K_GLYPH_MOTION,2);}
obs_properties_t* properties(void*){auto* p=obs_properties_create();
    obs_properties_add_int_slider(p,K_COLORS,"Poster colors",2,32,1);
    obs_properties_add_int_slider(p,K_TRACE_WIDTH,"Trace lattice width",160,960,16);
    obs_properties_add_int_slider(p,K_SEGMENTS,"Segments per frame",64,8192,64);
    obs_properties_add_float_slider(p,K_PERSISTENCE,"Trace history",0.0,0.98,0.01);
    obs_properties_add_int_slider(p,K_GLYPHS,"Independent glyph particles",0,2048,32);
    auto* e=obs_properties_add_list(p,K_EFFECT,"Effect",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(e,"Phosphor traces",0);obs_property_list_add_int(e,"Liquid metal",1);obs_property_list_add_int(e,"Emboss + color sheen",2);
    auto* m=obs_properties_add_list(p,K_GLYPH_MOTION,"Glyph motion",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(m,"Falling",0);obs_property_list_add_int(m,"Arcing",1);obs_property_list_add_int(m,"Mixed",2);return p;}
}

obs_source_info info{};
struct InfoInit { InfoInit() {
    info.id="realtime_vector_fx"; info.type=OBS_SOURCE_TYPE_FILTER;
    info.output_flags=OBS_SOURCE_ASYNC_VIDEO; info.get_name=name; info.create=create;
    info.destroy=destroy; info.update=update; info.get_defaults=defaults;
    info.get_properties=properties; info.filter_video=video;
} } info_init;

bool obs_module_load(void){obs_register_source(&info);rvfx_register_gpu_filter();
    blog(LOG_INFO,"[Realtime Vector FX] CPU, GPU FX, and Optimal OKLCH Posterizer filters registered");return true;}
