#include "rvfx/engine.hpp"
#include <obs-module.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr const char* K_COLORS="rvfx_colors";
constexpr const char* K_TRACE_WIDTH="rvfx_trace_width";
constexpr const char* K_SEGMENTS="rvfx_segments";
constexpr const char* K_PERSISTENCE="rvfx_persistence";
constexpr const char* K_EFFECT="rvfx_effect";
constexpr const char* K_GLYPHS="rvfx_glyphs";
constexpr const char* K_GLYPH_MOTION="rvfx_glyph_motion";
constexpr const char* K_SAMPLES="rvfx_palette_samples";
constexpr const char* K_LIGHTNESS="rvfx_lightness_weight";
constexpr const char* K_CHROMA="rvfx_chroma_weight";
constexpr const char* K_HUE="rvfx_hue_weight";
constexpr const char* K_ALPHA="rvfx_alpha_weight";
constexpr const char* K_SEPARATION="rvfx_node_separation";
constexpr const char* K_DETAIL="rvfx_detail_priority";
constexpr const char* K_POPULATION="rvfx_population_exponent";
constexpr const char* K_PRIOR="rvfx_prior_learning_rate";
constexpr const char* K_MIN_LEAF="rvfx_minimum_leaf";
constexpr const char* K_REFINEMENT="rvfx_bifurcation_refinement";
constexpr const char* K_CONTOUR="rvfx_contour_strength";
constexpr const char* K_INTERIOR_INK="rvfx_interior_ink";
constexpr const char* K_LINE_REACH="rvfx_line_reach";
constexpr const char* K_SATURATION="rvfx_look_saturation";
constexpr const char* K_CONTRAST="rvfx_look_contrast";
constexpr std::size_t VERTEX_CAPACITY=262144;

const char* POSTER_EFFECT=R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d label_image;
uniform texture2d palette_image;
uniform float2 label_texel;
uniform float2 source_texel;
uniform float contour_strength;
uniform float interior_ink;
uniform float line_reach;
uniform float look_saturation;
uniform float look_contrast;
uniform float look_enabled;

sampler_state linear_sampler { Filter=Linear; AddressU=Clamp; AddressV=Clamp; };
sampler_state point_sampler { Filter=Point; AddressU=Clamp; AddressV=Clamp; };
struct VertData { float4 pos : POSITION; float2 uv : TEXCOORD0; };
VertData VSDefault(VertData v_in) {
    VertData v_out;
    v_out.pos=mul(float4(v_in.pos.xyz,1.0),ViewProj);
    v_out.uv=v_in.uv;
    return v_out;
}
float4 PSPoster(VertData v_in) : TARGET {
    float index=floor(label_image.Sample(point_sampler,v_in.uv).r*255.0+0.5);
    float4 poster=palette_image.Sample(point_sampler,float2((index+0.5)/64.0,0.5));
    float4 source=image.Sample(linear_sampler,v_in.uv);
    poster.a=source.a;
    if (look_enabled<0.5) return poster;
    float reach=max(line_reach,0.0);
    float left=label_image.Sample(point_sampler,v_in.uv-float2(label_texel.x*reach,0.0)).r;
    float right=label_image.Sample(point_sampler,v_in.uv+float2(label_texel.x*reach,0.0)).r;
    float up=label_image.Sample(point_sampler,v_in.uv-float2(0.0,label_texel.y*reach)).r;
    float down=label_image.Sample(point_sampler,v_in.uv+float2(0.0,label_texel.y*reach)).r;
    float center=index/255.0;
    float contour=max(max(step(0.5/255.0,abs(center-left)),step(0.5/255.0,abs(center-right))),
                      max(step(0.5/255.0,abs(center-up)),step(0.5/255.0,abs(center-down))));
    float source_left=dot(image.Sample(linear_sampler,v_in.uv-float2(source_texel.x*reach,0.0)).rgb,
                          float3(0.2126,0.7152,0.0722));
    float source_right=dot(image.Sample(linear_sampler,v_in.uv+float2(source_texel.x*reach,0.0)).rgb,
                           float3(0.2126,0.7152,0.0722));
    float source_up=dot(image.Sample(linear_sampler,v_in.uv-float2(0.0,source_texel.y*reach)).rgb,
                        float3(0.2126,0.7152,0.0722));
    float source_down=dot(image.Sample(linear_sampler,v_in.uv+float2(0.0,source_texel.y*reach)).rgb,
                          float3(0.2126,0.7152,0.0722));
    float detail=saturate(2.5*(abs(source_right-source_left)+abs(source_down-source_up)));
    float luminance=dot(poster.rgb,float3(0.2126,0.7152,0.0722));
    poster.rgb=lerp(float3(luminance,luminance,luminance),poster.rgb,look_saturation);
    poster.rgb=saturate((poster.rgb-0.5)*look_contrast+0.5);
    poster.rgb*=1.0-saturate(contour_strength*contour+interior_ink*detail);
    return poster;
}
technique Draw { pass { vertex_shader=VSDefault(v_in); pixel_shader=PSPoster(v_in); } }
)";

const char* OVERLAY_EFFECT=R"(
uniform float4x4 ViewProj;
struct VertData { float4 pos : POSITION; float4 color : COLOR; };
VertData VSDefault(VertData v_in) {
    VertData v_out;
    v_out.pos=mul(float4(v_in.pos.xyz,1.0),ViewProj);
    v_out.color=v_in.color;
    return v_out;
}
float4 PSColor(VertData v_in) : TARGET { return v_in.color; }
technique Draw { pass { vertex_shader=VSDefault(v_in); pixel_shader=PSColor(v_in); } }
)";

const char* TRAIL_EFFECT=R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform float decay;
sampler_state linear_sampler { Filter=Linear; AddressU=Clamp; AddressV=Clamp; };
struct VertData { float4 pos : POSITION; float2 uv : TEXCOORD0; };
VertData VSDefault(VertData v_in) {
    VertData v_out;
    v_out.pos=mul(float4(v_in.pos.xyz,1.0),ViewProj);
    v_out.uv=v_in.uv;
    return v_out;
}
float4 PSTrail(VertData v_in) : TARGET {
    float4 value=image.Sample(linear_sampler,v_in.uv);
    value.a*=decay;
    return value;
}
technique Draw { pass { vertex_shader=VSDefault(v_in); pixel_shader=PSTrail(v_in); } }
)";

struct GpuFilter {
    obs_source_t* source=nullptr;
    bool poster_only=false;
    std::mutex settings_mutex;
    rvfx::Config pending;
    rvfx::Config active;
    rvfx::Engine engine;
    gs_effect_t* poster_effect=nullptr;
    gs_effect_t* overlay_effect=nullptr;
    gs_effect_t* trail_effect=nullptr;
    gs_texrender_t* analysis_render=nullptr;
    std::array<gs_texrender_t*,2> trail_render{};
    std::size_t trail_current=0;
    bool trail_ready=false;
    std::array<gs_stagesurf_t*,2> stage{};
    std::array<bool,2> written{};
    std::size_t stage_index=0;
    gs_texture_t* label_texture=nullptr;
    gs_texture_t* palette_texture=nullptr;
    gs_vertbuffer_t* vertices=nullptr;
    std::vector<std::uint8_t> label_bytes;
    std::array<std::uint8_t,64*4> palette_bytes{};
    std::uint32_t target_width=0,target_height=0,analysis_width=0,analysis_height=0;
    bool ready=false;
    struct PosterLook {
        float contour=0.0f,interior_ink=0.0f,line_reach=1.0f;
        float saturation=1.0f,contrast=1.0f;
    } pending_look,active_look;
};

void destroy_graphics(GpuFilter* f) {
    if(f->vertices){gs_vertexbuffer_destroy(f->vertices);f->vertices=nullptr;}
    if(f->label_texture){gs_texture_destroy(f->label_texture);f->label_texture=nullptr;}
    if(f->palette_texture){gs_texture_destroy(f->palette_texture);f->palette_texture=nullptr;}
    for(auto& stage:f->stage){if(stage)gs_stagesurface_destroy(stage);stage=nullptr;}
    if(f->analysis_render){gs_texrender_destroy(f->analysis_render);f->analysis_render=nullptr;}
    for(auto& trail:f->trail_render){if(trail)gs_texrender_destroy(trail);trail=nullptr;}
    if(f->poster_effect){gs_effect_destroy(f->poster_effect);f->poster_effect=nullptr;}
    if(f->overlay_effect){gs_effect_destroy(f->overlay_effect);f->overlay_effect=nullptr;}
    if(f->trail_effect){gs_effect_destroy(f->trail_effect);f->trail_effect=nullptr;}
}

rvfx::Config read_config(obs_data_t* settings) {
    rvfx::Config c;
    c.palette_colors=static_cast<std::uint32_t>(obs_data_get_int(settings,K_COLORS));
    c.trace_width=static_cast<std::uint32_t>(obs_data_get_int(settings,K_TRACE_WIDTH));
    c.segments_per_frame=static_cast<std::uint32_t>(obs_data_get_int(settings,K_SEGMENTS));
    c.trace_persistence=static_cast<float>(obs_data_get_double(settings,K_PERSISTENCE));
    c.effect=static_cast<rvfx::EffectMode>(obs_data_get_int(settings,K_EFFECT));
    c.glyph_particles=static_cast<std::uint32_t>(obs_data_get_int(settings,K_GLYPHS));
    c.glyph_layer=c.glyph_particles>0;
    c.glyph_motion=static_cast<rvfx::GlyphMotion>(obs_data_get_int(settings,K_GLYPH_MOTION));
    return c;
}

rvfx::Config read_poster_config(obs_data_t* settings) {
    rvfx::Config c;
    c.posterize_only=true;c.glyph_layer=false;c.glyph_particles=0;
    c.palette_colors=static_cast<std::uint32_t>(obs_data_get_int(settings,K_COLORS));
    c.trace_width=static_cast<std::uint32_t>(obs_data_get_int(settings,K_TRACE_WIDTH));
    c.palette_samples=static_cast<std::uint32_t>(obs_data_get_int(settings,K_SAMPLES));
    c.lightness_weight=static_cast<float>(obs_data_get_double(settings,K_LIGHTNESS));
    c.chroma_weight=static_cast<float>(obs_data_get_double(settings,K_CHROMA));
    c.hue_weight=static_cast<float>(obs_data_get_double(settings,K_HUE));
    c.alpha_weight=static_cast<float>(obs_data_get_double(settings,K_ALPHA));
    c.node_separation=static_cast<float>(obs_data_get_double(settings,K_SEPARATION));
    c.detail_priority=static_cast<float>(obs_data_get_double(settings,K_DETAIL));
    c.population_exponent=static_cast<float>(obs_data_get_double(settings,K_POPULATION));
    c.prior_learning_rate=static_cast<float>(obs_data_get_double(settings,K_PRIOR));
    c.minimum_leaf=static_cast<std::uint32_t>(obs_data_get_int(settings,K_MIN_LEAF));
    c.bifurcation_refinement=static_cast<std::uint32_t>(obs_data_get_int(settings,K_REFINEMENT));
    return c;
}

GpuFilter::PosterLook read_poster_look(obs_data_t* settings) {
    GpuFilter::PosterLook look;
    look.contour=static_cast<float>(obs_data_get_double(settings,K_CONTOUR));
    look.interior_ink=static_cast<float>(obs_data_get_double(settings,K_INTERIOR_INK));
    look.line_reach=static_cast<float>(obs_data_get_double(settings,K_LINE_REACH));
    look.saturation=static_cast<float>(obs_data_get_double(settings,K_SATURATION));
    look.contrast=static_cast<float>(obs_data_get_double(settings,K_CONTRAST));
    return look;
}

void* gpu_create_impl(obs_data_t* settings,obs_source_t* source,bool poster_only) {
    auto* f=new GpuFilter;f->source=source;f->poster_only=poster_only;
    f->pending=poster_only?read_poster_config(settings):read_config(settings);f->active=f->pending;
    if(poster_only)f->pending_look=f->active_look=read_poster_look(settings);
    f->engine.set_config(f->active);char* errors=nullptr;
    obs_enter_graphics();
    f->poster_effect=gs_effect_create(POSTER_EFFECT,"realtime-vector-poster.effect",&errors);
    if(errors){blog(LOG_ERROR,"[Realtime Vector FX GPU] poster shader: %s",errors);bfree(errors);errors=nullptr;}
    if(!poster_only){
        f->overlay_effect=gs_effect_create(OVERLAY_EFFECT,"realtime-vector-overlay.effect",&errors);
        if(errors){blog(LOG_ERROR,"[Realtime Vector FX GPU] overlay shader: %s",errors);bfree(errors);errors=nullptr;}
        f->trail_effect=gs_effect_create(TRAIL_EFFECT,"realtime-vector-trail.effect",&errors);
        if(errors){blog(LOG_ERROR,"[Realtime Vector FX GPU] trail shader: %s",errors);bfree(errors);errors=nullptr;}
        if(f->poster_effect&&f->overlay_effect&&f->trail_effect){
            auto* data=gs_vbdata_create();data->num=VERTEX_CAPACITY;
            data->points=static_cast<vec3*>(bzalloc(sizeof(vec3)*VERTEX_CAPACITY));
            data->colors=static_cast<std::uint32_t*>(bzalloc(sizeof(std::uint32_t)*VERTEX_CAPACITY));
            f->vertices=gs_vertexbuffer_create(data,GS_DYNAMIC);
        }
    }
    obs_leave_graphics();
    if(!f->poster_effect||(!poster_only&&(!f->overlay_effect||!f->trail_effect||!f->vertices))){
        obs_enter_graphics();destroy_graphics(f);obs_leave_graphics();delete f;return nullptr;
    }
    return f;
}

void* gpu_create(obs_data_t* settings,obs_source_t* source){return gpu_create_impl(settings,source,false);}
void* poster_create(obs_data_t* settings,obs_source_t* source){return gpu_create_impl(settings,source,true);}

void gpu_destroy(void* data) {
    auto* f=static_cast<GpuFilter*>(data);obs_enter_graphics();destroy_graphics(f);obs_leave_graphics();delete f;
}

void gpu_update(void* data,obs_data_t* settings) {
    auto* f=static_cast<GpuFilter*>(data);std::lock_guard<std::mutex> lock(f->settings_mutex);
    f->pending=f->poster_only?read_poster_config(settings):read_config(settings);
    if(f->poster_only)f->pending_look=read_poster_look(settings);
}

bool ensure_analysis(GpuFilter* f,std::uint32_t width,std::uint32_t height) {
    const auto aw=std::max(32u,std::min(width,f->active.trace_width));
    const auto ah=std::max(1u,static_cast<std::uint32_t>(std::lround(static_cast<double>(height)*aw/width)));
    if(width==f->target_width&&height==f->target_height&&aw==f->analysis_width&&ah==f->analysis_height)
        return true;
    if(f->label_texture){gs_texture_destroy(f->label_texture);f->label_texture=nullptr;}
    if(f->palette_texture){gs_texture_destroy(f->palette_texture);f->palette_texture=nullptr;}
    for(auto& stage:f->stage){if(stage)gs_stagesurface_destroy(stage);stage=nullptr;}
    if(f->analysis_render)gs_texrender_destroy(f->analysis_render);
    f->analysis_render=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    for(auto& trail:f->trail_render){if(trail)gs_texrender_destroy(trail);trail=nullptr;
        if(!f->poster_only)trail=gs_texrender_create(GS_RGBA,GS_ZS_NONE);}
    f->stage[0]=gs_stagesurface_create(aw,ah,GS_RGBA);
    f->stage[1]=gs_stagesurface_create(aw,ah,GS_RGBA);
    f->label_bytes.assign(static_cast<std::size_t>(aw)*ah,0);
    const std::uint8_t* label_data=f->label_bytes.data();
    f->label_texture=gs_texture_create(aw,ah,GS_R8,1,&label_data,GS_DYNAMIC);
    const std::uint8_t* palette_data=f->palette_bytes.data();
    f->palette_texture=gs_texture_create(64,1,GS_RGBA,1,&palette_data,GS_DYNAMIC);
    f->target_width=width;f->target_height=height;f->analysis_width=aw;f->analysis_height=ah;
    f->written={false,false};f->stage_index=0;f->ready=false;f->trail_ready=false;
    f->trail_current=0;f->engine.reset();
    return f->analysis_render&&(f->poster_only||(f->trail_render[0]&&f->trail_render[1]))&&
        f->stage[0]&&f->stage[1]&&f->label_texture&&f->palette_texture;
}

void upload_state(GpuFilter* f) {
    const auto& labels=f->engine.labels();
    for(std::size_t i=0;i<labels.size();++i)f->label_bytes[i]=static_cast<std::uint8_t>(labels[i]);
    std::fill(f->palette_bytes.begin(),f->palette_bytes.end(),0);
    const auto& palette=f->engine.palette();
    for(std::size_t i=0;i<palette.size();++i){const auto& c=palette[i];const auto q=4*i;
        f->palette_bytes[q]=c.r;f->palette_bytes[q+1]=c.g;f->palette_bytes[q+2]=c.blue;f->palette_bytes[q+3]=c.opacity;}
    gs_texture_set_image(f->label_texture,f->label_bytes.data(),f->analysis_width,false);
    gs_texture_set_image(f->palette_texture,f->palette_bytes.data(),64*4,false);
}

std::uint32_t color(std::uint8_t r,std::uint8_t g,std::uint8_t b,float opacity) {
    const auto a=static_cast<std::uint8_t>(std::clamp(std::lround(255.0f*opacity),0l,255l));
    return r|(static_cast<std::uint32_t>(g)<<8)|(static_cast<std::uint32_t>(b)<<16)|
           (static_cast<std::uint32_t>(a)<<24);
}

std::array<std::uint8_t,7> glyph_bitmap(std::uint8_t glyph) {
    switch(glyph){
    case '0': return {14,17,19,21,25,17,14};
    case '1': return {4,12,4,4,4,4,14};
    case 'Z': return {31,1,2,4,8,16,31};
    case 'X': return {17,17,10,4,10,17,17};
    case 'K': return {17,18,20,24,20,18,17};
    case 'M': return {17,27,21,21,17,17,17};
    case 'S': return {15,16,16,14,1,1,30};
    case 'V': return {17,17,17,17,10,10,4};
    default: return {4,4,4,31,4,4,4};
    }
}

void draw_commands(GpuFilter* f) {
    auto* data=gs_vertexbuffer_get_data(f->vertices);if(!data)return;
    std::size_t count=0;const float sx=static_cast<float>(f->target_width)/f->analysis_width;
    const float sy=static_cast<float>(f->target_height)/f->analysis_height;
    const auto add=[&](float x,float y,std::uint32_t rgba){if(count>=VERTEX_CAPACITY)return;
        vec3_set(&data->points[count],x,y,0.0f);data->colors[count]=rgba;++count;};
    const auto line=[&](float x1,float y1,float x2,float y2,float width,std::uint32_t rgba){
        if(count+6>VERTEX_CAPACITY)return;
        float dx=x2-x1,dy=y2-y1,length=std::sqrt(dx*dx+dy*dy);
        if(length<0.001f){dx=1.0f;dy=0.0f;length=1.0f;x1-=.5f*width;x2+=.5f*width;}
        const float nx=-dy*(.5f*width/length),ny=dx*(.5f*width/length);
        add(x1+nx,y1+ny,rgba);add(x1-nx,y1-ny,rgba);add(x2+nx,y2+ny,rgba);
        add(x2+nx,y2+ny,rgba);add(x1-nx,y1-ny,rgba);add(x2-nx,y2-ny,rgba);
    };
    for(const auto& c:f->engine.commands()){
        const float x1=c.x1*sx,y1=c.y1*sy,x2=c.x2*sx,y2=c.y2*sy;
        const auto rgba=color(c.r,c.g,c.b,c.opacity);
        if(c.kind==rvfx::CommandKind::Glyph){
            const auto rows=glyph_bitmap(c.glyph);
            const float pixel=std::max(1.0f,std::min(sx,sy));
            const float ox=x1-2.5f*pixel,oy=y1-3.5f*pixel;
            const auto glow=color(c.r,c.g,c.b,c.opacity*c.glow*.16f);
            for(std::size_t gy=0;gy<rows.size();++gy)for(int gx=0;gx<5;++gx)
                if(rows[gy]&(1u<<(4-gx))){
                    const float px=ox+(static_cast<float>(gx)+.5f)*pixel;
                    const float py=oy+(static_cast<float>(gy)+.5f)*pixel;
                    if(c.glow>0.01f){
                        line(px-.75f*pixel,py,px+.75f*pixel,py,2.8f*pixel,glow);
                    }
                    line(px-.5f*pixel,py,px+.5f*pixel,py,pixel,rgba);
                }
        }else{
            const float width=std::clamp(c.width*std::sqrt(sx*sy),1.0f,10.0f);
            if(c.glow>0.01f){const auto glow=color(c.r,c.g,c.b,c.opacity*c.glow*.18f);
                line(x1,y1,x2,y2,width+2.0f+4.0f*c.glow,glow);}
            line(x1,y1,x2,y2,width,rgba);
        }
    }
    data->num=count;gs_vertexbuffer_flush(f->vertices);gs_load_vertexbuffer(f->vertices);gs_load_indexbuffer(nullptr);
    gs_projection_push();gs_matrix_push();gs_matrix_identity();
    gs_ortho(0.0f,static_cast<float>(f->target_width),0.0f,static_cast<float>(f->target_height),-100.0f,100.0f);
    gs_blend_state_push();gs_blend_function(GS_BLEND_SRCALPHA,GS_BLEND_INVSRCALPHA);
    while(gs_effect_loop(f->overlay_effect,"Draw"))gs_draw(GS_TRIS,0,static_cast<std::uint32_t>(count));
    gs_blend_state_pop();gs_matrix_pop();gs_projection_pop();
}

void draw_trail_texture(GpuFilter* f,gs_texture_t* texture,float decay,bool blend) {
    if(!texture)return;
    gs_effect_set_texture(gs_effect_get_param_by_name(f->trail_effect,"image"),texture);
    gs_effect_set_float(gs_effect_get_param_by_name(f->trail_effect,"decay"),decay);
    gs_blend_state_push();gs_enable_blending(blend);
    if(blend)gs_blend_function(GS_BLEND_SRCALPHA,GS_BLEND_INVSRCALPHA);
    while(gs_effect_loop(f->trail_effect,"Draw"))
        gs_draw_sprite(texture,0,f->target_width,f->target_height);
    gs_blend_state_pop();
}

void update_trails(GpuFilter* f) {
    const std::size_t write=f->trail_ready?1u-f->trail_current:0u;
    auto* destination=f->trail_render[write];gs_texrender_reset(destination);
    gs_viewport_push();gs_projection_push();gs_matrix_push();
    if(gs_texrender_begin(destination,f->target_width,f->target_height)){
        vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0.0f,0);
        gs_matrix_identity();gs_ortho(0.0f,static_cast<float>(f->target_width),0.0f,
            static_cast<float>(f->target_height),-100.0f,100.0f);
        if(f->trail_ready)
            draw_trail_texture(f,gs_texrender_get_texture(f->trail_render[f->trail_current]),
                f->active.trace_persistence,false);
        draw_commands(f);gs_texrender_end(destination);f->trail_current=write;f->trail_ready=true;
    }
    gs_matrix_pop();gs_projection_pop();gs_viewport_pop();
}

void composite_trails(GpuFilter* f) {
    if(!f->trail_ready)return;
    gs_projection_push();gs_matrix_push();gs_matrix_identity();
    gs_ortho(0.0f,static_cast<float>(f->target_width),0.0f,
        static_cast<float>(f->target_height),-100.0f,100.0f);
    draw_trail_texture(f,gs_texrender_get_texture(f->trail_render[f->trail_current]),1.0f,true);
    gs_matrix_pop();gs_projection_pop();
}

void stage_target(GpuFilter* f) {
    auto* target=obs_filter_get_target(f->source);if(!target)return;
    auto* parent=obs_filter_get_parent(f->source);
    // A texrender refuses every begin after its first completed render until
    // it is reset. Without this, analysis silently remains on frame one.
    gs_texrender_reset(f->analysis_render);
    gs_viewport_push();gs_projection_push();gs_matrix_push();
    if(gs_texrender_begin(f->analysis_render,f->analysis_width,f->analysis_height)){
        vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0.0f,0);
        gs_matrix_identity();gs_ortho(0.0f,static_cast<float>(f->target_width),0.0f,
            static_cast<float>(f->target_height),-100.0f,100.0f);
        const auto flags=parent?obs_source_get_output_flags(parent):0u;
        const bool custom=(flags&OBS_SOURCE_CUSTOM_DRAW)!=0,async=(flags&OBS_SOURCE_ASYNC)!=0;
        if(target==parent&&!custom&&!async)obs_source_default_render(target);
        else obs_source_video_render(target);
        gs_texrender_end(f->analysis_render);
        gs_stage_texture(f->stage[f->stage_index],gs_texrender_get_texture(f->analysis_render));
        f->written[f->stage_index]=true;
    }
    gs_matrix_pop();gs_projection_pop();gs_viewport_pop();
}

void gpu_render(void* data,gs_effect_t*) {
    auto* f=static_cast<GpuFilter*>(data);auto* target=obs_filter_get_target(f->source);
    if(!target){obs_source_skip_video_filter(f->source);return;}
    {std::lock_guard<std::mutex> lock(f->settings_mutex);f->active=f->pending;f->active_look=f->pending_look;}
    f->engine.set_config(f->active);
    const auto width=obs_source_get_base_width(target),height=obs_source_get_base_height(target);
    if(!width||!height||!ensure_analysis(f,width,height)){obs_source_skip_video_filter(f->source);return;}
    const std::size_t read=(f->stage_index+1)%2;
    if(f->written[read]){
        std::uint8_t* mapped=nullptr;std::uint32_t stride=0;
        if(gs_stagesurface_map(f->stage[read],&mapped,&stride)){
            rvfx::FrameView frame{mapped,f->analysis_width,f->analysis_height,
                static_cast<std::ptrdiff_t>(stride),rvfx::PixelFormat::RGBA};
            f->engine.process(frame);gs_stagesurface_unmap(f->stage[read]);upload_state(f);f->ready=true;
        }
    }
    stage_target(f);f->stage_index=read;
    if(!f->ready){obs_source_skip_video_filter(f->source);return;}
    if(!obs_source_process_filter_begin(f->source,GS_RGBA,OBS_NO_DIRECT_RENDERING))return;
    gs_effect_set_texture(gs_effect_get_param_by_name(f->poster_effect,"label_image"),f->label_texture);
    gs_effect_set_texture(gs_effect_get_param_by_name(f->poster_effect,"palette_image"),f->palette_texture);
    vec2 label_texel,source_texel;
    vec2_set(&label_texel,1.0f/f->analysis_width,1.0f/f->analysis_height);
    vec2_set(&source_texel,1.0f/width,1.0f/height);
    gs_effect_set_vec2(gs_effect_get_param_by_name(f->poster_effect,"label_texel"),&label_texel);
    gs_effect_set_vec2(gs_effect_get_param_by_name(f->poster_effect,"source_texel"),&source_texel);
    gs_effect_set_float(gs_effect_get_param_by_name(f->poster_effect,"contour_strength"),f->active_look.contour);
    gs_effect_set_float(gs_effect_get_param_by_name(f->poster_effect,"interior_ink"),f->active_look.interior_ink);
    gs_effect_set_float(gs_effect_get_param_by_name(f->poster_effect,"line_reach"),f->active_look.line_reach);
    gs_effect_set_float(gs_effect_get_param_by_name(f->poster_effect,"look_saturation"),f->active_look.saturation);
    gs_effect_set_float(gs_effect_get_param_by_name(f->poster_effect,"look_contrast"),f->active_look.contrast);
    gs_effect_set_float(gs_effect_get_param_by_name(f->poster_effect,"look_enabled"),f->poster_only?1.0f:0.0f);
    obs_source_process_filter_end(f->source,f->poster_effect,width,height);
    if(!f->poster_only){update_trails(f);composite_trails(f);}
}

const char* gpu_name(void*){return "Realtime Vector FX (GPU)";}
const char* poster_name(void*){return "Optimal OKLCH Posterizer";}
void gpu_defaults(obs_data_t* s){obs_data_set_default_int(s,K_COLORS,8);obs_data_set_default_int(s,K_TRACE_WIDTH,480);
    obs_data_set_default_int(s,K_SEGMENTS,2048);obs_data_set_default_int(s,K_EFFECT,0);
    obs_data_set_default_double(s,K_PERSISTENCE,.86);obs_data_set_default_int(s,K_GLYPHS,256);
    obs_data_set_default_int(s,K_GLYPH_MOTION,2);}
obs_properties_t* gpu_properties(void*){auto* p=obs_properties_create();
    obs_properties_add_int_slider(p,K_COLORS,"Poster colors",2,32,1);
    obs_properties_add_int_slider(p,K_TRACE_WIDTH,"Analysis/trace width",160,960,16);
    obs_properties_add_int_slider(p,K_SEGMENTS,"Segments per frame",64,8192,64);
    obs_properties_add_float_slider(p,K_PERSISTENCE,"Trace history",0.0,0.98,0.01);
    obs_properties_add_int_slider(p,K_GLYPHS,"Independent glyph particles",0,2048,32);
    auto* e=obs_properties_add_list(p,K_EFFECT,"Effect",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(e,"Phosphor traces",0);obs_property_list_add_int(e,"Liquid metal",1);
    obs_property_list_add_int(e,"Emboss + color sheen",2);
    auto* m=obs_properties_add_list(p,K_GLYPH_MOTION,"Glyph motion",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(m,"Falling",0);obs_property_list_add_int(m,"Arcing",1);obs_property_list_add_int(m,"Mixed",2);return p;}

void poster_defaults(obs_data_t* s){
    obs_data_set_default_int(s,K_COLORS,24);obs_data_set_default_int(s,K_TRACE_WIDTH,480);
    obs_data_set_default_int(s,K_SAMPLES,4096);obs_data_set_default_double(s,K_LIGHTNESS,1.0);
    obs_data_set_default_double(s,K_CHROMA,1.0);obs_data_set_default_double(s,K_HUE,1.0);
    obs_data_set_default_double(s,K_ALPHA,.7);obs_data_set_default_double(s,K_SEPARATION,1.08);
    obs_data_set_default_double(s,K_DETAIL,2.0);obs_data_set_default_double(s,K_POPULATION,.65);
    obs_data_set_default_double(s,K_PRIOR,.14);obs_data_set_default_int(s,K_MIN_LEAF,8);
    obs_data_set_default_int(s,K_REFINEMENT,4);
    obs_data_set_default_double(s,K_CONTOUR,.16);obs_data_set_default_double(s,K_INTERIOR_INK,.06);
    obs_data_set_default_double(s,K_LINE_REACH,.65);obs_data_set_default_double(s,K_SATURATION,1.06);
    obs_data_set_default_double(s,K_CONTRAST,1.04);
}

obs_properties_t* poster_properties(void*){auto* p=obs_properties_create();
    obs_properties_add_int_slider(p,K_COLORS,"Colors",2,64,1);
    obs_properties_add_float_slider(p,K_CONTOUR,"Graphic contour strength",0.0,1.0,0.01);
    obs_properties_add_float_slider(p,K_INTERIOR_INK,"Interior detail ink",0.0,0.5,0.01);
    obs_properties_add_float_slider(p,K_LINE_REACH,"Ink line reach",0.0,2.5,0.05);
    obs_properties_add_float_slider(p,K_SATURATION,"Look saturation",0.5,1.8,0.01);
    obs_properties_add_float_slider(p,K_CONTRAST,"Look contrast",0.5,1.8,0.01);
    obs_properties_add_int_slider(p,K_TRACE_WIDTH,"Analysis resolution (width)",160,960,16);
    obs_properties_add_int_slider(p,K_SAMPLES,"Palette sample budget",512,16384,512);
    obs_properties_add_float_slider(p,K_SEPARATION,"Node separation",0.0,2.5,0.01);
    obs_properties_add_float_slider(p,K_LIGHTNESS,"Lightness weight",0.0,4.0,0.05);
    obs_properties_add_float_slider(p,K_CHROMA,"Chroma weight",0.0,4.0,0.05);
    obs_properties_add_float_slider(p,K_HUE,"Hue weight",0.0,4.0,0.05);
    obs_properties_add_float_slider(p,K_ALPHA,"Alpha weight",0.0,4.0,0.05);
    obs_properties_add_float_slider(p,K_DETAIL,"Detail priority",0.0,8.0,0.1);
    obs_properties_add_float_slider(p,K_POPULATION,"Area exponent",0.1,1.0,0.01);
    obs_properties_add_float_slider(p,K_PRIOR,"Temporal prior learning",0.01,1.0,0.01);
    obs_properties_add_int_slider(p,K_MIN_LEAF,"Minimum bifurcation leaf",1,64,1);
    obs_properties_add_int_slider(p,K_REFINEMENT,"Bifurcation refinement passes",0,12,1);
    return p;
}

obs_source_info gpu_info{};
obs_source_info poster_info{};
struct GpuInfoInit { GpuInfoInit(){gpu_info.id="realtime_vector_fx_gpu";gpu_info.type=OBS_SOURCE_TYPE_FILTER;
    gpu_info.output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_SRGB;gpu_info.get_name=gpu_name;gpu_info.create=gpu_create;
    gpu_info.destroy=gpu_destroy;gpu_info.update=gpu_update;gpu_info.get_defaults=gpu_defaults;
    gpu_info.get_properties=gpu_properties;gpu_info.video_render=gpu_render;
    poster_info.id="optimal_oklch_posterizer";poster_info.type=OBS_SOURCE_TYPE_FILTER;
    poster_info.output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_SRGB;poster_info.get_name=poster_name;
    poster_info.create=poster_create;poster_info.destroy=gpu_destroy;poster_info.update=gpu_update;
    poster_info.get_defaults=poster_defaults;poster_info.get_properties=poster_properties;
    poster_info.video_render=gpu_render;} } gpu_info_init;

} // namespace

void rvfx_register_gpu_filter(){obs_register_source(&gpu_info);obs_register_source(&poster_info);}
