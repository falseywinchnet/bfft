// GPU lifecycle and target rendering follow this plugin's gpu-filter.cpp.
#include "rvfx/entropy_stretch.hpp"
#include <obs-module.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>

namespace {
const char* shader=R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d profile;
uniform float2 source_size;
uniform float2 lattice_size;
uniform float2 phase;
uniform float strength;
sampler_state point_sampler { Filter=Point; AddressU=Clamp; AddressV=Clamp; };
sampler_state linear_sampler { Filter=Linear; AddressU=Clamp; AddressV=Clamp; };
struct VertData { float4 pos : POSITION; float2 uv : TEXCOORD0; };
VertData VSDefault(VertData v_in) {
    VertData v_out;v_out.pos=mul(float4(v_in.pos.xyz,1.0),ViewProj);v_out.uv=v_in.uv;return v_out;
}
float4 PSLattice(VertData v_in) : TARGET {
    float2 cell=floor(v_in.uv*lattice_size);
    float2 pixel=min(source_size-1.0,floor((cell+phase)*source_size/lattice_size));
    return image.Sample(point_sampler,(pixel+0.5)/source_size);
}
float4 PSStretch(VertData v_in) : TARGET {
    float4 src=image.Sample(linear_sampler,v_in.uv);
    float3 rgb=saturate(src.rgb/max(src.a,0.00001));
    float3 p=rgb*32.0;
    float blue=min(floor(p.b),31.0);
    float2 uv0=float2((blue*33.0+p.r+0.5)/1089.0,(p.g+0.5)/33.0);
    float2 uv1=uv0+float2(33.0/1089.0,0.0);
    float3 result=lerp(profile.Sample(linear_sampler,uv0).rgb,
                       profile.Sample(linear_sampler,uv1).rgb,p.b-blue);
    return float4(lerp(rgb,result,strength)*src.a,src.a);
}
technique Sample { pass { vertex_shader=VSDefault(v_in); pixel_shader=PSLattice(v_in); } }
technique Draw { pass { vertex_shader=VSDefault(v_in); pixel_shader=PSStretch(v_in); } }
)";
struct Settings { rvfx::entropy::Config config;int width=128;double hz=10;float strength=1;bool freeze=false; };
struct Filter {
    obs_source_t* source=nullptr;std::mutex mutex;Settings pending;
    rvfx::entropy::Profile profile;std::vector<uint16_t> upload;
    gs_effect_t* effect=nullptr;gs_texrender_t* capture=nullptr;gs_texrender_t* lattice=nullptr;
    gs_texture_t* lut=nullptr;std::array<gs_stagesurf_t*,2> stages{};std::array<bool,2> written{};
    int slot=0,phase=0,aw=0,ah=0;uint32_t width=0,height=0;
    std::chrono::steady_clock::time_point previous{},last_sample{};
    uint64_t last_frame=0;bool has_frame=false;
};
void release_analysis(Filter* f){for(auto& s:f->stages){if(s)gs_stagesurface_destroy(s);s=nullptr;}f->written={false,false};f->slot=0;}
void destroy(void* data){auto* f=static_cast<Filter*>(data);if(!f)return;obs_enter_graphics();release_analysis(f);
    if(f->lut)gs_texture_destroy(f->lut);if(f->capture)gs_texrender_destroy(f->capture);
    if(f->lattice)gs_texrender_destroy(f->lattice);if(f->effect)gs_effect_destroy(f->effect);obs_leave_graphics();delete f;}
void update(void* data,obs_data_t* s){auto* f=static_cast<Filter*>(data);Settings a;
    a.config.decorrelation=float(obs_data_get_double(s,"es_decorrelation"));
    a.config.allocation=float(obs_data_get_double(s,"es_allocation"));
    a.config.noise_floor=std::clamp(float(obs_data_get_double(s,"es_noise")),.001f,.1f);
    a.config.max_gain=std::clamp(float(obs_data_get_double(s,"es_gain")),1.f,16.f);
    a.config.adaptation_seconds=std::clamp(float(obs_data_get_double(s,"es_seconds")),.05f,20.f);
    a.config.max_change_per_second=std::clamp(float(obs_data_get_double(s,"es_slew")),.01f,2.f);
    a.width=std::clamp(int(obs_data_get_int(s,"es_width")),32,256);
    a.hz=std::clamp(obs_data_get_double(s,"es_hz"),1.,30.);
    a.strength=std::clamp(float(obs_data_get_double(s,"es_strength")),0.f,1.f);
    a.freeze=obs_data_get_bool(s,"es_freeze");std::lock_guard<std::mutex> lock(f->mutex);f->pending=a;
}
void pack_profile(Filter* f){const auto& a=f->profile.atlas();f->upload.resize(a.size());
    for(std::size_t i=0;i<a.size();++i)f->upload[i]=uint16_t(a[i]*65535.f+.5f);
}
void* create(obs_data_t* s,obs_source_t* source){auto* f=new Filter;f->source=source;update(f,s);
    obs_enter_graphics();char* errors=nullptr;f->effect=gs_effect_create(shader,"entropy-stretch.effect",&errors);
    if(errors){blog(LOG_WARNING,"[Entropy Stretch] %s",errors);bfree(errors);}
    f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);f->lattice=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    // RGBA16 UNORM avoids the OBS 32.2.1 Metal 128-bit upload bug and
    // retains finer precision than either the source or final SDR frame.
    pack_profile(f);const uint8_t* bytes=reinterpret_cast<const uint8_t*>(f->upload.data());
    f->lut=gs_texture_create(1089,33,GS_RGBA16,1,&bytes,GS_DYNAMIC);obs_leave_graphics();
    if(!f->effect||!f->capture||!f->lattice||!f->lut){destroy(f);return nullptr;}return f;
}
bool resources(Filter* f,uint32_t w,uint32_t h,int requested){
    int aw=std::min(int(w),requested),ah=std::max(1,std::min(256,int(double(aw)*h/w+.5)));
    if(f->width==w&&f->height==h&&f->aw==aw&&f->ah==ah&&f->stages[0]&&f->stages[1])return true;
    release_analysis(f);f->has_frame=false;f->width=w;f->height=h;f->aw=aw;f->ah=ah;
    f->stages[0]=gs_stagesurface_create(aw,ah,GS_RGBA);f->stages[1]=gs_stagesurface_create(aw,ah,GS_RGBA);
    // Keep the current assignment across resolution changes; discard stale samples.
    return f->stages[0]&&f->stages[1];
}
void vec(gs_effect_t* e,const char* key,float x,float y){vec2 v;vec2_set(&v,x,y);gs_effect_set_vec2(gs_effect_get_param_by_name(e,key),&v);}
bool capture(Filter* f,obs_source_t* target){
    gs_texrender_reset(f->capture);bool ok=false;gs_viewport_push();gs_projection_push();gs_matrix_push();
    if(gs_texrender_begin(f->capture,f->width,f->height)){
        vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0,0);
        gs_matrix_identity();gs_ortho(0,float(f->width),0,float(f->height),-100,100);
        gs_blend_state_push();
        gs_blend_function_separate(GS_BLEND_SRCALPHA,GS_BLEND_INVSRCALPHA,GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
        auto* parent=obs_filter_get_parent(f->source);auto flags=parent?obs_source_get_output_flags(parent):0;
        if(target==parent&&!(flags&(OBS_SOURCE_CUSTOM_DRAW|OBS_SOURCE_ASYNC)))obs_source_default_render(target);
        else obs_source_video_render(target);
        gs_blend_state_pop();gs_texrender_end(f->capture);ok=true;
    }
    gs_matrix_pop();gs_projection_pop();gs_viewport_pop();return ok;
}
void sample(Filter* f){
    gs_texrender_reset(f->lattice);gs_viewport_push();gs_projection_push();gs_matrix_push();
    gs_blend_state_push();gs_enable_blending(false);
    if(gs_texrender_begin(f->lattice,f->aw,f->ah)){
        gs_matrix_identity();gs_ortho(0,float(f->aw),0,float(f->ah),-100,100);
        auto* texture=gs_texrender_get_texture(f->capture);
        gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"image"),texture);
        vec(f->effect,"source_size",float(f->width),float(f->height));vec(f->effect,"lattice_size",float(f->aw),float(f->ah));
        // Four deterministic lattice phases; temporal profile smoothing absorbs sampling variation.
        vec(f->effect,"phase",(f->phase&1)?.75f:.25f,(f->phase&2)?.75f:.25f);f->phase=(f->phase+1)%4;
        while(gs_effect_loop(f->effect,"Sample"))gs_draw_sprite(texture,0,f->aw,f->ah);
        gs_texrender_end(f->lattice);gs_stage_texture(f->stages[f->slot],gs_texrender_get_texture(f->lattice));f->written[f->slot]=true;
        f->slot=1-f->slot;
    }
    gs_blend_state_pop();gs_matrix_pop();gs_projection_pop();gs_viewport_pop();
}
void render(void* data,gs_effect_t*){auto* f=static_cast<Filter*>(data);auto* target=obs_filter_get_target(f->source);
    if(!target){obs_source_skip_video_filter(f->source);return;}
    Settings s;{std::lock_guard<std::mutex> lock(f->mutex);s=f->pending;}
    // This is an SDR display-color transform. Do not silently quantize HDR.
    const gs_color_space preferred[]={GS_CS_SRGB,GS_CS_SRGB_16F,GS_CS_709_EXTENDED};
    auto space=obs_source_get_color_space(target,3,preferred);
    if(space!=GS_CS_SRGB||gs_get_color_space()!=GS_CS_SRGB||s.strength==0){obs_source_skip_video_filter(f->source);return;}
    auto w=obs_source_get_base_width(target),h=obs_source_get_base_height(target);
    if(!w||!h||!resources(f,w,h,s.width)){obs_source_skip_video_filter(f->source);return;}
    bool srgb=gs_framebuffer_srgb_enabled();gs_enable_framebuffer_srgb(false);
    bool linear=gs_set_linear_srgb(false);
    const uint64_t frame=obs_get_video_frame_time();
    const bool fresh=!f->has_frame||frame!=f->last_frame;
    if(fresh){
        f->has_frame=true;f->last_frame=frame;
        if(!capture(f,target)){f->has_frame=false;gs_set_linear_srgb(linear);gs_enable_framebuffer_srgb(srgb);obs_source_skip_video_filter(f->source);return;}
        auto now=std::chrono::steady_clock::now();
        double dt=f->previous.time_since_epoch().count()?std::chrono::duration<double>(now-f->previous).count():0.;f->previous=now;
        if(!s.freeze){
            // Read only a preceding frame's submission, never this render's sample.
            for(int i=0;i<2;++i)if(f->written[i]){uint8_t* mapped=nullptr;uint32_t stride=0;
                if(gs_stagesurface_map(f->stages[i],&mapped,&stride)){
                    f->profile.analyze(mapped,f->aw,f->ah,stride,s.config);gs_stagesurface_unmap(f->stages[i]);f->written[i]=false;}}
            if(f->profile.advance(dt,s.config)){pack_profile(f);gs_texture_set_image(f->lut,reinterpret_cast<const uint8_t*>(f->upload.data()),1089*4*sizeof(uint16_t),false);}
            if(std::chrono::duration<double>(now-f->last_sample).count()>=1./s.hz){sample(f);f->last_sample=now;}
        }else f->written={false,false};
    }
    gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"image"),gs_texrender_get_texture(f->capture));
    gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"profile"),f->lut);
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"strength"),s.strength);
    gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
    while(gs_effect_loop(f->effect,"Draw"))gs_draw_sprite(gs_texrender_get_texture(f->capture),0,w,h);
    gs_blend_state_pop();gs_set_linear_srgb(linear);gs_enable_framebuffer_srgb(srgb);
}
gs_color_space color_space(void* data,size_t count,const gs_color_space* preferred){
    auto* target=obs_filter_get_target(static_cast<Filter*>(data)->source);
    return target?obs_source_get_color_space(target,count,preferred):GS_CS_SRGB;
}
const char* name(void*){return "Entropy-Guided Decorrelation Stretch";}
void defaults(obs_data_t* s){obs_data_set_default_double(s,"es_strength",1);obs_data_set_default_double(s,"es_decorrelation",.65);
    obs_data_set_default_double(s,"es_allocation",.8);obs_data_set_default_double(s,"es_noise",.015);
    obs_data_set_default_double(s,"es_gain",6);obs_data_set_default_double(s,"es_seconds",1.5);
    obs_data_set_default_double(s,"es_slew",.25);obs_data_set_default_int(s,"es_width",128);
    obs_data_set_default_double(s,"es_hz",10);obs_data_set_default_bool(s,"es_freeze",false);}
obs_properties_t* properties(void*){auto* p=obs_properties_create();
    obs_properties_add_float_slider(p,"es_strength","Effect strength",0,1,.01);
    obs_properties_add_float_slider(p,"es_decorrelation","Decorrelation stretch",0,1,.01);
    obs_properties_add_float_slider(p,"es_allocation","Entropy color-range allocation",0,1,.01);
    obs_properties_add_float_slider(p,"es_seconds","Profile adaptation (seconds)",.05,20,.05);
    obs_properties_add_float_slider(p,"es_slew","Maximum color change per second",.01,2,.01);
    obs_properties_add_int_slider(p,"es_width","Lattice columns",32,256,8);
    obs_properties_add_float_slider(p,"es_hz","Analysis updates per second",1,30,1);
    obs_properties_add_float_slider(p,"es_noise","Noise floor",.001,.1,.001);
    obs_properties_add_float_slider(p,"es_gain","Maximum decorrelation gain",1,16,.25);
    obs_properties_add_bool(p,"es_freeze","Freeze current color assignment");return p;
}
}
void rvfx_register_entropy_filter(){obs_source_info info{};info.id="entropy_decorrelation_stretch";
    info.type=OBS_SOURCE_TYPE_FILTER;info.output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_SRGB;
    info.get_name=name;info.create=create;info.destroy=destroy;info.update=update;info.get_defaults=defaults;
    info.get_properties=properties;info.video_render=render;info.video_get_color_space=color_space;obs_register_source(&info);}
