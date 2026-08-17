#include <obs.h>
#include <util/base.h>

#import <AppKit/AppKit.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

namespace {

constexpr const char* CPU_ID="realtime_vector_fx";
constexpr const char* GPU_ID="realtime_vector_fx_gpu";
constexpr const char* POSTER_ID="optimal_oklch_posterizer";
#ifndef RVFX_SMOKE_WIDTH
#define RVFX_SMOKE_WIDTH 640
#endif
#ifndef RVFX_SMOKE_HEIGHT
#define RVFX_SMOKE_HEIGHT 360
#endif
constexpr std::uint32_t WIDTH=RVFX_SMOKE_WIDTH,HEIGHT=RVFX_SMOKE_HEIGHT;

struct LogState { log_handler_t previous=nullptr;void* parameter=nullptr;int errors=0; };

void capture_log(int level,const char* message,va_list arguments,void* parameter) {
    auto* state=static_cast<LogState*>(parameter);
    if(level==LOG_ERROR)++state->errors;
    if(state->previous){
        va_list copy;va_copy(copy,arguments);
        state->previous(level,message,copy,state->parameter);
        va_end(copy);
    }
}

bool registered(const char* wanted) {
    for(std::size_t i=0;;++i){
        const char* id=nullptr;
        if(!obs_enum_source_types(i,&id))return false;
        if(id&&std::strcmp(id,wanted)==0)return true;
    }
}

struct SyntheticSource {
    gs_texture_t* texture=nullptr;
    std::vector<std::uint8_t> pixels;
    std::uint32_t render_count=0;
};

const char* synthetic_name(void*) { return "RVFX synthetic smoke source"; }
void* synthetic_create(obs_data_t*,obs_source_t*) {
    auto* source=new SyntheticSource;source->pixels.resize(static_cast<std::size_t>(WIDTH)*HEIGHT*4u);
    for(std::uint32_t y=0;y<HEIGHT;++y)for(std::uint32_t x=0;x<WIDTH;++x){
        const auto q=4u*(static_cast<std::size_t>(y)*WIDTH+x);
        source->pixels[q]=static_cast<std::uint8_t>((3u*x)&255u);
        source->pixels[q+1]=static_cast<std::uint8_t>((2u*y)&255u);
        source->pixels[q+2]=static_cast<std::uint8_t>((x+y)&255u);
        source->pixels[q+3]=255;
    }
    const std::uint8_t* pixels=source->pixels.data();
    obs_enter_graphics();source->texture=gs_texture_create(WIDTH,HEIGHT,GS_RGBA,1,&pixels,GS_DYNAMIC);obs_leave_graphics();
    if(!source->texture){delete source;return nullptr;}return source;
}
void synthetic_destroy(void* data) {
    auto* source=static_cast<SyntheticSource*>(data);obs_enter_graphics();
    gs_texture_destroy(source->texture);obs_leave_graphics();delete source;
}
void synthetic_render(void* data,gs_effect_t*) {
    auto* source=static_cast<SyntheticSource*>(data);
    const auto phase=source->render_count++;
    for(std::uint32_t y=0;y<HEIGHT;++y)for(std::uint32_t x=0;x<WIDTH;++x){
        const bool cell=((((x+3u*phase)/48u)+(y/40u))&1u)!=0;
        const auto q=4u*(static_cast<std::size_t>(y)*WIDTH+x);
        source->pixels[q]=static_cast<std::uint8_t>(cell?150u+(3u*phase)%90u:20u+(5u*phase)%55u);
        source->pixels[q+1]=static_cast<std::uint8_t>(cell?45u+(7u*phase)%80u:95u+(2u*phase)%80u);
        source->pixels[q+2]=static_cast<std::uint8_t>(cell?35u+(2u*phase)%65u:145u+(5u*phase)%100u);
        source->pixels[q+3]=255;
    }
    gs_texture_set_image(source->texture,source->pixels.data(),WIDTH*4u,false);
    obs_source_draw(source->texture,0,0,WIDTH,HEIGHT,false);
}
std::uint32_t synthetic_width(void*) { return WIDTH; }
std::uint32_t synthetic_height(void*) { return HEIGHT; }

obs_source_info synthetic_info{};
void register_synthetic_source() {
    synthetic_info.id="rvfx_smoke_source";synthetic_info.type=OBS_SOURCE_TYPE_INPUT;
    synthetic_info.output_flags=OBS_SOURCE_VIDEO;synthetic_info.get_name=synthetic_name;
    synthetic_info.create=synthetic_create;synthetic_info.destroy=synthetic_destroy;
    synthetic_info.video_render=synthetic_render;synthetic_info.get_width=synthetic_width;
    synthetic_info.get_height=synthetic_height;obs_register_source(&synthetic_info);
}

bool render_filter_chain(obs_source_t* source,std::uint32_t frames,const char* capture_path,double* mean_ms,
                         std::uint32_t* changed_frames=nullptr) {
    bool pixels_seen=false;obs_enter_graphics();
    auto* output=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    auto* stage=gs_stagesurface_create(WIDTH,HEIGHT,GS_RGBA);
    std::uint32_t previous_pixel=0,changes=0;bool have_previous=false;
    const auto started=std::chrono::steady_clock::now();
    for(std::uint32_t frame=0;frame<frames;++frame){
        gs_texrender_reset(output);
        if(!gs_texrender_begin(output,WIDTH,HEIGHT))continue;
        vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0.0f,0);
        gs_matrix_identity();gs_ortho(0.0f,static_cast<float>(WIDTH),0.0f,static_cast<float>(HEIGHT),-100.0f,100.0f);
        obs_source_video_render(source);gs_texrender_end(output);
        if(changed_frames){
            gs_stage_texture(stage,gs_texrender_get_texture(output));
            std::uint8_t* mapped=nullptr;std::uint32_t mapped_stride=0;
            if(gs_stagesurface_map(stage,&mapped,&mapped_stride)){
                std::uint32_t pixel=0;std::memcpy(&pixel,mapped,sizeof(pixel));
                if(have_previous&&pixel!=previous_pixel)++changes;
                previous_pixel=pixel;have_previous=true;
                gs_stagesurface_unmap(stage);
            }
        }
    }
    if(changed_frames)*changed_frames=changes;
    gs_stage_texture(stage,gs_texrender_get_texture(output));
    std::uint8_t* pixels=nullptr;std::uint32_t stride=0;
    if(gs_stagesurface_map(stage,&pixels,&stride)){
        std::FILE* capture=capture_path?std::fopen(capture_path,"wb"):nullptr;
        if(capture)std::fprintf(capture,"P6\n%u %u\n255\n",WIDTH,HEIGHT);
        for(std::uint32_t y=0;y<HEIGHT&&!pixels_seen;y+=17)
            for(std::uint32_t x=0;x<WIDTH;x+=19){
                const auto* p=pixels+static_cast<std::size_t>(y)*stride+4u*x;
                if(p[0]||p[1]||p[2]||p[3]){pixels_seen=true;break;}
            }
        if(capture){
            for(std::uint32_t y=0;y<HEIGHT;++y)for(std::uint32_t x=0;x<WIDTH;++x){
                const auto* p=pixels+static_cast<std::size_t>(y)*stride+4u*x;
                const std::uint8_t rgb[3]={p[0],p[1],p[2]};std::fwrite(rgb,1,3,capture);
            }
            std::fclose(capture);
        }
        gs_stagesurface_unmap(stage);
    }
    const auto finished=std::chrono::steady_clock::now();
    if(mean_ms)*mean_ms=std::chrono::duration<double,std::milli>(finished-started).count()/frames;
    gs_stagesurface_destroy(stage);gs_texrender_destroy(output);obs_leave_graphics();return pixels_seen;
}

} // namespace

int main(int argc,char** argv) {
    @autoreleasepool {
        LogState logs;base_get_log_handler(&logs.previous,&logs.parameter);
        base_set_log_handler(capture_log,&logs);
        [NSApplication sharedApplication];
        const char* module=argc>1?argv[1]:
            "/tmp/rvfx-obs/realtime-vector-fx.plugin/Contents/MacOS/realtime-vector-fx";
        const char* graphics=argc>2?argv[2]:
            "/Applications/OBS.app/Contents/Frameworks/libobs-metal.dylib";
        const char* capture=argc>3?argv[3]:nullptr;
        if(!obs_startup("en-US",nullptr,nullptr)){
            std::fprintf(stderr,"obs_startup failed\n");return 2;
        }
        obs_video_info video{};
        video.graphics_module=graphics;video.fps_num=30;video.fps_den=1;
        video.base_width=WIDTH;video.base_height=HEIGHT;video.output_width=WIDTH;video.output_height=HEIGHT;
        video.output_format=VIDEO_FORMAT_RGBA;video.adapter=0;video.gpu_conversion=true;
        video.colorspace=VIDEO_CS_709;video.range=VIDEO_RANGE_FULL;video.scale_type=OBS_SCALE_BILINEAR;
        const int video_result=obs_reset_video(&video);
        if(video_result!=OBS_VIDEO_SUCCESS){
            std::fprintf(stderr,"obs_reset_video failed: %d\n",video_result);obs_shutdown();return 3;
        }
        obs_module_t* loaded=nullptr;
        const int open_result=obs_open_module(&loaded,module,"/tmp/rvfx-obs");
        if(open_result!=MODULE_SUCCESS||!obs_init_module(loaded)){
            std::fprintf(stderr,"module load failed: %d\n",open_result);obs_shutdown();return 4;
        }
        register_synthetic_source();
        if(!registered(CPU_ID)||!registered(GPU_ID)||!registered(POSTER_ID)){
            std::fprintf(stderr,"expected filter registration missing\n");obs_shutdown();return 5;
        }
        obs_source_t* cpu=obs_source_create_private(CPU_ID,"rvfx-cpu-smoke",nullptr);
        obs_source_t* gpu=obs_source_create_private(GPU_ID,"rvfx-gpu-smoke",nullptr);
        obs_source_t* poster=obs_source_create_private(POSTER_ID,"rvfx-poster-smoke",nullptr);
        obs_source_t* source=obs_source_create_private("rvfx_smoke_source","rvfx-input-smoke",nullptr);
        double baseline_ms=0.0,filtered_ms=0.0;
        const bool baseline_warm=source&&render_filter_chain(source,12,nullptr,nullptr);
        const bool baseline=baseline_warm&&render_filter_chain(source,120,nullptr,&baseline_ms);
        if(source&&gpu)obs_source_filter_add(source,gpu);
        const bool warmed=source&&gpu&&render_filter_chain(source,12,nullptr,nullptr);
        const bool rendered=warmed&&render_filter_chain(source,120,capture,&filtered_ms);
        std::uint32_t changed_frames=0;
        if(gpu){
            obs_data_t* settings=obs_source_get_settings(gpu);
            obs_data_set_int(settings,"rvfx_colors",2);
            obs_data_set_int(settings,"rvfx_segments",64);
            obs_data_set_int(settings,"rvfx_glyphs",0);
            obs_data_set_double(settings,"rvfx_persistence",0.0);
            obs_source_update(gpu,settings);obs_data_release(settings);
        }
        const bool moving=source&&gpu&&
            render_filter_chain(source,24,nullptr,nullptr,&changed_frames)&&changed_frames>=12;
        bool modes=true;
        const char* mode_names[]={"phosphor","liquid","emboss"};
        for(int mode=0;mode<3&&source&&gpu;++mode){
            obs_data_t* settings=obs_source_get_settings(gpu);
            obs_data_set_int(settings,"rvfx_effect",mode);
            obs_data_set_int(settings,"rvfx_glyph_motion",mode);
            obs_data_set_double(settings,"rvfx_persistence",0.0);
            obs_source_update(gpu,settings);obs_data_release(settings);
            modes=modes&&render_filter_chain(source,1,nullptr,nullptr);
            settings=obs_source_get_settings(gpu);
            obs_data_set_double(settings,"rvfx_persistence",.72+.08*mode);
            obs_source_update(gpu,settings);obs_data_release(settings);
            std::string mode_capture;
            if(capture){mode_capture=capture;const auto dot=mode_capture.rfind('.');
                if(dot!=std::string::npos)mode_capture.resize(dot);
                mode_capture+="-";mode_capture+=mode_names[mode];mode_capture+=".ppm";}
            modes=modes&&render_filter_chain(source,8,capture?mode_capture.c_str():nullptr,nullptr);
        }
        if(source&&gpu)obs_source_filter_remove(source,gpu);
        if(source&&poster)obs_source_filter_add(source,poster);
        if(poster){
            obs_data_t* settings=obs_source_get_settings(poster);
            obs_data_set_double(settings,"rvfx_contour_strength",.45);
            obs_data_set_double(settings,"rvfx_interior_ink",.18);
            obs_data_set_double(settings,"rvfx_line_reach",1.25);
            obs_data_set_double(settings,"rvfx_look_saturation",1.18);
            obs_data_set_double(settings,"rvfx_look_contrast",1.12);
            obs_source_update(poster,settings);obs_data_release(settings);
        }
        std::uint32_t poster_changes=0;
        double poster_ms=0.0;std::string poster_capture;
        if(capture){poster_capture=capture;const auto dot=poster_capture.rfind('.');
            if(dot!=std::string::npos)poster_capture.resize(dot);poster_capture+="-poster-look.ppm";}
        const bool posterized=source&&poster&&render_filter_chain(source,24,
            capture?poster_capture.c_str():nullptr,&poster_ms,&poster_changes)&&
            poster_changes>=12;
        if(!cpu||!gpu||!poster||!source||!baseline||!rendered||!moving||!modes||!posterized||logs.errors){
            std::fprintf(stderr,"filter creation failed: cpu=%p gpu=%p poster=%p moving=%d changes=%u poster_changes=%u errors=%d\n",
                static_cast<void*>(cpu),static_cast<void*>(gpu),static_cast<void*>(poster),moving,changed_frames,
                poster_changes,logs.errors);
            if(source&&poster)obs_source_filter_remove(source,poster);
            if(source)obs_source_release(source);if(cpu)obs_source_release(cpu);if(gpu)obs_source_release(gpu);
            if(poster)obs_source_release(poster);
            obs_shutdown();return 6;
        }
        obs_source_filter_remove(source,poster);obs_source_release(source);
        obs_source_release(poster);obs_source_release(gpu);obs_source_release(cpu);obs_shutdown();
        std::printf("rvfx Metal %ux%u baseline %.3f ms, filtered %.3f ms, overhead %.3f ms\n",
            WIDTH,HEIGHT,baseline_ms,filtered_ms,filtered_ms-baseline_ms);
        std::printf("rvfx poster look total %.3f ms/frame\n",poster_ms);
        std::puts("rvfx OBS module, Optimal OKLCH Posterizer, all GPU effects/motions, Metal shaders, and changing filter chains passed");
        return 0;
    }
}
