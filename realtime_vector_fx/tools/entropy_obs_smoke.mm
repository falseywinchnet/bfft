// Actual libobs/Metal regression, using the existing RVFX smoke harness pattern.
#include <obs.h>
#include <util/base.h>
#include "rvfx/entropy_stretch.hpp"
#import <AppKit/AppKit.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
namespace {
#ifndef ES_SMOKE_WIDTH
#define ES_SMOKE_WIDTH 640
#define ES_SMOKE_HEIGHT 360
#endif
constexpr uint32_t W=ES_SMOKE_WIDTH,H=ES_SMOKE_HEIGHT;
int errors=0;log_handler_t logger=nullptr;void* logger_arg=nullptr;
void log(int level,const char* message,va_list args,void*){if(level==LOG_ERROR)++errors;if(logger)logger(level,message,args,logger_arg);}
struct Source {gs_texture_t* texture=nullptr;gs_color_space space=GS_CS_SRGB;};
void* create(obs_data_t*,obs_source_t*){
    std::vector<uint8_t> pixels(W*H*4,255);
    for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x){auto i=4*(y*W+x);int xx=int(x*128/W),yy=int(y*72/H);
        if(x<W/2){pixels[i]=30+15*x/(W/2);pixels[i+1]=pixels[i];pixels[i+2]=pixels[i];}
        else{pixels[i]=140+(xx*13+yy*7)%50;pixels[i+1]=130+(xx*7+yy*17)%50;pixels[i+2]=120+(xx*19+yy*3)%50;}
        // Exercise premultiplied alpha at capture and final composition.
        if(y<20)pixels[i+3]=128;if(y>=20&&y<30)pixels[i+3]=0;
    }
    auto* s=new Source;const uint8_t* bytes=pixels.data();obs_enter_graphics();s->texture=gs_texture_create(W,H,GS_RGBA,1,&bytes,0);obs_leave_graphics();return s;
}
void destroy(void* data){auto* s=static_cast<Source*>(data);obs_enter_graphics();gs_texture_destroy(s->texture);obs_leave_graphics();delete s;}
void draw(void* data,gs_effect_t*){obs_source_draw(static_cast<Source*>(data)->texture,0,0,W,H,false);}
const char* name(void*){return "Entropy regression source";}
void source_update(void* data,obs_data_t* settings){static_cast<Source*>(data)->space=static_cast<gs_color_space>(obs_data_get_int(settings,"space"));}
gs_color_space source_space(void* data,size_t,const gs_color_space*){return static_cast<Source*>(data)->space;}
uint32_t width(void*){return W;}uint32_t height(void*){return H;}
void register_source(){obs_source_info i{};i.id="entropy_test_source";i.type=OBS_SOURCE_TYPE_INPUT;i.output_flags=OBS_SOURCE_VIDEO;
    i.create=create;i.destroy=destroy;i.update=source_update;i.video_get_color_space=source_space;i.video_render=draw;i.get_name=name;i.get_width=width;i.get_height=height;obs_register_source(&i);}
struct Renderer {
    gs_texrender_t* out=nullptr;gs_stagesurf_t* stage=nullptr;std::vector<uint8_t> pixels;double elapsed=0;
    Renderer(){obs_enter_graphics();out=gs_texrender_create(GS_RGBA,GS_ZS_NONE);stage=gs_stagesurface_create(W,H,GS_RGBA);obs_leave_graphics();pixels.resize(W*H*4);}
    ~Renderer(){obs_enter_graphics();gs_stagesurface_destroy(stage);gs_texrender_destroy(out);obs_leave_graphics();}
    bool render(obs_source_t* source,bool read=true){
        obs_enter_graphics();auto start=std::chrono::steady_clock::now();gs_texrender_reset(out);
        if(!gs_texrender_begin(out,W,H)){obs_leave_graphics();return false;}
        gs_blend_state_push();gs_blend_function_separate(GS_BLEND_SRCALPHA,GS_BLEND_INVSRCALPHA,GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
        vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0,0);gs_matrix_identity();gs_ortho(0,float(W),0,float(H),-100,100);
        obs_source_video_render(source);gs_blend_state_pop();gs_texrender_end(out);
        bool ok=true;if(read){gs_stage_texture(stage,gs_texrender_get_texture(out));uint8_t* data=nullptr;uint32_t stride=0;
            if(gs_stagesurface_map(stage,&data,&stride)){for(uint32_t y=0;y<H;++y)std::memcpy(pixels.data()+y*W*4,data+y*stride,W*4);gs_stagesurface_unmap(stage);}else ok=false;}
        elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();obs_leave_graphics();return ok;
    }
};
void check(bool value,const char* why){if(!value){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(10);}}
void settings(obs_source_t* filter,bool freeze,double strength=1){auto* s=obs_source_get_settings(filter);obs_data_set_double(s,"es_seconds",.05);
    obs_data_set_double(s,"es_slew",2);obs_data_set_double(s,"es_hz",30);obs_data_set_double(s,"es_strength",strength);obs_data_set_bool(s,"es_freeze",freeze);
    obs_source_update(filter,s);obs_data_release(s);
    // Video-filter settings are applied by OBS on its next video tick.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
void save(const char* path,const std::vector<uint8_t>& pixels){if(!path)return;auto* f=std::fopen(path,"wb");if(!f)return;std::fprintf(f,"P6\n%u %u\n255\n",W,H);
    for(std::size_t i=0;i<pixels.size();i+=4)std::fwrite(pixels.data()+i,1,3,f);std::fclose(f);}
}
int main(int argc,char** argv){@autoreleasepool {
    if(argc<3)return 2;[NSApplication sharedApplication];base_get_log_handler(&logger,&logger_arg);base_set_log_handler(log,nullptr);
    check(obs_startup("en-US",nullptr,nullptr),"OBS startup");obs_video_info v{};v.graphics_module=argv[2];v.fps_num=60;v.fps_den=1;
    v.base_width=W;v.base_height=H;v.output_width=W;v.output_height=H;v.output_format=VIDEO_FORMAT_RGBA;v.gpu_conversion=true;v.colorspace=VIDEO_CS_709;v.range=VIDEO_RANGE_FULL;
    check(obs_reset_video(&v)==OBS_VIDEO_SUCCESS,"Metal startup");obs_module_t* module=nullptr;
    check(obs_open_module(&module,argv[1],"/tmp")==MODULE_SUCCESS&&obs_init_module(module),"module load");register_source();
    auto* source=obs_source_create_private("entropy_test_source","entropy-source",nullptr);
    auto* filter=obs_source_create_private("entropy_decorrelation_stretch","entropy-filter",nullptr);check(source&&filter,"filter registration/create");
    {
        Renderer renderer;check(renderer.render(source),"baseline");auto baseline=renderer.pixels;
        double baseline_ms=0;for(int i=0;i<30;++i){check(renderer.render(source),"baseline repeat");baseline_ms+=renderer.elapsed/30;}
        settings(filter,true);obs_source_filter_add(source,filter);check(renderer.render(source),"identity draw");int identity_error=0;
        for(std::size_t i=0;i<baseline.size();++i)identity_error=std::max(identity_error,std::abs(int(baseline[i])-renderer.pixels[i]));
        std::printf("identity maximum byte error %d\n",identity_error);check(identity_error<=1,"identity LUT preserves gamma and alpha");
        settings(filter,false);double total=0,peak=0;
        for(int frame=0;frame<150;++frame){std::this_thread::sleep_for(std::chrono::milliseconds(20));check(renderer.render(source),"adaptive draw");total+=renderer.elapsed;peak=std::max(peak,renderer.elapsed);}
        auto result=renderer.pixels;double difference=0;int alpha_error=0;
        for(std::size_t i=0;i<result.size();i+=4){for(int c=0;c<3;++c)difference+=std::abs(int(result[i+c])-baseline[i+c]);alpha_error=std::max(alpha_error,std::abs(int(result[i+3])-baseline[i+3]));}
        difference/=(W*H*3);check(difference>2,"profile actually updates and affects pixels");check(alpha_error<=1,"output alpha unchanged");
        // These source cells are 5x5; all four lattice phases see the same
        // sample except a slow gray ramp (sub-byte differences allowed below).
        std::vector<uint8_t> lattice(128*72*4);
        for(int y=0;y<72;++y)for(int x=0;x<128;++x)std::copy_n(baseline.data()+4*(int((y+.25)*H/72)*W+int((x+.25)*W/128)),4,lattice.data()+4*(y*128+x));
        rvfx::entropy::Profile oracle;rvfx::entropy::Config cfg;check(oracle.analyze(lattice.data(),128,72,512,cfg),"CPU oracle");
        for(int i=0;i<1200;++i)oracle.advance(1./60,cfg);
        double cpu_error=0;int max_error=0;
        for(std::size_t i=0;i<result.size();i+=4){float a=baseline[i+3]/255.f;if(a<.5f)continue;
            rvfx::entropy::Color c;for(int j=0;j<3;++j)c[j]=std::min(1.f,baseline[i+j]/(255*a));auto expected=oracle.apply(c);
            for(int j=0;j<3;++j){int e=std::abs(int(std::lround(expected[j]*a*255))-result[i+j]);cpu_error+=e;max_error=std::max(max_error,e);}}
        cpu_error/=(W*H*3);std::printf("CPU/GPU mean byte error %.4f maximum %d\n",cpu_error,max_error);check(cpu_error<1.2&&max_error<=5,"GPU atlas agrees with CPU transform");
        settings(filter,true);check(renderer.render(source),"freeze start");auto frozen=renderer.pixels;
        for(int i=0;i<10;++i){std::this_thread::sleep_for(std::chrono::milliseconds(20));check(renderer.render(source),"freeze draw");check(renderer.pixels==frozen,"freeze holds profile exactly");}
        settings(filter,false,0);check(renderer.render(source),"bypass draw");check(renderer.pixels==baseline,"zero strength exact bypass");
        if(argc>3)save(argv[3],result);
        // New controls are optional, with a neutral migration for existing scenes.
        auto* props=obs_source_properties(filter);auto* group=obs_properties_get(props,"es_chains");
        check(group&&obs_property_group_type(group)==OBS_GROUP_CHECKABLE,"checkable Chains group");
        auto* chain_props=obs_property_group_content(group);
        check(obs_properties_get(chain_props,"es_rgb_amount")&&obs_properties_get(chain_props,"es_brightness_amount")&&
              obs_properties_get(chain_props,"es_contrast_amount"),"independent amount controls");
        check(obs_property_list_item_count(obs_properties_get(chain_props,"es_chain_order"))==6,"six application orders");
        obs_properties_destroy(props);
        auto* controls=obs_source_get_settings(filter);
        check(!obs_data_get_bool(controls,"es_chains")&&obs_data_get_double(controls,"es_rgb_amount")==1&&
              obs_data_get_double(controls,"es_brightness_amount")==0&&obs_data_get_double(controls,"es_contrast_amount")==0,"neutral saved-setting defaults");
        obs_data_set_bool(controls,"es_chains",true);obs_data_set_double(controls,"es_strength",1);
        obs_data_set_bool(controls,"es_freeze",false);obs_data_set_double(controls,"es_rgb_amount",.7);
        obs_data_set_double(controls,"es_brightness_amount",.2);obs_data_set_double(controls,"es_contrast_amount",.2);
        cfg.chains=true;cfg.rgb_amount=.7;cfg.brightness_amount=.2;cfg.contrast_amount=.2;
        for(int order=0;order<6;++order){
            obs_data_set_int(controls,"es_chain_order",order);obs_source_update(filter,controls);
            for(int frame=0;frame<75;++frame){std::this_thread::sleep_for(std::chrono::milliseconds(20));check(renderer.render(source),"chain draw");}
            cfg.chain_order=rvfx::entropy::ChainOrder(order);oracle.analyze(lattice.data(),128,72,512,cfg);
            for(int i=0;i<1200;++i)oracle.advance(1./60,cfg);
            double sum=0;int maximum=0,alpha=0;
            for(std::size_t i=0;i<baseline.size();i+=4){float a=baseline[i+3]/255.f;alpha=std::max(alpha,std::abs(int(baseline[i+3])-renderer.pixels[i+3]));if(a<.5f)continue;
                rvfx::entropy::Color in;for(int c=0;c<3;++c)in[c]=std::min(1.f,baseline[i+c]/(255*a));auto expected=oracle.apply(in);
                for(int c=0;c<3;++c){int e=std::abs(int(std::lround(expected[c]*a*255))-renderer.pixels[i+c]);sum+=e;maximum=std::max(maximum,e);}}
            std::printf("chain order %d CPU/GPU mean byte error %.4f maximum %d alpha %d\n",order,sum/(W*H*3),maximum,alpha);
            check(sum/(W*H*3)<1.2&&maximum<=5&&alpha<=1,"composed CPU/GPU agreement and alpha");
        }
        settings(filter,true);check(renderer.render(source),"chain freeze");auto frozen_chain=renderer.pixels;
        for(int i=0;i<5;++i){std::this_thread::sleep_for(std::chrono::milliseconds(20));check(renderer.render(source),"chain frozen frame");check(renderer.pixels==frozen_chain,"chain freeze exact");}
        obs_data_set_double(controls,"es_rgb_amount",0);obs_data_set_double(controls,"es_brightness_amount",0);
        obs_data_set_double(controls,"es_contrast_amount",0);obs_data_set_bool(controls,"es_freeze",false);obs_source_update(filter,controls);
        for(int i=0;i<75;++i){std::this_thread::sleep_for(std::chrono::milliseconds(20));check(renderer.render(source),"zero chain amounts");}
        int zero_error=0;for(std::size_t i=0;i<baseline.size();++i)zero_error=std::max(zero_error,std::abs(int(baseline[i])-renderer.pixels[i]));
        check(zero_error<=1,"zero chain amounts settle to identity");obs_data_release(controls);
        std::puts("Chains defaults, properties, six orders, independent amounts, GPU composition, alpha, freeze and identity passed");
        settings(filter,true,1);
        for(auto space:{GS_CS_SRGB_16F,GS_CS_709_EXTENDED}){
            auto* s=obs_source_get_settings(source);obs_data_set_int(s,"space",space);obs_source_update(source,s);obs_data_release(s);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            check(obs_source_get_color_space(filter,1,&space)==space,"upstream color space forwarded");
            obs_source_filter_remove(source,filter);check(renderer.render(source),"non-SDR baseline");auto unfiltered=renderer.pixels;
            obs_source_filter_add(source,filter);check(renderer.render(source),"non-SDR bypass");
            check(renderer.pixels==unfiltered,"floating SDR/HDR bypass exact");
        }
        check(errors==0,"no libobs errors");
        std::printf("Metal %ux%u static source + filter + full readback: baseline %.3f ms, adaptive mean %.3f ms, peak %.3f ms; mean change %.3f codes; alpha error %d\n",W,H,baseline_ms,total/150,peak,difference,alpha_error);
        std::puts("entropy OBS registration, shader, identity/gamma/alpha, adaptation, CPU/GPU equivalence, freeze, zero-strength bypass, and floating SDR/HDR passthrough passed");
        obs_source_filter_remove(source,filter);
    }
    obs_source_release(filter);obs_source_release(source);obs_shutdown();return 0;
}}
