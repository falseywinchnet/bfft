// Reuse the tested OBS source/renderer without changing production timing APIs.
#define main entropy_existing_regression_main
#include "entropy_obs_smoke.mm"
#undef main
#include <fstream>
#include <iomanip>
#include <numeric>
int main(int argc,char** argv){@autoreleasepool {
    if(argc<4)return 2;const bool submit_only=argc>4&&std::strcmp(argv[4],"--submit-only")==0;[NSApplication sharedApplication];base_get_log_handler(&logger,&logger_arg);base_set_log_handler(log,nullptr);
    check(obs_startup("en-US",nullptr,nullptr),"OBS startup");obs_video_info v{};v.graphics_module=argv[2];v.fps_num=60;v.fps_den=1;
    v.base_width=W;v.base_height=H;v.output_width=W;v.output_height=H;v.output_format=VIDEO_FORMAT_RGBA;v.gpu_conversion=true;v.colorspace=VIDEO_CS_709;v.range=VIDEO_RANGE_FULL;
    check(obs_reset_video(&v)==OBS_VIDEO_SUCCESS,"Metal startup");obs_module_t* module=nullptr;
    check(obs_open_module(&module,argv[1],"/tmp")==MODULE_SUCCESS&&obs_init_module(module),"module load");register_source();
    auto* source=obs_source_create_private("entropy_test_source","entropy-benchmark-source",nullptr);
    auto* s=obs_data_create();
    // The user's saved settings; unspecified controls retain their defaults.
    obs_data_set_double(s,"es_decorrelation",.09);obs_data_set_double(s,"es_allocation",.11);
    obs_data_set_double(s,"es_gain",16);obs_data_set_int(s,"es_width",256);
    auto* filter=obs_source_create_private("entropy_decorrelation_stretch","entropy-benchmark-filter",s);obs_data_release(s);
    check(source&&filter,"source and filter");obs_source_filter_add(source,filter);
    {
        Renderer renderer;std::vector<double> timings;std::ofstream hashes(argv[3]);
        for(int frame=0;frame<180;++frame){
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const bool read=!submit_only||frame==179;
            check(renderer.render(source,read),"render");
            if(submit_only){obs_enter_graphics();auto flush_start=std::chrono::steady_clock::now();gs_flush();renderer.elapsed+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-flush_start).count();obs_leave_graphics();}
            if(frame>=30&&(!submit_only||frame<179))timings.push_back(renderer.elapsed);
            if(!read)continue;
            uint64_t hash=1469598103934665603ULL;for(uint8_t c:renderer.pixels){hash^=c;hash*=1099511628211ULL;}
            hashes<<std::hex<<hash<<'\n';
        }
        auto sorted=timings;std::sort(sorted.begin(),sorted.end());
        std::printf("BENCH {\"frames\":%zu,\"mean_ms\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"max_ms\":%.6f}\n",timings.size(),
            std::accumulate(timings.begin(),timings.end(),0.)/timings.size(),sorted[sorted.size()/2],sorted[sorted.size()*95/100],sorted.back());
        check(!errors,"OBS errors");obs_source_filter_remove(source,filter);
    }
    obs_source_release(filter);obs_source_release(source);obs_shutdown();return 0;
}}
