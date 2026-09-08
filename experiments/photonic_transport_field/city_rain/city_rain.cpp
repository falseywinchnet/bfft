#define main regime_scene_program_main
#include "../native/regime_scene_native.cpp"
#undef main
#include <barrier>
#include <functional>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static std::atomic<bool> rain_city_frozen{false};
static std::atomic<std::uint64_t> rain_forbidden_city_calls{0};
#include "city_scene.hpp"
#include "volume_field.hpp"
#include "rain_normals.hpp"

struct RowWorkers {
    unsigned count;std::barrier<> start,finish;std::vector<std::thread> workers;
    std::atomic<int> next{0};int rows=0;bool stop=false;std::function<void(int)> work;
    RowWorkers():count(std::min(8u,std::max(1u,std::thread::hardware_concurrency()))),start(count+1),finish(count+1){
        for(unsigned i=0;i<count;++i)workers.emplace_back([&]{for(;;){start.arrive_and_wait();if(stop)return;
            for(;;){int y=next.fetch_add(1);if(y>=rows)break;work(y);}finish.arrive_and_wait();}});
    }
    void run(int n,std::function<void(int)> f){rows=n;work=std::move(f);next=0;start.arrive_and_wait();finish.arrive_and_wait();}
    ~RowWorkers(){stop=true;start.arrive_and_wait();for(auto& w:workers)w.join();}
};
std::array<std::uint8_t,65536> make_rain_tone_lut(){std::array<std::uint8_t,65536> lut;
    for(int i=0;i<65536;++i)lut[i]=tone_byte(i*(16./65535));return lut;}
void render_frozen_volume(const FrozenVolumeField& field,const RainNormalTexture& normals,float time,
    RowWorkers& pool,std::vector<std::uint8_t>& pixels,std::vector<std::uint8_t>* normal_image=nullptr){
    static const auto tone=make_rain_tone_lut();const int width=field.output_width,height=field.output_height;
    pool.run(height,[&](int y){for(int x=0;x<width;++x){auto n=normals.normal(x,y,time);
        const RGB color=field.gather(x,y,n[0],n[1]);const std::size_t offset=(std::size_t(y)*width+x)*3;
        for(int c=0;c<3;++c)pixels[offset+c]=tone[std::clamp(int(color[c]*(65535./16)+.5),0,65535)];
        if(normal_image){const Vec3 unit_normal=unit(Vec3{n[0],n[1],1});
            (*normal_image)[offset]=std::uint8_t(std::clamp(127.5+127.5*unit_normal.x,0.,255.));
            (*normal_image)[offset+1]=std::uint8_t(std::clamp(127.5+127.5*unit_normal.y,0.,255.));
            (*normal_image)[offset+2]=std::uint8_t(std::clamp(127.5+127.5*unit_normal.z,0.,255.));}
    }});
}
void rain_ppm(const std::string& path,int w,int h,const std::vector<std::uint8_t>& pixels){
    std::ofstream out(path,std::ios::binary);out<<"P6\n"<<w<<" "<<h<<"\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());update_require(bool(out),"image write failed");
}
FrozenVolumeField load_volume_field(const std::string& path){FrozenVolumeField field;std::ifstream in(path,std::ios::binary);
    update_require(bool(in),"cannot open frozen field");std::uint64_t h[8];in.read(reinterpret_cast<char*>(h),sizeof(h));
    update_require(h[0]==0x31564c464e494152ULL,"invalid frozen field format");
    field.width=int(h[1]);field.height=int(h[2]);field.normals=int(h[3]);field.output_width=int(h[4]);field.output_height=int(h[5]);field.generation=h[6];field.city_queries=h[7];
    update_require(field.width>1&&field.height>1&&field.normals>1&&field.width<4096&&field.height<4096&&field.normals<129,"invalid field dimensions");
    in.read(reinterpret_cast<char*>(&field.normal_extent),sizeof(double));
    update_require(bool(in),"truncated field header");
    const std::size_t bytes=sizeof(h)+sizeof(double)+field.sample_count()*sizeof(PackedRadiance);
    const int fd=open(path.c_str(),O_RDONLY);update_require(fd>=0,"cannot map frozen field");
    struct stat info{};if(fstat(fd,&info)!=0||std::uint64_t(info.st_size)!=bytes){close(fd);throw std::runtime_error("truncated frozen field");}
    void* mapped=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    update_require(mapped!=MAP_FAILED,"read-only field mapping failed");
    field.mapped_owner=std::shared_ptr<void>(mapped,[bytes](void* address){munmap(address,bytes);});
    field.mapped_response=reinterpret_cast<const PackedRadiance*>(static_cast<const char*>(mapped)+sizeof(h)+sizeof(double));
    return field;
}
void test_city_rain(){
    FrozenVolumeField f;f.width=f.height=f.normals=3;f.output_width=f.output_height=5;
    f.response.resize(81);f.generation=17;
    for(int b=0;b<3;++b)for(int a=0;a<3;++a)for(int y=0;y<3;++y)for(int x=0;x<3;++x){
        const float v=x+.125f*y+.25f*a+.5f*b;f.response[f.sample_index(x,y,a,b)]={_Float16(v),_Float16(v*.5),_Float16(v*.25)};}
    for(int i=0;i<100;++i){double x=(i%10)*4/9.,y=(i/10)*4/9.,nx=.4*std::sin(i),ny=.4*std::cos(i);
        const double expected=x*.5+.125*y*.5+.25*(nx/f.normal_extent+1)+.5*(ny/f.normal_extent+1);
        const auto v=f.gather(x,y,nx,ny);update_require(std::abs(v[0]-expected)<1e-12,"four-dimensional affine gather failed");}
    f.camera=make_camera(5,5);
    for(int y=0;y<5;++y)for(int x=0;x<5;++x){auto site=f.site(x,y);Vec3 outgoing;
        update_require(refract(site.internal,-f.camera.forward,f.index,outgoing),"flat volume exit failed");
        update_require(norm2(outgoing-site.incident)<1e-24,"parallel slab direction not preserved");}
    RainNormalTexture a(320,180),b(320,180);
    for(int frame=0;frame<600;++frame){const float time=frame/30.f;a.update(time,1.f/30);b.update(time,1.f/30);
        for(int y=0;y<180;y+=13)for(int x=0;x<320;x+=13){const auto p=a.normal(x,y,time),q=b.normal(x,y,time);
            update_require(p==q,"rain simulation is not deterministic");
            update_require(std::isfinite(p[0])&&std::isfinite(p[1])&&std::abs(p[0])<.551&&std::abs(p[1])<.551,"normal left response domain");}}
    const auto path=(std::filesystem::temp_directory_path()/("city_rain_test_"+std::to_string(Clock::now().time_since_epoch().count())+".field")).string();
    save_volume_field(f,path);auto restored=load_volume_field(path);std::filesystem::remove(path);
    update_require(f.checksum()==restored.checksum()&&restored.generation==f.generation,"frozen field round-trip failed");
    CityScene city;BeamField beam;TransportField transport;TraceContext ctx{city.scene,beam,transport};
    rain_city_frozen=true;bool caught=false;try{evaluate_city(city,ctx,{});}catch(const std::logic_error&){caught=true;}
    update_require(caught&&rain_forbidden_city_calls==1,"frozen city evaluation guard failed");rain_city_frozen=false;rain_forbidden_city_calls=0;
    std::cerr<<"city rain checks passed: 4D gather, slab identity, deterministic bounded normals, field round-trip, freeze guard\n";
}

#ifndef CITY_RAIN_LIBRARY
int main(int argc,char** argv){try{
    std::string mode=argc>1?argv[1]:"",field_path,stats_path,images_prefix;
    int width=1280,height=720,fw=641,fh=361,normals=49,fps=60;double duration=18;
    for(int i=2;i<argc;++i){std::string arg=argv[i];auto value=[&](){update_require(i+1<argc,"missing option value");return std::string(argv[++i]);};
        if(arg=="--field")field_path=value();else if(arg=="--stats")stats_path=value();else if(arg=="--images")images_prefix=value();
        else if(arg=="--width")width=std::stoi(value());else if(arg=="--height")height=std::stoi(value());
        else if(arg=="--field-width")fw=std::stoi(value());else if(arg=="--field-height")fh=std::stoi(value());
        else if(arg=="--normals")normals=std::stoi(value());else if(arg=="--fps")fps=std::stoi(value());else if(arg=="--duration")duration=std::stod(value());
        else throw std::invalid_argument("unknown option "+arg);
    }
    if(mode=="test"){test_city_rain();return 0;}
    update_require(!field_path.empty()&&!stats_path.empty(),"--field and --stats are required");
    if(mode=="bake"){
        update_require(width>1&&height>1&&fw>1&&fh>1&&normals>=3&&normals%2==1,"invalid bake grid");
        const auto start=Clock::now();CityScene city=make_rain_city();std::cerr<<"city "<<city.buildings<<" buildings, "<<city.scene.primitives.size()<<" primitives\n";
        auto beams=compile_beam_field(city.scene);auto transport=compile_transport_field(city.scene,beams,true);
        const double solve_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        std::cerr<<"static transport solved in "<<solve_ms<<" ms\n";TraceContext ctx{city.scene,beams,transport};
        FrozenVolumeField field;field.width=fw;field.height=fh;field.normals=normals;field.output_width=width;field.output_height=height;
        field.camera=make_look_camera(width,height,{10,7.5,17},{0,2.5,-9},{0,1,0},23.5);
        const auto field_start=Clock::now();compile_volume_field(field,city,ctx);
        const double field_ms=std::chrono::duration<double,std::milli>(Clock::now()-field_start).count();
        std::cerr<<"field complete: "<<field.response.size()<<" responses, "<<field_ms<<" ms\n";
        std::uint64_t validation_queries=0;double squared=0;int maximum=0;constexpr int checks=512;
        std::uint32_t rng=81573;auto random=[&](){rng=rng*1664525+1013904223;return (rng>>8)/double(1u<<24);};
        for(int i=0;i<checks;++i){const double x=random()*(width-1),y=random()*(height-1),nx=(random()*2-1)*.45,ny=(random()*2-1)*.45;
            const RGB exact=evaluate_lens_response(field,field.site(x,y),nx,ny,city,ctx,validation_queries),approx=field.gather(x,y,nx,ny);
            for(int c=0;c<3;++c){const int error=int(tone_byte(exact[c]))-int(tone_byte(approx[c]));squared+=error*error;maximum=std::max(maximum,std::abs(error));}}
        const auto checksum=field.checksum();save_volume_field(field,field_path);
        RowWorkers pool;RainNormalTexture rain(width,height);std::vector<std::uint8_t> pixels(std::size_t(width)*height*3);
        render_frozen_volume(field,rain,0,pool,pixels);if(!images_prefix.empty())rain_ppm(images_prefix+"_clear.ppm",width,height,pixels);
        std::ofstream out(stats_path);out<<std::setprecision(17)<<"{\"buildings\":"<<city.buildings<<",\"primitives\":"<<city.scene.primitives.size()
            <<",\"transport_nodes\":"<<transport.node_primitives.size()<<",\"transport_generation\":"<<transport.generation<<",\"transport_residual\":"<<transport.final_residual
            <<",\"static_transport_ms\":"<<solve_ms<<",\"response_precompute_ms\":"<<field_ms<<",\"field_width\":"<<fw<<",\"field_height\":"<<fh<<",\"normal_axis_samples\":"<<normals
            <<",\"response_count\":"<<field.response.size()<<",\"field_payload_bytes\":"<<field.response.size()*sizeof(PackedRadiance)<<",\"city_queries\":"<<field.city_queries
            <<",\"field_checksum\":\""<<checksum<<"\",\"held_out_queries\":"<<checks<<",\"held_out_rms_byte_error\":"<<std::sqrt(squared/(checks*3))<<",\"held_out_max_byte_error\":"<<maximum<<"}\n";
    }else if(mode=="reference"){
        // Independent direct city evaluation, deliberately outside playback.
        FrozenVolumeField field=load_volume_field(field_path);width=field.output_width;height=field.output_height;
        field.camera=make_look_camera(width,height,{10,7.5,17},{0,2.5,-9},{0,1,0},23.5);
        CityScene city=make_rain_city();auto beams=compile_beam_field(city.scene);auto transport=compile_transport_field(city.scene,beams,true);
        TraceContext ctx{city.scene,beams,transport};RainNormalTexture rain(width,height);RowWorkers pool;
        const int ticks=720;for(int frame=0;frame<=ticks;++frame)rain.update(frame/60.f,1.f/60);
        std::vector<std::uint8_t> cached(std::size_t(width)*height*3),direct(cached.size()),normal(cached.size());
        render_frozen_volume(field,rain,12,pool,cached,&normal);const auto tone=make_rain_tone_lut();std::vector<std::uint64_t> queries(height);
        const auto start=Clock::now();pool.run(height,[&](int y){for(int x=0;x<width;++x){auto n=rain.normal(x,y,12);
            const RGB value=evaluate_lens_response(field,field.site(x,y),n[0],n[1],city,ctx,queries[y]);
            const std::size_t offset=(std::size_t(y)*width+x)*3;
            for(int c=0;c<3;++c)direct[offset+c]=tone[std::clamp(int(value[c]*(65535./16)+.5),0,65535)];}});
        const double direct_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        double squared=0,absolute=0;int maximum=0;std::uint64_t over1=0,over8=0;
        for(std::size_t p=0;p<direct.size();p+=3){int pixel=0;for(int c=0;c<3;++c){const int e=std::abs(int(cached[p+c])-direct[p+c]);
            pixel=std::max(pixel,e);squared+=e*e;absolute+=e;maximum=std::max(maximum,e);}over1+=pixel>1;over8+=pixel>8;}
        if(!images_prefix.empty()){rain_ppm(images_prefix+"_direct.ppm",width,height,direct);rain_ppm(images_prefix+"_gather.ppm",width,height,cached);rain_ppm(images_prefix+"_normals.ppm",width,height,normal);}
        std::ofstream out(stats_path);out<<std::setprecision(17)<<"{\"time_seconds\":12,\"width\":"<<width<<",\"height\":"<<height
            <<",\"direct_reference_ms\":"<<direct_ms<<",\"direct_city_queries\":"<<std::accumulate(queries.begin(),queries.end(),std::uint64_t(0))
            <<",\"rms_byte_error\":"<<std::sqrt(squared/direct.size())<<",\"mean_absolute_byte_error\":"<<absolute/direct.size()<<",\"maximum_byte_error\":"<<maximum
            <<",\"pixels_over_1\":"<<over1<<",\"pixels_over_8\":"<<over8<<"}\n";
    }else if(mode=="play"){
        // This mode constructs no Scene or TransportField. The normal texture
        // and immutable response array are the entire playback input.
        rain_city_frozen=true;const auto load_start=Clock::now();const FrozenVolumeField field=load_volume_field(field_path);
        const double load_ms=std::chrono::duration<double,std::milli>(Clock::now()-load_start).count();
        const auto scan_start=Clock::now();const auto before=field.checksum();
        const double scan_ms=std::chrono::duration<double,std::milli>(Clock::now()-scan_start).count();
        width=field.output_width;height=field.output_height;
        RainNormalTexture rain(width,height);RowWorkers pool;
        std::vector<std::uint8_t> pixels(std::size_t(width)*height*3),normal_image(pixels.size());
        const auto warm_start=Clock::now();render_frozen_volume(field,rain,0,pool,pixels);
        const double warm_ms=std::chrono::duration<double,std::milli>(Clock::now()-warm_start).count();
        const int frames=int(std::lround(duration*fps));update_require(fps>0&&frames>0,"invalid clip duration");
        std::vector<double> frame_ms;frame_ms.reserve(frames);double update_ms=0,write_ms=0;
        const auto sequence_start=Clock::now();
        for(int frame=0;frame<frames;++frame){const float time=float(frame)/fps;
            const auto begin=Clock::now();rain.update(time,1.f/fps);const auto updated=Clock::now();
            const bool snapshot=frame==0||frame==int(frames*.33)||frame==int(frames*.66)||frame==frames-1;
            render_frozen_volume(field,rain,time,pool,pixels,snapshot?&normal_image:nullptr);
            const auto rendered=Clock::now();frame_ms.push_back(std::chrono::duration<double,std::milli>(rendered-begin).count());
            update_ms+=std::chrono::duration<double,std::milli>(updated-begin).count();
            std::cout.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());update_require(bool(std::cout),"video pipe failed");
            write_ms+=std::chrono::duration<double,std::milli>(Clock::now()-rendered).count();
            if(snapshot&&!images_prefix.empty()){rain_ppm(images_prefix+"_"+std::to_string(frame)+".ppm",width,height,pixels);
                rain_ppm(images_prefix+"_normal_"+std::to_string(frame)+".ppm",width,height,normal_image);}
            if(frame%120==0)std::cerr<<"playback "<<frame<<"/"<<frames<<" compute "<<frame_ms.back()<<" ms\n";
        }
        std::cout.flush();const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-sequence_start).count();
        const auto after=field.checksum();update_require(before==after,"frozen response changed during playback");
        update_require(rain_forbidden_city_calls==0,"city evaluation attempted during frozen playback");
        auto sorted=frame_ms;std::sort(sorted.begin(),sorted.end());const double mean=std::accumulate(frame_ms.begin(),frame_ms.end(),0.)/frames;
        std::ofstream out(stats_path);out<<std::setprecision(17)<<"{\"width\":"<<width<<",\"height\":"<<height<<",\"frames\":"<<frames<<",\"fps\":"<<fps<<",\"duration_seconds\":"<<double(frames)/fps
            <<",\"field_storage\":\"read-only memory map\",\"field_integrity_scan_ms\":"<<scan_ms<<",\"field_load_ms\":"<<load_ms<<",\"warm_frame_ms\":"<<warm_ms<<",\"mean_compute_ms\":"<<mean<<",\"median_compute_ms\":"<<sorted[sorted.size()/2]
            <<",\"p95_compute_ms\":"<<sorted[std::min(sorted.size()-1,std::size_t(sorted.size()*.95))]<<",\"rain_state_update_total_ms\":"<<update_ms<<",\"pipe_write_total_ms\":"<<write_ms
            <<",\"sequence_wall_ms\":"<<elapsed<<",\"city_evaluations_during_playback\":"<<rain_forbidden_city_calls.load()<<",\"static_transport_updates\":0,\"field_rebuilds\":0,\"normal_texture_updates\":"<<rain.generation
            <<",\"coalescences\":"<<rain.merges<<",\"field_checksum_before\":\""<<before<<"\",\"field_checksum_after\":\""<<after<<"\",\"frame_compute_ms\":[";
        for(std::size_t i=0;i<frame_ms.size();++i){if(i)out<<",";out<<frame_ms[i];}out<<"]}\n";
    }else throw std::invalid_argument("mode must be bake, play, reference or test");
    return 0;
}catch(const std::exception& e){std::cerr<<"city rain failed: "<<e.what()<<"\n";return 1;}}

#endif
