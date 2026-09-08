#define CITY_RAIN_LIBRARY
#include "city_rain.cpp"
#include "night_scene.hpp"

struct IlluminationScreen {
    int width=1280,height=720,pad=192,scale=2,sw=3328,sh=2208;
    std::vector<PackedRadiance> pixels;
    struct Level {int w,h;std::vector<PackedRadiance> data;};std::vector<Level> mips;
    void build_mips(){mips.clear();int w=sw,h=sh;const auto* previous=&pixels;
        while(w>=8&&h>=8){int nw=w/2,nh=h/2;Level level{nw,nh,{}};level.data.resize(std::size_t(nw)*nh);
            for(int y=0;y<nh;++y)for(int x=0;x<nw;++x){RGB v{};for(int j=0;j<2;++j)for(int i=0;i<2;++i){const auto& p=(*previous)[std::size_t(y*2+j)*w+x*2+i];v+=RGB{double(p.r),double(p.g),double(p.b)}*.25;}level.data[std::size_t(y)*nw+x]={_Float16(v[0]),_Float16(v[1]),_Float16(v[2])};}
            mips.push_back(std::move(level));previous=&mips.back().data;w=nw;h=nh;}
    }
    std::size_t resident_bytes()const{std::size_t n=pixels.size();for(const auto& l:mips)n+=l.data.size();return n*6;}
    std::uint64_t checksum()const{std::uint64_t h=1469598103934665603ULL;auto scan=[&](const std::vector<PackedRadiance>& v){const auto* p=reinterpret_cast<const unsigned char*>(v.data());for(std::size_t i=0;i<v.size()*6;++i){h^=p[i];h*=1099511628211ULL;}};scan(pixels);for(const auto& l:mips)scan(l.data);return h;}
    RGB sample_level(double x,double y,int level)const{
        double divisor=std::ldexp(1.,level),sx=(x+pad+.5)*scale/divisor-.5,sy=(y+pad+.5)*scale/divisor-.5;
        const auto& data=level?mips[level-1].data:pixels;int w=level?mips[level-1].w:sw,h=level?mips[level-1].h:sh;
        sx=std::clamp(sx,0.,double(w-1));sy=std::clamp(sy,0.,double(h-1));int ix=std::min(int(sx),w-2),iy=std::min(int(sy),h-2);double ax=sx-ix,ay=sy-iy;RGB v{};
        for(int j=0;j<2;++j)for(int i=0;i<2;++i){const auto& p=data[std::size_t(iy+j)*w+ix+i];double weight=(i?ax:1-ax)*(j?ay:1-ay);v+=RGB{double(p.r),double(p.g),double(p.b)}*weight;}return v;
    }
    RGB sample(double x,double y,double lod=0)const{
        double sx=(x+pad+.5)*scale-.5,sy=(y+pad+.5)*scale-.5;
        update_require(sx>=0&&sy>=0&&sx<sw-1&&sy<sh-1,"deformation exceeded captured support");
        lod=std::clamp(lod,0.,double(mips.size()));int level=int(lod);RGB a=sample_level(x,y,level);double t=lod-level;
        if(t>0&&level<int(mips.size()))return a*(1-t)+sample_level(x,y,level+1)*t;return a;
    }
    void save(const std::string& path)const{std::ofstream out(path,std::ios::binary);std::uint32_t h[]={0x32524353, std::uint32_t(width),std::uint32_t(height),std::uint32_t(pad),std::uint32_t(scale),std::uint32_t(sw),std::uint32_t(sh)};out.write((const char*)h,sizeof(h));out.write((const char*)pixels.data(),pixels.size()*6);update_require(bool(out),"screen save failed");}
    static IlluminationScreen load(const std::string& path){IlluminationScreen s;std::ifstream in(path,std::ios::binary);std::uint32_t h[7]{};in.read((char*)h,sizeof(h));update_require(bool(in)&&h[0]==0x32524353,"invalid screen");s.width=h[1];s.height=h[2];s.pad=h[3];s.scale=h[4];s.sw=h[5];s.sh=h[6];update_require(s.sw==(s.width+2*s.pad)*s.scale&&s.sh==(s.height+2*s.pad)*s.scale&&s.sw<10000&&s.sh<10000,"invalid screen dimensions");s.pixels.resize(std::size_t(s.sw)*s.sh);in.read((char*)s.pixels.data(),s.pixels.size()*6);update_require(bool(in),"truncated screen");s.build_mips();return s;}
};
struct ScreenScratch {std::vector<std::array<float,2>> normals;};
void render_screen(const IlluminationScreen& s,const RainNormalTexture& rain,float t,RowWorkers& pool,ScreenScratch& scratch,std::vector<std::uint8_t>& image){
    static auto tone=make_rain_tone_lut();
    // Paraxial slope-to-screen transport. n=0 is exactly identity. Every
    // dynamic output channel comes only from the fixed illumination screen.
    const double strength=s.height/(2*std::tan(26*std::numbers::pi/180))*(1.333-1);
    scratch.normals.resize(std::size_t(s.width)*s.height);
    pool.run(s.height,[&](int y){for(int x=0;x<s.width;++x)scratch.normals[std::size_t(y)*s.width+x]=rain.normal(x,y,t);});
    pool.run(s.height,[&](int y){for(int x=0;x<s.width;++x){
        const auto n=scratch.normals[std::size_t(y)*s.width+x];
        const auto dx=scratch.normals[std::size_t(y)*s.width+(x==s.width-1?x-1:x+1)];
        const auto dy=scratch.normals[std::size_t(y==s.height-1?y-1:y+1)*s.width+x];
        const double signx=x==s.width-1?-1:1,signy=y==s.height-1?-1:1;
        double a=1+strength*(dx[0]-n[0])*signx,b=strength*(dy[0]-n[0])*signy;
        double c=-strength*(dx[1]-n[1])*signx,d=1-strength*(dy[1]-n[1])*signy;
        double aa=a*a+c*c,bb=b*b+d*d,ab=a*b+c*d;
        double rho=std::sqrt(.5*(aa+bb+std::sqrt((aa-bb)*(aa-bb)+4*ab*ab)));
        double lod=std::log2(std::max(1.,s.scale*rho));
        RGB v=s.sample(x+strength*n[0],y-strength*n[1],lod);
        const auto offset=(std::size_t(y)*s.width+x)*3;for(int c=0;c<3;++c)image[offset+c]=tone[std::clamp(int(v[c]*(65535./16)+.5),0,65535)];}});
}
void screen_test(){
    IlluminationScreen s;s.width=8;s.height=6;s.pad=2;s.scale=2;s.sw=24;s.sh=20;s.pixels.resize(480);
    for(int y=0;y<s.sh;++y)for(int x=0;x<s.sw;++x)s.pixels[y*s.sw+x]={_Float16(x*.125),_Float16(y*.125),_Float16(1)};
    for(int j=0;j<20;++j){double x=j*.3,y=j*.2;auto v=s.sample(x,y);update_require(std::abs(v[0]-((x+2.5)*2-.5)*.125)<1e-12,"screen affine x");update_require(std::abs(v[1]-((y+2.5)*2-.5)*.125)<1e-12,"screen affine y");}
    const auto sum=s.checksum();RainNormalTexture rain(8,6);RowWorkers pool;ScreenScratch scratch;std::vector<std::uint8_t> a(144);render_screen(s,rain,0,pool,scratch,a);update_require(sum==s.checksum(),"screen changed");
    for(int y=0;y<6;++y)for(int x=0;x<8;++x){auto v=s.sample(x,y);for(int c=0;c<3;++c)update_require(std::abs(int(a[(y*8+x)*3+c])-int(tone_byte(v[c])))<=1,"clear-screen identity");}
    bool caught=false;try{s.sample(-100,0);}catch(const std::exception&){caught=true;}update_require(caught,"support guard");
    rain_city_frozen=true;NightCity n;BeamField b;TransportField f;TraceContext ctx{n.city.scene,b,f};caught=false;try{evaluate_night(n,ctx,{});}catch(const std::exception&){caught=true;}update_require(caught,"freeze guard");rain_city_frozen=false;rain_forbidden_city_calls=0;
    IlluminationScreen constant=s;for(auto& p:constant.pixels)p={_Float16(.5),_Float16(1),_Float16(2)};constant.build_mips();
    for(double lod=0;lod<=constant.mips.size();lod+=.25){auto v=constant.sample(2.3,1.7,lod);update_require(v==RGB{.5,1,2},"mip constant preservation");}
    auto path=(std::filesystem::temp_directory_path()/("screen_test_"+std::to_string(Clock::now().time_since_epoch().count()))).string();constant.save(path);auto restored=IlluminationScreen::load(path);std::filesystem::remove(path);update_require(constant.checksum()==restored.checksum(),"screen and mip round trip");
    test_city_rain();std::cerr<<"screen checks passed: affine reconstruction, identity, immutable sample, support guard, night freeze guard\n";
}
int main(int argc,char** argv){try{
    std::string mode=argc>1?argv[1]:"test",dir=argc>2?argv[2]:"/tmp/night_screen";
    if(mode=="test"){screen_test();return 0;}std::filesystem::create_directories(dir);
    if(mode=="bake"||mode=="bake-legacy"){
        source_interval_extinction=mode=="bake";source_extinction_counts={};
        auto start=Clock::now();NightCity n=make_night_city();auto beams=compile_beam_field(n.city.scene);auto field=compile_transport_field(n.city.scene,beams,true);TraceContext ctx{n.city.scene,beams,field};
        double transport_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();auto extinction=source_extinction_counts;std::cerr<<"night city transport "<<transport_ms<<" ms, "<<n.city.scene.primitives.size()<<" primitives\n";
        IlluminationScreen s;s.pixels.resize(std::size_t(s.sw)*s.sh);RowWorkers pool;ScreenScratch scratch;Camera camera=make_look_camera(s.width,s.height,{10,8,19},{0,4,-10},{0,1,0},26);
        auto capture=Clock::now();pool.run(s.sh,[&](int y){for(int x=0;x<s.sw;++x){double px=(x+.5)/s.scale-.5-s.pad,py=(y+.5)/s.scale-.5-s.pad;RGB v=evaluate_night(n,ctx,camera_ray(camera,s.width,s.height,px,py));s.pixels[std::size_t(y)*s.sw+x]={_Float16(v[0]),_Float16(v[1]),_Float16(v[2])};}});
        double capture_ms=std::chrono::duration<double,std::milli>(Clock::now()-capture).count();s.build_mips();s.save(dir+"/illumination.screen");
        RainNormalTexture rain(s.width,s.height);std::vector<std::uint8_t> image(std::size_t(s.width)*s.height*3);render_screen(s,rain,0,pool,scratch,image);rain_ppm(dir+"/night_clear.ppm",s.width,s.height,image);
        std::ofstream out(dir+"/bake.json");out<<"{\"buildings\":"<<n.city.buildings<<",\"vehicles\":"<<n.vehicles<<",\"red_beacons\":"<<n.beacons<<",\"point_lights\":"<<n.lights.size()<<",\"primitives\":"<<n.city.scene.primitives.size()<<",\"transport_ms\":"<<transport_ms<<",\"source_interval_extinction\":"<<(source_interval_extinction?"true":"false")<<",\"source_calls\":"<<extinction.source_calls<<",\"source_intervals\":"<<extinction.intervals<<",\"extinguished_intervals\":"<<extinction.extinguished<<",\"primitive_certificate_checks\":"<<extinction.primitive_checks<<",\"fallback_intervals\":"<<extinction.fallbacks<<",\"capture_ms\":"<<capture_ms<<",\"screen_width\":"<<s.sw<<",\"screen_height\":"<<s.sh<<",\"screen_bytes\":"<<s.pixels.size()*6<<",\"screen_with_mips_bytes\":"<<s.resident_bytes()<<",\"capture_queries\":"<<s.pixels.size()<<",\"checksum\":\""<<s.checksum()<<"\"}\n";
    }else if(mode=="play"){
        // City, geometry, lighting and transport objects do not exist here.
        rain_city_frozen=true;const auto s=IlluminationScreen::load(dir+"/illumination.screen");auto before=s.checksum();RowWorkers pool;ScreenScratch scratch;RainNormalTexture rain(s.width,s.height);std::vector<std::uint8_t> image(std::size_t(s.width)*s.height*3);render_screen(s,rain,0,pool,scratch,image);
        constexpr int fps=60,frames=1080;std::vector<double> ms;auto start=Clock::now();
        for(int f=0;f<frames;++f){float t=f/float(fps);auto begin=Clock::now();rain.update(t,1.f/fps);render_screen(s,rain,t,pool,scratch,image);ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
            std::cout.write((const char*)image.data(),image.size());update_require(bool(std::cout),"video pipe failed");if(f==0||f==360||f==720||f==1079)rain_ppm(dir+"/frame_"+std::to_string(f)+".ppm",s.width,s.height,image);
            if(f%180==0)std::cerr<<f<<"/"<<frames<<" "<<ms.back()<<"ms\n";
        }std::cout.flush();double wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();auto after=s.checksum();update_require(before==after&&rain_forbidden_city_calls==0,"freeze invariant failed");auto sorted=ms;std::sort(sorted.begin(),sorted.end());
        std::ofstream out(dir+"/playback.json");out<<"{\"frames\":1080,\"fps\":60,\"duration\":18,\"median_ms\":"<<sorted[540]<<",\"p95_ms\":"<<sorted[1026]<<",\"max_ms\":"<<sorted.back()<<",\"sequence_wall_ms\":"<<wall<<",\"city_evaluations\":0,\"screen_updates\":0,\"checksum_before\":\""<<before<<"\",\"checksum_after\":\""<<after<<"\",\"frame_ms\":[";for(int i=0;i<frames;++i){if(i)out<<',';out<<ms[i];}out<<"]}\n";
    }else throw std::runtime_error("use test, bake, or play");return 0;
}catch(const std::exception& e){std::cerr<<"night screen failed: "<<e.what()<<'\n';return 1;}}
