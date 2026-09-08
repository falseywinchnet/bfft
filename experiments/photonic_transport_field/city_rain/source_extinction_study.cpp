#define CITY_RAIN_LIBRARY
#include "city_rain.cpp"
#include "night_scene.hpp"

int main(int argc,char**){try{
    if(argc>1){
        for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
            Scene scene=build_demonstrator_scene(name);auto beams=compile_beam_field(scene);
            source_interval_extinction=false;auto reference=compile_transport_field(scene,beams,true);
            source_interval_extinction=true;auto accelerated=compile_transport_field(scene,beams,true);
            require_same_source_fields(reference,accelerated);
            std::cout<<name<<": direct, bounce, radiance, coupling and complete irradiance atlases identical\n";
        }
        return 0;
    }
    const auto night=make_night_city();const auto& scene=night.city.scene;
    struct Query{Vec3 point,normal;int receiver;};std::vector<Query> queries;
    std::uint32_t state=54719;auto random=[&](){state=state*1664525u+1013904223u;return (state>>8)/double(1u<<24);};
    for(int i=0;i<2500;++i){int id=int(random()*scene.primitives.size());const auto& p=scene.primitives[id];
        if(p.shape!=Shape::Rectangle||scene.materials[p.material].kind==MaterialKind::Emissive)continue;
        double u=random(),v=random();if(i%7==0)u=0;if(i%11==0)v=1;
        queries.push_back({p.origin+p.u*u+p.v*v+p.normal*2e-4,p.normal,id});}
    std::vector<RGB> baseline;source_interval_extinction=false;auto begin=Clock::now();
    for(auto q:queries)baseline.push_back(area_irradiance(scene,q.point,q.normal,q.receiver));
    double legacy_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    source_interval_extinction=true;source_extinction_counts={};begin=Clock::now();double maximum=0;int unequal=0;
    for(std::size_t i=0;i<queries.size();++i){auto q=queries[i];auto v=area_irradiance(scene,q.point,q.normal,q.receiver);unequal+=v!=baseline[i];
        for(int c=0;c<3;++c)maximum=std::max(maximum,std::abs(v[c]-baseline[i][c]));}
    double fast_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();auto stats=source_extinction_counts;
    update_require(maximum<1e-12,"city source extinction changed irradiance");
    // Mixed dielectric support must retain the original path.
    Scene mixed;int emit=add_material(mixed,{"source",MaterialKind::Emissive,{1,1,1},{2,3,4},0,0});
    int opaque=add_material(mixed,{"blocker",MaterialKind::Diffuse,{.5,.5,.5},{},.8,.4});
    int glass=add_material(mixed,{"glass",MaterialKind::Dielectric,{.9,.95,1},{},0,.01});
    int light=add_rect(mixed,"source",{-2,4,-2},{4,0,0},{0,0,4},{0,-1,0},emit,false);mixed.area_lights.push_back({light,{2,3,4}});
    add_rect(mixed,"glass",{-1,1,-1},{2,0,0},{0,0,2},{0,1,0},glass,false);
    add_rect(mixed,"partial blocker",{-.7,2,-.7},{.7,0,0},{0,0,1.4},{0,1,0},opaque,false);build_scene_bvh(mixed);
    source_interval_extinction=false;auto a=area_irradiance(mixed,{0,0,0},{0,1,0},-1);
    source_interval_extinction=true;source_extinction_counts={};auto b=area_irradiance(mixed,{0,0,0},{0,1,0},-1);
    update_require(a==b&&source_extinction_counts.extinguished==0,"mixed medium failed to fall back");
    // A sphere uses the legacy conic/visibility path, including grazing rows.
    mixed.primitives[1].intersectable=false;mixed.primitives[2].intersectable=false;
    add_sphere(mixed,"opaque sphere",{0,2,0},.4,opaque,false);build_scene_bvh(mixed);
    for(int i=0;i<20;++i){Vec3 q{-.7+i*.07,0,.03};source_interval_extinction=false;auto expected=area_irradiance(mixed,q,{0,1,0},-1);
        source_interval_extinction=true;auto actual=area_irradiance(mixed,q,{0,1,0},-1);update_require(expected==actual,"sphere fallback changed");}
    std::cout<<std::setprecision(17)<<"{\"queries\":"<<queries.size()<<",\"legacy_ms\":"<<legacy_ms<<",\"extinction_ms\":"<<fast_ms
        <<",\"maximum_error\":"<<maximum<<",\"unequal_queries\":"<<unequal<<",\"intervals\":"<<stats.intervals<<",\"extinguished\":"<<stats.extinguished
        <<",\"primitive_checks\":"<<stats.primitive_checks<<",\"fallback_intervals\":"<<stats.fallbacks<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
