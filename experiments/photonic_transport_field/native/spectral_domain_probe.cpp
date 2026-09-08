// Experimental comparison: event decomposition and own-domain quadrature.
#include <functional>
#define main regime_scene_program_main
#include "regime_scene_native.cpp"
#undef main
#include "spectral_domains.hpp"
int main(){
    Scene scene=build_demonstrator_scene("aperture-canyon");scene.use_bvh=false;
    const BeamField beams=compile_beam_field(scene);
    const TransportField field=compile_transport_field(scene,beams,true);
    TraceContext ctx{scene,beams,field};Camera camera=make_camera(800,600);
    std::cout<<std::setprecision(17)<<"{\"cases\":[";bool comma=false;
    int checked=0;double maximum_error=0;
    for(const auto xy:std::array<std::array<double,2>,3>{{{275.,324.},{275.25,324.25},{275.25,323.75}}}){
        const Ray ray=camera_ray(camera,800,600,xy[0],xy[1]);const Hit hit=first_hit(scene,ray);
        if(!hit.valid)continue;const Material& material=scene.materials[scene.primitives[hit.primitive].material];
        if(material.kind!=MaterialKind::Dielectric)continue;
        for(int channel=0;channel<3;++channel){
            for(int i=0;i<33;++i){double t=channel-.5+(i+.37)/33;
                SpectralDomainStats stats;double a=spectral_event_point(ctx,ray,hit,channel,t,stats);
                double b=trace_primary_dielectric_sample(ctx,ray,hit,material,channel,t).value;
                maximum_error=std::max(maximum_error,std::abs(a-b));++checked;
                if(std::abs(a-b)>1e-11*(1+std::abs(b)))throw std::runtime_error("pointwise event decomposition mismatch");}
            TraceStats old_stats,new_stats;ctx.stats=&old_stats;
            const auto start=Clock::now();double old_value=integrate_spectral_band(ctx,ray,hit,material,channel,nullptr);
            const double old_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            ctx.stats=&new_stats;SpectralDomainStats domain;
            const auto next=Clock::now();double new_value=integrate_spectral_domains(ctx,ray,hit,channel,nullptr,&domain);
            const double new_ms=std::chrono::duration<double,std::milli>(Clock::now()-next).count();ctx.stats=nullptr;
            std::array<double,2> dense{};
            for(int j=0;j<2;++j){int count=j?2048:512;
                for(int i=0;i<count;++i)dense[j]+=trace_primary_dielectric_sample(ctx,ray,hit,material,channel,
                    channel-.5+(i+.5)/count).value/count;}
            if(comma)std::cout<<",";comma=true;
            std::cout<<"{\"x\":"<<xy[0]<<",\"y\":"<<xy[1]<<",\"channel\":"<<channel
                <<",\"paths\":"<<old_value<<",\"domains\":"<<new_value<<",\"dense512\":"<<dense[0]
                <<",\"dense2048\":"<<dense[1]<<",\"paths_ms\":"<<old_ms<<",\"domains_ms\":"<<new_ms
                <<",\"paths_quadrature\":"<<old_stats.emitter_quadrature_samples
                <<",\"domains_quadrature\":"<<new_stats.emitter_quadrature_samples
                <<",\"event_steps\":"<<domain.steps<<",\"event_domains\":"<<domain.domains
                <<",\"topology_limits\":"<<domain.topology_limits<<",\"shading_limits\":"<<domain.shading_limits
                <<",\"estimated_error\":"<<domain.estimated_error<<"}"<<std::flush;
        }
    }
    if(checked==0)throw std::runtime_error("no dielectric test rays");
    std::cout<<"],\"pointwise_comparisons\":"<<checked<<",\"maximum_pointwise_error\":"<<maximum_error<<"}\n";
}
