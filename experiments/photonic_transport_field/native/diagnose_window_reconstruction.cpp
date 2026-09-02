#define main window_transport_native_entry
#include "window_transport_native.cpp"
#undef main

RGB exact_surface_radiance(const Scene& scene,const std::vector<Relation>& relations,
                           const MarchResult& marched,int receiver,const SurfacePoint& point){
    RGB incident{};
    for(int i=0;i<static_cast<int>(relations.size());++i){
        const Relation& r=relations[i];if(r.receiver!=receiver||r.factor<=1e-14)continue;
        const double exact_density=r.scale*relation_kernel(scene,r.source,r.receiver,point).density;
        incident+=marched.total_delivered[i]*(exact_density/r.factor);}
    return scene.objects[receiver].emission+multiply(incident,scene.objects[receiver].albedo)*(1.0/pi);
}

std::vector<std::uint8_t> render_exact_lookup(const Scene& scene,const std::vector<Relation>& relations,
                                               const MarchResult& marched,int width,int height){
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3,0);
    const Vec3 camera{0,2.15,6.8},target{0,1.55,-2.15};const Vec3 forward=unit(target-camera);
    const Vec3 right=unit(cross(forward,{0,1,0})),up=cross(right,forward);
    const double scale=std::tan(24*pi/180.0),aspect=double(width)/height;std::atomic<int> next_row{0};
    std::vector<std::thread> threads;const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&]{for(;;){
        const int y=next_row.fetch_add(1);if(y>=height)break;
        for(int x=0;x<width;++x){const double px=(2*(x+.5)/width-1)*aspect*scale;
            const double py=(1-2*(y+.5)/height)*scale;
            const Hit hit=first_hit(scene,camera,unit(forward+right*px+up*py));RGB linear{};
            if(hit.valid)linear=(hit.object==Floor||scene.objects[hit.object].shape==Shape::Sphere)
                ?exact_surface_radiance(scene,relations,marched,hit.object,hit.surface)
                :surface_radiance(scene,relations,marched,hit.object,hit.surface);
            const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
            for(int c=0;c<3;++c){const double mapped=std::pow(std::clamp(
                    1-std::exp(-.72*std::max(linear[c],0.0)),0.0,1.0),1/2.2);
                image[offset+c]=static_cast<std::uint8_t>(std::lround(255*mapped));}}}});
    for(auto& thread:threads)thread.join();return image;
}

int main()try{
    constexpr int width=800,height=600;Limits limits;limits.max_nodes=800000;limits.max_build_seconds=50;
    const Scene scene=build_scene();const auto relations=build_relations(scene,limits);
    const auto op=build_window_operator(scene,relations);
    const auto marched=march(scene,relations,op,24,.001);
    const auto diagnostic=render_exact_lookup(scene,relations,marched,width,height);
    write_ppm("/tmp/window_base_exact_receiver.ppm",diagnostic,width,height);
    std::cout<<"{\n  \"floor_and_spheres_exact_receiver_lookup\": true,\n  \"march_depths\": "
        <<marched.depths<<"\n}\n";return 0;
}catch(const std::exception& error){std::cerr<<"error: "<<error.what()<<"\n";return 2;}
