#define main regime_scene_program_main
#include "/tmp/direct_spectral_scene.cpp"
#undef main

int main(int argc,char** argv){
    const std::string name=argc>2?argv[2]:"aperture-canyon";
    Scene scene=build_demonstrator_scene(name);scene.use_bvh=false;scene.child_boundaries=true;
    refresh_optical_children(scene);auto beams=compile_beam_field(scene);auto field=compile_transport_field(scene,beams,true);
    TraceContext ctx{scene,beams,field};std::vector<std::uint8_t> reference;std::ofstream out(argv[1]);
    out<<std::setprecision(17)<<"{\"scene\":\""<<name<<"\",\"frames\":[";bool comma=false;
    for(int repeat=0;repeat<(argc>4?std::stoi(argv[4]):(argc>3?1:2));++repeat){std::vector<int> modes=repeat?std::vector<int>{64,32,16,8,4,0}:std::vector<int>{0,4,8,16,32,64};
        if(argc>3){modes={0,std::stoi(argv[3])};if(repeat%2)std::reverse(modes.begin(),modes.end());}
        for(int mode:modes){direct_spectral_samples=mode;RenderStats stats;
            std::uint64_t before=0;for(auto& c:scene.optical_children)before+=c->classification_evaluations;
            const auto start=Clock::now();auto pixels=render_visible_edge_field(ctx,800,600,1,stats);
            const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            std::uint64_t after=0;for(auto& c:scene.optical_children)after+=c->classification_evaluations;
            if(reference.empty())reference=pixels;
            std::uint64_t changed=0,over1=0;double absolute=0,square=0;int maximum=0;
            for(std::size_t i=0;i<pixels.size();++i){int delta=std::abs(int(pixels[i])-reference[i]);absolute+=delta;square+=delta*delta;
                maximum=std::max(maximum,delta);changed+=delta>0;over1+=delta>1;}
            if(comma)out<<",";comma=true;
            out<<"{\"samples_per_band\":"<<mode<<",\"repeat\":"<<repeat<<",\"ms\":"<<ms
                <<",\"classification_calls\":"<<after-before<<",\"primary_samples\":"<<stats.exact_samples
                <<",\"quadrature\":"<<stats.trace.emitter_quadrature_samples
                <<",\"mean_byte_error\":"<<absolute/pixels.size()<<",\"rms_byte_error\":"<<std::sqrt(square/pixels.size())
                <<",\"max_byte_error\":"<<maximum<<",\"changed_bytes\":"<<changed<<",\"over1_bytes\":"<<over1<<"}";out.flush();
            if(repeat==0){std::ofstream image(std::string(argv[1])+"."+std::to_string(mode)+".ppm",std::ios::binary);
                image<<"P6\n800 600\n255\n";image.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());}
        }}out<<"]}\n";
}
