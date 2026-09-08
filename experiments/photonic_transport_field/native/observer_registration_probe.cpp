#define main regime_scene_program_main
#include "regime_scene_native.cpp"
#undef main

void equal_response(const ChildBoundaryResponse& a,const ChildBoundaryResponse& b){
    update_require(a.ports.size()==b.ports.size()&&a.source_weight==b.source_weight&&
        a.unresolved_weight==b.unresolved_weight,"shared law changed boundary values");
    for(std::size_t p=0;p<a.ports.size();++p){const auto& x=a.ports[p];const auto& y=b.ports[p];
        update_require(x.weight==y.weight&&x.primitive==y.primitive&&
            norm2(x.ray.origin-y.ray.origin)==0&&norm2(x.ray.direction-y.ray.direction)==0,
            "shared law changed a boundary port");}
}

int main(int argc,char** argv){
    Scene shared=build_demonstrator_scene("aperture-canyon");shared.child_boundaries=true;
    Scene exact=shared;exact.exact_child_states=true;refresh_optical_children(exact);
    const auto registered=shared.optical_children;std::size_t checks=0;
    for(std::size_t c=0;c<registered.size();++c)for(const auto& face:registered[c]->faces)
        for(int k=0;k<64;++k){const auto& primitive=face.second;
            const Vec3 normal=primitive.shape==Shape::Sphere?unit(Vec3{std::cos(k*.13),.3,std::sin(k*.13)}):primitive.normal;
            const Vec3 target=primitive.center+(primitive.shape==Shape::Rectangle?primitive.u*(.2*std::sin(k*.17)):Vec3{});
            const Ray ray{target+normal*1.5,-normal};const Hit hit=intersect_primitive(primitive,face.first,ray);
            update_require(hit.valid,"invalid boundary comparison ray");
            const double coordinate=-.25+2.5*k/63.;const int channel=k%3;
            const auto a=registered[c]->response(ray,hit,channel,coordinate,1e-5,32);
            const auto b=exact.optical_children[c]->response(ray,hit,channel,coordinate,1e-5,32);
            equal_response(a,b);
            auto prefix=registered[c]->classification_response(ray,hit,channel,coordinate,1e-5,32);
            auto expected=b;expected.ports.resize(std::min<std::size_t>(2,expected.ports.size()));
            expected.source_weight=prefix.source_weight;expected.unresolved_weight=prefix.unresolved_weight;
            equal_response(prefix,expected);++checks;}
    for(const auto& child:registered)update_require(child->unit_compilations==0&&child->unit_cache_bytes==0&&!child->responses,
        "observation registered a query state");
    refresh_optical_children(shared);
    update_require(shared.optical_children==registered,"unchanged law was registered again");
    std::vector<std::thread> threads;
    for(int t=0;t<8;++t)threads.emplace_back([&]{const auto& face=registered.front()->faces.front();
        const Ray ray{face.second.center+face.second.normal,-face.second.normal};
        const Hit hit=intersect_primitive(face.second,face.first,ray);
        for(int k=0;k<32;++k)registered.front()->response(ray,hit,k%3,.01*k,1e-5,32);});
    for(auto& thread:threads)thread.join();
    update_require(registered.front()->unit_compilations==0,"parallel observers mutated registrations");
    Primitive moved=shared.primitives[shared.glass_sheet];moved.origin.x+=.1;
    edit_scene_geometry(shared,{{shared.glass_sheet,moved}});
    update_require(shared.optical_children.size()==3&&shared.optical_children[0]==registered[0]&&
        shared.optical_children[1]!=registered[1]&&shared.optical_children[2]==registered[2],
        "geometry edit did not replace precisely its one child law");
    std::cout<<"shared registration checks passed: "<<checks<<" exact response comparisons; 3 child laws, 7 boundaries; zero query registrations\n";
    if(argc<=1)return 0;
    std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"frames\":[";bool comma=false;
    for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        std::array<Scene,2> scenes;std::array<BeamField,2> beams;std::array<TransportField,2> fields;
        std::array<double,2> build_ms{};std::vector<std::uint8_t> reference;
        for(int mode=0;mode<2;++mode){const auto start=Clock::now();scenes[mode]=build_demonstrator_scene(name);
            scenes[mode].use_bvh=false;scenes[mode].child_boundaries=true;scenes[mode].exact_child_states=mode==0;
            refresh_optical_children(scenes[mode]);beams[mode]=compile_beam_field(scenes[mode]);
            fields[mode]=compile_transport_field(scenes[mode],beams[mode],true);
            build_ms[mode]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
        for(int repeat=0;repeat<3;++repeat)for(int slot=0;slot<2;++slot){const int mode=(slot+repeat)%2;
            TraceContext ctx{scenes[mode],beams[mode],fields[mode]};RenderStats stats;
            std::uint64_t before_calls=0,before_hits=0,before_classes=0;for(const auto& child:scenes[mode].optical_children){
                before_calls+=child->evaluations;before_hits+=child->evaluated_internal_hits;before_classes+=child->classification_evaluations;}
            const auto start=Clock::now();const auto pixels=render_visible_edge_field(ctx,800,600,1,stats);
            const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            if(reference.empty())reference=pixels;
            update_require(pixels==reference,"shared observer law changed rendered bytes");
            std::uint64_t entries=0,bytes=0,calls=0,hits=0,classes=0;
            for(const auto& child:scenes[mode].optical_children){entries+=child->unit_compilations;bytes+=child->unit_cache_bytes;
                calls+=child->evaluations;hits+=child->evaluated_internal_hits;classes+=child->classification_evaluations;}
            if(mode)update_require(entries==0&&bytes==0,"renderer created sample registrations");
            if(comma)out<<",";comma=true;
            out<<"{\"scene\":\""<<name<<"\",\"mode\":\""<<(mode?"shared":"exact")<<"\",\"repeat\":"<<repeat
                <<",\"render_ms\":"<<ms<<",\"build_ms\":"<<build_ms[mode]<<",\"child_laws\":"<<scenes[mode].optical_children.size()
                <<",\"exact_state_entries\":"<<entries<<",\"exact_state_payload_bytes\":"<<bytes
                <<",\"classification_evaluations\":"<<classes-before_classes
                <<",\"evaluations\":"<<calls-before_calls<<",\"evaluated_internal_hits\":"<<hits-before_hits
                <<",\"quadrature\":"<<stats.trace.emitter_quadrature_samples<<",\"identical\":true}";out.flush();}
    }
    out<<"]}\n";
}
