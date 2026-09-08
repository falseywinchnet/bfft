#define main regime_scene_program_main
#include "regime_scene_native.cpp"
#undef main

double boundary_environment(Vec3 d){return 1+.2*d.x+.3*d.y+.4*d.z;}
double independent_boundary_reference(const Scene& scene,const Ray& ray,int channel,int depth){
    const Hit hit=first_hit(scene,ray);
    if(!hit.valid)return boundary_environment(ray.direction);
    if(depth<=0)return 0;
    const Material& m=scene.materials[scene.primitives[hit.primitive].material];
    const double ni=hit.front?1:m.ior_rgb[channel],nt=hit.front?m.ior_rgb[channel]:1;
    Vec3 transmitted;const bool exits=refract(ray.direction,hit.normal,ni/nt,transmitted);
    const double f=exits?schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt):1;
    const Vec3 reflected=unit(reflect(ray.direction,hit.normal));
    double value=f*independent_boundary_reference(scene,{hit.position+reflected*1e-7,reflected},channel,depth-1);
    if(exits)value+=(1-f)*independent_boundary_reference(scene,{hit.position+transmitted*1e-7,transmitted},channel,depth-1);
    return m.base[channel]*std::exp(-m.absorption[channel]*.24)*value;
}

int main(int argc,char** argv){
    Scene scene=build_demonstrator_scene("aperture-canyon");
    scene.exact_child_states=true;refresh_optical_children(scene);
    const int id=scene.prism_volume;
    ConvexOpticalChild child(scene,id);
    update_require(child.faces.size()==5,"prism ownership incomplete");
    child.material.base={1,1,1};child.material.absorption={0,0,0};
    int tested=0;
    for(const auto& face:child.faces){
        const Vec3 position=face.second.center;
        Ray ray{position+face.second.normal*.2,-face.second.normal};
        const Hit entry=intersect_primitive(face.second,face.first,ray);
        if(!entry.valid)continue;
        const auto response=child.response(ray,entry,0,0,1e-12,512);
        double weight=response.unresolved_weight;
        for(const auto& port:response.ports){weight+=port.weight;
            const auto it=std::find_if(child.faces.begin(),child.faces.end(),[&](const auto& f){return f.first==port.primitive;});
            update_require(it!=child.faces.end(),"foreign child egress");
            update_require(dot(port.ray.direction,it->second.normal)>0,"internal direction leaked to parent");}
        update_require(std::abs(weight-1)<1e-10,"child energy balance failed");
        Scene private_oracle;private_oracle.materials.push_back(child.material);
        for(const auto& owned:child.faces){auto p=owned.second;p.material=0;private_oracle.primitives.push_back(p);}
        double observed=0;for(const auto& port:response.ports)observed+=port.weight*boundary_environment(port.ray.direction);
        const double reference=independent_boundary_reference(private_oracle,ray,0,129);
        update_require(std::abs(observed-reference)<1e-8+2*response.unresolved_weight,
            "owned supporting-plane response disagrees with independent finite-face transport");
        // Destroying the source registry cannot alter this retained owned operator.
        const auto saved_geometry=scene.primitives;
        for(auto& p:scene.primitives){p.center={1e6,1e6,1e6};p.radius=1e6;}
        const auto private_recompute=child.evaluate_law(ray,entry,0,0,1e-12,128);
        update_require(private_recompute.ports.size()==response.ports.size(),"private computation read mutated parent geometry");
        for(std::size_t p=0;p<response.ports.size();++p)
            update_require(private_recompute.ports[p].weight==response.ports[p].weight&&
                norm2(private_recompute.ports[p].ray.direction-response.ports[p].ray.direction)==0,
                "private boundary response changed after parent mutation");
        const auto marches=child.unit_compilations.load();
        const auto again=child.response(ray,entry,0,0,1e-6,16);
        update_require(again.cache_hit&&child.unit_compilations.load()==marches,"unit response recompiled for a new parent budget");
        scene.primitives=saved_geometry;
        update_require(again.ports.size()==response.ports.size(),"operator depends on foreign scene");
        ++tested;
    }
    update_require(tested==5,"boundary coverage incomplete");
    // Both sides of a zero-thickness subsystem are public boundaries.
    {ConvexOpticalChild sheet(scene,scene.glass_sheet);sheet.material.base={1,1,1};sheet.material.absorption={0,0,0};
        const auto& face=sheet.faces.front();
        for(double side:{-1.,1.}){const Ray ray{face.second.center+face.second.normal*(side*.2),face.second.normal*(-side)};
            const Hit hit=intersect_primitive(face.second,face.first,ray);
            const auto response=sheet.response(ray,hit,1,1,1e-12,64);double sum=response.unresolved_weight;
            for(const auto& port:response.ports)sum+=port.weight;
            update_require(response.ports.size()==2&&std::abs(sum-1)<1e-12,"two-sided sheet energy");}}
    Scene boundary_scene=scene;boundary_scene.child_boundaries=true;
    // Same straight source-light law, composed in each child's private domain.
    double source_error=0;int source_checks=0;
    for(int k=0;k<600;++k){const Vec3 a{-4.5+9.*(k%17)/16,.15+4.*(k%13)/12,-9.5+10.*(k%19)/18};
        const Vec3 b{-4.3+8.6*(k%23)/22,.25+5.*(k%11)/10,-9.+9.*(k%29)/28};
        const RGB old=segment_transmittance(scene,a,b),now=segment_transmittance(boundary_scene,a,b);
        for(int c=0;c<3;++c)source_error=std::max(source_error,std::abs(old[c]-now[c]));
        const auto program=discover_segment_program(boundary_scene,a,b);
        RGB value;
        try{value=evaluate_segment_radiance(boundary_scene,program,a,b,{1,1,1},nullptr);}
        catch(const std::exception& error){std::cerr<<"source k="<<k<<" a="<<a.x<<","<<a.y<<","<<a.z
            <<" b="<<b.x<<","<<b.y<<","<<b.z<<" word=";
            for(int w=0;w<program.count;++w)std::cerr<<program.primitive[w]<<",";
            std::cerr<<" "<<error.what()<<"\n";throw;}
        for(int c=0;c<3;++c)update_require(std::abs(value[c]-now[c])<1e-9,"child source program disagreement");
        ++source_checks;}
    update_require(source_error<1e-9,"source boundary closure changed straight visibility");
    // Reusing a source word must not confuse an ingress with an egress.
    const Vec3 pa=scene.primitives[scene.prism_volume].center;
    const Vec3 pn=scene.primitives[scene.prism_volume].normal;
    const Vec3 aa=pa+pn*.5,bb=pa-pn*.5;
    auto reverse_program=discover_segment_program(boundary_scene,aa,bb);
    const RGB reversed=evaluate_segment_radiance(boundary_scene,reverse_program,bb,aa,{1,1,1},nullptr);
    const RGB reversed_exact=segment_transmittance(boundary_scene,bb,aa);
    for(int c=0;c<3;++c)update_require(std::abs(reversed[c]-reversed_exact[c])<1e-9,"boundary role reuse failed");

    const BeamField owned_beams=compile_beam_field(boundary_scene);
    const TransportField owned_field=compile_transport_field(boundary_scene,owned_beams,true);
    update_require(owned_field.index_by_primitive[scene.jelly_core]<0,"private core leaked into parent field");
    update_require(owned_field.index_by_primitive[scene.jelly_volume]>=0,"participating boundary mode missing");
    // Source intensity is not in the retained optical operator.
    const auto retained=boundary_scene.optical_children.front();
    const auto& retained_face=retained->faces.front();
    const Ray unit_ray{retained_face.second.center+retained_face.second.normal*.2,-retained_face.second.normal};
    const Hit unit_hit=intersect_primitive(retained_face.second,retained_face.first,unit_ray);
    retained->response(unit_ray,unit_hit,1,1,1e-8,32);
    const auto compiled=retained->unit_compilations.load();
    boundary_scene.area_lights.front().radiance=boundary_scene.area_lights.front().radiance*2;
    refresh_optical_children(boundary_scene);
    update_require(boundary_scene.optical_children.front().get()==retained.get(),"intensity invalidated geometry response");
    update_require(retained->response(unit_ray,unit_hit,1,1,1e-6,16).cache_hit&&
        retained->unit_compilations.load()==compiled,"intensity caused another child march");
    const auto before_parallel=retained->unit_compilations.load();
    std::vector<std::thread> workers;
    for(int worker=0;worker<8;++worker)workers.emplace_back([&]{for(int repeat=0;repeat<32;++repeat)
        retained->response(unit_ray,unit_hit,1,.8123456,1e-8,32);});
    for(auto& worker:workers)worker.join();
    update_require(retained->unit_compilations.load()==before_parallel+1,"parallel unit response compiled more than once");
    const auto other_child=boundary_scene.optical_children.back();
    std::vector<std::pair<int,Primitive>> edits;
    for(int face=scene.prism_volume;face<=scene.prism_top;++face){Primitive moved=scene.primitives[face];
        moved.origin.x+=.1;moved.a.x+=.1;moved.b.x+=.1;moved.c.x+=.1;edits.emplace_back(face,moved);}
    Scene dynamic_scene=scene;dynamic_scene.child_boundaries=true;
    RetainedSceneState state(dynamic_scene);state.apply_geometry(edits);
    const TransportField cold=compile_transport_field(state.geometry(),state.beams(),true);
    double update_error=0;
    update_require(cold.node_primitives==state.field().node_primitives,"child mode changed node identity");
    for(std::size_t i=0;i<cold.radiance.size();++i)for(int c=0;c<3;++c)
        update_error=std::max(update_error,std::abs(cold.radiance[i][c]-state.field().radiance[i][c]));
    update_require(update_error<1e-8,"child incremental update differs from cold field");
    edit_scene_geometry(boundary_scene,std::move(edits));
    update_require(boundary_scene.optical_children.front().get()!=retained.get(),"geometry left stale child response");
    update_require(boundary_scene.optical_children.back()==other_child,"unrelated child response was invalidated");
    std::cout<<"child boundary ownership and energy checks passed: "<<tested<<" faces; "<<source_checks
        <<" source checks, max error "<<source_error<<"; private core excluded; revisions checked\n";
    if(argc>1){
        std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"frames\":[";bool comma=false;
        for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
            std::array<Scene,2> scenes;std::array<BeamField,2> beams;std::array<TransportField,2> fields;
            std::array<double,2> build_ms{};std::array<std::vector<std::uint8_t>,2> images;
            for(int mode=0;mode<2;++mode){const auto start=Clock::now();
                scenes[mode]=build_demonstrator_scene(name);scenes[mode].use_bvh=false;scenes[mode].child_boundaries=mode;
                scenes[mode].exact_child_states=true;refresh_optical_children(scenes[mode]);
                beams[mode]=compile_beam_field(scenes[mode]);fields[mode]=compile_transport_field(scenes[mode],beams[mode],true);
                build_ms[mode]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
            for(int repeat=0;repeat<3;++repeat)for(int slot=0;slot<2;++slot){const int mode=(slot+repeat)%2;
                TraceContext ctx{scenes[mode],beams[mode],fields[mode]};RenderStats stats;
                auto count=[&](int which){std::uint64_t n=0;for(const auto& child:scenes[mode].optical_children)
                    n+=which==0?child->unit_compilations.load():which==1?child->unit_internal_hits.load():child->unit_cache_bytes.load();return n;};
                const auto before=count(0),before_hits=count(1);
                const auto start=Clock::now();images[mode]=render_visible_edge_field(ctx,800,600,1,stats);
                const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if(comma)out<<",";comma=true;
                out<<"{\"scene\":\""<<name<<"\",\"mode\":"<<mode<<",\"repeat\":"<<repeat
                    <<",\"build_ms\":"<<build_ms[mode]<<",\"render_ms\":"<<ms
                    <<",\"secondary\":"<<stats.trace.secondary<<",\"quadrature\":"<<stats.trace.emitter_quadrature_samples
                    <<",\"child_responses\":"<<stats.trace.child_responses<<",\"child_internal_hits\":"<<stats.trace.child_internal_hits
                    <<",\"new_unit_responses\":"<<(count(0)-before)<<",\"new_unit_internal_hits\":"<<(count(1)-before_hits)
                    <<",\"resident_unit_responses\":"<<count(0)<<",\"unit_payload_bytes\":"<<count(2)
                    <<",\"child_ports\":"<<stats.trace.child_egress_ports<<",\"unresolved_weight_sum\":"<<stats.trace.child_unresolved_weight<<"}";
                out.flush();}
            for(int mode=0;mode<2;++mode){std::ofstream image("/tmp/child_boundary_"+name+"_"+std::to_string(mode)+".ppm",std::ios::binary);
                image<<"P6\n800 600\n255\n";image.write(reinterpret_cast<const char*>(images[mode].data()),images[mode].size());}
        }
        out<<"]}\n";
    }
}
