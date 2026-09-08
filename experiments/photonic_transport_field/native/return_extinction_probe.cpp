#define main regime_scene_program_main
#include "regime_scene_native.cpp"
#undef main

Scene return_fixture(bool dielectric){
    Scene scene;
    Material material{"return surface",dielectric?MaterialKind::Dielectric:MaterialKind::Metal,
        {.8,.8,.8},{},.12,.24};
    if(dielectric){material.base={1,1,1};material.diffuse=0;material.ior=5;
        material.ior_rgb={5,5,5};material.thin=true;material.absorption={0,0,0};}
    const int surface=add_material(scene,material);
    add_rect(scene,"return A",{-2,-2,0},{4,0,0},{0,4,0},{0,0,1},surface);
    add_rect(scene,"return B",{-2,-2,2},{4,0,0},{0,4,0},{0,0,-1},surface);
    if(dielectric){
        const int left=add_material(scene,{"left source",MaterialKind::Emissive,{1,1,1},{2,2,2},0});
        const int right=add_material(scene,{"right source",MaterialKind::Emissive,{1,1,1},{5,5,5},0});
        add_rect(scene,"left exit",{-2,-2,-1},{4,0,0},{0,4,0},{0,0,1},left,false);
        add_rect(scene,"right exit",{-2,-2,3},{4,0,0},{0,4,0},{0,0,-1},right,false);
    }else{
        const int light=add_material(scene,{"side source",MaterialKind::Emissive,{1,1,1},{8,8,8},0});
        const int id=add_rect(scene,"side source",{-1,3,0},{2,0,0},{0,0,2},{0,-1,0},light,false);
        scene.area_lights.push_back({id,{8,8,8}});
    }
    build_scene_bvh(scene);scene.use_bvh=false;return scene;
}

void return_tests(){
    ReturnEvent state;state.hit.valid=true;state.hit.primitive=1;state.hit.position={0,0,0};
    state.hit.normal=state.hit.geometric_normal={0,0,1};state.incoming.ray.direction={0,0,-1};
    update_require(same_return_state(state,state.incoming,state.hit),"identical state rejected");
    Hit displaced=state.hit;displaced.position.x=1e-12;
    update_require(!same_return_state(state,state.incoming,displaced),"translated return admitted");
    OpticalPacket excluded=state.incoming;excluded.exclude_area_emitters=true;
    update_require(!same_return_state(state,excluded,state.hit),"source exclusion state merged");
    OpticalPacket reversed=state.incoming;reversed.ray.direction.z=1;
    update_require(!same_return_state(state,reversed,state.hit),"opposite incoming direction merged");
    for(bool dielectric:{false,true}){
        Scene scene=return_fixture(dielectric);const BeamField beams=compile_beam_field(scene);
        const TransportField field=compile_transport_field(scene,beams,true);
        TraceStats stats;TraceContext ctx{scene,beams,field,&stats,true,1e-12};
        const Ray ray{{0,0,1},{0,0,1}};
        return_mode=ReturnMode::Off;double reference=trace_channel_sealed(ctx,ray,128,0,nullptr);
        const auto reference_steps=stats.secondary;
        stats={};return_mode=ReturnMode::StateClosure;double closed=trace_channel_sealed(ctx,ray,128,0,nullptr);
        update_require(stats.return_closures>0,"fixture did not close a return");
        update_require(std::abs(reference-closed)<1e-9*(1+std::abs(reference)),"closed return differs from deep reference");
        if(dielectric){double r=schlick(1,1,5),analytic=(1-r)*(5+r*2)/(1-r*r);
            update_require(std::abs(analytic-closed)<1e-12,"leaking cavity differs from geometric series");}
        const auto closed_steps=stats.secondary;stats={};return_mode=ReturnMode::StateExtinction;
        double extinguished=trace_channel_sealed(ctx,ray,128,0,nullptr);
        update_require(stats.return_extinctions>0&&extinguished<reference,"extinction failed to remove cycle contribution");
        std::cout<<std::setprecision(17)<<"fixture "<<(dielectric?"dielectric exits":"rough sources")
            <<" reference="<<reference<<" closure="<<closed<<" extinction="<<extinguished
            <<" reference_steps="<<reference_steps<<" closure_steps="<<closed_steps<<"\n";
        stats={};return_mode=ReturnMode::StateExtinction;
        trace_channel_sealed(ctx,{{0,0,1},unit(Vec3{.03,0,1})},16,0,nullptr);
        update_require(stats.return_matches==0,"drifting path was treated as an exact return");
        if(dielectric){
            ReturnEvent event;event.incoming.weight=1;event.count=1;event.source=1;
            OpticalPacket returned;returned.weight=1;returned.return_parent=0;returned.return_edge=0;
            double added=0;std::vector<ReturnEvent> history{event};
            update_require(!close_return_cycle(ctx,history,returned,0,0,added),"driven unit-gain loop admitted");
            history[0].source=0;
            update_require(close_return_cycle(ctx,history,returned,0,0,added)&&added==0,"dark unit-gain loop rejected");
            returned.weight=.5;history[0].count=2;history[0].children[1].ray=ray;
            update_require(!close_return_cycle(ctx,history,returned,0,0,added),"nonterminal side exit admitted");
            history[0].count=1;history[0].pruned=true;
            update_require(!close_return_cycle(ctx,history,returned,0,0,added),"incomplete return cycle admitted");
        }
    }
    return_mode=ReturnMode::Off;std::cout<<"return-state invariants: ok\n";
}

int main(int argc,char** argv){
    if(argc==1){return_tests();return 0;}
    std::ofstream out(argv[1]);if(!out)throw std::runtime_error("cannot write probe output");
    const int width=800,height=600;
    out<<std::setprecision(17)<<"{\"width\":800,\"height\":600,\"frames\":[";bool comma=false;
    const std::array<ReturnMode,4> modes{ReturnMode::Off,ReturnMode::PathExtinction,
        ReturnMode::StateExtinction,ReturnMode::StateClosure};
    for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene scene=build_demonstrator_scene(name);scene.use_bvh=false;
        const BeamField beams=compile_beam_field(scene);const TransportField field=compile_transport_field(scene,beams,true);
        TraceContext ctx{scene,beams,field};std::vector<std::uint8_t> reference;
        return_mode=ReturnMode::Off;RenderStats initial;reference=render_visible_edge_field(ctx,width,height,1,initial);
        for(int repeat=0;repeat<3;++repeat)for(int slot=0;slot<4;++slot){
            return_mode=modes[(slot+repeat)%4];RenderStats stats;
            auto image=render_visible_edge_field(ctx,width,height,1,stats);
            int maximum=0;std::uint64_t changed=0;double sum=0;
            for(std::size_t i=0;i<image.size();++i){int d=std::abs(int(image[i])-int(reference[i]));
                maximum=std::max(maximum,d);changed+=d!=0;sum+=d;}
            if(comma)out<<",";comma=true;
            out<<"{\"scene\":\""<<name<<"\",\"mode\":\""<<return_mode_label()<<"\",\"repeat\":"<<repeat
                <<",\"raster_ms\":"<<stats.raster_ms<<",\"secondary\":"<<stats.trace.secondary
                <<",\"quadrature\":"<<stats.trace.emitter_quadrature_samples
                <<",\"tests\":"<<stats.trace.return_tests<<",\"matches\":"<<stats.trace.return_matches
                <<",\"extinctions\":"<<stats.trace.return_extinctions<<",\"closures\":"<<stats.trace.return_closures
                <<",\"rejections\":"<<stats.trace.return_rejections
                <<",\"changed_channels\":"<<changed<<",\"maximum_byte_error\":"<<maximum
                <<",\"mean_byte_error\":"<<sum/image.size()<<"}"<<std::flush;
            std::cerr<<name<<" "<<return_mode_label()<<" "<<repeat<<" ms="<<stats.raster_ms
                <<" matches="<<stats.trace.return_matches<<" extinct="<<stats.trace.return_extinctions
                <<" closed="<<stats.trace.return_closures<<" max="<<maximum<<"\n";
            if(repeat==0&&name=="aperture-canyon")write_ppm("/tmp/return_"+std::string(return_mode_label())+".ppm",image,width,height);
        }
    }
    out<<"]}\n";return_mode=ReturnMode::Off;
}
