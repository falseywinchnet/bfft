// Included by the integrated renderer: the eager atlas is the independent
// representation oracle, with unchanged refinement and quadrature rules.
void retained_camera_self_test(){
    const Scene scene=build_regime_scene();const BeamField beams=compile_beam_field(scene);
    const TransportField field=compile_transport_field(scene,beams,true);TraceContext context{scene,beams,field};
    const Vec3 eye=make_camera(800,600).origin;
    const auto eager=compile_viewer_origin_specular_field(context,eye,false);
    const auto demand=compile_viewer_origin_specular_field(context,eye,true);
    update_require(camera_demand_counts(demand)[0]==0,"demand field integrated before a gather");
    TraceContext ctx=context;ctx.camera_specular=&demand;
    struct Query{int index;Vec3 point;RGB expected;};std::vector<Query> queries;
    const std::array<double,9> coordinates{0,1e-12,.03125,.125,.4999,.5,.875,1-1e-12,1};
    for(std::size_t index=0;index<eager.atlas.size();++index){const auto& atlas=eager.atlas[index];
        const Primitive& primitive=scene.primitives[atlas.primitive];
        auto add=[&](double u,double v){Vec3 point;
            if(atlas.sphere){const double ny=1-2*v,r=std::sqrt(std::max(0.0,1-ny*ny)),angle=2*pi*u-pi;
                point=primitive.center+Vec3{r*std::cos(angle),ny,r*std::sin(angle)}*primitive.radius;
            }else point=primitive.origin+primitive.u*u+primitive.v*v;
            queries.push_back({static_cast<int>(index),point,sample_surface_atlas(atlas,primitive,point)});};
        for(double u:coordinates)for(double v:coordinates)add(u,v);
        for(int i=0;i<256;++i)add(std::fmod(i*.61803398875,1.0),std::fmod(i*.41421356237,1.0));
    }
    std::atomic<bool> matches{true};std::vector<std::thread> workers;
    // Every worker revisits all queries in a different order. This stresses
    // simultaneous first requests for shared samples and refinement decisions.
    for(int worker=0;worker<8;++worker)workers.emplace_back([&,worker]{
        for(std::size_t i=0;i<queries.size();++i){const Query& q=queries[(i+worker*31)%queries.size()];
            const RGB actual=gather_camera_specular(ctx,*demand.demand_atlas[q.index],q.point);
            for(int c=0;c<3;++c)if(actual[c]!=q.expected[c])matches=false;}});
    for(auto& worker:workers)worker.join();update_require(matches,"concurrent demand gather differs from eager atlas");
    const auto populated=camera_demand_counts(demand);
    for(const auto& q:queries)gather_camera_specular(ctx,*demand.demand_atlas[q.index],q.point);
    update_require(populated==camera_demand_counts(demand),"repeat gathers reintegrated retained samples");
    update_require(populated[0]>0&&populated[0]<=eager.samples,"invalid demand sample count");
    Scene inserted=scene;inserted.use_bvh=true;Primitive added=scene.primitives.front();
    added.shape=Shape::Sphere;added.center=eye+make_camera(800,600).forward*2;added.radius=.05;
    edit_scene_geometry(inserted,{}, {added});std::vector<int> candidates;
    query_camera_primitives(inserted,make_camera(800,600),candidates);
    update_require(std::find(candidates.begin(),candidates.end(),static_cast<int>(inserted.primitives.size())-1)!=candidates.end(),
        "camera contour discovery missed buffered insertion");
}

int run_camera_gather_benchmark(const std::string& out,int width,int height){
    if(width<16||height<16)throw std::invalid_argument("invalid camera benchmark dimensions");
    const Scene scene=build_regime_scene();const BeamField beams=compile_beam_field(scene);
    const TransportField field=compile_transport_field(scene,beams,true);TraceContext ctx{scene,beams,field};
    const Camera initial=make_camera(width,height);
    const std::array<Camera,4> cameras{initial,
        make_look_camera(width,height,initial.origin,initial.origin+initial.forward*4+initial.right*.6,{0,1,0},27),
        make_look_camera(width,height,initial.origin+Vec3{.1,0,0},initial.origin+initial.forward*4,{0,1,0},27),
        make_prism_overhead_camera(width,height)};
    std::ofstream record(out);if(!record)throw std::runtime_error("cannot write camera benchmark");
    record<<std::setprecision(12)<<"{\"width\": "<<width<<", \"height\": "<<height
        <<", \"scene\": \"standard\", \"terminal_error\": 1, \"cases\": [\n";
    bool comma=false;std::array<std::vector<std::uint8_t>,4> references;
    auto report=[&](int pose,const char* mode,int repeat,double elapsed,const RenderStats& stats){
        if(comma)record<<",\n";comma=true;
        record<<"{\"pose\": "<<pose<<", \"mode\": \""<<mode<<"\", \"repeat\": "<<repeat
            <<", \"camera_ms\": "<<elapsed<<", \"setup_ms\": "<<stats.camera_specular_build_ms
            <<", \"adaptive_ms\": "<<stats.adaptive_ms<<", \"edges_ms\": "<<stats.edge_discovery_ms
            <<", \"boundary_ms\": "<<stats.boundary_reconstruction_ms<<", \"source_samples\": "<<stats.camera_specular_samples
            <<", \"new_source_samples\": "<<stats.camera_specular_new_samples
            <<", \"new_quadrature_samples\": "<<stats.camera_specular_quadrature_samples
            <<", \"new_cell_tests\": "<<stats.camera_specular_cell_tests
            <<", \"edge_crossing_storage_bytes\": "<<stats.edge_crossing_storage_bytes
            <<", \"boundary_samples\": "<<stats.boundary_samples
            <<", \"secondary_specular_calls\": "<<stats.trace.secondary_specular_area_calls
            <<", \"exact_image_equal\": true}";
    };
    for(int pose=0;pose<4;++pose)for(int repeat=0;repeat<3;++repeat)for(int order=0;order<3;++order){
        const int mode=(order+repeat)%3;const bool demand=mode==2;
        camera_demand_gather=demand;camera_sparse_crossings=mode!=0;RenderStats stats;
        const auto start=Clock::now();const auto image=render_visible_edge_field(ctx,width,height,1,stats,&cameras[pose]);
        const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        if(references[pose].empty())references[pose]=image;
        update_require(image==references[pose],"camera demand/eager images differ");
        report(pose,demand?"demand":(mode==0?"eager-dense":"eager"),repeat,elapsed,stats);
    }
    camera_demand_gather=true;camera_sparse_crossings=true;const auto retained=compile_viewer_origin_specular_field(ctx,initial.origin,true);
    for(int step=0;step<3;++step){const int pose=step==2?1:0;RenderStats stats;
        const auto start=Clock::now();const auto image=render_visible_edge_field(ctx,width,height,1,stats,&cameras[pose],&retained);
        const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        update_require(image==references[pose],"retained camera rotation differs from eager image");
        if(step==1)update_require(stats.camera_specular_new_samples==0&&stats.camera_specular_quadrature_samples==0,
            "identical camera frame reintegrated source samples");
        report(pose,"retained-demand",step,elapsed,stats);
    }
    record<<"\n]}\n";std::cout<<"camera gather benchmark: "<<out<<"\n";return 0;
}
