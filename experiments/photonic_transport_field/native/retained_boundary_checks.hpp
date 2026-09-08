// Tests and paired benchmarks for exact boundary information reuse.
bool same_boundary_hit(const Hit& a,const Hit& b){
    auto same=[](Vec3 x,Vec3 y){return x.x==y.x&&x.y==y.y&&x.z==y.z;};
    return a.valid==b.valid&&a.primitive==b.primitive&&a.t==b.t&&a.front==b.front&&
        same(a.position,b.position)&&same(a.normal,b.normal)&&same(a.geometric_normal,b.geometric_normal);
}
void retained_boundary_self_test(){
    Scene scene=build_regime_scene();const BeamField beams=compile_beam_field(scene);
    const TransportField field=compile_transport_field(scene,beams,true);TraceContext ctx{scene,beams,field};
    const Camera camera=make_camera(800,600);boundary_shared_filter=true;boundary_path_capacity=128;
    for(bool bvh:{false,true}){scene.use_bvh=bvh;
        for(int capacity:{1,128}){OpticalQueryMemo memo(capacity);
            for(int i=0;i<2048;++i){const Ray ray=camera_ray(camera,800,600,(i*97)%800,(i*193)%600);
                const Hit first=first_hit(scene,ray);
                for(double limit:{std::numeric_limits<double>::infinity(),first.valid?first.t:1.,first.valid?first.t+1e-7:10.})
                    for(int ignore:{-1,first.primitive}){
                        const Hit expected=first_hit(scene,ray,limit,ignore);
                        update_require(same_boundary_hit(expected,memo.query(scene,ray,limit,ignore)),"optical query cache changed hit");
                        const auto reused=memo.reused;
                        update_require(same_boundary_hit(expected,memo.query(scene,ray,limit,ignore))&&memo.reused==reused+1,
                            "identical optical query did not reuse its geometric result");}}
        }
        VisibleEdgeField edges;edges.kind.resize(800*600);std::vector<int> pixels;
        auto add=[&](int x,int y){const int index=y*800+x;if(edges.kind[index])return;edges.kind[index]=10;pixels.push_back(index);};
        for(auto centre:std::array<std::pair<int,int>,5>{{{0,0},{797,597},{330,370},{510,280},{410,480}}})
            for(int y=centre.second;y<std::min(600,centre.second+3);++y)
                for(int x=centre.first;x<std::min(800,centre.first+3);++x)add(x,y);
        const auto shared=compile_boundary_filter_field(ctx,camera,800,600,edges,pixels);
        update_require(!shared.labels.empty()&&shared.labels.size()*16<shared.requests,"overlapping filters did not share queries");
        for(int pixel:pixels)for(int sy=0;sy<8;++sy)for(int sx=0;sx<8;++sx){
            const double x=pixel%800-1+(sx+.5)*2/8,y=pixel/800-1+(sy+.5)*2/8;
            const Ray ray=camera_ray(camera,800,600,x,y);const Hit hit=first_hit(scene,ray);
            const TerminalLabel expected=hit.valid?TerminalLabel{hit.primitive,visible_path_signature(ctx,ray,hit)}:TerminalLabel{};
            update_require(shared.at(pixel%800,pixel/800,sx,sy)==expected,"shared filter altered an optical label");}
        VisibleEdgeField sparse;sparse.kind.resize(800*600);sparse.kind[0]=10;
        update_require(compile_boundary_filter_field(ctx,camera,800,600,sparse,{0}).labels.empty(),
            "shared prepass admitted an isolated filter with no reuse");
    }
    RenderStats reference_stats,reused_stats;boundary_shared_filter=false;boundary_path_capacity=0;
    const auto reference=render_visible_edge_field(ctx,160,120,1,reference_stats);
    boundary_shared_filter=true;boundary_path_capacity=128;
    const auto reused=render_visible_edge_field(ctx,160,120,1,reused_stats);
    update_require(reference==reused&&reused_stats.boundary_optical_reused>0,
        "complete boundary reuse render differs or failed to reuse optical geometry");
    OpticalQueryMemo disabled(0);const Ray ray=camera_ray(camera,800,600,400,300);
    update_require(same_boundary_hit(first_hit(scene,ray),disabled.query(scene,ray,std::numeric_limits<double>::infinity(),-1)),
        "zero-capacity query oracle differs");
}

int run_boundary_discovery_benchmark(const std::string& out,int width,int height){
    if(width<16||height<16)throw std::invalid_argument("invalid boundary benchmark dimensions");
    const bool previous_audit=boundary_audit;boundary_audit=false;
    std::ofstream record(out);if(!record)throw std::runtime_error("cannot write boundary benchmark");
    record<<std::setprecision(12)<<"{\"width\": "<<width<<", \"height\": "<<height<<", \"terminal_error\": 1, \"cases\": [\n";
    bool comma=false;
    for(const std::string scene_name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene scene=build_demonstrator_scene(scene_name);
        const int poses=scene_name=="standard"?5:1;
        for(int pose=0;pose<poses;++pose){
            // Record both the CLI's small-scene linear policy and the BVH used
            // by the earlier retained-update/camera benchmarks.
            scene.use_bvh=pose==4||scene.primitives.size()>=64;
            const BeamField beams=compile_beam_field(scene);const TransportField field=compile_transport_field(scene,beams,true);
            TraceContext ctx{scene,beams,field};const Camera initial=make_camera(width,height);
            const Camera camera=pose==1?make_look_camera(width,height,initial.origin,
                    initial.origin+initial.forward*4+initial.right*.6,{0,1,0},27):
                pose==2?make_look_camera(width,height,initial.origin+Vec3{.1,0,0},initial.origin+initial.forward*4,{0,1,0},27):
                pose==3?make_prism_overhead_camera(width,height):initial;
            std::vector<std::uint8_t> reference;
            for(int repeat=0;repeat<3;++repeat)for(int order=0;order<4;++order){
                const int mode=(order+repeat)%4;boundary_shared_filter=(mode&1)!=0;boundary_path_capacity=(mode&2)?128:0;
                RenderStats stats;const auto start=Clock::now();const auto image=render_visible_edge_field(ctx,width,height,1,stats,&camera);
                const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if(reference.empty())reference=image;update_require(image==reference,"boundary reuse changed the camera image");
                if(comma)record<<",\n";comma=true;
                record<<"{\"scene\": \""<<scene_name<<"\", \"pose\": "<<pose<<", \"acceleration\": \""<<(scene.use_bvh?"bvh":"linear")
                    <<"\", \"mode\": "<<mode<<", \"repeat\": "<<repeat<<", \"camera_ms\": "<<elapsed
                    <<", \"adaptive_ms\": "<<stats.adaptive_ms<<", \"edges_ms\": "<<stats.edge_discovery_ms
                    <<", \"boundary_ms\": "<<stats.boundary_reconstruction_ms<<", \"shared_build_ms\": "<<stats.shared_filter_build_ms
                    <<", \"shared_requests\": "<<stats.shared_filter_requests<<", \"shared_queries\": "<<stats.shared_filter_queries
                    <<", \"shared_storage_bytes\": "<<stats.shared_filter_storage_bytes
                    <<", \"optical_queries\": "<<stats.boundary_optical_queries<<", \"optical_reused\": "<<stats.boundary_optical_reused
                    <<", \"labels\": "<<stats.boundary_samples<<", \"radiance_samples\": "<<stats.boundary_radiance_samples
                    <<", \"source_quadrature\": "<<stats.trace.emitter_quadrature_samples<<", \"exact_image_equal\": true}";
            }
            std::cout<<"boundary comparison: "<<scene_name<<" pose "<<pose<<" "<<(scene.use_bvh?"bvh":"linear")<<" ok\n";
        }
    }
    boundary_shared_filter=true;boundary_path_capacity=128;boundary_audit=previous_audit;
    record<<"\n]}\n";return 0;
}
