// Finite source support: preserve the numerical transport operator while
// eliminating annihilated arithmetic and angle reconstruction.
void require_same_source_fields(const TransportField& a,const TransportField& b){
    update_require(a.direct==b.direct&&a.bounce==b.bounce&&a.radiance==b.radiance&&a.raw_coupling==b.raw_coupling&&
        a.direct_atlas_by_primitive==b.direct_atlas_by_primitive&&a.direct_atlas.size()==b.direct_atlas.size(),
        "source support changed fresh transport field");
    for(std::size_t i=0;i<a.direct_atlas.size();++i){const auto& x=a.direct_atlas[i];const auto& y=b.direct_atlas[i];
        update_require(x.value==y.value&&x.refined_value==y.refined_value&&x.refined_offset==y.refined_offset,
            "source support changed fresh direct-light atlas");}
}

void retained_source_self_test(){
    source_zero_elision=true;source_cone_algebraic=true;
    std::uint64_t comparisons=0,certificates=0,cone_tests=0;
    auto compare=[&](const Scene& scene,const SegmentProgram& program,Vec3 receiver,Vec3 source,
                     int ignore_receiver,int ignore_source){
        for(const RGB incident:{RGB{},RGB{1.2,.7,3.1}}){const RGB medium{.3,1.1,.7};RGB a{},b{};
            source_zero_elision=false;const RGB reference=evaluate_segment_radiance(scene,program,receiver,source,
                incident,&medium,ignore_receiver,ignore_source,&a);
            source_zero_elision=true;TraceStats stats;const RGB value=evaluate_segment_radiance(scene,program,receiver,source,
                incident,&medium,ignore_receiver,ignore_source,&b,&stats);
            update_require(reference==value&&a==b,"zero suffix changed radiance or affine medium response");
            certificates+=stats.source_zero_certificates;++comparisons;}
    };
    for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene scene=build_demonstrator_scene(name);
        for(int receiver=0;receiver<int(scene.primitives.size());++receiver){
            const auto samples=surface_samples(scene.primitives[receiver]);
            for(std::size_t sample=0;sample<std::min<std::size_t>(3,samples.size());++sample){
                const Vec3 point=samples[sample].position+samples[sample].normal*2e-4;
                for(const AreaLight& area:scene.area_lights){const Primitive& light=scene.primitives[area.primitive];
                    const SourceCone cone(light,point);
                    for(const Primitive& blocker:scene.primitives){
                        update_require(cone.overlaps(light,blocker,point)==blocker_may_overlap_emitter_angles(light,blocker,point),
                            "algebraic source cone changed a scene candidate");++cone_tests;}
                    for(double u:{.1,.5,.9}){const Vec3 centre=light.origin+light.u*u+light.v*.5;
                        const auto program=discover_segment_program(scene,point,centre,receiver,area.primitive);
                        // Include ordinates outside the discovered membership, both ignores,
                        // and zero incident light with a nonzero participating-medium source.
                        for(double x:{-.1,.17,.5,.83,1.1})for(double y:{.13,.5,.87}){
                            const Vec3 source=light.origin+light.u*x+light.v*y;
                            compare(scene,program,point,source,receiver,area.primitive);
                            if(program.last_opaque>=0)compare(scene,program,point,source,
                                program.primitive[program.last_opaque],area.primitive);}
                    }
                }
            }
        }
    }
    for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene scene=build_demonstrator_scene(name);const BeamField beams=compile_beam_field(scene);
        source_zero_elision=false;source_cone_algebraic=false;
        const TransportField reference=compile_transport_field(scene,beams,true);
        source_zero_elision=true;source_cone_algebraic=true;
        const TransportField value=compile_transport_field(scene,beams,true);
        require_same_source_fields(reference,value);
    }
    // Probe the angular decision at and on both sides of grazing boundaries,
    // including zero-sized bounds and receiver horizons.
    Primitive light,blocker;light.shape=blocker.shape=Shape::Sphere;
    for(double alpha:{0.,1e-8,.01,.6,1.4,pi*.5-.01,pi*.5-1e-4,pi*.5-1e-6,pi*.5-1e-8})
        for(double beta:{0.,1e-8,.01,.6,1.4,pi*.5-.01,pi*.5-1e-4,pi*.5-1e-6,pi*.5-1e-8})
            for(double epsilon:{-1e-7,-1e-12,-1e-14,0.,1e-14,1e-12,1e-7}){
                light.center={0,0,10};light.radius=10*std::sin(alpha);
                const double theta=alpha+beta+epsilon;
                blocker.center={5*std::sin(theta),0,5*std::cos(theta)};blocker.radius=5*std::sin(beta);
                update_require(blocker_may_overlap_emitter(light,blocker,{})==
                    blocker_may_overlap_emitter_angles(light,blocker,{}),"algebraic cone changed grazing decision");++cone_tests;
            }
    update_require(certificates>100&&comparisons>10000&&cone_tests>10000,"source support tests did not exercise enough cases");
    std::cout<<"source support comparisons="<<comparisons<<" zero certificates="<<certificates<<" cone tests="<<cone_tests<<"\n";
}

int run_source_expansion_benchmark(const std::string& out,int width,int height){
    if(width<16||height<16)throw std::invalid_argument("invalid source benchmark dimensions");
    std::ofstream record(out);if(!record)throw std::runtime_error("cannot write source benchmark");
    expansion_audit_out.clear();boundary_audit=false;
    record<<std::setprecision(12)<<"{\"width\":"<<width<<",\"height\":"<<height<<",\"frames\":[";
    bool comma=false;
    for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene scene=build_demonstrator_scene(name);const BeamField beams=compile_beam_field(scene);
        source_zero_elision=false;source_cone_algebraic=false;
        const TransportField field=compile_transport_field(scene,beams,true);TraceContext ctx{scene,beams,field};
        for(bool bvh:{false,true}){scene.use_bvh=bvh;
            const int poses=name=="standard"?3:1;
            for(int pose=0;pose<poses;++pose){const Camera camera=pose==0?make_camera(width,height):
                make_look_camera(width,height,journey_pose(.13*pose,name).position,journey_pose(.13*pose,name).target,
                    {0,1,0},journey_pose(.13*pose,name).half_fov_degrees);
                std::vector<std::uint8_t> reference;
                for(int repeat=0;repeat<3;++repeat)for(int slot=0;slot<3;++slot){
                    const int mode=(slot+repeat)%3;source_zero_elision=mode!=0;source_cone_algebraic=mode==2;
                    RenderStats stats;const auto image=render_visible_edge_field(ctx,width,height,1,stats,&camera);
                    if(reference.empty())reference=image;
                    update_require(reference==image,"source support changed complete frame");
                    if(comma)record<<",";comma=true;
                    record<<"{\"scene\":\""<<name<<"\",\"acceleration\":\""<<(bvh?"bvh":"linear")
                        <<"\",\"pose\":"<<pose<<",\"repeat\":"<<repeat<<",\"mode\":"<<mode
                        <<",\"raster_ms\":"<<stats.raster_ms<<",\"boundary_ms\":"<<stats.boundary_reconstruction_ms
                        <<",\"adaptive_ms\":"<<stats.adaptive_ms<<",\"quadrature\":"<<stats.trace.emitter_quadrature_samples
                        <<",\"zero_certificates\":"<<stats.trace.source_zero_certificates
                        <<",\"zero_lobes\":"<<stats.trace.specular_zero_lobes<<",\"identical\":true}";
                    record.flush();
                }
            }
        }
    }
    record<<"],\"compilation\":[";comma=false;
    for(const std::string name:{"standard","aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene scene=build_demonstrator_scene(name);const BeamField beams=compile_beam_field(scene);
        for(int repeat=0;repeat<3;++repeat)for(int slot=0;slot<2;++slot){const bool optimized=(slot+repeat)%2;
            source_zero_elision=optimized;source_cone_algebraic=optimized;
            const auto start=Clock::now();const TransportField field=compile_transport_field(scene,beams,true);
            const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            if(comma)record<<",";comma=true;record<<"{\"scene\":\""<<name<<"\",\"repeat\":"<<repeat
                <<",\"optimized\":"<<(optimized?"true":"false")<<",\"field_ms\":"<<elapsed<<"}";record.flush();
        }
    }
    record<<"]}\n";source_zero_elision=true;source_cone_algebraic=true;return 0;
}
