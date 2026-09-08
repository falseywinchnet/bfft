// Included inside the proof renderer's namespace: exercises its actual geometry,
// retained operator, area emitters, optical trace, and camera implementations.
void update_require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}

double update_field_error(const TransportField& a,const TransportField& b){
    update_require(a.node_primitives==b.node_primitives,"updated transport node IDs differ");
    update_require(a.incoming_offset==b.incoming_offset,"updated edge support differs");
    double error=0;
    for(std::size_t i=0;i<a.coupling.size();++i){
        update_require(a.coupling[i].source==b.coupling[i].source,"updated edge source differs");
        update_require(std::abs(a.raw_coupling[i]-b.raw_coupling[i])<1e-12,"updated raw coupling differs");
        update_require(std::abs(a.coupling[i].weight-b.coupling[i].weight)<1e-12,"updated normalized coupling differs");}
    for(std::size_t i=0;i<a.radiance.size();++i)for(int c=0;c<3;++c){
        update_require(std::abs(a.direct[i][c]-b.direct[i][c])<1e-10,"updated direct irradiance differs");
        error+=std::abs(a.radiance[i][c]-b.radiance[i][c]);}
    update_require(error<=a.certified_tail+b.certified_tail+1e-10,"updated radiance exceeds both tail bounds");
    update_require(a.direct_atlas_by_primitive==b.direct_atlas_by_primitive,"updated atlas IDs differ");
    for(std::size_t i=0;i<a.direct_atlas.size();++i){const auto& x=a.direct_atlas[i];const auto& y=b.direct_atlas[i];
        update_require(x.refined_offset==y.refined_offset,"updated atlas refinement differs");
        update_require(x.value.size()==y.value.size()&&x.refined_value.size()==y.refined_value.size(),"updated atlas size differs");
        for(std::size_t j=0;j<x.value.size();++j)for(int c=0;c<3;++c)
            if(std::abs(x.value[j][c]-y.value[j][c])>=1e-9){
                std::cerr<<std::setprecision(17)<<"atlas mismatch primitive="<<x.primitive<<" sample="<<j<<" channel="<<c
                    <<" reused="<<x.value[j][c]<<" fresh="<<y.value[j][c]<<"\n";
                throw std::runtime_error("updated atlas value differs");}
        for(std::size_t j=0;j<x.refined_value.size();++j)for(int c=0;c<3;++c)
            update_require(std::abs(x.refined_value[j][c]-y.refined_value[j][c])<1e-9,"updated refined atlas value differs");}
    return error;
}

void check_updated_intersections(const Scene& scene){
    Scene linear=scene;linear.use_bvh=false;Scene rebuilt=scene;build_scene_bvh(rebuilt);rebuilt.use_bvh=true;
    for(int i=0;i<2048;++i){const int target=i%static_cast<int>(scene.primitives.size());
        Vec3 origin{7*std::sin(i*.81),5*std::cos(i*.39)+2,9*std::sin(i*.17)};
        const Ray ray{origin,unit(scene.primitives[target].center-origin)};
        for(double limit:{std::numeric_limits<double>::infinity(),.5,2.0,10.0}){
            const Hit x=first_hit(scene,ray,limit),y=first_hit(linear,ray,limit),z=first_hit(rebuilt,ray,limit);
            update_require(x.valid==y.valid&&x.valid==z.valid,"dynamic BVH hit validity differs");
            if(x.valid)update_require(x.primitive==y.primitive&&x.primitive==z.primitive&&
                std::abs(x.t-y.t)<1e-10&&std::abs(x.t-z.t)<1e-10,"dynamic BVH closest hit differs");}
    }
    for(int id=0;id<static_cast<int>(scene.primitives.size());++id){std::vector<int> candidates;
        query_scene_bounds(scene,primitive_bounds(scene.primitives[id]),candidates);
        update_require(std::find(candidates.begin(),candidates.end(),id)!=candidates.end(),"dynamic bounds omitted primitive");}
}

Scene update_fixture(){Scene s;const int diffuse=add_material(s,{"matte",MaterialKind::Diffuse,{.6,.7,.8},{},.9});
    const int glow=add_material(s,{"light",MaterialKind::Emissive,{1,1,1},{5,4,3},0});
    for(int room=0;room<3;++room){const double x=room*12;
        add_rect(s,"floor",{x-2,0,-2},{4,0,0},{0,0,4},{0,1,0},diffuse);
        add_rect(s,"ceiling",{x-2,4,-2},{4,0,0},{0,0,4},{0,-1,0},diffuse);
        add_rect(s,"back",{x-2,0,-2},{4,0,0},{0,4,0},{0,0,1},diffuse);
        const int light=add_rect(s,"emitter",{x-.5,3.8,-.5},{1,0,0},{0,0,1},{0,-1,0},glow,false);
        s.area_lights.push_back({light,{5,4,3}});
    }
    add_sphere(s,"moving blocker",{0,2,0},.45,diffuse,false);build_scene_bvh(s);return s;
}

bool retained_update_self_test(){
    // A blocker crossing the receiver horizon used to fail vertex projection.
    // Its source window is still a finite half-plane cut, independently checked
    // against ray/primitive intersections at every source test point.
    Scene clip_scene;add_material(clip_scene,{"matte",MaterialKind::Diffuse,{.7,.7,.7}});
    const int light_id=add_rect(clip_scene,"light",{-1,-1,2},{2,0,0},{0,2,0},{0,0,-1},0,false);
    const int blocker_id=add_rect(clip_scene,"horizon blocker",{.2,-2,-1},{0,4,0},{0,0,4},{1,0,0},0,false);
    const Primitive& light=clip_scene.primitives[light_id];
    const Primitive& blocker_geometry=clip_scene.primitives[blocker_id];
    for(Vec3 target:{Vec3{0,0,0},Vec3{-.3,.2,-.4},Vec3{.8,.1,.2}}){
        const ProjectedPolygon polygon=clip_blocker_to_emitter(light,blocker_geometry,target,blocker_id);
        for(int y=0;y<39;++y)for(int x=0;x<41;++x){Vec2 uv{(x+.371)/41,(y+.619)/39};
            bool inside=polygon.count>=3;
            for(int i=0;i<polygon.count;++i){Vec2 a=polygon.vertex[i],b=polygon.vertex[(i+1)%polygon.count];
                if((b.x-a.x)*(uv.y-a.y)-(b.y-a.y)*(uv.x-a.x)<-1e-12)inside=false;}
            Vec3 delta=light.origin+light.u*uv.x+light.v*uv.y-target;
            const Hit hit=intersect_primitive(blocker_geometry,blocker_id,{target,unit(delta)},norm(delta));
            update_require(inside==hit.valid,"finite source half-space clipping differs from ray visibility");
        }
    }

    Scene indirect;const int indirect_matte=add_material(indirect,{"indirect_matte",MaterialKind::Diffuse,{.7,.7,.7},{},.9});
    const int emitter_material=add_material(indirect,{"light",MaterialKind::Emissive,{1,1,1},{4,4,4},0});
    const int receiver=add_rect(indirect,"fully shadowed receiver",{-.5,0,-.5},{1,0,0},{0,0,1},{0,1,0},indirect_matte);
    add_rect(indirect,"relay",{2,0,-1},{0,3,0},{0,0,2},{-1,0,0},indirect_matte);
    add_rect(indirect,"occluder",{-1,2,-1},{2,0,0},{0,0,2},{0,1,0},indirect_matte,false);
    const int lamp=add_rect(indirect,"lamp",{-.5,4,-.5},{1,0,0},{0,0,1},{0,-1,0},emitter_material,false);
    indirect.area_lights.push_back({lamp,{4,4,4}});build_scene_bvh(indirect);
    const auto indirect_beams=compile_beam_field(indirect);const auto indirect_field=compile_transport_field(indirect,indirect_beams);
    const int receiver_node=indirect_field.index_by_primitive[receiver];
    update_require(rgb_energy(indirect_field.direct[receiver_node])<1e-12&&
        rgb_energy(indirect_field.bounce[receiver_node])>1e-5,"strictly indirect-only illumination failed");

    // Independently evaluate the ordinary source-valued integral and the
    // retained affine response at several source spectra, including zero.
    const Scene standard=build_regime_scene();bool saw_volume_response=false;
    for(int id=0;id<static_cast<int>(standard.primitives.size());++id){
        const auto samples=surface_samples(standard.primitives[id]);if(samples.empty())continue;
        const auto& q=samples[samples.size()/2];const Vec3 point=q.position+q.normal*2e-4;RGB coefficient{};
        const RGB base=area_irradiance(standard,point,q.normal,id,nullptr,nullptr,&coefficient);
        saw_volume_response|=coefficient[0]>1e-12;
        for(RGB source:{RGB{},RGB{.1,.7,.2},RGB{3,1,5}}){
            const RGB direct=area_irradiance(standard,point,q.normal,id,nullptr,&source);
            const RGB retained=base+multiply(coefficient,source);
            for(int c=0;c<3;++c)update_require(std::abs(direct[c]-retained[c])<1e-10,
                "retained affine source response differs from direct integration");
        }
    }
    update_require(saw_volume_response,"affine response test did not cross the volume");
    Scene scene=update_fixture();BeamField beams=compile_beam_field(scene);
    TransportField field=compile_transport_field(scene,beams);
    Scene linear_fixture=scene;linear_fixture.use_bvh=false;
    update_field_error(field,compile_transport_field(linear_fixture,beams));
    const int blocker=static_cast<int>(scene.primitives.size())-1;
    bool saw_reuse=false,saw_negative=false,saw_increased_light=false;
    auto check=[&](const GeometryChanges& changes){
        check_updated_intersections(scene);beams=compile_beam_field(scene);
        TransportField updated=compile_transport_field(scene,beams,true,&field,&changes);
        const TransportField fresh=compile_transport_field(scene,beams,true);
        update_field_error(updated,fresh);
        saw_reuse|=updated.reused_pairs>0;saw_negative|=updated.negative_defect_channels>0;
        for(std::size_t i=0;i<field.direct.size()&&i<updated.direct.size();++i)
            saw_increased_light|=updated.direct[i][0]>field.direct[i][0]+1e-4;
        field=std::move(updated);
    };
    Primitive moved=scene.primitives[blocker];moved.center.x=2.5;
    check(edit_scene_geometry(scene,{{blocker,moved}}));
    moved.center={0,1,0};moved.radius=.65;check(edit_scene_geometry(scene,{{blocker,moved}}));
    moved.intersectable=false;check(edit_scene_geometry(scene,{{blocker,moved}}));
    moved.intersectable=true;moved.transport=true;check(edit_scene_geometry(scene,{}, {moved}));
    Primitive floor=scene.primitives[0];floor.u={3.5,.1,.4};check(edit_scene_geometry(scene,{{0,floor}}));
    // Edges with zero old visibility must become nonzero when an intervening
    // opaque sheet is moved out. This is separate from direct-source relighting.
    Scene closed;const int matte=add_material(closed,{"matte",MaterialKind::Diffuse,{.7,.7,.7},{},.9});
    add_rect(closed,"source",{-1,0,-1},{2,0,0},{0,0,2},{0,1,0},matte);
    add_rect(closed,"receiver",{-1,2,-1},{2,0,0},{0,0,2},{0,-1,0},matte);
    const int barrier=add_rect(closed,"barrier",{-2,1,-2},{4,0,0},{0,0,4},{0,1,0},matte,false);
    build_scene_bvh(closed);auto cb=compile_beam_field(closed);auto cf=compile_transport_field(closed,cb);
    update_require(cf.nonzeros==0,"closed relation fixture unexpectedly connected");
    Primitive removed=closed.primitives[barrier];removed.intersectable=false;
    auto cc=edit_scene_geometry(closed,{{barrier,removed}});
    auto opened=compile_transport_field(closed,cb,true,&cf,&cc);auto cold=compile_transport_field(closed,cb,true);
    update_require(opened.nonzeros==2,"newly opened zero relation was not rediscovered");update_field_error(opened,cold);
    // Force addition-delta rebuild, including primitives outside the old root.
    std::vector<Primitive> additions(40,moved);
    for(std::size_t i=0;i<additions.size();++i){additions[i].center={50+double(i),2,0};additions[i].transport=false;}
    const auto appended=edit_scene_geometry(scene,{},additions);
    update_require(appended.rebuilt,"addition delta failed to compact");check_updated_intersections(scene);
    const auto revision=scene.geometry_revision;Primitive invalid=moved;invalid.radius=-1;bool rejected=false;
    try{edit_scene_geometry(scene,{{blocker,invalid}});}catch(const std::invalid_argument&){rejected=true;}
    update_require(rejected&&scene.geometry_revision==revision,"invalid edit was not rejected before mutation");
    // Empty and all-delta scenes, including transition into first hierarchy.
    Scene empty;auto eb=compile_beam_field(empty);auto ef=compile_transport_field(empty,eb);
    update_require(ef.radiance.empty()&&ef.certified_tail==0,"empty retained scene failed");
    add_material(empty,{"matte",MaterialKind::Diffuse,{.7,.7,.7}});moved.material=0;
    auto ec=edit_scene_geometry(empty,{}, {moved});check_updated_intersections(empty);
    eb=compile_beam_field(empty);auto added=compile_transport_field(empty,eb,true,&ef,&ec);
    update_require(added.node_primitives.size()==1,"first diffuse primitive not registered");
    update_require(saw_reuse&&saw_negative&&saw_increased_light,"update fixtures missed reuse, subtraction, or disocclusion");
    RetainedSceneState owned(update_fixture());const auto generation=owned.field().generation;
    const auto no_change=owned.apply_geometry({});
    update_require(no_change.primitives.empty()&&owned.field().generation==generation&&owned.last_field_ms==0,
        "no-op edit recompiled the retained state");
    return true;
}

int run_retained_update_benchmark(const std::string& out,int width,int height){
    if(width<16||height<16)throw std::invalid_argument("invalid update benchmark dimensions");
    // Standard proof scene, unchanged physical and retained renderer path.
    auto start=Clock::now();RetainedSceneState state(build_regime_scene());
    const Scene& scene=state.geometry();const BeamField& beams=state.beams();const TransportField& field=state.field();
    const double initial_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    const Camera camera=make_camera(width,height);TraceContext before{scene,beams,field};
    const CameraSpecularField stale=compile_viewer_origin_specular_field(before,camera.origin);
    int blocker=-1;for(int i=0;i<static_cast<int>(scene.primitives.size());++i)
        if(scene.primitives[i].shape==Shape::Sphere&&scene.primitives[i].transport&&
           scene.materials[scene.primitives[i].material].kind==MaterialKind::Diffuse){blocker=i;break;}
    if(blocker<0)blocker=scene.cavity_target;
    std::ofstream record(out);if(!record)throw std::runtime_error("cannot write update benchmark");
    record<<"{\n  \"width\": "<<width<<",\n  \"height\": "<<height<<",\n";
    record<<std::setprecision(12)<<"  \"scene\": \"standard\",\n  \"initial_compile_ms\": "<<initial_ms<<",\n  \"updates\": [\n";
    for(int step=0;step<3;++step){
        Primitive p=scene.primitives[blocker];
        if(p.shape==Shape::Sphere)p.center.x+=.1;
        else p.origin.x+=.1;
        const GeometryChanges changes=step==2?state.apply_geometry({}, {p}):state.apply_geometry({{blocker,p}});
        const double geometry_ms=state.last_geometry_ms,update_ms=state.last_field_ms;
        const TransportField& updated=state.field();
        start=Clock::now();const auto fresh_beams=compile_beam_field(scene);
        const auto fresh=compile_transport_field(scene,fresh_beams,true);
        const double fresh_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        const double error=update_field_error(updated,fresh);check_updated_intersections(scene);
        const Camera moving=make_look_camera(width,height,camera.origin+Vec3{.1*step,0,0},
            camera.origin+camera.forward*4,{0,1,0},27);
        TraceContext ctx{scene,beams,updated},oracle{scene,fresh_beams,fresh};RenderStats a,b;
        start=Clock::now();const auto image=render_visible_edge_field(ctx,width,height,1,a,&moving,&stale);
        const double frame_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        const auto reference=render_visible_edge_field(oracle,width,height,1,b,&moving);
        int max_byte=0;std::size_t changed_bytes=0;
        for(std::size_t i=0;i<image.size();++i){max_byte=std::max(max_byte,std::abs(int(image[i])-int(reference[i])));
            changed_bytes+=image[i]!=reference[i];}
        update_require(max_byte<=1&&!a.viewer_origin_field_reused,"edited render diverged or reused stale viewer field");
        write_ppm(out+"."+std::to_string(step)+".ppm",image,width,height);
        if(step)record<<",\n";
        record<<"    {\"step\": "<<step<<", \"kind\": \""<<(step==2?"insert":"move")<<"\", \"geometry_ms\": "<<geometry_ms
              <<", \"refit_nodes\": "<<changes.refit_nodes<<", \"update_compile_ms\": "<<update_ms
              <<", \"fresh_compile_ms\": "<<fresh_ms<<", \"reused_pairs\": "<<updated.reused_pairs
              <<", \"recomputed_pairs\": "<<updated.recomputed_pairs<<", \"reused_atlases\": "<<updated.reused_atlases
              <<", \"reused_atlas_samples\": "<<updated.reused_atlas_samples<<", \"evaluated_atlas_samples\": "<<updated.direct_atlas_samples
              <<", \"radiance_l1_error\": "<<error<<", \"certified_tail\": "<<updated.certified_tail
              <<", \"contraction_bound\": "<<updated.contraction_bound<<", \"negative_defect_channels\": "<<updated.negative_defect_channels
              <<", \"warm_waves\": "<<updated.iterations<<", \"cold_waves\": "<<fresh.iterations
              <<", \"warm_solve_ms\": "<<updated.transport_solve_ms<<", \"cold_solve_ms\": "<<fresh.transport_solve_ms
              <<", \"render_ms\": "<<frame_ms<<", \"frame_with_update_ms\": "<<geometry_ms+update_ms+frame_ms
              <<", \"maximum_byte_error\": "<<max_byte<<", \"changed_bytes\": "<<changed_bytes<<"}";
    }
    record<<"\n  ]\n}\n";std::cout<<"retained update benchmark: "<<out<<"\n";return 0;
}

int run_geometry_update_benchmark(const std::string& out){
    std::ofstream record(out);if(!record)throw std::runtime_error("cannot write geometry benchmark");
    record<<std::setprecision(12)<<"{\"benchmark\": \"retained_geometry_updates\", \"cases\": [\n";bool comma=false;
    for(int count:{10000,100000})for(int changed:{1,64,1024}){
        Scene scene;add_material(scene,{"matte",MaterialKind::Diffuse,{.7,.7,.7}});
        const int side=static_cast<int>(std::ceil(std::sqrt(double(count))));
        scene.primitives.reserve(count);
        for(int i=0;i<count;++i)add_sphere(scene,"",{.8*(i%side),.8*(i/side),-.2*(i%7)},.2,0,false);
        build_scene_bvh(scene);std::vector<double> refits,rebuilds;std::size_t nodes=0;
        for(int repeat=0;repeat<9;++repeat){std::vector<std::pair<int,Primitive>> edits;edits.reserve(changed);
            for(int j=0;j<changed;++j){int id=(j*7919+repeat*31)%count;Primitive p=scene.primitives[id];
                p.center.z+=.025;edits.push_back({id,std::move(p)});}
            auto start=Clock::now();const auto changes=edit_scene_geometry(scene,std::move(edits));
            refits.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());nodes=changes.refit_nodes;
            Scene rebuilt=scene;start=Clock::now();build_scene_bvh(rebuilt);
            rebuilds.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
            for(int i=0;i<128;++i){int target=(i*7919+repeat*31)%count;
                const Vec3 origin{.4*side,.4*side,12};Ray ray{origin,unit(scene.primitives[target].center-origin)};
                const Hit a=first_hit(scene,ray),b=first_hit(rebuilt,ray);
                update_require(a.valid==b.valid&&a.primitive==b.primitive&&std::abs(a.t-b.t)<1e-10,
                    "large scene refit differs from rebuild");}
        }
        std::sort(refits.begin(),refits.end());std::sort(rebuilds.begin(),rebuilds.end());
        if(comma)record<<",\n";comma=true;
        record<<"{\"primitives\": "<<count<<", \"edited\": "<<changed<<", \"repeats\": 9, \"refit_nodes_last\": "<<nodes
            <<", \"total_bvh_nodes\": "<<scene.bvh_nodes.size()<<", \"refit_median_ms\": "<<refits[4]
            <<", \"refit_max_ms\": "<<refits.back()<<", \"rebuild_median_ms\": "<<rebuilds[4]<<"}";
    }
    record<<"\n]}\n";std::cout<<"geometry update benchmark: "<<out<<"\n";return 0;
}
