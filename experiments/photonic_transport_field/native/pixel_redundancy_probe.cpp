#define PHOTONIC_PIXEL_REUSE_AUDIT
#define main regime_scene_program_main
#include "regime_scene_native.cpp"
#undef main

struct PixelBucket{TerminalLabel label{};double x=0,y=0,first_x=0,first_y=0;int count=0;};
std::vector<PixelBucket> pixel_buckets(const TraceContext& ctx,const Camera& camera,int px,int py){
    std::vector<PixelBucket> buckets;
    for(int sy=0;sy<16;++sy)for(int sx=0;sx<16;++sx){
        const double x=px-.5+(sx+.5)/16,y=py-.5+(sy+.5)/16;
        const Ray ray=camera_ray(camera,800,600,x,y);const Hit hit=first_hit(ctx.scene,ray);
        const TerminalLabel label{hit.primitive,visible_path_signature(ctx,ray,hit)};
        auto it=std::find_if(buckets.begin(),buckets.end(),[&](const auto& b){return b.label==label;});
        if(it==buckets.end()){buckets.push_back({label,0,0,x,y,0});it=buckets.end()-1;}
        it->x+=x;it->y+=y;++it->count;
    }
    return buckets;
}

int main(int argc,char** argv){
    const std::string path=argc>1?argv[1]:"/tmp/pixel_redundancy.json";
    const int repeats=argc>2?std::stoi(argv[2]):3;
    if(repeats<1||repeats>100)throw std::invalid_argument("invalid repeat count");
    {PixelReuseAudit audit;audit.mode=1;Hit hit;hit.primitive=1;hit.normal={0,0,1};
        auto& entry=audit.request(1,hit,{0,0,1},true,false);
        update_require(!audit.reuse(entry),"uninitialized reuse entry admitted");
        audit.record(entry,{1,2,3},false);
        update_require(audit.reuse(entry),"identical cached state missed");
        ++audit.bucket;update_require(!audit.reuse(entry),"bucket cache escaped its scope");
        audit.mode=2;update_require(audit.reuse(entry),"pixel cache failed to share across buckets");
        hit.position.x=1e-12;auto& moved=audit.request(1,hit,{0,0,1},true,false);
        update_require(!audit.reuse(moved),"nearby coordinates merged");}
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot write pixel audit");
    Scene scene=build_demonstrator_scene("aperture-canyon");scene.use_bvh=false;
    const BeamField beams=compile_beam_field(scene);const TransportField field=compile_transport_field(scene,beams,true);
    TraceContext ctx{scene,beams,field};Camera camera=make_camera(800,600);
    out<<std::setprecision(17)<<"{\"primitive_count\":"<<scene.primitives.size()<<",\"area_lights\":"<<scene.area_lights.size()
        <<",\"runs\":[";bool comma=false;
    for(const auto xy:std::array<std::array<int,2>,2>{{{275,324},{268,323}}}){
        ctx.stats=nullptr;ctx.expansion=nullptr;ctx.optical_queries=nullptr;
        const auto buckets=pixel_buckets(ctx,camera,xy[0],xy[1]);RGB reference{};bool have_reference=false;
        for(int repeat=0;repeat<repeats;++repeat)for(int slot=0;slot<4;++slot){
            // Begin each pixel with the uninstrumented reference; rotate later runs.
            const int mode=(slot+repeat)%4-1;
            PixelReuseAudit audit;audit.mode=mode;pixel_reuse_audit=mode<0?nullptr:&audit;
            TraceStats stats;ctx.stats=&stats;ExpansionAudit expansion;expansion.receivers.resize(scene.primitives.size());
            ctx.expansion=&expansion;OpticalQueryMemo queries(128);ctx.optical_queries=&queries;
            RGB value{};const auto start=Clock::now();
            for(const auto& b:buckets){++audit.bucket;double x=b.x/b.count,y=b.y/b.count;
                Ray ray=camera_ray(camera,800,600,x,y);Hit hit=optical_first_hit(ctx,ray);
                if(!(TerminalLabel{hit.primitive,visible_path_signature(ctx,ray,hit)}==b.label)){
                    x=b.first_x;y=b.first_y;ray=camera_ray(camera,800,600,x,y);hit=optical_first_hit(ctx,ray);}
                value+=trace_primary(ctx,ray,hit,nullptr)*(double(b.count)/256.);}
            const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            pixel_reuse_audit=nullptr;
            if(!have_reference){reference=value;have_reference=true;}
            update_require(value==reference,"exact surface reuse changed pixel radiance");
            update_require(audit.inconsistent==0,"same full surface key produced different RGB");
            std::uint64_t recomputed=0;for(const auto& pair:audit.entries)if(pair.second.computed>1)recomputed+=pair.second.computed-1;
            if(comma)out<<",";comma=true;
            out<<"{\"x\":"<<xy[0]<<",\"y\":"<<xy[1]<<",\"mode\":"<<mode<<",\"repeat\":"<<repeat
                <<",\"buckets\":"<<buckets.size()<<",\"ms\":"<<elapsed<<",\"identical\":true"
                <<",\"surface_requests\":"<<audit.requests<<",\"unique_surface_states\":"<<audit.entries.size()
                <<",\"unique_surface_geometry\":"<<audit.geometry.size()<<",\"surface_computations\":"<<audit.computed
                <<",\"small_memo_hits\":"<<audit.small_hits<<",\"wide_memo_hits\":"<<audit.wide_hits
                <<",\"exact_recomputations\":"<<recomputed
                <<",\"specular_calls\":"<<expansion.pixel.specular<<",\"unique_specular_geometry\":"<<audit.specular_geometry.size()
                <<",\"all_quadrature\":"<<stats.emitter_quadrature_samples
                <<",\"adjacent_same_word\":"<<audit.adjacent_same_word
                <<",\"glossy_intervals\":"<<audit.glossy_intervals
                <<",\"glossy_adjacent_same_word\":"<<audit.glossy_adjacent_same_word
                <<",\"unique_glossy_words\":"<<audit.glossy_words.size()
                <<",\"source_intervals\":"<<audit.source_intervals<<",\"unique_source_words\":"<<audit.source_words.size()
                <<",\"quadrature\":"<<expansion.pixel.quadrature<<",\"zero_quadrature\":"<<expansion.pixel.zero_quadrature
                <<",\"packets\":"<<expansion.pixel.packets<<",\"spectral_regions\":"<<expansion.pixel.regions
                <<",\"rgb\":["<<value[0]<<","<<value[1]<<","<<value[2]<<"],\"source_relationships\":[";
            bool word_comma=false;for(const auto& pair:audit.source_words){if(word_comma)out<<",";word_comma=true;
                out<<"{\"receiver\":"<<pair.first[0]<<",\"emitter\":"<<pair.first[1]<<",\"word\":[";
                for(std::size_t i=2;i<pair.first.size();++i){if(i>2)out<<",";out<<pair.first[i];}
                out<<"],\"intervals\":"<<pair.second<<",\"glossy_intervals\":"<<audit.glossy_words[pair.first]<<"}";}
            out<<"]}"<<std::flush;
            std::cerr<<xy[0]<<","<<xy[1]<<" mode="<<mode<<" ms="<<elapsed<<" requests="<<audit.requests
                <<" unique="<<audit.entries.size()<<" recomputed="<<recomputed<<" source="<<expansion.pixel.quadrature<<"\n";
        }
    }
    out<<"]}\n";
}
